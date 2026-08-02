from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import json
import logging
import math
import os
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Concatenate, Literal, TypeVar, Union, overload

from griffe import DocstringStyle
from pydantic import BaseModel as PydanticModel, ValidationError
from typing_extensions import ParamSpec

from troopai.adk.exceptions import (
    AgentToolDeferral,
    GuardrailTripwireTriggered,
    ToolGuardrailTripwireTriggered,
    ToolRetry,
)
from troopai.adk.run.context import RunContext
from troopai.adk.schemas import SchemaEnforcement, enforce_schema
from troopai.adk.schemas.function_schema import (
    FunctionSchema,
    FunctionToolSchema,
    function_schema as generate_function_schema,
)
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.tools.tool_guardrails import ToolGuardrails
from troopai.adk.types.tools import ApprovalPolicy
from troopai.adk.types.tools.tool_rate_limit import ToolRateLimit
from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.tools.tool_search import ToolSearchState

TContext = TypeVar("TContext")

ToolParams = ParamSpec("ToolParams")

ToolFunctionWithoutContext = Callable[ToolParams, Any]
ToolFunctionWithContext = Callable[Concatenate[ToolContext, ToolParams], Any]
ToolFunctionWithToolContext = Callable[Concatenate[ToolContext, ToolParams], Any]
ToolFunction = Union[
    ToolFunctionWithoutContext[ToolParams],
    ToolFunctionWithContext[ToolParams],
    ToolFunctionWithToolContext[ToolParams],
]

ToolTimeoutBehavior = Literal["error_as_result", "raise_exception"]
ToolErrorFunction = Callable[[RunContext[Any], Exception], MaybeAwaitable[str]]

ToolInvokeFunction = Callable[[ToolContext[Any], str], Awaitable[Any]]
"""Function type for tool invocation.

Takes a ToolContext and JSON input string, returns the tool result.
Must be async.
"""

ToolCacheScope = Literal["run", "process"]
ToolCacheKeyBuilder = Callable[[ToolContext[Any], str], str]

logger = logging.getLogger(__name__)


# Framework control-flow signals that must propagate out of a tool's
# ``on_invoke`` unchanged rather than being converted to an error-as-result
# string. Each is handled explicitly upstream — ``ToolRetry`` carries a retry
# hint for the LLM, ``AgentToolDeferral`` drives human-in-the-loop approval,
# and the guardrail tripwires are halt verdicts. Masking any of them behind
# the failure handler would silently drop a retry, a deferral, or a safety
# verdict.
_CONTROL_FLOW_EXCEPTIONS: tuple[type[Exception], ...] = (
    ToolRetry,
    AgentToolDeferral,
    GuardrailTripwireTriggered,
    ToolGuardrailTripwireTriggered,
)


@dataclass(frozen=True, kw_only=True)
class ToolCachePolicy:
    """Policy controlling result-cache scope, bounds, expiry, and keys."""

    scope: ToolCacheScope = "run"
    """Cache scope. ``"run"`` is isolated to one ``RunContext``; ``"process"``
    persists on the tool instance across runs."""

    max_entries: int = 128
    """Maximum entries retained before least-recently-used eviction."""

    ttl_seconds: float | None = None
    """Optional maximum entry age in seconds."""

    max_bytes: int | None = None
    """Optional approximate byte budget for retained values."""

    key_builder: ToolCacheKeyBuilder | None = None
    """Optional custom key builder receiving the ``ToolContext`` and raw JSON."""

    def __post_init__(self) -> None:
        if self.max_entries <= 0:
            raise ValueError(f"ToolCachePolicy.max_entries must be positive, got {self.max_entries}")
        if self.ttl_seconds is not None and (self.ttl_seconds <= 0 or not math.isfinite(self.ttl_seconds)):
            raise ValueError(f"ToolCachePolicy.ttl_seconds must be positive and finite, got {self.ttl_seconds}")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError(f"ToolCachePolicy.max_bytes must be positive, got {self.max_bytes}")


@dataclass
class _ToolCacheEntry:
    value: Any
    expires_at: float | None
    size: int


def default_tool_error_function(ctx: RunContext[Any], error: Exception) -> str:  # noqa: ARG001
    """The default tool error function, which just returns a generic error message."""
    return f"An error occurred while running the tool. Please try again. Error: {error!s}"


@dataclass
class FunctionTool:
    """A tool that wraps a Python function for use in an agent.

    Attributes:
        name: The name of the tool.
        description: A description of the tool's purpose.
        schema: The Pydantic model defining the tool's input schema.
        schema_enforcement: Controls how the schema is processed.
        guardrails: ``ToolGuardrails`` config holding per-phase
            (``input`` / ``output``) tool-level guardrails. ``None``
            (default) means no guardrails — fast-path skip in the
            executor.
        enabled: Whether the tool is enabled (can be dynamic).
        requires_approval: Whether this tool requires human approval (bool or callable).
        on_invoke: The tool invocation callback.
    """

    name: str
    """The name of the tool."""

    schema: FunctionToolSchema
    """The Pydantic model or JSON schema defining the tool's input parameters.
    Used for validation and to generate the JSON schema for the LLM."""

    description: str | None = None
    """A description of the tool's purpose."""

    schema_enforcement: SchemaEnforcement = SchemaEnforcement.NORMALIZED
    """Controls how the tool's JSON schema is processed before being sent to the LLM.

    - ``NONE``: The raw schema is sent as-is with no transformation.
    - ``NORMALIZED`` (default): Provider-agnostic defaults are applied (type, description,
      required) without imposing strict-mode constraints.  Use this for providers
      that do not support strict schemas.
    - ``STRICT``: Full OpenAI strict-mode compliance — all properties
      required, no additionalProperties, oneOf→anyOf, etc.  Guarantees that API
      responses strictly match the schema on supported models.
    """

    guardrails: ToolGuardrails | None = None
    """Per-phase tool-level guardrails registered on this tool.

    ``None`` (default) means no guardrails configured — the executor
    skips both phases entirely on the fast path. When set, the
    :class:`~troopai.adk.tools.tool_guardrails.ToolGuardrails` config
    holds two phase-typed lists:

    - ``guardrails.input``: validates parsed arguments before
      ``on_invoke``. A ``raise_exception`` verdict raises
      ``ToolGuardrailTripwireTriggered``; a ``reject_content`` verdict
      replaces the result the LLM sees with the rejection message.
    - ``guardrails.output``: validates the result after ``on_invoke``.
      Same verdict shape; ``reject_content`` swaps the result text
      that flows back to the LLM.
    """

    enabled: bool | Callable[[RunContext[Any]], Any] | MaybeAwaitable[bool] = True
    """Whether the tool is enabled. Either a boolean or a callable that takes the run context and returns a boolean.
    This can be used to enable/disable tools dynamically based on the run context."""

    requires_approval: ApprovalPolicy = False
    """Whether this tool requires human approval before execution.

    Can be a bool or a callable that takes a ToolContext and returns a bool
    (sync or async). If True or the callable returns True, the tool call
    will be deferred for human approval instead of being executed immediately.

    Example::

        # Static approval
        @function_tool(name="deploy", description="Deploy", requires_approval=True)
        def deploy():
            pass

        # Conditional approval
        async def require_in_prod(ctx: ToolContext) -> bool:
            return ctx.context.get("environment") == "production"

        @function_tool(name="deploy", description="Deploy", requires_approval=require_in_prod)
        def deploy():
            pass
    """

    max_result_tokens: int | None = None
    """Maximum token count for this tool's result string.

    When set, the Runner truncates any result exceeding this limit
    before inserting it into the message history. Prevents a single
    tool from bloating context — e.g. a RAG tool returning 3,000
    tokens with ``max_result_tokens=500``, over 8 remaining turns,
    saves (3000 - 500) * 8 = 20,000 input tokens.

    ``None`` (default) means no limit. **Production tools returning
    variable-size results (RAG, web search, file readers) SHOULD set
    a bound** — unbounded results are re-sent every turn until the
    context editor clears them, and that editor is itself opt-in (see
    :attr:`ContextEditingConfig.clear_tool_results`).
    """

    max_retries: int | None = None
    """LLM retry budget for this tool.

    Controls how many times the tool may fail before being disabled
    for the rest of the run.

    - ``None`` (default): No enforcement at the tool level. LLM can
      retry freely (still bounded by ``max_turns``). This default is
      load-bearing for skill-governance precedence: a tool with
      ``max_retries=None`` defers to ``SkillGovernance.max_retries``
      when the tool sits inside a governed skill. A bounded default
      (e.g. 3) would silently override governance — a hidden cost
      change for developers who configured governance explicitly.
      Developers wanting per-tool retry caps set this directly.
    - ``0``: No retries allowed. Tool disabled on first failure.
    - ``N > 0``: After N failed executions, tool is removed from the
      LLM's tool list.

    Failures include exceptions and timeouts (with error_as_result behavior).
    Does not count guardrail rejections or HITL deferrals."""

    timeout: float | None = None
    """Per-tool timeout in seconds. Wraps execution with asyncio.wait_for().
    Must be positive and finite."""

    timeout_behavior: ToolTimeoutBehavior = "error_as_result"
    """What happens on timeout:
    - "error_as_result": Return error message as tool result (LLM sees it, can retry).
    - "raise_exception": Raise ToolTimeoutError (halts execution)."""

    timeout_error: ToolErrorFunction | None = None
    """Custom function to generate timeout error message.
    Receives (RunContext, ToolTimeoutError) → str. Only used with "error_as_result"."""

    on_invoke: ToolInvokeFunction | None = None
    """The tool invocation callback.

    Takes (ToolContext, raw_args_json_str) and returns the result string.
    Set by the @function_tool decorator or passed directly.
    """

    execution_aware: bool = False
    """Whether this tool expects :class:`ExecutionAwareToolContext`.

    When ``True``, the Runner constructs an
    :class:`ExecutionAwareToolContext` with read-only execution state
    snapshots (usage, turn count, message count, token estimate)
    instead of a plain :class:`ToolContext`.

    Set automatically by ``@function_tool`` when the function's first
    parameter is annotated as ``ExecutionAwareToolContext``, or manually
    when constructing ``FunctionTool`` directly.
    """

    history_aware: bool = False
    """Whether this tool expects :class:`HistoryAwareToolContext`.

    When ``True``, the Runner constructs a
    :class:`HistoryAwareToolContext` with a read-only snapshot of the
    conversation history as Layer 3 RunItems, in addition to all
    execution state from :class:`ExecutionAwareToolContext`.

    Implies ``execution_aware=True`` (enforced in ``__post_init__``).

    Set automatically by ``@function_tool`` when the function's first
    parameter is annotated as ``HistoryAwareToolContext``, or manually
    when constructing ``FunctionTool`` directly.
    """

    cache: bool | ToolCachePolicy = False
    """Whether to cache tool results by input arguments.

    ``False`` disables caching. ``True`` enables the safe default
    :class:`ToolCachePolicy` (run-scoped, bounded LRU). Pass an explicit
    policy with ``scope="process"`` for cache reuse across runs. Cache
    keys are canonical JSON strings unless a policy supplies
    ``key_builder``. Output guardrails and post-processing still apply to
    cached results.
    """

    cache_function: Callable[[str, str], bool] | None = None
    """Conditional cache function for selective caching.

    Called with ``(input_args_json, result_string)`` after tool execution.
    Returns ``True`` to cache, ``False`` to skip caching.  Only used when
    caching is enabled.  When ``None`` (default), all successful results are cached.

    Example: cache successful responses but not errors::

        cache_function=lambda args, result: "error" not in result.lower()
    """

    response_format: str = "text"
    """How the tool's return value is interpreted.

    - ``"text"`` (default): Return value is stringified and sent to the LLM.
    - ``"content_and_artifact"``: Return value must be a ``tuple[str, Any]``.
      The first element (content) is sent to the LLM; the second (artifact)
      is stored on ``FunctionToolCallResult.artifact`` for the application.
      Useful for RAG tools, chart generators, etc. where the LLM needs a
      summary but the app needs the full data.

    Example::

        @function_tool(name="rag", response_format="content_and_artifact")
        def rag_search(query: str) -> tuple[str, list[Document]]:
            docs = retrieve(query)
            return f"Found {len(docs)} results", docs
    """

    return_direct: bool = False
    """Whether this tool's result should become the final output directly.

    When ``True``, the tool's result skips LLM post-processing and becomes
    the run's ``final_output`` immediately.  Saves one LLM round-trip for
    tools that produce polished, user-ready output (formatted reports,
    generated images, pre-built responses).

    Equivalent to per-tool ``tool_use_behavior="stop_on_first_tool"`` but
    only triggers for this specific tool.
    """

    prepare: Callable | None = None
    """Dynamic tool modifier called before each LLM step.

    A callable that receives ``(RunContext, FunctionTool)`` and returns
    a modified ``FunctionTool`` (via ``dataclasses.replace()``) or
    ``None`` to exclude the tool for that step.

    More powerful than ``enabled`` — can modify description, restrict
    parameters, or adapt the tool contextually::

        from dataclasses import replace

        def prepare_search(ctx, tool):
            remaining = ctx.context.get("api_calls_remaining", 0)
            if remaining <= 0:
                return None  # Exclude tool
            return replace(
                tool,
                description=f"Search (API calls remaining: {remaining})",
            )

        @function_tool(name="search", prepare=prepare_search)
        def search(query: str) -> str: ...
    """

    requires_env: tuple[str, ...] = ()
    """Environment variables that MUST be set (and non-empty) for this
    tool to function.

    Validated by ``Agent.__post_init__`` — any missing variable raises
    :class:`troopai.adk.exceptions.ToolDependencyError` listing every
    unsatisfied requirement across every tool. Failing fast at agent
    construction surfaces misconfiguration before the first LLM call,
    rather than as a cryptic error mid-turn.

    Example::

        @function_tool(name="slack_notify", requires_env=("SLACK_TOKEN",))
        def slack_notify(message: str) -> str: ...
    """

    requires_packages: tuple[str, ...] = ()
    """Python packages (PEP 508 requirement strings) that MUST be
    importable for this tool to function.

    Each entry is a requirement spec parseable by
    ``packaging.requirements.Requirement`` —
    e.g. ``"slack-sdk"``, ``"slack-sdk>=3.0"``,
    ``"requests>=2.30,<3"``. Validation checks both that the
    distribution is installed AND that the installed version satisfies
    the specifier.

    Validated alongside ``requires_env`` in ``Agent.__post_init__``.

    Example::

        @function_tool(
            name="slack_notify",
            requires_packages=("slack-sdk>=3.0",),
        )
        def slack_notify(message: str) -> str: ...
    """

    defer_loading: bool = False
    """Whether to hide this tool from the LLM until explicitly revealed.

    When ``True``, the tool is filtered out of the per-step tool list
    that ``build_tools()`` emits to the LLM. It becomes visible only
    after ``build_tool_search()`` (or any equivalent reveal mechanism)
    adds the tool's name to a per-run ``revealed`` set.

    Use this to keep oversized tool registries (50+ tools) off the
    system prompt: the LLM sees only a small set of core tools plus a
    ``tool_search``-style discovery tool, then reveals specialised
    tools on demand. Saves the per-turn token cost of every unused
    tool definition.

    The ``Runner`` resets the revealed set at the start of each
    ``Runner.arun()`` call, so every run begins with a clean slate
    regardless of what prior runs revealed.
    """

    rate_limit: ToolRateLimit | None = None
    """Optional sliding-window rate limit for this tool.

    When set, the executor checks the limit before invocation. On
    saturation:

    - ``behavior="wait"`` (default): the executor sleeps until a slot
      opens, keeping the LLM unaware of throttling.
    - ``behavior="error"``: the executor returns a rate-limit error
      result the LLM can react to in its next turn.

    State is stored on the tool instance via :meth:`acquire_rate_slot`
    and persists across invocations within the same process. Use one
    ``FunctionTool`` instance per limited backend so the window is
    enforced consistently.

    Example::

        from troopai.adk.tools import ToolRateLimit, function_tool

        @function_tool(
            name="search_api",
            description="Query the search API.",
            rate_limit=ToolRateLimit(rpm=30),
        )
        def search_api(query: str) -> str: ...
    """

    streaming: bool = False
    """Whether this tool yields incremental progress events.

    When ``True``, ``on_invoke`` MUST return
    ``AsyncIterator[ToolStreamEvent]`` rather than a single value.
    The executor drains the iterator, surfacing each non-``"done"``
    event to consumers of ``Runner.arun(stream=True)`` as a
    ``RunItemType.TOOL_PARTIAL_OUTPUT`` event. The LLM still sees
    exactly one tool-result message — the value carried on the
    terminal ``"done"`` event.

    Approval gates: ``streaming=True`` coexists with
    ``requires_approval``. The HITL gate runs first; the iterator
    only starts after approval is granted (and on resumption).
    Approval is a gate; streaming is the body.

    Incoherent with: ``cache=True`` (cache stores a single value, not
    a stream), ``cache_function`` (same reason),
    ``response_format == "content_and_artifact"`` (artifact channel
    needs the full payload), ``return_direct=True`` (return-direct
    semantics don't apply to a streaming intermediary). All four
    combinations raise ``ValueError`` at construction.

    Running a streaming tool under the non-streaming path
    (``Runner.arun()`` without ``stream=True``) drains the iterator
    silently and emits a ``logger.warning`` — partial events are
    discarded but the final value still flows back to the LLM.
    """

    metadata: Mapping[str, str] = field(default_factory=dict)
    """Arbitrary string-valued labels for tracing / telemetry.

    Useful for attaching tool-class / cost-tier / owner labels that
    flow through observability without participating in execution
    semantics. NOT shown to the LLM. NOT validated for shape — the
    expected use is simple ``{"owner": "platform-team",
    "cost_tier": "expensive"}`` style dictionaries.
    """

    tool_namespace: str | None = None
    """Optional namespace prefix for grouping tools in provider UIs.

    Currently surfaced through the OpenAI Responses tool-naming
    boundary as a flat prefix; falls through silently on providers
    that don't carry a namespace concept.
    """

    # Internal state (not constructor fields)
    _cache: OrderedDict[str, _ToolCacheEntry] = field(init=False, repr=False, default_factory=OrderedDict)
    _agent: Agent | None = field(init=False, repr=False, default=None)
    _rate_state: deque[float] = field(init=False, repr=False, default_factory=deque)
    _rate_lock: asyncio.Lock | None = field(init=False, repr=False, default=None)
    _search_state: ToolSearchState | None = field(init=False, repr=False, default=None)
    """When this FunctionTool is the product of ``build_tool_search()``,
    holds the closure state that owns the deferred-tool registry and
    the per-run ``revealed`` set. ``None`` for ordinary tools. Access
    via ``get_search_state()`` — never read this attribute directly
    from another module."""

    def __post_init__(self) -> None:
        # history_aware implies execution_aware
        if self.history_aware and not self.execution_aware:
            object.__setattr__(self, "execution_aware", True)
        if self.max_result_tokens is not None and self.max_result_tokens <= 0:
            raise ValueError(f"max_result_tokens must be positive, got {self.max_result_tokens}")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.timeout is not None and (self.timeout <= 0 or not math.isfinite(self.timeout)):
            raise ValueError(f"timeout must be positive and finite, got {self.timeout}")
        # ``ToolRateLimit`` is frozen and validates rpm > 0 in its own
        # __post_init__, so an instance with a non-positive rpm cannot
        # reach this point. No re-check needed here.
        # Eagerly initialise the rate-limit lock so that clones (produced by
        # clone()) always inherit a valid shared Lock object.  Lazy init would
        # leave _rate_lock as None on un-used tools; if two concurrent
        # coroutines call acquire_rate_slot() on two different clones of such a
        # tool, each would create its own independent Lock, breaking mutual
        # exclusion over the shared _rate_state deque.
        if self.rate_limit is not None:
            object.__setattr__(self, "_rate_lock", asyncio.Lock())
        if self.streaming:
            # Each rejected combination is incoherent: cache stores a
            # single value not a stream; artifact channel needs the
            # full payload; return_direct doesn't make sense for a
            # streaming intermediary. Reject at construction so the
            # author sees the conflict immediately, not at first call.
            if self.cache:
                raise ValueError(
                    f"FunctionTool '{self.name}': streaming=True is incoherent with cache=True "
                    "(cache stores a single value, not a stream)."
                )
            if self.cache_function is not None:
                raise ValueError(
                    f"FunctionTool '{self.name}': streaming=True is incoherent with cache_function "
                    "(cache_function is only consulted when cache=True)."
                )
            if self.response_format == "content_and_artifact":
                raise ValueError(
                    f"FunctionTool '{self.name}': streaming=True is incoherent with "
                    "response_format='content_and_artifact' (artifact channel needs the full payload)."
                )
            if self.return_direct:
                raise ValueError(
                    f"FunctionTool '{self.name}': streaming=True is incoherent with return_direct=True "
                    "(return_direct semantics don't apply to a streaming intermediary)."
                )

    # ------------------------------------------------------------------
    # Internal metadata accessors
    # ------------------------------------------------------------------

    def get_delegate_agent(self) -> Agent | None:
        """Return the Agent this tool delegates to, or ``None``.

        Set by ``Agent.as_tool()`` — not for direct use.
        """
        return self._agent

    def get_search_state(self) -> ToolSearchState | None:
        """Return the closure state set by ``build_tool_search()``, or ``None``.

        Used by ``build_tools()`` to discover the deferred-tool registry
        and the per-run ``revealed`` set this tool maintains. Returns
        ``None`` for ordinary FunctionTools.
        """
        return self._search_state

    def set_search_state(self, state: ToolSearchState | None) -> None:
        """Set the closure state produced by ``build_tool_search()``.

        Internal use only — called by the tool-search factory to attach
        the deferred-tool registry and per-run ``revealed`` set onto the
        materialised search tool. External code MUST NOT touch the
        backing field directly.

        Uses ``object.__setattr__`` to match the established
        internal-state-write convention in ``clone()`` (other private
        slots like ``_agent`` / ``_cache`` use the same pattern). This
        also stays correct if ``FunctionTool`` is ever made frozen.
        """
        object.__setattr__(self, "_search_state", state)

    def resolve_cache_policy(self) -> ToolCachePolicy | None:
        """Return the effective cache policy, or ``None`` when disabled."""
        if isinstance(self.cache, ToolCachePolicy):
            return self.cache
        if self.cache is True:
            return ToolCachePolicy()
        return None

    def get_cached(
        self,
        raw_args: str,
        tool_ctx: ToolContext[Any] | None = None,
        run_context: RunContext[Any] | None = None,
    ) -> Any | None:
        """Return cached result for these args, or ``None`` on miss."""
        policy = self.resolve_cache_policy()
        if policy is None:
            return None
        cache = self._get_cache_store(policy, run_context, create=False)
        if cache is None:
            return None
        key = self._build_cache_key(policy, raw_args, tool_ctx)
        if key is None:
            return None
        entry = cache.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= time.monotonic():
            del cache[key]
            return None
        cache.move_to_end(key)
        return entry.value

    def set_cached(
        self,
        raw_args: str,
        result: Any,
        tool_ctx: ToolContext[Any] | None = None,
        run_context: RunContext[Any] | None = None,
    ) -> None:
        """Store a result in the configured cache."""
        policy = self.resolve_cache_policy()
        if policy is None:
            return
        cache = self._get_cache_store(policy, run_context, create=True)
        if cache is None:
            return
        key = self._build_cache_key(policy, raw_args, tool_ctx)
        if key is None:
            return
        expires_at = time.monotonic() + policy.ttl_seconds if policy.ttl_seconds is not None else None
        cache[key] = _ToolCacheEntry(value=result, expires_at=expires_at, size=self._approx_cache_size(result))
        cache.move_to_end(key)
        self._evict_cache_entries(cache, policy)

    def clear_cache(self) -> None:
        """Clear process-scoped cached results."""
        self._cache.clear()

    def _get_cache_store(
        self,
        policy: ToolCachePolicy,
        run_context: RunContext[Any] | None,
        *,
        create: bool,
    ) -> OrderedDict[str, _ToolCacheEntry] | None:
        if policy.scope == "process":
            return self._cache
        namespace = id(self._cache)
        if run_context is None:
            return None
        existing = run_context.get_tool_cache(namespace)
        if isinstance(existing, OrderedDict):
            return existing
        if not create:
            return None
        cache: OrderedDict[str, _ToolCacheEntry] = OrderedDict()
        run_context.set_tool_cache(namespace, cache)
        return cache

    def _build_cache_key(
        self,
        policy: ToolCachePolicy,
        raw_args: str,
        tool_ctx: ToolContext[Any] | None,
    ) -> str | None:
        if policy.key_builder is not None:
            if tool_ctx is None:
                return None
            return policy.key_builder(tool_ctx, raw_args)
        try:
            parsed = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError:
            return raw_args
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _approx_cache_size(value: Any) -> int:
        try:
            encoded = json.dumps(value, default=str, separators=(",", ":")).encode()
        except (TypeError, ValueError):
            encoded = repr(value).encode()
        return len(encoded)

    @staticmethod
    def _evict_cache_entries(
        cache: OrderedDict[str, _ToolCacheEntry],
        policy: ToolCachePolicy,
    ) -> None:
        while len(cache) > policy.max_entries:
            cache.popitem(last=False)
        if policy.max_bytes is None:
            return
        total = sum(entry.size for entry in cache.values())
        while total > policy.max_bytes and len(cache) > 0:
            _, removed = cache.popitem(last=False)
            total -= removed.size

    def clone(
        self,
        *,
        name: str | None = None,
        on_invoke: ToolInvokeFunction | None = None,
    ) -> FunctionTool:
        """Return a clone of this tool with internal state preserved.

        Used by toolset materialisation (renaming) and middleware
        wrapping (replacing ``on_invoke`` with a chained version).
        Internal slots — the delegate-agent reference set by
        ``Agent.as_tool()``, the result cache, the rate-limit window
        state, and the deferred-tool reveal closure — are preserved
        by reference so downstream behaviour is identical.

        Cache-dict and rate-state references are **shared** with the
        original on purpose: cache hits and rate-limit windows must
        follow the underlying tool, not the surface name. Two clones
        produced from the same tool therefore share both caches —
        which is correct when both clones represent the same backing
        operation (the typical case) and a deliberate constraint
        otherwise.

        Args:
            name: Override the cloned tool's ``name`` field. ``None``
                keeps the original name.
            on_invoke: Override the cloned tool's ``on_invoke``
                callable (e.g. a middleware-wrapped version).
                ``None`` keeps the original.

        Returns:
            A new ``FunctionTool`` with the requested overrides
            applied and internal state copied from ``self``.
        """
        replacements: dict[str, Any] = {}
        if name is not None and name != self.name:
            replacements["name"] = name
        if on_invoke is not None and on_invoke is not self.on_invoke:
            replacements["on_invoke"] = on_invoke
        if len(replacements) == 0:
            return self
        cloned = dataclasses.replace(self, **replacements)
        # Internal-state copy. Sharing references — see docstring above
        # for the cache / rate-state contract.
        object.__setattr__(cloned, "_agent", self._agent)
        object.__setattr__(cloned, "_cache", self._cache)
        object.__setattr__(cloned, "_rate_state", self._rate_state)
        object.__setattr__(cloned, "_rate_lock", self._rate_lock)
        object.__setattr__(cloned, "_search_state", self._search_state)
        return cloned

    def get_json_schema(self) -> dict[str, Any]:
        """Generate the JSON schema for this tool, applying schema enforcement.

        This is used when sending the tool definition to the LLM API.

        Returns:
            The JSON schema dict for this tool's parameters.
        """
        # Get the raw JSON schema from the Pydantic model
        if isinstance(self.schema, type) and issubclass(self.schema, PydanticModel):
            json_schema = self.schema.model_json_schema()
        elif isinstance(self.schema, dict):
            # Deep-copy so that enforce_schema's in-place mutations (for
            # STRICT/COMPACT modes) do not permanently alter self.schema.
            json_schema = copy.deepcopy(self.schema)
        else:
            raise TypeError(f"Unexpected schema type: {type(self.schema)}")

        return enforce_schema(json_schema, self.schema_enforcement)

    async def check_enabled(self, context: RunContext[Any] | None = None) -> bool:
        """Check if this tool is enabled for the given run context.

        Args:
            context: The run context. Passed to callable ``enabled`` values;
                ignored when ``enabled`` is a plain bool.

        Returns:
            True if the tool should be included in the LLM's tool list.
        """
        if callable(self.enabled):
            if context is None:
                return True
            result = self.enabled(context)
            if inspect.isawaitable(result):
                return bool(await result)
            return bool(result)
        # ``enabled`` is part of ``MaybeAwaitable[bool]``, so it may be a bare
        # awaitable (not a callable). ``bool(coroutine)`` is always truthy, so
        # awaiting first is the only way to read the intended value.
        if inspect.isawaitable(self.enabled):
            return bool(await self.enabled)
        return bool(self.enabled)

    async def check_requires_approval(self, ctx: ToolContext) -> bool:
        """Check if this tool requires approval for the given context.

        Args:
            ctx: The tool context for this invocation.

        Returns:
            True if approval is required, False otherwise.
        """
        if callable(self.requires_approval):
            result = self.requires_approval(ctx)
            if inspect.isawaitable(result):
                return await result
            return result
        return self.requires_approval

    async def acquire_rate_slot(self) -> bool:
        """Acquire a rate-limit slot for this tool.

        Returns ``True`` when the call may proceed. Returns ``False``
        when:

        - ``rate_limit.behavior == "error"`` AND the window is saturated
          — caller surfaces the error to the LLM.
        - ``rate_limit.behavior == "wait"`` AND ``max_wait_seconds`` is
          set AND the next sleep would exceed the cap — falls back to
          error semantics so a single acquire cannot block indefinitely.

        Under ``behavior == "wait"`` (no cap) the coroutine sleeps as
        long as needed, then returns ``True``.

        No-op (returns ``True`` immediately) when ``rate_limit`` is
        ``None``.

        The asyncio Lock is allocated eagerly in ``__post_init__`` for
        tools that configure a ``rate_limit``. This ensures clones share
        the same Lock object and mutual exclusion over the shared
        ``_rate_state`` deque is preserved across concurrent callers.

        Concurrent waiters under parallel tool execution: each waiter
        sleeps independently. With N waiters all blocked on the same
        saturated window, the worst-case wall-clock hold is roughly
        ``N * window``, not ``window``. Set ``max_wait_seconds`` to cap
        this in production deployments.
        """
        if self.rate_limit is None:
            return True

        # _rate_lock is guaranteed non-None when rate_limit is set:
        # __post_init__ initialises it eagerly to avoid a race condition in
        # clone() scenarios where two concurrent coroutines could each create
        # an independent Lock over the shared _rate_state deque.
        lock = self._rate_lock
        if lock is None:
            raise RuntimeError("FunctionTool rate-limit lock missing despite configured rate_limit")

        rpm = self.rate_limit.rpm
        max_wait = self.rate_limit.max_wait_seconds
        window = 60.0
        accumulated_sleep = 0.0
        while True:
            async with lock:
                now = time.monotonic()
                # Drop timestamps older than the window
                while len(self._rate_state) > 0 and now - self._rate_state[0] >= window:
                    self._rate_state.popleft()
                if len(self._rate_state) < rpm:
                    self._rate_state.append(now)
                    return True
                retry_after = window - (now - self._rate_state[0])

            if self.rate_limit.behavior == "error":
                logger.info(
                    "Tool '%s' rate-limit saturated (rpm=%d); returning error to LLM",
                    self.name,
                    rpm,
                )
                return False

            if max_wait is not None and accumulated_sleep + retry_after > max_wait:
                logger.info(
                    "Tool '%s' rate-limit wait would exceed cap (%.3fs); returning error to LLM",
                    self.name,
                    max_wait,
                )
                return False

            logger.debug(
                "Tool '%s' rate-limit saturated (rpm=%d); sleeping %.3fs",
                self.name,
                rpm,
                retry_after,
            )
            sleep_for = max(retry_after, 0.0)
            await asyncio.sleep(sleep_for)
            accumulated_sleep += sleep_for

    def validate_dependencies(self) -> list[str]:
        """Return the list of unsatisfied runtime dependencies for this tool.

        Walks ``requires_env`` and ``requires_packages`` and returns one
        entry per unsatisfied requirement, prefixed with ``env:`` or
        ``package:`` so the caller can render a clear error message.
        An empty list means every declared dependency is healthy.

        Env-var validation: an entry is considered unsatisfied when
        the variable is unset OR set to an empty string.

        Package validation: each entry is parsed as a PEP 508 requirement
        string. The tool is considered to depend on a distribution that
        is installed AND whose installed version satisfies the optional
        version specifier. Unparseable specs (``InvalidRequirement``) are
        treated as missing — the developer should fix the spec.
        """
        missing: list[str] = []

        for env_var in self.requires_env:
            value = os.environ.get(env_var, "")
            if len(value) == 0:
                missing.append(f"env:{env_var}")

        for spec in self.requires_packages:
            if not _package_satisfied(spec):
                missing.append(f"package:{spec}")

        return missing


def _package_satisfied(spec: str) -> bool:
    """Return True if the PEP 508 requirement ``spec`` is satisfied.

    Module-private. ``packaging`` and ``importlib.metadata`` are imported
    lazily so the cost stays off the module-load path; both are called
    only when an agent declares ``requires_packages``.

    URL-based specs (``"pkg @ git+https://..."``) are treated as
    unsatisfied — provenance verification is a pip-level concern that
    ``importlib.metadata`` cannot perform, so a silent pass would
    misrepresent the check.
    """
    import importlib.metadata

    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        # Fallback when ``packaging`` is unavailable (highly unlikely;
        # it is a transitive dep of pydantic / litellm). Strip any PEP
        # 508 environment marker (``; python_version >= '3.8'``) and
        # version operators so the bare name reaches ``version()``.
        clean = spec.split(";")[0]
        bare = clean.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
        try:
            importlib.metadata.version(bare)
        except importlib.metadata.PackageNotFoundError:
            return False
        return True

    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return False

    if req.url is not None:
        logger.debug(
            "URL spec %r not verifiable via importlib.metadata; treating as unsatisfied",
            spec,
        )
        return False

    try:
        installed = importlib.metadata.version(canonicalize_name(req.name))
    except importlib.metadata.PackageNotFoundError:
        return False

    if len(req.specifier) == 0:
        return True

    return req.specifier.contains(installed, prereleases=True)


def _extract_json_decode_error(error: Exception) -> str | None:
    """Extract a JSON decode error message if the exception is related to JSON parsing."""
    if isinstance(error, json.JSONDecodeError):
        return str(error)
    return None


class _ArgumentError(Exception):
    """Marks a tool-argument parse/validation failure.

    Raised only by the argument-preparation stages of the ``on_invoke``
    wrapper (JSON decode, schema validation). It lets the wrapper route
    malformed LLM input to graceful error-as-result degradation — the tool
    body never ran, so it is never a tool *failure* the executor should count
    or re-raise — while genuine tool-body exceptions propagate untouched.
    """


async def _apply_error_function(
    error_function: ToolErrorFunction,
    tool_name: str,
    ctx: ToolContext,
    error: Exception,
) -> Any:
    """Turn a tool failure into an error-as-result string via the failure handler.

    Falls back to a generic message (logged) when the handler itself raises.

    Args:
        error_function: The configured tool-failure handler.
        tool_name: Name of the failing tool, for the log line.
        ctx: The tool context passed through to the handler.
        error: The exception to render.

    Returns:
        The handler's string, or a generic fallback if the handler raises.
    """
    try:
        # ``ToolErrorFunction`` annotates its context parameter as
        # ``RunContext``; at this call site we only have a ``ToolContext``.
        # The runtime contract is that error handlers read ``ctx.context``
        # and the exception, both available on a ``ToolContext``. Widen to
        # ``Callable[..., Any]`` to document that this is a deliberate shape
        # decision, not an unchecked mismatch.
        error_dispatch: Callable[..., Any] = error_function
        result = error_dispatch(ctx, error)
        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as err_fn_exc:
        # If error function itself fails, log and return a generic message
        logger.warning(
            "Tool %r error function raised: %s",
            tool_name,
            err_fn_exc,
            exc_info=True,
        )
        json_error = _extract_json_decode_error(error)
        if json_error is not None:
            return (
                f"An error occurred while parsing tool arguments. Please try again with valid JSON. Error: {json_error}"
            )
        return f"An error occurred while running the tool. Please try again. Error: {error!s}"


def _create_invoke_wrapper(
    func: ToolFunction,
    func_schema: FunctionSchema,
    error_function: ToolErrorFunction,
) -> ToolInvokeFunction:
    """Create the on_invoke_tool wrapper for a function.

    This wrapper handles:
    - Parsing JSON input
    - Validating input against Pydantic model
    - Converting validated data to function arguments using to_call_args()
    - Calling the function with appropriate context
    - Handling async/sync functions
    - Error handling with failure_error_function
    """

    async def _on_invoke_tool_impl(ctx: ToolContext, input: str) -> Any:
        # Stage 1: Parse JSON string. A malformed payload is the wrapper's own
        # input concern — the tool body never ran — so it is raised as an
        # ``_ArgumentError`` for graceful degradation, never a tool failure.
        try:
            json_data: dict[str, Any] = json.loads(input) if len(input) > 0 else {}
        except json.JSONDecodeError as e:
            raise _ArgumentError(f"Invalid JSON input for tool {func_schema.name}: {e}") from e

        # Stage 2: Validate against Pydantic model
        if not isinstance(func_schema.schema, type):
            raise _ArgumentError(f"Expected Pydantic model class, got {type(func_schema.schema)}")
        try:
            parsed = func_schema.schema(**json_data) if json_data else func_schema.schema()
        except ValidationError as e:
            raise _ArgumentError(f"Invalid input for tool {func_schema.name}: {e}") from e

        # Stage 3: Convert to call arguments
        args, kwargs_dict = func_schema.to_call_args(parsed)

        # Stage 4: Call function with appropriate context and arguments.
        # ``ToolFunction`` is a union over three callable shapes — with
        # ``RunContext``, with ``ToolContext``, or no context at all — and
        # dispatch is selected at runtime by ``func_schema.takes_context``.
        # The type checker can't narrow a callable union by an external
        # boolean flag, so we widen to ``Callable[..., Any]`` locally.
        # This is a legitimate widening (every ToolFunction member
        # satisfies ``Callable[..., Any]``), not a silenced mismatch.
        dispatch: Callable[..., Any] = func
        if inspect.isasyncgenfunction(func):
            # Streaming tool: calling the async-gen function returns
            # the async generator object directly. The executor's
            # terminal drains it and forwards chunks to the stream
            # sink; awaiting here would TypeError.
            if func_schema.takes_context:
                return dispatch(ctx, *args, **kwargs_dict)
            return dispatch(*args, **kwargs_dict)
        if inspect.iscoroutinefunction(func):
            if func_schema.takes_context:
                return await dispatch(ctx, *args, **kwargs_dict)
            return await dispatch(*args, **kwargs_dict)
        # Sync tool body: run it in a worker thread. Invoking it inline
        # would block the event loop for the whole call, freezing every
        # other concurrent tool/turn and — because the executor guards
        # the await with ``asyncio.wait_for()`` — defeating the per-tool
        # timeout, which can only fire while control is back on the loop.
        if func_schema.takes_context:
            return await asyncio.to_thread(dispatch, ctx, *args, **kwargs_dict)
        return await asyncio.to_thread(dispatch, *args, **kwargs_dict)

    async def on_invoke_tool(ctx: ToolContext, input: str) -> Any:
        try:
            return await _on_invoke_tool_impl(ctx, input)
        except _ArgumentError as e:
            # Malformed LLM input: the tool body never ran, so always degrade
            # to a helpful string the model can correct — never a tool failure.
            return await _apply_error_function(error_function, func_schema.name, ctx, e)
        except _CONTROL_FLOW_EXCEPTIONS:
            # Framework control-flow signals (retry hint, HITL deferral,
            # guardrail verdict) are handled explicitly upstream, so let them
            # propagate rather than masking them behind the failure handler.
            raise
        except Exception as e:
            # A genuine tool-body failure. With the framework-default handler,
            # propagate so the executor owns error policy — ``fail_on_tool_error``,
            # retry-budget accounting, and the configurable ``tool_execution_error``
            # message — all unreachable while this wrapper swallowed every
            # exception into a string. A developer-supplied handler instead keeps
            # the error-as-result contract.
            if error_function is default_tool_error_function:
                raise
            return await _apply_error_function(error_function, func_schema.name, ctx, e)

    return on_invoke_tool


# Overloads let type checkers narrow the return type based on call shape:
# bare-decorator application (``@function_tool``) returns ``FunctionTool``
# directly, while parameterised application (``@function_tool(name=...)``)
# returns the decorator callable. Without this, the combined return type
# ``FunctionTool | Callable[...]`` triggers ``reportCallIssue`` at every
# decorator site because pyright sees ``FunctionTool`` as non-callable.


@overload
def function_tool(function: ToolFunction, /) -> FunctionTool: ...


@overload
def function_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parse_docstring: bool = True,
    docstring_style: DocstringStyle | None = None,
    on_tool_call_fails: ToolErrorFunction = ...,
    guardrails: ToolGuardrails | None = None,
    enabled: bool | Callable[[RunContext[Any]], Any] = True,
    requires_approval: ApprovalPolicy = False,
    max_result_tokens: int | None = None,
    max_retries: int | None = None,
    timeout: float | None = None,
    timeout_behavior: Literal["error_as_result", "raise_exception"] = "error_as_result",
    timeout_error_function: ToolErrorFunction | None = None,
    schema_enforcement: SchemaEnforcement = SchemaEnforcement.NORMALIZED,
    cache: bool | ToolCachePolicy = False,
    cache_function: Callable[[str, str], bool] | None = None,
    response_format: str = "text",
    return_direct: bool = False,
    prepare: Callable | None = None,
    requires_env: tuple[str, ...] = (),
    requires_packages: tuple[str, ...] = (),
    rate_limit: ToolRateLimit | None = None,
    max_calls_per_minute: int | None = None,
    defer_loading: bool = False,
    streaming: bool = False,
) -> Callable[[ToolFunction], FunctionTool]: ...


def function_tool(
    function: ToolFunction | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parse_docstring: bool = True,
    docstring_style: DocstringStyle | None = None,
    on_tool_call_fails: ToolErrorFunction = default_tool_error_function,
    guardrails: ToolGuardrails | None = None,
    enabled: bool | Callable[[RunContext[Any]], Any] = True,
    requires_approval: ApprovalPolicy = False,
    max_result_tokens: int | None = None,
    max_retries: int | None = None,
    timeout: float | None = None,
    timeout_behavior: Literal["error_as_result", "raise_exception"] = "error_as_result",
    timeout_error_function: ToolErrorFunction | None = None,
    schema_enforcement: SchemaEnforcement = SchemaEnforcement.NORMALIZED,
    cache: bool | ToolCachePolicy = False,
    cache_function: Callable[[str, str], bool] | None = None,
    response_format: str = "text",
    return_direct: bool = False,
    prepare: Callable | None = None,
    requires_env: tuple[str, ...] = (),
    requires_packages: tuple[str, ...] = (),
    rate_limit: ToolRateLimit | None = None,
    max_calls_per_minute: int | None = None,
    defer_loading: bool = False,
    streaming: bool = False,
) -> FunctionTool | Callable[[ToolFunction], FunctionTool]:
    """Create a FunctionTool from a function.

    This decorator creates a FunctionTool that wraps a Python function.
    The function can optionally accept a ToolContext or RunContext as its
    first parameter.

    Args:
        function: The function to wrap (if using decorator without parentheses).
        name: The name of the tool.
        description: A description of the tool's purpose.
        parse_docstring: Whether to parse the docstring for parameter descriptions.
        docstring_style: The docstring style ('google', 'numpy', 'sphinx'). Auto-detected if None.
        schema_enforcement: How to enforce the schema.
        on_tool_call_fails: Function to handle errors.
        guardrails: ``ToolGuardrails`` config holding ``input`` and
            ``output`` phase-typed lists. Default ``None`` skips both
            guardrail phases on the executor's fast path.
        enabled: Whether the tool is enabled (can be dynamic).
        requires_approval: Whether this tool requires human approval (bool or callable).
        max_result_tokens: Max tokens for this tool's result. Truncated by Runner if exceeded.
        max_retries: LLM retry budget (None=no limit, 0=no retries, N=N retries).
        timeout: Timeout in seconds for this tool's execution.
        timeout_behavior: What happens on timeout ("error_as_result" or "raise_exception").
        timeout_error_function: Custom function to generate timeout error message.
        cache: Whether to cache results by input arguments.
        cache_function: Conditional cache: ``(args, result) -> bool``.
        response_format: ``"text"`` or ``"content_and_artifact"`` (dual return).
        return_direct: If True, tool result becomes final output immediately.
        prepare: Dynamic tool definition modifier per LLM step.
        requires_env: Environment variables required for this tool to function.
            Validated at agent construction; missing vars raise
            :class:`troopai.adk.exceptions.ToolDependencyError`.
        requires_packages: PEP 508 package requirements (e.g.
            ``("slack-sdk>=3.0",)``) required for this tool to function.
            Validated at agent construction.
        rate_limit: Sliding-window rate-limit config. Mutually exclusive
            with ``max_calls_per_minute``.
        max_calls_per_minute: Shorthand for
            ``rate_limit=ToolRateLimit(rpm=N)``. Use ``rate_limit=`` for
            non-default behavior (``"error"`` instead of ``"wait"``).
        defer_loading: Hide the tool from the LLM's per-step tool list
            until ``build_tool_search()`` reveals it.
        streaming: When ``True``, the wrapped function MUST return
            ``AsyncIterator[ToolStreamEvent]``. The executor drains
            the iterator, surfacing each non-``"done"`` event as a
            ``RunItemType.TOOL_PARTIAL_OUTPUT`` event to streaming
            consumers; the LLM still sees one final tool result
            (the value carried on the terminal ``"done"`` event).
            Mutually incoherent with ``cache=True``,
            ``cache_function``, ``response_format='content_and_artifact'``,
            and ``return_direct=True`` — those combinations raise
            ``ValueError`` at construction.

    Returns:
        A FunctionTool instance.

    Example:
        @function_tool(
            name="greet",
            description="Greet someone by name",
            requires_approval=True
        )
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        @function_tool(
            name="search",
            description="Search the database"
        )
        def search(ctx: ToolContext, query: str) -> str:
            # Access context
            user_id = ctx.context.get("user_id")
            return f"Results for {query}"
    """

    if rate_limit is not None and max_calls_per_minute is not None:
        raise ValueError(
            "Pass either rate_limit= or max_calls_per_minute=, not both. "
            "max_calls_per_minute is shorthand for rate_limit=ToolRateLimit(rpm=N)."
        )
    resolved_rate_limit: ToolRateLimit | None
    if rate_limit is not None:
        resolved_rate_limit = rate_limit
    elif max_calls_per_minute is not None:
        resolved_rate_limit = ToolRateLimit(rpm=max_calls_per_minute)
    else:
        resolved_rate_limit = None

    def function_tool_decorator(func: ToolFunction) -> FunctionTool:
        # Generate FunctionSchema from the function
        func_schema = generate_function_schema(
            func,
            name=name,
            description=description,
            parse_docstring=parse_docstring,
            docstring_style=docstring_style,
        )

        # Create the invocation wrapper with proper validation
        on_invoke_tool = _create_invoke_wrapper(func, func_schema, on_tool_call_fails)

        return FunctionTool(
            name=name if name is not None else func_schema.name,
            description=description if description is not None else func_schema.description,
            schema=func_schema.schema,  # Store Pydantic model
            schema_enforcement=schema_enforcement,
            guardrails=guardrails,
            enabled=enabled,
            requires_approval=requires_approval,
            max_result_tokens=max_result_tokens,
            max_retries=max_retries,
            timeout=timeout,
            timeout_behavior=timeout_behavior,
            timeout_error=timeout_error_function,
            on_invoke=on_invoke_tool,
            execution_aware=func_schema.execution_aware,
            history_aware=func_schema.history_aware,
            cache=cache,
            cache_function=cache_function,
            response_format=response_format,
            return_direct=return_direct,
            prepare=prepare,
            requires_env=requires_env,
            requires_packages=requires_packages,
            rate_limit=resolved_rate_limit,
            defer_loading=defer_loading,
            streaming=streaming,
        )

    if function is not None:
        return function_tool_decorator(function)

    return function_tool_decorator
