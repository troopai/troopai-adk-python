"""Run configuration for agent execution.

This module provides configuration options that control how agents are executed.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
)

from troopai.adk.agents.agent_guardrails import AgentGuardrails

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.audit import AuditSink
    from troopai.adk.budgets import CostLedger, TenantBudget
    from troopai.adk.context.context_config import ContextManagementConfig
    from troopai.adk.llms.llm import LLM
    from troopai.adk.llms.llm_usage import LLMUsageLimits
    from troopai.adk.llms.routing import LLMRouter
    from troopai.adk.run.messages import RunMessages
    from troopai.adk.sandbox.config import SandboxRunConfig
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.verbose.config import VerboseConfig

# TypeVar for CallModelData generic parameter. Local to this module to
# avoid importing from ``run/context.py`` (which in turn could lead to
# import ordering issues — config is loaded very early).
TContext = TypeVar("TContext")

logger = logging.getLogger(__name__)


class ErrorHandler(Protocol):
    """Protocol for exception recovery handlers on :attr:`RunConfig.error_handlers`.

    A callable that receives the raised exception and returns a fallback
    ``final_output`` value.  Both synchronous and asynchronous callables
    are accepted: the runner awaits the return value when it is
    :func:`inspect.isawaitable`.

    The return value is used directly as the run's ``final_output``; no
    schema re-validation is attempted.
    """

    def __call__(self, exc: Exception) -> Any | Awaitable[Any]: ...


# Type alias for history processors — functions that transform the
# RunItem list between context management and the LLM call.
type OnMaxTurnsHandler = Callable[
    ["Agent[Any]", int],
    Awaitable[str | None],
]
"""Async handler invoked when a per-agent ``max_turns`` cap is hit.

Receives the agent that exhausted its budget and the turn count.
Return a string to use as the run's final output (in which case the
run completes normally) or ``None`` to let the :class:`MaxTurnsExceeded`
exception propagate.

The swarm-level :attr:`RunConfig.max_total_turns` cap is **not** routed
through this handler — it represents a runaway workflow, not a
per-agent budget, and always raises.
"""


HistoryProcessor = Callable[[list["RunItem"]], list["RunItem"]]
"""A function that transforms the conversation items before the LLM call.

Applied after context management (compaction/editing) and before the LLM
call. Receives Layer 3 RunItems and must return RunItems. Items are frozen
dataclasses — use ``dataclasses.replace()`` to create modified copies.

Example::

    import dataclasses
    from troopai.adk.types.items import UserItem

    def redact_user_input(items):
        import re
        result = []
        for item in items:
            if isinstance(item, UserItem) and isinstance(item.raw.get("content"), str):
                new_raw = {**item.raw, "content": re.sub(r'[\\w.-]+@[\\w.-]+', '[REDACTED]', item.raw["content"])}
                result.append(dataclasses.replace(item, raw=new_raw))
            else:
                result.append(item)
        return result
"""


@dataclass
class ModelInputData:
    """Container for the input items that will be sent to the LLM.

    Wraps the Layer 1 input items list. Kept as a dataclass (not a bare
    list) so future metadata (cache control flags, turn index, etc.) can
    be added without a breaking change.

    Attributes:
        input: The Layer 1 input items list that will be sent to the LLM.

    Note:
        Unlike OpenAI's ``ModelInputData``, this class does *not* carry a
        separate ``instructions`` field. TroopAI's :meth:`LLM.acomplete`
        takes the system prompt in-band as part of ``messages``; a filter
        that wants to rewrite the system prompt edits the message with
        ``role="system"`` in-place on ``input``.
    """

    input: list[LLMInputContentItem]
    """The Layer 1 input items list that will be sent to the LLM."""


@dataclass
class CallModelData[TContext]:
    """Payload handed to a :data:`CallModelInputFilter`.

    Wraps the mutable input plus the agent identity and run context so
    the filter can make context-aware edits (e.g. inject a per-user
    system prompt, truncate based on ``ctx.usage``, etc.).

    Attributes:
        model_data: The input items wrapper. Filters return a (possibly
            new) ``ModelInputData`` with the rewritten input.
        agent: The agent the LLM call is being made for.
        context: The user-provided context (unwrapped from the internal
            ``RunContext``), or ``None`` if no context was provided.
    """

    model_data: ModelInputData
    """The input items wrapper."""

    agent: Agent[TContext]
    """The agent the LLM call is being made for."""

    context: TContext | None
    """The user-provided context, or ``None``."""


type CallModelInputFilter = Callable[
    [CallModelData[Any]],
    ModelInputData | Awaitable[ModelInputData],
]
"""Pre-LLM-call hook that rewrites the input items list.

Runs on every turn, immediately before the LLM call, after context
management and history processors, and before ``hooks.on_llm_start``.
Receives a :class:`CallModelData` with the current agent, unwrapped run
context, and a :class:`ModelInputData` wrapping a shallow copy of the
input items list. Must return a :class:`ModelInputData` — either the
same instance with edits applied, or a new one. May be sync or async.

Use this to:

- Inject per-request system messages based on run context
- Truncate input items based on token budget
- Add diagnostic shims in development
- Implement application-level caching or deduplication

For pure Layer 3 (``RunItem``) transforms, use
:attr:`RunConfig.history_processors` instead.
"""


@dataclass
class RunConfig:
    """Configuration for agent execution.

    Controls various aspects of how the Runner executes an agent,
    including model settings, tracing, execution limits, and
    cost controls.

    Attributes:
        model: Default model to use if the agent does not specify one.
        tenant_id: Opaque tenant identifier for this run. ``None`` =
            untenanted.
        tracing_enabled: Whether to enable OTel span emission.
        tracing_metadata: Additional metadata included on the root span.
        metrics_enabled: Whether to emit OTel metric instruments.
        max_tool_calls_per_turn: Maximum tool calls allowed per turn
            before forcing a response.
        fail_on_tool_error: Whether to raise on tool errors or continue
            with an error message.
        verbose: Optional verbose output configuration. ``None`` disables
            verbose output entirely.
        context_management: Optional context management configuration
            (compaction, editing, token budget). ``None`` disables
            context management.
        compaction_llm: Explicit :class:`~troopai.adk.llms.llm.LLM`
            instance for compaction calls. Falls back to the agent's
            primary LLM when ``None``.
        usage_limits: Token usage limits checked after each LLM
            response. ``None`` disables limits.
        tenant_budget: Per-tenant dollar budget. ``None`` = no cap.
        cost_ledger: Cross-run cost store for per-period budgets.
            Required when ``tenant_budget.dollars_per_period`` is set.
        audit_sink: Append-only sink for tool-call audit events.
            ``None`` = audit logging off.
        audit_strict: When ``True``, an audit-sink failure re-raises
            instead of logging a warning and continuing.
        router: Optional LLM router. ``None`` = no routing.
        history_processors: Pre-LLM-call hooks that transform the
            Layer 3 RunItem list. ``None`` = no processors.
        call_model_input_filter: Optional pre-LLM-call hook that
            rewrites the Layer 1 input items list (sync or async).
            ``None`` = no filter.
        on_max_turns: Handler invoked when per-agent ``max_turns`` is
            exhausted. ``None`` = raise ``MaxTurnsExceeded``.
        max_total_turns: Cross-agent cumulative turn limit for swarms.
            ``None`` disables the safety net.
        guardrails: Run-scope agent-level guardrails applied across all
            agents. Empty lists mean no run-scope guardrails.
        can_use_tool: Per-invocation permission callback (Layer 0).
            ``None`` = all tools permitted.
        tenant_tool_allowlist: Per-tenant allowed tool names. ``None``
            disables the feature.
        tenant_allowlist_default_deny: When ``True``, a tenant absent
            from the allowlist is denied all tools.
        tenant_allowlist_soft_deny: When ``True``, a forbidden tool
            call returns a denial message instead of raising.
        messages: Configurable messages for tool execution and handoffs.
            ``None`` uses default English messages.
        sandbox: Per-run sandbox configuration. ``None`` = no sandbox.
        max_parallel_tools: Maximum concurrent function tools per turn when
            parallel execution is active. ``None`` = unbounded gather (default).
            A positive integer *N* throttles via an asyncio semaphore.
            ``0`` or negative raises :class:`ValueError` at execution time.
        error_handlers: Mapping from exception type to a recovery handler
            that returns a fallback ``final_output``.  ``None`` (the
            default) — all exceptions propagate unchanged.  Recovered runs
            mark ``result.recovered`` and SKIP session/memory persistence
            (the turn's items are partial); the result carries the live
            run context, with ``new_items``/guardrail data reflecting only
            the progress made before the error.
        include_hook_events: When ``True``, hook lifecycle moments (tool
            start/end, guardrail start/end) are emitted as first-class typed
            :class:`~troopai.adk.run.stream.HookLifecycleEvent` stream events
            during streaming runs.  Off by default.

    Example:
        config = RunConfig(
            model="gpt-4o",
            tracing_enabled=True,
            max_tool_calls_per_turn=10,
        )
        result = await Runner.arun(agent, "Hello!", run_config=config)
    """

    model: str | None = None
    """Default model to use if agent doesn't specify one."""

    tenant_id: str | None = None
    """Opaque tenant identifier for this run. Threaded to status records,
    quotas, spans (``troopai.tenant.id``), the ``tenant`` metric dimension,
    and logs. ``None`` = untenanted (default)."""

    tracing_enabled: bool = False
    """Whether to enable execution tracing."""

    tracing_metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata to include in traces.

    Keys and values are surfaced on the root ``AgentSpanData.metadata``
    field and **recorded verbatim** by whichever tracer is installed
    (OTel emits them as ``troopai.metadata.<key>`` span attributes). Do
    NOT include secrets, credentials, API keys, or personally
    identifiable information — span payloads typically leave the
    application boundary (collectors, vendor backends, log files) and
    should be treated as externally visible.
    """

    metrics_enabled: bool = False
    """Whether to emit OTel metric instruments for this run.

    Independent of :attr:`tracing_enabled`: when ``True``, span objects
    are created at every emission seam so a composed
    :class:`~troopai.adk.tracing.metrics.MetricsTracer` records
    instruments, even when span export is off. Default ``False`` — opt-in.
    """

    max_tool_calls_per_turn: int = 10
    """Maximum tool calls allowed per turn before forcing a response."""

    fail_on_tool_error: bool = True
    """Whether to raise exceptions when tools fail, or continue with error message."""

    verbose: VerboseConfig | None = None
    """Configurable colourful output during execution.

    When set to a :class:`~troopai.adk.verbose.VerboseConfig` with
    ``enabled=True``, the runner installs a
    :class:`~troopai.adk.verbose.VerboseHooks` instance that renders
    lifecycle events (agent start/end, LLM calls, tool calls,
    handoffs, guardrails, skills, sessions) with per-event colours,
    icons, and prefixes.

    Per-agent overrides via ``Agent.verbose`` take precedence over
    this run-level default at emit time — useful for silencing a
    summariser in a handoff chain while keeping the coordinator
    loud.

    ``None`` (the default) disables verbose output entirely and the
    framework pays zero rendering cost.

    Example::

        from troopai.adk.verbose import VerboseConfig
        config = RunConfig(verbose=VerboseConfig())
        result = await Runner.arun(agent, "Hello", run_config=config)
    """

    context_management: ContextManagementConfig | None = None
    """Optional context management configuration (compaction, editing, token budget).

    Controls how conversation history is managed as it grows:
    compaction (LLM summarization), editing (clearing old tool results),
    and cache strategy for tool lists.

    Example::

        config = RunConfig(
            context_management=ContextManagementConfig(
                compaction=CompactionConfig(enabled=True, trigger_tokens=100_000),
                cache_strategy=CacheStrategy.STABLE,
            )
        )
    """

    compaction_llm: LLM | None = None
    """Explicit ``LLM`` instance for context-compaction calls.

    When set, every compaction call (the ``ContextManager`` pipeline,
    JIT ``CompactDirective``, and ``HandoffStrategy.SUMMARY``) routes
    through this instance via :meth:`LLM.acomplete`. When ``None``
    (the default), compaction falls back to the agent's resolved
    ``LLM`` — the same instance used for the main turn.

    Provide an explicit override when the agent's primary model is
    expensive and a cheaper model is acceptable for summarisation
    (e.g. primary ``claude-opus-4-7`` + compaction ``claude-haiku-4-5``).
    Because the call goes through the ``LLM`` ABC, compaction tokens
    land in :attr:`RunContext.usage` and ``Agent.middleware.llms`` sees
    the call.

    Example::

        from troopai.adk.llms import LiteLLM

        config = RunConfig(
            compaction_llm=LiteLLM(model="claude-haiku-4-5-20251001"),
        )
    """

    usage_limits: LLMUsageLimits | None = None
    """Token usage limits for the run.

    Checked after each LLM response. Raises :class:`UsageLimitExceeded`
    when any limit is exceeded. Each limit can be set independently;
    ``None`` disables that specific limit.

    Example::

        config = RunConfig(
            usage_limits=LLMUsageLimits(
                total_tokens_limit=50_000,
                request_limit=20,
            )
        )
    """

    tenant_budget: TenantBudget | None = None
    """Per-tenant dollar budget for the run (default ``None`` = no cap).

    Enforced pre-call against ``RunContext.cost_usd`` (per-run) and, when
    ``dollars_per_period`` is set, the ``cost_ledger`` (per-period). Requires
    ``RunContext.tenant_id``; a per-period cap requires ``cost_ledger``.
    """

    cost_ledger: CostLedger | None = None
    """Cross-run cost-accounting store for per-period budgets (default
    ``None``). Required when ``tenant_budget.dollars_per_period`` is set."""

    ledger_fail_open: bool = False
    """How the per-period dollar gate behaves when the ``cost_ledger`` is
    unreachable (e.g. a Redis/Postgres outage) — default ``False`` (fail
    **closed**).

    When ``False`` (default), an unreadable ledger cannot prove the tenant is
    under its period cap, so the pre-call gate refuses to silently permit the
    spend: it treats the outage as a breach and applies ``kill_on_exceed``
    (raise ``TenantBudgetExceeded`` by default; warn-and-continue when
    ``kill_on_exceed=False``). This keeps the dollar cap meaningful during a
    ledger outage rather than letting cost run unbounded — the developer never
    opts out of a cost they did not choose.

    Set ``True`` to restore permissive behavior (log the error and proceed as
    if zero had been spent this period). Use only when availability matters
    more than the period cap.
    """

    audit_sink: AuditSink | None = None
    """Append-only sink for tool-call audit events (default ``None`` =
    audit logging off). Each tool-call resolution (executed, denied, or
    errored) is recorded as a privacy-preserving
    :class:`~troopai.adk.audit.AuditEvent` (hashes, not raw payloads).

    Scope: covers ``FunctionTool`` calls and framework-executed built-ins
    on the normal and HITL-resume paths, including tenant-allowlist denials.
    Other early denials (for example, a ``can_use_tool`` rejection) are not
    yet audited."""

    audit_strict: bool = False
    """When ``True``, an audit-sink failure re-raises (fail-closed) instead
    of logging a warning and continuing. Default ``False`` = best-effort:
    a sink outage never takes down a run."""

    router: LLMRouter | None = None
    """Optional model router (default ``None`` = no routing — the single
    resolved LLM is used). When set, the loop tries the router's ordered
    candidates, escalating to a pricier one on failure."""

    history_processors: list[HistoryProcessor] | None = None
    """Pre-LLM-call hooks that transform the message list.

    Applied after context management (compaction/editing) and before
    the LLM call. Processors run in order; each receives the output
    of the previous one.

    Example::

        def redact_pii(messages):
            # ... transform messages ...
            return messages

        config = RunConfig(history_processors=[redact_pii])
    """

    call_model_input_filter: CallModelInputFilter | None = None
    """Optional pre-LLM-call hook that rewrites the input items list.

    Runs on every turn, immediately before the LLM call, after context
    management and history processors, and before
    :meth:`RunHooks.on_llm_start`. The filter receives a
    :class:`CallModelData` with the current agent, unwrapped run context,
    and a :class:`ModelInputData` wrapping a shallow copy of the input
    items list. It must return a :class:`ModelInputData` — either the
    same instance with edits applied, or a new one. May be sync or async.

    Use this to:

    - Inject per-request system messages based on run context
    - Truncate input items based on token budget
    - Add diagnostic shims in development
    - Implement application-level caching or deduplication

    For pure Layer 3 (``RunItem``) transforms, use
    :attr:`history_processors` instead.

    Example::

        async def inject_debug_note(payload):
            messages = list(payload.model_data.input)
            messages.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[debug]"}],
            })
            return ModelInputData(input=messages)

        config = RunConfig(call_model_input_filter=inject_debug_note)
    """

    on_max_turns: OnMaxTurnsHandler | None = None
    """Handler invoked when per-agent ``max_turns`` is exhausted.

    When set, the runner awaits the handler instead of raising
    :class:`MaxTurnsExceeded` at the end of the agent loop. A string
    return value becomes the run's final output; ``None`` falls through
    to the exception. The swarm-level :attr:`max_total_turns` cap is
    deliberately **not** routed through this handler — it represents a
    runaway workflow, not a per-agent budget.

    Example::

        async def salvage(agent, turns):
            return f"[Partial answer — {agent.name} hit {turns}-turn cap]"

        config = RunConfig(on_max_turns=salvage)
    """

    max_total_turns: int | None = 500
    """Cross-agent cumulative turn limit for multi-agent swarms.

    When set, the Runner tracks total turns across all agents
    (including handoffs) and raises :class:`MaxTurnsExceeded` when
    exceeded. Distinct from per-agent ``max_turns`` which resets
    on each agent.

    **Safety-net semantics**: this is the "absolute safety net" for
    multi-agent runs — the only cross-agent cap that catches a
    runaway swarm after per-agent ``max_turns`` have reset on every
    handoff. The default is a bounded ``500`` so a run is always capped
    unless the developer explicitly opts out — never silently unbounded.
    Set to ``None`` explicitly to disable the safety net for
    a specific run. A swarm using :class:`~troopai.adk.swarms.Swarm`
    complements this with :attr:`SwarmConfig.max_handoffs` and
    :attr:`SwarmConfig.max_total_tokens` — three different exhaustion
    dimensions (turns, handoffs, tokens).

    Example::

        # Allow 5 turns per agent, but max 20 total across all handoffs
        result = await Runner.arun(
            agent, "Hello!",
            max_turns=5,
            run_config=RunConfig(max_total_turns=20),
        )
    """

    guardrails: AgentGuardrails = field(default_factory=AgentGuardrails)
    """Run-scope agent-level guardrails applied across all agents in a run.

    A single :class:`~troopai.adk.agents.agent_guardrails.AgentGuardrails`
    config object holds ``input`` and ``output`` phase-typed lists.
    Empty lists (the default) mean no run-scope guardrails — the
    agent's own ``Agent.guardrails`` still applies.

    Run-scope guardrails merge with agent-level guardrails: run-scope
    runs first, then agent-scope. This is useful for applying
    organization-wide safety checks across all agents in a multi-agent
    workflow without wiring each agent individually.

    For input guardrails, each entry's ``run_in_parallel`` flag is
    respected: guardrails with ``run_in_parallel=True`` race alongside
    the agent loop, while ``run_in_parallel=False`` guardrails block
    before it starts. Set ``run_in_parallel=False`` on run-scope
    guardrails to ensure they always gate the LLM call.

    Example::

        from troopai.adk.agents import AgentGuardrails

        config = RunConfig(
            guardrails=AgentGuardrails(
                input=[pii_guardrail, jailbreak_guardrail],
                output=[content_filter, compliance_check],
            ),
        )
    """

    can_use_tool: Callable[[Agent, str, ToolContext], bool | Awaitable[bool]] | None = None
    """Per-invocation permission callback (Layer 0).

    Called before input guardrails for every tool invocation. Receives
    the agent, tool name, and tool context.  Returns ``False`` to deny
    the call (the LLM sees an error message and can choose a different
    tool).

    Use for per-user/per-role tool access control that shouldn't be
    baked into the tool definition.

    Example::

        def only_admin_tools(agent, tool_name, ctx):
            if tool_name.startswith("admin_"):
                return ctx.context.get("role") == "admin"
            return True

        config = RunConfig(can_use_tool=only_admin_tools)
    """

    tenant_tool_allowlist: Mapping[str, set[str]] | None = None
    """Per-tenant tool policy: ``tenant_id`` -> the set of tool names that
    tenant may call. ``None`` (default) disables the feature. A tenant key
    mapping to an empty set denies that tenant all tools. Untenanted runs
    (``RunContext.tenant_id is None``) are never governed by this map."""

    tenant_allowlist_default_deny: bool = False
    """When ``True``, a ``tenant_id`` absent from ``tenant_tool_allowlist``
    is denied all tools (fail-closed for tenants the operator did not
    configure). Default ``False`` = an absent tenant is unrestricted."""

    tenant_allowlist_soft_deny: bool = False
    """When ``True``, a forbidden tool call returns a denial message to the
    model instead of raising :class:`ToolNotPermittedForTenant`. Default
    ``False`` = fail-fast (the tool never executes)."""

    messages: RunMessages | None = None
    """Configurable messages for tool execution, handoffs, and memory.

    Override to customize strings sent to the LLM or surfaced to users.
    When ``None``, uses default English messages.

    Example::

        from troopai.adk.run.messages import RunMessages
        config = RunConfig(messages=RunMessages(tool_rejected="Refusé."))
    """

    sandbox: SandboxRunConfig | None = None
    """Per-run sandbox configuration.

    When set, the Runner brackets the agent loop with a sandbox
    session lifecycle (acquire → run → release). The session is
    resolved via the SandboxRunConfig fields: explicit ``session``
    takes precedence over ``session_state`` over ``client`` +
    ``manifest``.

    ``None`` (default) is the no-sandbox path — agents run with
    standard tools, no workspace, no isolation.

    Example::

        from troopai.adk.sandbox.config import SandboxRunConfig
        # The actual client classes ship in P18-P26; for now, any
        # backend with the BaseSandboxClient interface works.
        config = RunConfig(
            sandbox=SandboxRunConfig(client=some_client, manifest=...),
        )
    """

    max_parallel_tools: int | None = None
    """Maximum number of function tools that may run concurrently within one
    turn when the agent's ``tool_execution_mode`` is ``"parallel"``.

    ``None`` (default) preserves unbounded
    :func:`asyncio.gather` behaviour — all tools in the batch start at
    once.  Set to a positive integer *N* to bound concurrency to *N*
    simultaneous tool coroutines via an :class:`asyncio.Semaphore`;
    the remaining tools queue and start as slots become free.

    This is a concurrency knob for resource-sensitive back-ends (e.g.
    rate-limited APIs, connection-pooled databases) rather than a
    token-cost knob, so the ``None`` default is intentional — unlike
    most cost-affecting fields, no opt-out is required.

    Raises :class:`ValueError` at execution time when set to ``0`` or
    any negative value.  The sequential execution path
    (``tool_execution_mode != "parallel"``) is unaffected.

    Example::

        config = RunConfig(max_parallel_tools=3)
        result = await Runner.arun(agent, "Run all tools", run_config=config)
    """

    error_handlers: dict[type[Exception], ErrorHandler] | None = None
    """Mapping from exception type to a recovery handler (default ``None``).

    When set, the runner walks the MRO of the raised exception (most-derived
    first) and calls the first handler whose key is a superclass-or-exact-match
    of the exception's type.  The handler's return value becomes the run's
    ``final_output``; no schema re-validation is attempted.  A recovered run
    sets ``result.recovered = True`` and skips session/memory persistence in
    both the plain and streamed paths — the interrupted turn's items are
    partial and persisting them would seed the next turn with a half-formed
    exchange.

    MRO semantics — given::

        error_handlers = {
            ModelRefusalError: lambda e: "refused",
            TroopAIError: lambda e: "base-fallback",
        }

    and a raised ``ModelRefusalError``: the runner walks
    ``ModelRefusalError.__mro__`` → ``[ModelRefusalError, TroopAIError, …]``
    and finds ``ModelRefusalError`` as the first matching key, so the most
    specific handler wins.

    Both sync and async handlers are supported: an
    :func:`inspect.isawaitable` return value is awaited automatically.

    ``None`` (the default) disables error recovery entirely — all exceptions
    propagate unchanged.

    Example::

        async def on_refusal(exc):
            return "I can't help with that right now."

        config = RunConfig(
            error_handlers={ModelRefusalError: on_refusal},
        )
    """

    include_hook_events: bool = False
    """Emit hook lifecycle moments as stream events during streaming runs.

    When ``True``, the runner wraps the active :class:`~troopai.adk.hooks.RunHooks`
    instance with an emitter that publishes a
    :class:`~troopai.adk.run.stream.HookLifecycleEvent` to the stream queue at
    each tool-start, tool-end, guardrail-input-start, guardrail-input-end,
    guardrail-output-start, and guardrail-output-end call site.

    Off by default (``False``) — zero overhead unless enabled.
    Non-streaming :meth:`~troopai.adk.run.runner.Runner.arun` calls are
    unaffected regardless of this flag.
    """

    def snapshot(self) -> RunConfig:
        """Return an isolated run-owned copy of this configuration.

        ``RunConfig`` is mutable because fluent profiles and direct callers
        edit it before execution. A snapshot copies value configuration that
        could otherwise leak between profiles/runs, while deliberately sharing
        runtime handles and callbacks such as LLMs, routers, ledgers, sinks,
        sandbox clients/sessions, and hook functions.
        """
        copied = dataclasses.replace(self)
        copied.tracing_metadata = copy.deepcopy(self.tracing_metadata)
        if self.usage_limits is not None:
            copied.usage_limits = dataclasses.replace(self.usage_limits)
        if self.verbose is not None:
            copied.verbose = dataclasses.replace(
                self.verbose,
                styles=copy.deepcopy(self.verbose.styles),
            )
        if self.context_management is not None:
            copied.context_management = self.context_management.model_copy(deep=True)
        if self.messages is not None:
            copied.messages = dataclasses.replace(self.messages)
        copied.guardrails = AgentGuardrails(
            input=list(self.guardrails.input),
            output=list(self.guardrails.output),
        )
        if self.history_processors is not None:
            copied.history_processors = list(self.history_processors)
        if self.tenant_tool_allowlist is not None:
            copied.tenant_tool_allowlist = {tenant: set(tools) for tenant, tools in self.tenant_tool_allowlist.items()}
        if self.error_handlers is not None:
            copied.error_handlers = dict(self.error_handlers)
        if self.sandbox is not None:
            copied.sandbox = self._snapshot_sandbox_config()
        return copied

    def _snapshot_sandbox_config(self) -> SandboxRunConfig:
        """Copy sandbox value fields while preserving live runtime handles."""
        if self.sandbox is None:
            raise ValueError("cannot snapshot missing sandbox config")

        sandbox = dataclasses.replace(self.sandbox)
        sandbox.options = copy.deepcopy(self.sandbox.options)
        if self.sandbox.session_state is not None:
            sandbox.session_state = self.sandbox.session_state.model_copy(deep=True)
        if self.sandbox.manifest is not None:
            sandbox.manifest = self.sandbox.manifest.model_copy(deep=True)
        if self.sandbox.snapshot is not None:
            sandbox.snapshot = self.sandbox.snapshot.model_copy(deep=True)
        if self.sandbox.resource_limits is not None:
            sandbox.resource_limits = copy.deepcopy(self.sandbox.resource_limits)
        if self.sandbox.network_policy is not None:
            sandbox.network_policy = copy.deepcopy(self.sandbox.network_policy)
        if self.sandbox.iac is not None:
            sandbox.iac = copy.deepcopy(self.sandbox.iac)
        if self.sandbox.requirements is not None:
            sandbox.requirements = copy.deepcopy(self.sandbox.requirements)
        if self.sandbox.candidates is not None:
            from troopai.adk.sandbox.selector import SandboxCandidate

            sandbox.candidates = [
                dataclasses.replace(candidate, options=copy.deepcopy(candidate.options))
                if isinstance(candidate, SandboxCandidate)
                else copy.deepcopy(candidate)
                for candidate in self.sandbox.candidates
            ]
        return sandbox


# Singleton default messages — avoids re-creating on every call.
_DEFAULT_MESSAGES: RunMessages | None = None


def get_messages(config: RunConfig) -> RunMessages:
    """Resolve the RunMessages instance from a RunConfig.

    Returns ``config.messages`` if set, or a shared default instance.
    """
    if config.messages is not None:
        return config.messages
    global _DEFAULT_MESSAGES
    if _DEFAULT_MESSAGES is None:
        from troopai.adk.run.messages import RunMessages

        _DEFAULT_MESSAGES = RunMessages()
    return _DEFAULT_MESSAGES


# Default configuration
DEFAULT_RUN_CONFIG = RunConfig()

# Default max turns for the agent loop
DEFAULT_MAX_TURNS = 10

# Default model when neither agent nor RunConfig specifies one
DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"
