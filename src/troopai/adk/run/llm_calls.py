"""LLM interaction layer — call, resolve, build tools.

Extracted from ``Runner``: LLM instance resolution, tool definition
building, output schema resolution, and the actual LLM call methods.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, cast

from troopai.adk.exceptions import NoRoutingCandidateError, TroopAIError, UsageLimitExceeded
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.run.cost import check_request_limit_before_call, check_usage_limits
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.verbose.run_bridge import fire_stream_chunk

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.budgets import BudgetPeriod
    from troopai.adk.context.context_config import CacheStrategy
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.llms import LLM
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext, TContext
    from troopai.adk.run.stream import (
        RunResultStreaming,
    )
    from troopai.adk.run.tools_executor import FunctionToolFailureCounts
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.local.apply_patch_tool import ApplyPatchTool
    from troopai.adk.tools.local.shell_tool import ShellTool
    from troopai.adk.types.responses.llm_response import LLMResponse
    from troopai.adk.types.tools import ToolChoice

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Period-key helper
# ------------------------------------------------------------------


def _current_period_key(period: BudgetPeriod) -> str:
    """Return the calendar bucket key for the current UTC wall-clock time.

    Wraps ``period_key`` so call sites avoid repeating the UTC import.
    """
    from datetime import UTC, datetime

    from troopai.adk.budgets import period_key

    return period_key(period, datetime.now(UTC))


# ------------------------------------------------------------------
# Cache for default LLM instances (keyed by model name)
#
# Using functools.cache makes the cache scope explicit, avoids a
# module-level mutable dict (R6 in python-conventions.md), and
# exposes get_default_llm.cache_clear() for test isolation.
# ------------------------------------------------------------------


@functools.cache
def get_default_llm(model: str) -> LLM:
    """Get or create a default ``LiteLLM`` instance for the given model.

    Lazily instantiated and cached by model name so different agents
    using different models get separate instances.  The cache can be
    cleared between tests via ``get_default_llm.cache_clear()``.

    Args:
        model: The model identifier.

    Returns:
        A ``LiteLLM`` instance for the given model.
    """
    from troopai.adk.llms import LiteLLM

    return LiteLLM(model=model)


def resolve_model_name(agent: Agent, config: RunConfig) -> str:
    """Resolve the model name from agent/config without creating an LLM.

    Used for non-LLM purposes: token counting, context compaction,
    handoff budget estimation, etc.

    Args:
        agent: The agent with optional ``llm`` override.
        config: Run configuration with optional ``model`` default.

    Returns:
        The resolved model name string.
    """
    from troopai.adk.llms import LLM as _LLM

    if isinstance(agent.llm, _LLM):
        return getattr(agent.llm, "model", None) or config.model or DEFAULT_MODEL
    return (agent.llm if isinstance(agent.llm, str) else None) or config.model or DEFAULT_MODEL


def resolve_llm(agent: Agent, config: RunConfig) -> LLM:
    """Resolve the LLM instance from agent/config.

    Resolution rules:

    - If ``agent.llm`` is an ``LLM`` instance, use it directly.
    - If ``agent.llm`` is a string, it's the model name — creates
      a default ``LiteLLM(model=...)`` instance.
    - If ``agent.llm`` is ``None``, uses ``config.model`` or
      ``DEFAULT_MODEL``.

    Args:
        agent: The agent with optional ``llm`` override.
        config: Run configuration with optional ``model`` default.

    Returns:
        The resolved ``LLM`` instance.
    """
    from troopai.adk.llms import LLM as _LLM

    if isinstance(agent.llm, _LLM):
        return agent.llm
    model = (agent.llm if isinstance(agent.llm, str) else None) or config.model or DEFAULT_MODEL
    return get_default_llm(model)


def resolve_compaction_llm(agent: Agent, config: RunConfig) -> LLM:
    """Resolve the LLM instance for context-compaction calls.

    Resolution rules:

    - If ``config.compaction_llm`` is set, use it directly.
    - Otherwise, fall back to :func:`resolve_llm` (the agent's
      configured ``LLM`` — same instance used for the main turn).

    Used by every framework site that needs to invoke an LLM for
    summarisation: the ``ContextManager`` pipeline, JIT
    ``CompactDirective``, and ``HandoffStrategy.SUMMARY``. Routing
    through this helper keeps compaction usage observable in
    :attr:`RunContext.usage` and visible to ``Agent.middleware.llms``.

    Args:
        agent: The agent whose primary ``LLM`` is the fallback.
        config: Run configuration with optional ``compaction_llm`` override.

    Returns:
        The resolved compaction ``LLM`` instance.
    """
    if config.compaction_llm is not None:
        return config.compaction_llm
    return resolve_llm(agent, config)


def resolve_output_schema(agent: Agent) -> AgentOutputSchemaBase | None:
    """Resolve the agent's output_schema to an ``AgentOutputSchemaBase``.

    - ``None`` → ``None`` (no structured output)
    - ``AgentOutputSchemaBase`` instance → returned as-is
    - Raw type (``BaseModel``, ``int``, etc.) → wrapped in
      ``AgentOutputSchema``

    Args:
        agent: The agent with optional ``output_schema``.

    Returns:
        An ``AgentOutputSchemaBase`` or ``None``.
    """
    if agent.output_schema is None:
        return None

    from troopai.adk.schemas import AgentOutputSchemaBase as _Base

    if isinstance(agent.output_schema, _Base):
        return agent.output_schema

    # Raw type — wrap in AgentOutputSchema
    from troopai.adk.schemas import AgentOutputSchema

    return AgentOutputSchema(agent.output_schema)


def resolve_llm_config(
    agent: Agent,
    tool_choice_override: ToolChoice | None = None,
) -> LLMConfig | None:
    """Resolve the effective ``LLMConfig`` for an LLM call.

    Starts with the agent's ``llm_config`` and optionally overrides
    the ``tool_choice`` field.  Used by the agent loop to implement
    ``reset_tool_choice`` and forced tool use.

    Args:
        agent: The agent with optional ``llm_config``.
        tool_choice_override: If provided, overrides the agent's
            ``llm_config.tool_choice`` for this call.

    Returns:
        The resolved ``LLMConfig``, or ``None`` if no config is needed.
    """
    config = agent.llm_config
    if tool_choice_override is None:
        return config

    from dataclasses import replace

    from troopai.adk.llms.llm_config import LLMConfig

    if config is None:
        return LLMConfig(tool_choice=tool_choice_override)
    return replace(config, tool_choice=tool_choice_override)


async def enforce_tenant_budget(
    agent: Agent,
    config: RunConfig,
    context: RunContext[Any],
    messages: list[LLMInputContentItem],
    model: str,
    llm: LLM,
    hooks: RunHooks,
) -> None:
    """Pre-call dollar gate. No-op unless a budget + tenant_id are present.

    Raises ``TenantBudgetExceeded`` when ``kill_on_exceed`` (default);
    otherwise logs a warning + emits the breach event and returns.

    On a per-period cap, if the ``cost_ledger`` is unreachable the gate fails
    **closed** by default (``RunConfig.ledger_fail_open=False``): the period is
    treated as fully spent so the configured ``kill_on_exceed`` policy applies,
    keeping the dollar cap meaningful during an outage. Set
    ``ledger_fail_open=True`` to proceed as if zero had been spent instead.
    """
    budget = config.tenant_budget
    if budget is None or context.tenant_id is None:
        return
    from troopai.adk.exceptions import TenantBudgetExceeded
    from troopai.adk.run.cost import check_tenant_budget
    from troopai.adk.verbose.hooks import emit_budget_exceeded

    llm_config = resolve_llm_config(agent)
    max_out = llm_config.max_output_tokens if llm_config is not None else None
    estimate = llm.estimate_cost(messages, model, max_output_tokens=max_out).estimated_cost_usd

    spend = 0.0
    if budget.dollars_per_period is not None and config.cost_ledger is not None:
        key = _current_period_key(budget.period)
        try:
            spend = await config.cost_ledger.spend(context.tenant_id, key)
        except Exception:
            logger.exception(
                "cost ledger spend() failed for tenant=%s; period budget cannot be verified",
                context.tenant_id,
            )
            if config.ledger_fail_open:
                logger.warning(
                    "ledger_fail_open=True: proceeding despite unverified period spend for tenant=%s",
                    context.tenant_id,
                )
                spend = 0.0
            else:
                # Fail closed: an unreadable ledger cannot prove the tenant is
                # under its period cap. Treat the period as fully spent so the
                # check below trips and the configured kill_on_exceed policy
                # applies — rather than silently permitting unbounded spend.
                spend = float("inf")

    try:
        check_tenant_budget(budget, context.tenant_id, context.cost_usd, spend, estimate)
    except TenantBudgetExceeded as exc:
        emit_budget_exceeded(hooks, agent, str(exc))
        if budget.kill_on_exceed:
            raise
        logger.warning("tenant budget breach (continuing, kill_on_exceed=False): %s", exc)


async def record_tenant_spend(
    config: RunConfig,
    context: RunContext[Any],
    call_cost: float | None,
) -> None:
    """Post-call: add ``call_cost`` to the per-period ledger (best-effort)."""
    budget = config.tenant_budget
    if (
        budget is None
        or budget.dollars_per_period is None
        or config.cost_ledger is None
        or context.tenant_id is None
        or call_cost is None
    ):
        return
    if call_cost < 0:
        logger.warning(
            "negative call_cost (%.6f) not recorded to ledger for tenant=%s",
            call_cost,
            context.tenant_id,
        )
        return
    # Pre-call check and post-call record each capture wall-clock time
    # independently; a call that straddles an HOUR boundary may check bucket N
    # and record into N+1 — acceptable best-effort for HOUR budgets
    # (DAY and MONTH are unaffected in practice).
    key = _current_period_key(budget.period)
    try:
        await config.cost_ledger.record(context.tenant_id, key, call_cost)
    except Exception:
        # Post-call accounting: the call already happened, so this is
        # best-effort. But an unrecorded cost under-counts every later period
        # gate (it will read a too-low spend), so surface it at error level.
        logger.exception(
            "cost ledger record() failed (period accounting under-counted) for tenant=%s",
            context.tenant_id,
        )


async def build_tools(
    agent: Agent,
    context: RunContext[TContext] | None = None,
    tool_failure_counts: FunctionToolFailureCounts | None = None,
    cache_strategy: CacheStrategy | None = None,
    extra_tools: list[FunctionTool] | None = None,
) -> list[Tool] | None:
    """Build tool list for LLM call.

    Merges regular agent tools, built-in tools, and LLM handoff
    transfer tools when the agent uses LLM-orchestrated routing.
    The concrete ``LLM`` implementation converts these to
    wire-format dicts.

    Tools that have exhausted their retry budget (per ``max_retries``)
    are excluded from the list so the LLM cannot call them — unless
    ``cache_strategy`` is ``STABLE``, in which case they remain with
    an ``[UNAVAILABLE]`` description prefix to preserve prompt cache.

    Args:
        agent: The agent whose tools to build.
        context: Run context for dynamic ``enabled`` callbacks.
        tool_failure_counts: Per-tool failure counter.
        cache_strategy: How to handle disabled/exhausted tools for
            prompt caching.  ``None`` defaults to ``CacheStrategy.NONE``.
        extra_tools: Optional per-turn ephemeral tools appended verbatim
            after the agent's own tools. Used by the swarm driver to
            inject ``transfer_to_<name>`` and ``swarm_done`` tools
            without mutating ``agent.tools``. These bypass the
            enabled/retry/prepare filters by design — they are
            constructed per-turn and have no persistent identity.
    """
    from dataclasses import replace

    from troopai.adk.context.context_config import CacheStrategy
    from troopai.adk.tools.builtin.builtin_tool import BuiltinTool, ExecutableBuiltinTool
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.hosted import HostedTool
    from troopai.adk.tools.toolsets import Toolset

    effective_strategy: CacheStrategy = cache_strategy or CacheStrategy.NONE
    tools: list[Tool] = []

    # Locate the per-run reveal set in a separate pass before the
    # partition loop so the deferred-loading filter is correct
    # regardless of where the search tool appears in ``agent.tools``.
    # Folding this into the partition loop would mean a deferred tool
    # registered before the search tool would be incorrectly hidden
    # on the same turn its corresponding name was revealed.
    from troopai.adk.tools.tool_search import find_revealed_deferred_tools

    revealed_deferred: frozenset[str] = find_revealed_deferred_tools(agent.tools) if agent.tools else frozenset()

    # Materialise toolsets up front: each ``Toolset`` entry resolves to
    # a ``dict[str, FunctionTool]`` via ``await get_tools(ctx)``; the
    # flattened tools then flow through the same enabled / retry /
    # prepare pipeline as ad-hoc ``FunctionTool`` entries.
    #
    # Tracking the source of each materialised tool lets us aggregate
    # name-conflict errors with all contributing toolsets named in one
    # message, instead of silently letting the last-writer win.
    materialised_from_toolsets: list[FunctionTool] = []
    # Conflict detection only fires when at least one source is a
    # toolset. Pure standalone-vs-standalone collisions are left to
    # the pre-existing soft handling (e.g. multiple ``build_tool_search``
    # tools warn but coexist) — toolsets are the new surface where
    # silent name overwriting would actively confuse the developer.
    toolset_contributions: dict[str, list[str]] = {}
    standalone_names: dict[str, list[str]] = {}
    if agent.tools:
        for entry_index, entry in enumerate(agent.tools):
            if isinstance(entry, Toolset):
                inner_dict = await entry.get_tools(context)
                source_label = f"{type(entry).__name__}#{entry_index}"
                for inner_name, inner_tool in inner_dict.items():
                    toolset_contributions.setdefault(inner_name, []).append(source_label)
                    if inner_tool.name != inner_name:
                        logger.warning(
                            "Toolset %s returned tool with key '%s' but tool.name='%s'; "
                            "using key as the canonical name",
                            source_label,
                            inner_name,
                            inner_tool.name,
                        )
                        inner_tool = replace(inner_tool, name=inner_name)
                    materialised_from_toolsets.append(inner_tool)
            elif isinstance(entry, (FunctionTool, ExecutableBuiltinTool)):
                standalone_names.setdefault(entry.name, []).append(f"agent.tools[{entry_index}]")

    # A conflict surfaces when a toolset-contributed name appears more
    # than once across toolsets, OR when it collides with a standalone
    # FunctionTool name. Pure standalone-vs-standalone collisions are
    # not raised here — that path predates toolsets and would break
    # existing patterns like multiple ``build_tool_search`` instances.
    conflicts: dict[str, list[str]] = {}
    for name, ts_sources in toolset_contributions.items():
        all_sources = list(ts_sources)
        if name in standalone_names:
            all_sources.extend(standalone_names[name])
        if len(all_sources) > 1:
            conflicts[name] = all_sources
    if len(conflicts) > 0:
        from troopai.adk.exceptions import ToolsetNameConflictError

        raise ToolsetNameConflictError(agent.name, conflicts)

    # Partition agent.tools by type: FunctionTool vs HostedTool vs BuiltinTool vs Toolset
    if agent.tools:
        # Iterate the union of standalone tools and toolset-materialised
        # tools. Toolsets entries themselves are skipped (already flattened
        # into ``materialised_from_toolsets`` above); the loop body handles
        # FunctionTool and BuiltinTool variants from both sources. Typed
        # as ``list[Tool]`` (rather than ``list[Any]``) so the isinstance
        # narrowing chain in the loop body still flows through to
        # ``ExecutableBuiltinTool`` as an intersection with the ``Tool``
        # union — the same pattern the original code relied on before
        # toolset materialisation was introduced.
        all_tools: list[Tool] = []
        for entry in agent.tools:
            if isinstance(entry, Toolset):
                continue
            all_tools.append(entry)
        all_tools.extend(materialised_from_toolsets)

        for candidate in all_tools:
            tool = candidate.to_function_tool() if isinstance(candidate, ExecutableBuiltinTool) else candidate
            if isinstance(tool, FunctionTool):
                # Deferred-loading filter — hide unrevealed tools entirely
                # (no cache-stable carve-out, since the schema would be
                # missing from prior turns anyway).
                if tool.defer_loading and tool.name not in revealed_deferred:
                    continue

                # Function tools — filtered by enabled state and retry budget
                is_enabled: bool = await tool.check_enabled(context)
                is_exhausted: bool = (
                    tool_failure_counts is not None
                    and tool.max_retries is not None
                    and tool_failure_counts.get(tool.name, 0) > tool.max_retries
                )

                if not is_enabled or is_exhausted:
                    if effective_strategy == CacheStrategy.STABLE:
                        # Keep tool in list but mark unavailable.
                        # Schema stays identical → cache prefix preserved.
                        unavailable_tool = replace(
                            tool,
                            description=f"[UNAVAILABLE] {tool.description}",
                        )
                        tools.append(unavailable_tool)
                    else:
                        # NONE strategy: remove entirely (current behavior)
                        continue
                else:
                    current_tool: FunctionTool = tool
                    # Apply prepare function if configured
                    if tool.prepare is not None:
                        try:
                            prepared = tool.prepare(context, tool)
                            if isawaitable(prepared):
                                prepared = await prepared
                            if prepared is None:
                                continue  # Tool excluded for this step
                            current_tool = prepared
                        except Exception as e:
                            # Skip the tool (equivalent to prepare returning None)
                            # rather than including the unmodified original.
                            # Including the original silently defeats the filtering
                            # intent of prepare (e.g. hiding a tool for this step).
                            logger.warning(
                                "prepare function for tool '%s' raised: %s — "
                                "excluding tool for this step (same as returning None)",
                                tool.name,
                                e,
                                exc_info=True,
                            )
                            continue
                    tools.append(current_tool)
            elif isinstance(tool, HostedTool):
                # Provider-hosted tools (web search, code execution,
                # file search, image generation, URL context, hosted MCP,
                # …) are typed config dataclasses translated to wire
                # format by each LLM provider's converter. The framework
                # does NOT execute them — silent drops here would ship
                # the model zero tools and produce hallucinated answers.
                # Unsupported variants raise UnsupportedHostedToolError
                # at the converter boundary.
                tools.append(tool)
                logger.debug("Added hosted tool: %s", type(tool).__name__)
            elif isinstance(tool, BuiltinTool):
                from troopai.adk.tools.local.apply_patch_tool import ApplyPatchTool as _ApplyPatchTool
                from troopai.adk.tools.local.shell_tool import ShellTool as _ShellTool

                if isinstance(tool, _ShellTool) and tool.executor is not None:
                    # Wire ShellTool to a FunctionTool for this invocation.
                    # MUST NOT mutate agent.tools — Agent is configuration
                    # shared across concurrent Runner.arun() calls.
                    # Tool executor resolves the wired tool on demand via
                    # resolve_function_tool().
                    func_tool = _build_shell_function_tool(tool)
                    tools.append(func_tool)
                    logger.debug("Wired ShellTool with local executor as FunctionTool")
                elif isinstance(tool, _ApplyPatchTool) and tool.editor is not None:
                    # Wire ApplyPatchTool to a FunctionTool. Same non-mutation
                    # contract as ShellTool above.
                    func_tool = _build_apply_patch_function_tool(tool)
                    tools.append(func_tool)
                    logger.debug("Wired ApplyPatchTool with local editor as FunctionTool")
                else:
                    # Any remaining BuiltinTool subclass without a wired
                    # local execution path cannot be called by the Runner.
                    # Provider-native
                    # capabilities go through LLMConfig.extra_body / extra_args.
                    logger.warning(
                        "Tool %r is a BuiltinTool without a local executor — skipping. "
                        "Use LLMConfig.extra_body for provider-native capabilities, "
                        "or set ShellTool.executor / ApplyPatchTool.editor.",
                        tool.name,
                    )

    # Skill tools — merge tools from enabled skills with governance/guardrail defaults
    if agent.skills:
        from troopai.adk.skills.skill_rendering import check_skill_enabled

        for skill in agent.skills:
            if not await check_skill_enabled(skill, context):
                continue
            for candidate in skill.tools:
                tool = candidate.to_function_tool() if isinstance(candidate, ExecutableBuiltinTool) else candidate
                if isinstance(tool, FunctionTool):
                    # Deferred-loading filter — hide unrevealed tools entirely,
                    # mirroring the agent-tools branch. Without it a skill's
                    # ``defer_loading`` tool is always advertised, defeating the
                    # token-cost + capability gate the flag exists for.
                    if tool.defer_loading and tool.name not in revealed_deferred:
                        continue
                    governed_tool = _apply_skill_governance(tool, skill)
                    # Apply same enabled/retry-budget/prepare logic as regular tools
                    is_enabled = await governed_tool.check_enabled(context)
                    is_exhausted = (
                        tool_failure_counts is not None
                        and governed_tool.max_retries is not None
                        and tool_failure_counts.get(governed_tool.name, 0) > governed_tool.max_retries
                    )
                    if not is_enabled or is_exhausted:
                        if effective_strategy == CacheStrategy.STABLE:
                            unavailable_tool = replace(
                                governed_tool,
                                description=f"[UNAVAILABLE] {governed_tool.description}",
                            )
                            tools.append(unavailable_tool)
                        else:
                            continue
                    else:
                        current_skill_tool: FunctionTool = governed_tool
                        if governed_tool.prepare is not None:
                            try:
                                prepared = governed_tool.prepare(context, governed_tool)
                                if isawaitable(prepared):
                                    prepared = await prepared
                                if prepared is None:
                                    continue
                                current_skill_tool = prepared
                            except Exception as e:
                                # Skip the tool (equivalent to prepare returning None)
                                # rather than including the unmodified original.
                                # Including the original silently defeats the filtering
                                # intent of prepare (e.g. hiding a tool for this step).
                                logger.warning(
                                    "prepare function for skill tool '%s' raised: %s — "
                                    "excluding tool for this step (same as returning None)",
                                    governed_tool.name,
                                    e,
                                    exc_info=True,
                                )
                                continue
                        tools.append(current_skill_tool)
                elif isinstance(tool, BuiltinTool):
                    logger.warning(
                        "Skill %r declared tool %r as a raw BuiltinTool — skipping. "
                        "Use ExecutableBuiltinTool, FunctionTool, or LLMConfig.extra_body.",
                        skill.name,
                        tool.name,
                    )

    # LLM handoff tools
    if agent.handoffs is not None and isinstance(agent.handoffs, list):
        from troopai.adk.handoffs.handoff_helpers import (
            build_handoff_tools as _build_handoff_tools,
            normalize_handoffs as _normalize_handoffs,
        )

        normalized = await _normalize_handoffs(agent.handoffs)
        handoff_tools = await _build_handoff_tools(normalized, context)
        tools.extend(handoff_tools)

    # Per-turn ephemera injected by the swarm driver. These never
    # participate in enabled/retry/prepare filtering — they exist only
    # for the duration of this single call.
    if extra_tools is not None and len(extra_tools) > 0:
        tools.extend(extra_tools)

    return tools if len(tools) > 0 else None


async def call_llm(
    agent: Agent,
    messages: str | list[LLMInputContentItem],
    config: RunConfig,
    context: RunContext[TContext] | None = None,
    tool_failure_counts: FunctionToolFailureCounts | None = None,
    tool_choice_override: ToolChoice | None = None,
    extra_tools: list[FunctionTool] | None = None,
    llm: LLM | None = None,
) -> LLMResponse:
    """Call LLM via the ``LLM`` interface (default: ``LiteLLM``).

    Resolves the LLM instance, builds tool definitions, resolves
    output schema, and delegates to ``llm.acomplete()``.

    Args:
        agent: The agent to call the LLM for.
        messages: The conversation messages.
        config: Run configuration.
        context: Optional run context, passed to ``build_tools`` so that
            dynamic ``FunctionTool.enabled`` callables can inspect it.
        tool_failure_counts: Per-tool failure counter for filtering exhausted tools.
        tool_choice_override: If provided, overrides the agent's
            ``llm_config.tool_choice`` for this call. Used by
            the agent loop for ``reset_tool_choice`` and forced tool use.
        extra_tools: Optional per-turn ephemeral tools appended verbatim.
        llm: If provided, used directly instead of resolving from agent/config.
            Used by the routing layer to inject a specific candidate LLM.
            When ``None`` (the default), behavior is identical to before
            this parameter was added.

    Returns:
        An ``LLMResponse`` with content, tool_calls, usage, and
        optional structured output.
    """
    resolved_llm = llm if llm is not None else resolve_llm(agent, config)
    cache_strategy = (
        getattr(config.context_management, "cache_strategy", None) if config.context_management is not None else None
    )
    tools = await build_tools(
        agent,
        context=context,
        tool_failure_counts=tool_failure_counts,
        cache_strategy=cache_strategy,
        extra_tools=extra_tools,
    )
    output_schema = resolve_output_schema(agent)
    llm_config = resolve_llm_config(agent, tool_choice_override=tool_choice_override)

    # Fast path — no LLMMiddleware configured: invoke ``acomplete`` directly,
    # one async frame less than the composed chain (which would just call
    # the same terminal). Mirrors ``compose_tool_middleware``'s identity-on-
    # empty-list shape.
    if len(agent.middleware.llms) == 0:
        # stream defaults to False; the stream=False overload returns LLMResponse.
        if config.usage_limits is not None and context is not None:
            check_request_limit_before_call(config.usage_limits, context.usage)
        return await resolved_llm.acomplete(
            messages=messages,
            llm_config=llm_config,
            tools=tools,
            output_schema=output_schema,
        )

    # ``LLMMiddleware`` chain: compose around the actual ``acomplete`` call.
    from troopai.adk.llms.llm_middleware import compose_llm_middleware
    from troopai.adk.run.context import RunContext

    async def _llm_terminal(
        chain_messages: list[LLMInputContentItem],
        chain_llm_config: LLMConfig | None,
    ) -> LLMResponse:
        # The chain's outermost middleware passes whatever shape it was
        # given; ``call_llm`` itself accepts ``messages: str | list[...]``
        # but the middleware path always normalises to a list because
        # the Protocol's ``LLMMiddlewareNext`` signature commits to it.
        if config.usage_limits is not None:
            check_request_limit_before_call(config.usage_limits, chain_context.usage)
        return await resolved_llm.acomplete(
            messages=chain_messages,
            llm_config=chain_llm_config,
            tools=tools,
            output_schema=output_schema,
        )

    # ``messages`` may arrive as ``str`` from convenience callers; widen
    # to a list before crossing the middleware boundary so middleware
    # always sees the canonical Layer 1 shape.
    if isinstance(messages, str):
        from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage

        chain_initial_messages: list[LLMInputContentItem] = [
            LLMInputEasyMessage(role="user", content=messages),
        ]
    else:
        chain_initial_messages = messages

    chain_context: RunContext[Any] = context if context is not None else RunContext.make(None)
    chain = compose_llm_middleware(
        agent.middleware.llms,
        _llm_terminal,
        agent=agent,
        context=chain_context,
    )
    # ``LLMMiddlewareTermination`` is caught and unwrapped inside
    # ``compose_llm_middleware`` so the chain returns a normal
    # ``LLMResponse``; this caller receives the cached/synthetic
    # response like any other return value.
    return await chain(chain_initial_messages, llm_config)


# ------------------------------------------------------------------
# Routing escalation driver
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingOutcome:
    """Result of a routed call: the response plus which candidate served it.

    Attributes:
        response: The :class:`~troopai.adk.types.responses.LLMResponse`
            returned by the winning candidate.
        model: Model name of the candidate that served the request.
        llm: The :class:`~troopai.adk.llms.llm.LLM` instance of the
            candidate that served the request.
    """

    response: LLMResponse
    """The ``LLMResponse`` returned by the winning candidate."""

    model: str
    """Model name of the candidate that served the request."""

    llm: LLM
    """The LLM instance of the candidate that served the request."""


class _SchemaOrPredicateRejection(Exception):
    """Internal marker so a final NoRoutingCandidateError can chain a cause.

    Never raised — only assigned to ``last_error`` so it becomes the
    ``__cause__`` of ``NoRoutingCandidateError`` via exception chaining.
    """

    def __init__(self, model: str) -> None:
        super().__init__(f"candidate {model} rejected by schema/predicate")


def _is_routing_retryable(exc: Exception) -> bool:
    """Escalate on transient/provider errors; never on framework errors.

    Framework errors (``TroopAIError`` — budget exceeded, guardrail trips,
    user errors) are deliberate stops, not transient failures, so they
    propagate immediately rather than escalating to a pricier model.
    """
    return not isinstance(exc, TroopAIError)


def _routed_response_ok(
    router: LLMRouter,
    output_schema: AgentOutputSchemaBase | None,
    response: LLMResponse,
) -> bool:
    """Return ``False`` to escalate. Checks the router predicate + output-schema validity."""
    if router.should_escalate(response):
        return False
    if output_schema is not None and response.content is not None:
        try:
            output_schema.validate_json(response.content)
        except Exception as exc:  # broad catch is intentional; any validation failure drives escalation
            logger.debug("routing: response failed output-schema validation, escalating: %s", exc)
            return False
    return True


async def _record_rejected_routed_response(
    *,
    agent: Agent,
    candidate: RoutedModel,
    config: RunConfig,
    context: RunContext[TContext] | None,
    hooks: RunHooks[TContext],
    response: LLMResponse,
) -> None:
    """Account for a routed response that was rejected before fallback."""
    if context is None:
        return

    from troopai.adk.llms.llm_usage import LLMUsage
    from troopai.adk.verbose.hooks import emit_budget_exceeded, emit_usage_recorded

    if response.usage is not None:
        call_cost = candidate.llm.cost(candidate.model, response.usage)
        if call_cost is not None:
            context.cost_usd += call_cost
            await record_tenant_spend(config, context, call_cost)

    usage_delta = response.usage if response.usage is not None else LLMUsage(requests=1)
    context.usage = context.usage + usage_delta
    emit_usage_recorded(hooks, agent, context.usage)

    if config.usage_limits is not None:
        try:
            check_usage_limits(config.usage_limits, context.usage)
        except UsageLimitExceeded as exc:
            emit_budget_exceeded(hooks, agent, str(exc))
            raise


async def call_llm_with_routing(
    router: LLMRouter,
    agent: Agent,
    messages: list[LLMInputContentItem],
    config: RunConfig,
    hooks: RunHooks,
    context: RunContext[TContext] | None = None,
    tool_failure_counts: FunctionToolFailureCounts | None = None,
    tool_choice_override: ToolChoice | None = None,
    extra_tools: list[FunctionTool] | None = None,
) -> RoutingOutcome:
    """Try router candidates in order; escalate on failure.

    Escalates on non-``TroopAIError`` exceptions, ``router.should_escalate``
    returning ``True``, or output-schema validation failure. ``TroopAIError``
    subclasses propagate immediately. See :func:`call_llm` for shared args.

    Raises:
        NoRoutingCandidateError: Chains the last error when all candidates fail.
    """
    ctx = RoutingContext(
        messages=list(messages),
        tenant_id=context.tenant_id if context is not None else None,
        run_cost=context.cost_usd if context is not None else 0.0,
    )
    try:
        candidates = router.candidates(ctx)
    except Exception as exc:
        logger.error("router %s candidates() failed: %s", type(router).__name__, exc)
        raise NoRoutingCandidateError(f"router.candidates() raised: {type(exc).__name__}: {exc}") from exc
    # Routing layer's copy — for escalation logic only; call_llm passes output_schema
    # to acomplete() as a format constraint but does not validate the response content.
    output_schema = resolve_output_schema(agent)
    last_error: Exception | None = None
    for candidate in candidates:
        if context is not None:
            await enforce_tenant_budget(agent, config, context, messages, candidate.model, candidate.llm, hooks)
        try:
            response = await call_llm(
                agent,
                messages,
                config,
                context=context,
                tool_failure_counts=tool_failure_counts,
                tool_choice_override=tool_choice_override,
                extra_tools=extra_tools,
                llm=candidate.llm,
            )
        except Exception as exc:  # re-raised immediately if not retryable; see _is_routing_retryable
            if not _is_routing_retryable(exc):
                raise
            last_error = exc
            logger.warning("routing: candidate %s failed, escalating: %s", candidate.model, exc)
            continue
        if _routed_response_ok(router, output_schema, response):
            return RoutingOutcome(response=response, model=candidate.model, llm=candidate.llm)
        await _record_rejected_routed_response(
            agent=agent,
            candidate=candidate,
            config=config,
            context=context,
            hooks=hooks,
            response=response,
        )
        last_error = _SchemaOrPredicateRejection(candidate.model)
        logger.warning("routing: candidate %s rejected by validation/predicate, escalating", candidate.model)
    raise NoRoutingCandidateError(f"all {len(candidates)} routing candidate(s) failed") from last_error


async def call_llm_streamed_with_routing(
    router: LLMRouter,
    agent: Agent,
    messages: list[LLMInputContentItem],
    config: RunConfig,
    hooks: RunHooks,
    result: RunResultStreaming,
    context: RunContext[TContext] | None = None,
    tool_failure_counts: FunctionToolFailureCounts | None = None,
    tool_choice_override: ToolChoice | None = None,
    extra_tools: list[FunctionTool] | None = None,
) -> RoutingOutcome:
    """Streaming router: try candidates in order, escalating before the first token.

    Escalation is sound ONLY before any token reaches the consumer: a candidate that
    raises while ``result.has_emitted_tokens`` is False with a routing-retryable error
    escalates to the next candidate. Once streaming has begun, the error is surfaced
    (re-running on another model would double-bill and reorder output). There is no
    ``should_escalate`` / output-schema check here — tokens are already forwarded by
    the time the full response exists. Args mirror :func:`call_llm_streamed`. Each
    candidate is budget-gated before its attempt.

    Raises:
        NoRoutingCandidateError: when all candidates fail before emitting a token.
    """
    ctx = RoutingContext(
        messages=list(messages),
        tenant_id=context.tenant_id if context is not None else None,
        run_cost=context.cost_usd if context is not None else 0.0,
    )
    try:
        candidates = router.candidates(ctx)
    except Exception as exc:
        logger.error("router %s candidates() failed: %s", type(router).__name__, exc)
        raise NoRoutingCandidateError(f"router.candidates() raised: {type(exc).__name__}: {exc}") from exc
    last_error: Exception | None = None
    for candidate in candidates:
        if context is not None:
            await enforce_tenant_budget(agent, config, context, messages, candidate.model, candidate.llm, hooks)
        try:
            final = await call_llm_streamed(
                agent=agent,
                messages=messages,
                config=config,
                result=result,
                context=context,
                tool_failure_counts=tool_failure_counts,
                tool_choice_override=tool_choice_override,
                extra_tools=extra_tools,
                llm=candidate.llm,
            )
        except Exception as exc:  # broad: escalation needs the surface
            if not _is_routing_retryable(exc) or result.has_emitted_tokens:
                raise
            last_error = exc
            logger.warning(
                "routing(stream): %s failed pre-token, escalating: %s",
                candidate.model,
                exc,
            )
            continue
        return RoutingOutcome(response=final, model=candidate.model, llm=candidate.llm)
    raise NoRoutingCandidateError(f"all {len(candidates)} streaming candidate(s) failed") from last_error


async def call_llm_streamed(
    agent: Agent,
    messages: list[LLMInputContentItem],
    config: RunConfig,
    result: RunResultStreaming,
    context: RunContext[TContext] | None = None,
    tool_failure_counts: FunctionToolFailureCounts | None = None,
    tool_choice_override: ToolChoice | None = None,
    extra_tools: list[FunctionTool] | None = None,
    llm: LLM | None = None,
) -> LLMResponse:
    """Call LLM with streaming, emitting events for each token.

    Uses the ``LLM`` interface (default: ``LiteLLM``) with
    ``stream=True``.  Iterates over ``LLMStreamEvent`` objects,
    forwarding content deltas to the result's event stream.

    The final ``"done"`` event contains the complete ``LLMResponse``
    with content, tool_calls, usage, and optional structured output.

    Args:
        agent: The agent to call the LLM for.
        messages: The conversation messages.
        config: Run configuration.
        result: The streaming result to emit events to.
        context: Optional run context for tool enablement checks.
        tool_failure_counts: Per-tool failure counter for filtering exhausted tools.
        tool_choice_override: If provided, overrides the agent's
            ``llm_config.tool_choice`` for this call. Used by
            the agent loop for ``reset_tool_choice`` and forced tool use.
        llm: If provided, used directly instead of resolving from agent/config.
            Used by the routing layer to inject a specific candidate LLM.
            When ``None`` (the default), behavior is identical to before
            this parameter was added.

    Returns:
        The complete ``LLMResponse`` after streaming finishes.

    Streaming middleware:
        ``Agent.middleware.stream_llms`` (sibling of
        ``Agent.middleware.llms``) wraps each streaming call. When
        only the non-streaming ``llms`` slot is populated and a
        streaming call is issued, this function emits one
        ``logger.warning`` line per call so the user notices the
        path mismatch — the non-streaming middleware never applies
        to streaming calls because the return-type contracts differ
        (``LLMResponse`` vs. ``AsyncIterator[LLMStreamEvent]``).
    """
    from troopai.adk.llms.llm_stream_middleware import (
        LLMStreamMiddlewareNext,
        compose_llm_stream_middleware,
    )
    from troopai.adk.run.stream import RawResponseStreamEvent
    from troopai.adk.types.responses.llm_response import LLMStreamEvent

    has_non_streaming_only = len(agent.middleware.llms) > 0 and len(agent.middleware.stream_llms) == 0
    if has_non_streaming_only:
        logger.warning(
            "Non-streaming LLMMiddleware does not apply to streaming calls "
            "(agent='%s', llms=%d). Register streaming logic via "
            "Agent.middleware.stream_llms to wrap the streaming path.",
            agent.name,
            len(agent.middleware.llms),
        )

    resolved_llm = llm if llm is not None else resolve_llm(agent, config)
    cache_strategy = (
        getattr(config.context_management, "cache_strategy", None) if config.context_management is not None else None
    )
    tools = await build_tools(
        agent,
        context=context,
        tool_failure_counts=tool_failure_counts,
        cache_strategy=cache_strategy,
        extra_tools=extra_tools,
    )
    output_schema = resolve_output_schema(agent)
    llm_config = resolve_llm_config(agent, tool_choice_override=tool_choice_override)

    from troopai.adk.types.responses.llm_response import LLMResponseText

    final_response: LLMResponse | None = None

    # Build the streaming chain. The terminal makes the actual
    # provider call; outer middleware wrap the iterator returned
    # from each link, observing or replacing events as they flow.
    async def llm_stream_terminal(
        chain_messages: list[LLMInputContentItem],
        chain_llm_config: LLMConfig | None,
    ) -> AsyncIterator[LLMStreamEvent]:
        # The ``stream=True`` overload returns
        # ``AsyncIterator[LLMStreamEvent]``; pyright cannot narrow
        # the LLM overload union without a literal-bool branch on
        # this side of the boundary, so the cast carries the
        # discriminator narrowing the typed surface cannot.
        if config.usage_limits is not None:
            check_request_limit_before_call(config.usage_limits, chain_ctx.usage)
        return await resolved_llm.acomplete(  # type: ignore[return-value]
            messages=chain_messages,
            llm_config=chain_llm_config,
            tools=tools,
            output_schema=output_schema,
            stream=True,
        )

    # Compose unconditionally. ``compose_llm_stream_middleware``
    # already returns ``terminal`` unchanged when the chain is empty
    # (zero-overhead path), so a separate fast-path here would only
    # silently bypass middleware when ``context`` is ``None`` — the
    # exact silent-bypass shape removed elsewhere on this path.
    # Synthesize an empty context if the runner didn't supply one;
    # middleware bodies treat it as a sentinel.
    from troopai.adk.run.context import RunContext

    chain_ctx = context if context is not None else RunContext.make(None)
    chain: LLMStreamMiddlewareNext = compose_llm_stream_middleware(
        agent.middleware.stream_llms,
        llm_stream_terminal,
        agent=agent,
        context=chain_ctx,
    )

    stream_iter = await chain(messages, llm_config)
    # Track which part indices are text parts so we only forward text
    # deltas to RawResponseStreamEvent (downstream consumers like
    # demo.py render event.data as display text). Reasoning and
    # tool-call argument deltas share the "part_delta" event type but
    # must not be interleaved into the rendered response stream.
    text_part_indices: set[int] = set()
    # Tool-call argument parts get their own index set so the verbose
    # Live widget can route them to the yellow "🔧 Tool Arguments"
    # panel instead of the green "✅ Agent Final Answer" panel.
    tool_part_indices: set[int] = set()
    # Per-stream accumulators for the verbose Live widget. Plain
    # locals (not module-level) so concurrent agent streams never
    # share a buffer.
    text_buf: list[str] = []
    tool_buf: list[str] = []

    try:
        async for event in stream_iter:
            if event.type == "part_start" and event.index is not None:
                if isinstance(event.part, LLMResponseText):
                    text_part_indices.add(event.index)
                else:
                    tool_part_indices.add(event.index)
            elif (
                event.type == "part_delta"
                and event.delta is not None
                and event.index is not None
                and event.index in text_part_indices
            ):
                if len(event.delta) > 0:
                    # Only a non-empty content delta counts as consumer-visible output;
                    # empty deltas must not block a still-safe escalation.
                    result.has_emitted_tokens = True
                await result.put_event(RawResponseStreamEvent(data=event.delta))
                text_buf.append(event.delta)
                fire_stream_chunk("".join(text_buf), "text")
            elif (
                event.type == "part_delta"
                and event.delta is not None
                and event.index is not None
                and event.index in tool_part_indices
            ):
                tool_buf.append(event.delta)
                fire_stream_chunk("".join(tool_buf), "tool_call")
            elif event.type == "done":
                final_response = event.response
    finally:
        # Explicitly close the provider async generator so the
        # underlying HTTP/SSE connection is released promptly on
        # cancellation rather than relying on GC.  The declared type
        # is AsyncIterator which does not guarantee aclose(), but the
        # real runtime type is always an AsyncGenerator from the LLM
        # provider.  Guard with hasattr for type-checker safety.
        aclose = getattr(stream_iter, "aclose", None)
        if callable(aclose):
            close_stream = cast(Callable[[], Awaitable[None]], aclose)
            await close_stream()

    if final_response is None:
        from troopai.adk.types.responses.llm_response import LLMResponse

        final_response = LLMResponse(response_id="", model=getattr(resolved_llm, "model", ""))

    return final_response


def find_agent_by_name(root_agent: Agent, name: str) -> Agent | None:
    """Search the agent tree for an agent with the given name.

    Used during HITL state resumption when the run was interrupted
    after a handoff — the saved ``current_agent_name`` may differ
    from the root agent.

    Performs a recursive breadth-first search through handoff targets
    to handle deep handoff chains (A → B → C).

    Args:
        root_agent: The root agent to search from.
        name: The agent name to find.

    Returns:
        The matching ``Agent``, or ``None`` if not found.
    """
    if root_agent.name == name:
        return root_agent

    # BFS with visited set to prevent cycles
    visited: set[str] = {root_agent.name}
    queue: list[Agent] = [root_agent]

    while queue:
        current = queue.pop(0)
        if current.handoffs is None or not isinstance(current.handoffs, list):
            continue

        # Lazy import — ``Handoff`` lives in a module that would introduce a
        # cycle at module-load time if imported at the top. Only the runtime
        # class is needed for the isinstance narrowing below; the ``else``
        # branch narrows to ``Agent`` by union exhaustiveness.
        from troopai.adk.handoffs.handoff import Handoff as _Handoff

        for item in current.handoffs:
            # Union is exhaustive: item is either a Handoff (extract its
            # ``target``) or a raw Agent (is the target). Isinstance narrowing
            # is enough — no defensive ``else`` because the type is closed.
            if isinstance(item, _Handoff):
                agent_target: Agent = item.target
            else:
                agent_target = item

            if agent_target.name == name:
                return agent_target

            if agent_target.name not in visited:
                visited.add(agent_target.name)
                queue.append(agent_target)

    return None


# ------------------------------------------------------------------
# Builtin tool → FunctionTool converters (local executor wiring)
# ------------------------------------------------------------------


def _build_shell_function_tool(shell_tool: ShellTool) -> FunctionTool:
    """Create a FunctionTool that wraps a ShellTool's local executor.

    Callers MUST check ``shell_tool.executor is not None`` before calling —
    the wire-in loop in build_tools (the ``isinstance(tool, _ShellTool)`` branch)
    enforces that invariant.
    """
    from troopai.adk.tools.function_tool import FunctionTool

    if shell_tool.executor is None:
        raise ValueError("ShellTool.executor must be set before wiring")
    executor = shell_tool.executor

    async def shell_handler(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
        import json

        args = json.loads(raw_args) if len(raw_args) > 0 else {}
        command = str(args.get("command", ""))
        return await executor.execute(
            command,
            timeout=shell_tool.timeout,
            working_directory=shell_tool.working_directory,
            environment=shell_tool.environment,
        )

    return FunctionTool(
        name="shell",
        description="Execute a shell command and return the output.",
        schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
            },
            "required": ["command"],
        },
        on_invoke=shell_handler,
        requires_approval=shell_tool.approval,
    )


def _build_apply_patch_function_tool(patch_tool: ApplyPatchTool) -> FunctionTool:
    """Create a FunctionTool that wraps an ApplyPatchTool's local editor.

    Callers MUST check ``patch_tool.editor is not None`` before calling —
    the wire-in loop in build_tools (the ``isinstance(tool, _ApplyPatchTool)`` branch)
    enforces that invariant.
    """
    from troopai.adk.tools.function_tool import FunctionTool

    if patch_tool.editor is None:
        raise ValueError("ApplyPatchTool.editor must be set before wiring")
    editor = patch_tool.editor

    async def patch_handler(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
        import json

        args = json.loads(raw_args) if len(raw_args) > 0 else {}
        patch = str(args.get("patch", ""))
        return await editor.apply(patch)

    return FunctionTool(
        name="apply_patch",
        description="Apply a unified diff patch to modify files.",
        schema={
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "The unified diff patch to apply."},
            },
            "required": ["patch"],
        },
        on_invoke=patch_handler,
        requires_approval=patch_tool.approval,
    )


async def resolve_function_tool(
    agent: Agent,
    tool_name: str,
    ctx: RunContext[TContext] | None = None,
) -> FunctionTool | None:
    """Resolve a ``FunctionTool`` by name without mutating ``agent``.

    Shared lookup helper used by every execution path that needs to
    dispatch a tool call back to a concrete ``FunctionTool``:

    - Non-streaming tool execution (``_execute_single_tool_call``)
    - Streaming tool execution (same path)
    - Approved HITL tool execution (``execute_approved_tool``)
    - HITL resumption (``resumption.py``)
    - ``return_direct`` post-processing in turn resolution

    Resolution order:

    1. ``agent.tools`` — matching ``FunctionTool`` by name.
    2. Skill tools from ``agent.skills`` — matching ``FunctionTool`` by name.
    3. ``ShellTool`` / ``ApplyPatchTool`` in ``agent.tools`` with a local
       executor/editor — wrapped as a ``FunctionTool`` on demand.
    4. ``Toolset`` entries in ``agent.tools`` — materialised via
       ``await toolset.get_tools(ctx)`` (PrefixedToolset / RenamedToolset
       expose tools under names that don't appear on any standalone
       ``FunctionTool``, so the lookup falls through to the toolset
       layer).

    The wire-on-demand path replaces the old design where ``build_tools``
    mutated ``agent.tools`` to append the synthesized wrapper. That was a
    race condition under concurrent ``Runner.arun()`` on a shared ``Agent``
    instance.

    Args:
        agent: The agent whose tools to search.
        tool_name: The tool name to resolve.
        ctx: Run context. Forwarded to ``Toolset.get_tools(ctx)`` so
            context-dependent toolsets (e.g. ``FilteredToolset``)
            evaluate against the same context that selected the tool
            in the first place.

    Returns:
        The resolved ``FunctionTool``, or ``None`` when no match is found.
    """
    from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.local.apply_patch_tool import ApplyPatchTool
    from troopai.adk.tools.local.shell_tool import ShellTool
    from troopai.adk.tools.toolsets import Toolset

    if agent.tools:
        for tool in agent.tools:
            if isinstance(tool, FunctionTool) and tool.name == tool_name:
                return tool
            if isinstance(tool, ExecutableBuiltinTool) and tool.name == tool_name:
                return tool.to_function_tool()

    if agent.skills:
        from troopai.adk.skills.skill_rendering import check_skill_enabled

        for skill in agent.skills:
            if not await check_skill_enabled(skill, ctx):
                continue
            for skill_tool in skill.tools:
                if isinstance(skill_tool, FunctionTool) and skill_tool.name == tool_name:
                    return _apply_skill_governance(skill_tool, skill)
                if isinstance(skill_tool, ExecutableBuiltinTool) and skill_tool.name == tool_name:
                    return _apply_skill_governance(skill_tool.to_function_tool(), skill)

    if agent.tools:
        for tool in agent.tools:
            if isinstance(tool, ShellTool) and tool.executor is not None and tool.name == tool_name:
                return _build_shell_function_tool(tool)
            if isinstance(tool, ApplyPatchTool) and tool.editor is not None and tool.name == tool_name:
                return _build_apply_patch_function_tool(tool)

    # Toolset materialisation: each toolset re-runs ``get_tools(ctx)``
    # to find a match by name. The cost is one extra ``get_tools(ctx)``
    # call per concrete toolset; for the common case of a flat
    # FunctionToolset this is a dict construction, no I/O. Live wrappers
    # (filtered/prefixed/etc.) re-run their predicate / rename — both
    # are pure functions of ``ctx`` and the inner toolset, so they are
    # safe to call again.
    if agent.tools:
        for entry in agent.tools:
            if isinstance(entry, Toolset):
                materialised = await entry.get_tools(ctx)
                if tool_name in materialised:
                    return materialised[tool_name]

    return None


def _apply_skill_governance(tool: FunctionTool, skill: Any) -> FunctionTool:
    """Apply skill-level governance and guardrails as defaults for a tool.

    Governance values (timeout, max_result_tokens, max_retries) from the
    skill are applied only when the tool doesn't have its own value.
    Skill guardrails are prepended to the tool's guardrails so they run
    first.

    Args:
        tool: The function tool to apply governance to.
        skill: The skill providing governance and guardrails.

    Returns:
        A modified tool (via ``dataclasses.replace``) or the original
        if no changes were needed.
    """
    from dataclasses import replace as _replace

    replacements: dict[str, Any] = {}

    # Apply governance defaults
    governance = skill.governance
    if governance is not None:
        if governance.timeout is not None and tool.timeout is None:
            replacements["timeout"] = governance.timeout
        if governance.max_result_tokens is not None and tool.max_result_tokens is None:
            replacements["max_result_tokens"] = governance.max_result_tokens
        if governance.max_retries is not None and tool.max_retries is None:
            replacements["max_retries"] = governance.max_retries

    # Apply skill guardrails (prepend to tool's existing guardrails).
    # Both Skill and FunctionTool now use ``guardrails: ToolGuardrails | None``;
    # skill-level entries run first when both are configured.
    skill_input = (
        skill.guardrails.input if (skill.guardrails is not None and skill.guardrails.input is not None) else []
    )
    skill_output = (
        skill.guardrails.output if (skill.guardrails is not None and skill.guardrails.output is not None) else []
    )
    if len(skill_input) > 0 or len(skill_output) > 0:
        from troopai.adk.tools.tool_guardrails import ToolGuardrails

        existing_input = (
            tool.guardrails.input if (tool.guardrails is not None and tool.guardrails.input is not None) else []
        )
        existing_output = (
            tool.guardrails.output if (tool.guardrails is not None and tool.guardrails.output is not None) else []
        )
        merged_input = (
            list(skill_input) + list(existing_input)
            if len(skill_input) > 0
            else (existing_input if len(existing_input) > 0 else None)
        )
        merged_output = (
            list(skill_output) + list(existing_output)
            if len(skill_output) > 0
            else (existing_output if len(existing_output) > 0 else None)
        )
        replacements["guardrails"] = ToolGuardrails(
            input=merged_input if (merged_input is not None and len(merged_input) > 0) else None,
            output=merged_output if (merged_output is not None and len(merged_output) > 0) else None,
        )

    if len(replacements) > 0:
        logger.debug(
            "Applied skill '%s' governance to tool '%s': %s",
            skill.name,
            tool.name,
            list(replacements.keys()),
        )
        return _replace(tool, **replacements)
    return tool
