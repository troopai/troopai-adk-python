"""Runner — public facade for agent execution.

The Runner is the entry point for executing agents.  It delegates to
the extracted modules for the actual work:

- ``loop`` — agent loop (turn-by-turn execution cycle)
- ``llm_calls`` — LLM interaction (call, resolve, build tools)
- ``tools_executor`` — tool execution with 3-layer system
- ``guardrails_executor`` — input/output guardrails
- ``handoffs_executor`` — handoff strategies
- ``resumption`` — HITL state resumption
- ``cost`` — token cost optimization

Example:
    # Sync execution
    result = Runner.run(agent, "Hello!")

    # Async execution
    result = await Runner.arun(agent, "Hello!")

    # Streaming (sync return, async iteration)
    result = Runner.run(agent, "Hello!", stream=True)
    async for event in result.stream_events():
        logger.info(event)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, overload, override

from troopai.adk.exceptions import (
    AgentInputGuardrailTripwireTriggered,
    AgentOutputGuardrailTripwireTriggered,
)
from troopai.adk.hooks.hooks import RunHooks, compose_run_hooks
from troopai.adk.run.config import DEFAULT_MAX_TURNS, DEFAULT_RUN_CONFIG, RunConfig
from troopai.adk.run.context import RunContext, TContext
from troopai.adk.run.cost import validate_budget_config
from troopai.adk.run.guardrails_executor import (
    run_blocking_input_guardrails,
    run_output_guardrails,
    run_parallel_input_guardrails,
)
from troopai.adk.run.loop import run_agent_loop, run_agent_loop_streamed
from troopai.adk.run.resumption import resume_from_state, resume_from_state_streamed
from troopai.adk.run.state import RunState
from troopai.adk.run.stream import CancelMode, HookEventKind, HookLifecycleEvent, RunResultStreaming
from troopai.adk.tracing import agent_span
from troopai.adk.tracing.spans import swarm_span
from troopai.adk.types.items.items import MessageOutputItem
from troopai.adk.types.responses.llm_response import LLMResponseText
from troopai.adk.types.run import RunResult
from troopai.adk.types.tracing.span_data import CustomSpanData

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.flows.checkpoint import FlowCheckpoint
    from troopai.adk.flows.config import FlowConfig
    from troopai.adk.flows.flow import Flow
    from troopai.adk.flows.result import FlowRunResult, FlowRunResultStreaming
    from troopai.adk.flows.worker_backend import FlowWorkerBackend
    from troopai.adk.graphs.checkpointer import Checkpointer
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.hooks import GraphHooks, HookProvider
    from troopai.adk.graphs.interrupt import GraphResume
    from troopai.adk.graphs.result import (
        GraphRunResult,
        GraphRunResultStreaming,
    )
    from troopai.adk.graphs.state import GraphState
    from troopai.adk.memory.memory_config import MemoryConfig
    from troopai.adk.run.profile import RunnerProfile
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.swarms.checkpointer import SwarmCheckpointer
    from troopai.adk.swarms.interrupt import SwarmResume
    from troopai.adk.swarms.result import SwarmRunResult, SwarmRunResultStreaming
    from troopai.adk.swarms.state import SwarmState
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.tasks.task import Task
    from troopai.adk.tasks.task_group import TaskGroup, TaskGroupResult
    from troopai.adk.tasks.task_output import TaskOutput
    from troopai.adk.tasks.task_pipeline import TaskPipeline, TaskPipelineResult
    from troopai.adk.tasks.task_pipeline_state import TaskPipelineState
    from troopai.adk.tracing.spans import AgentSpanData, Span
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage
    from troopai.adk.types.input.llm_input_text import LLMInputText
    from troopai.adk.types.session import SessionStore

logger = logging.getLogger(__name__)


def _snapshot_run_config(run_config: RunConfig | None) -> RunConfig:
    """Return the run-owned config object for one public execution call."""
    if run_config is None:
        return DEFAULT_RUN_CONFIG.snapshot()
    return run_config.snapshot()


async def apply_output_transform(result: RunResult[Any] | RunResultStreaming, replacement: str) -> None:
    """Substitute a transformed text output everywhere the run exposes it.

    Sets ``final_output`` and rewrites the trailing assistant message in
    ``new_items`` so ``to_param()`` — and therefore the persisted session events
    and memory extraction — observe the masked text, not the raw output. Text
    outputs only: a run without a trailing text message only has its
    ``final_output`` replaced (a structured replacement is out of scope).
    """
    result.final_output = replacement
    for index in range(len(result.new_items) - 1, -1, -1):
        item = result.new_items[index]
        if isinstance(item, MessageOutputItem):
            result.new_items[index] = dataclasses.replace(item, raw=[LLMResponseText(text=replacement)])
            return
    logger.debug("Output transform set final_output; no trailing message item to rewrite")


def wrap_hooks_with_verbose(
    user_hooks: RunHooks[TContext] | None,
    config: RunConfig,
) -> RunHooks[TContext]:
    """Return the effective ``RunHooks`` for a run, layering verbose if enabled.

    When ``config.verbose`` is a :class:`VerboseConfig` with ``enabled=True``,
    a :class:`VerboseHooks` instance is composed with any user-provided hooks
    via :func:`compose_run_hooks`. Otherwise the user hooks are returned as-is
    (or a no-op :class:`RunHooks` when ``user_hooks`` is ``None``).

    Idempotent: when ``user_hooks`` already carries a :class:`VerboseHooks`
    layer (e.g. a streamed swarm member turn whose hooks the driver
    pre-wrapped), the chain is returned unchanged rather than stacking a
    second layer that would fire every verbose panel twice.
    """
    if config.verbose is not None and config.verbose.enabled:
        from troopai.adk.verbose.hooks import VerboseHooks, find_verbose_hooks

        if user_hooks is not None and len(find_verbose_hooks(user_hooks)) > 0:
            return user_hooks
        return compose_run_hooks(VerboseHooks(config.verbose), user_hooks)
    return user_hooks if user_hooks is not None else RunHooks()


def _sweep_verbose_panels(hooks: RunHooks[Any]) -> None:
    """Close verbose panels left open when a run ends by exception.

    Walks the effective hooks chain for :class:`VerboseHooks` layers and
    flushes each one's still-open generic block-tree panels (verdict
    ``"interrupted"``). A clean run closes its own blocks, so this is wired
    only into the exception teardown arms; a no-op when no verbose layer is
    installed.
    """
    from troopai.adk.verbose.hooks import find_verbose_hooks

    for verbose_hooks in find_verbose_hooks(hooks):
        verbose_hooks.close_all_panels()


def _activate_mcp_run_hooks_bridge(hooks: RunHooks[Any], ctx: Any) -> None:
    """Set the MCP module's ``ContextVar`` bridge so MCP-server lifecycle
    events fire ``RunHooks.on_mcp_*`` without explicit plumbing.

    Soft-imported: when the optional ``mcp`` extra is not installed,
    the bridge module is unavailable and we silently skip — the
    framework still works without MCP.
    """
    try:
        from troopai.adk.mcp.run_hooks_bridge import active_run_context, active_run_hooks
    except ImportError:
        return
    active_run_hooks.set(hooks)
    active_run_context.set(ctx)


def _resolve_swarm_id(
    initial_state: SwarmState[Any] | None,
) -> tuple[str, bool]:
    """Resolve effective swarm_id for a streamed run.

    Returns ``(swarm_id, regenerated)`` — ``regenerated=True`` indicates
    a caller-supplied ``initial_state`` was missing the field and a fresh
    UUID was generated. The caller logs the warning and stamps the new id
    back onto ``initial_state`` when applicable.
    """
    if initial_state is not None and initial_state.swarm_id is not None:
        return initial_state.swarm_id, False
    return str(uuid.uuid4()), True


@dataclasses.dataclass(slots=True)
class _RunLifecycle:
    """Context and hooks prepared for one Runner lifecycle bracket."""

    run_context: RunContext[Any]
    ctx_wrapper: RunContext[Any]
    hooks: RunHooks[Any]


def _build_run_lifecycle(
    context: Any,
    hooks: RunHooks[Any] | None,
    config: RunConfig,
) -> _RunLifecycle:
    """Build run context, wrapper, and resolved hooks for one run.

    Activates the MCP run-hooks bridge as a side-effect so MCP-server
    lifecycle events can flow through the same ``RunHooks`` instance as
    the rest of the run.
    """
    run_context: RunContext[Any] = RunContext.make(context, tenant_id=config.tenant_id)
    ctx_wrapper: RunContext[Any] = RunContext.from_run_context(run_context)
    resolved_hooks = wrap_hooks_with_verbose(hooks, config)
    _activate_mcp_run_hooks_bridge(resolved_hooks, ctx_wrapper)
    return _RunLifecycle(run_context=run_context, ctx_wrapper=ctx_wrapper, hooks=resolved_hooks)


def _build_swarm_run_context(
    context: Any,
    hooks: RunHooks[Any] | None,
    config: RunConfig,
) -> tuple[RunContext[Any], RunHooks[Any]]:
    """Build ctx_wrapper and resolved hooks for a streamed swarm run."""
    lifecycle = _build_run_lifecycle(context, hooks, config)
    return lifecycle.ctx_wrapper, lifecycle.hooks


async def _dispose_agent_toolsets(agent: Agent) -> None:
    """Dispose every ``Toolset`` instance on ``agent.tools``.

    Called from the ``finally`` block of both ``arun()`` and the
    streaming run impl so toolsets that hold network connections
    (e.g. ``MCPToolset``) release them after every run, even on
    exception. Cleanup runs sequentially in the same asyncio task
    that drove the run — ``MCPToolset``'s underlying anyio cancel
    scopes were entered in this task and MUST exit in it, so spawning
    ``asyncio.create_task`` per cleanup would break the invariant.

    Disposal runs in REVERSE registration order at the TOP level of
    ``agent.tools``. AnyIO stores cancel scopes per task on a LIFO
    stack; a toolset whose ``get_tools`` was awaited later in
    ``build_tools`` therefore sits on top of the one that ran first.
    Closing the older toolset first would try to pop a scope below
    the active head — that raises
    ``RuntimeError: Attempted to exit a cancel scope that isn't the
    current tasks's current cancel scope``. Mirrors OpenAI Agents
    SDK ``MCPServerManager.cleanup_all`` (``agents/mcp/manager.py``).

    Wrapper toolsets (``PrefixedToolset``, ``FilteredToolset``,
    ``RenamedToolset``) hold no cancel scopes themselves — they
    delegate ``adispose()`` to their wrapped target, so the top-level
    reversal correctly inverts the order of the underlying
    connections. ``CombinedToolset`` likewise reverses its own
    children on disposal so nested MCP topologies stay LIFO end-to-end.

    Per-toolset exceptions are logged at WARNING and do NOT block
    cleanup of the remaining toolsets.

    Only the entry-point agent's tools are disposed by this path.
    Toolsets contributed by handoff targets must be managed via an
    explicit ``MCPServerManager`` or ``async with``.
    """
    from troopai.adk.tools.toolsets import Toolset

    for entry in reversed(agent.tools):
        if isinstance(entry, Toolset):
            try:
                await entry.adispose()
            except Exception:
                logger.warning("Toolset adispose() raised; entry=%r", entry, exc_info=True)


@dataclasses.dataclass
class _GraphBracket:
    """Mutable handle for a graph-run verbose/trace bracket.

    The body may record a non-exception failure (e.g. a FAILED
    :class:`~troopai.adk.graphs.result.GraphRunStatus`) so the Task panel
    closes as failed even when no exception propagates.

    Attributes:
        error: Populated by :meth:`mark_failed` or by the context
            manager's exception handler. ``None`` means the run
            succeeded.
    """

    error: str | None = None
    """Populated by :meth:`mark_failed` or by the context manager on exception."""

    def mark_failed(self, message: str) -> None:
        """Record a non-exception failure reason for the Task panel."""
        self.error = message


@contextlib.asynccontextmanager
async def _graph_run_bracket(
    *,
    graph: Graph[Any],
    user_prompt: UserPrompt,
    config: RunConfig,
) -> AsyncIterator[_GraphBracket]:
    """Wrap a graph run in the verbose Task panel and OTel root span.

    On an exception the span is marked errored and the panel closes
    failed, then the exception propagates. A non-exception failure is
    reported by the body via :meth:`_GraphBracket.mark_failed`.
    """
    from troopai.adk.verbose.hooks import emit_task_end, emit_task_start

    verbose_hooks = wrap_hooks_with_verbose(None, config)
    task_id = str(uuid.uuid4())
    task_name = _derive_task_name(user_prompt)
    emit_task_start(verbose_hooks, graph, task_name, task_id)
    bracket = _GraphBracket()
    root_span = agent_span(
        name=f"graph:{graph.id}",
        tools=[],
        handoffs=[],
        output_type=None,
        metadata=config.tracing_metadata,
        tenant_id=config.tenant_id,
        disabled=not (config.tracing_enabled or config.metrics_enabled),
    )
    root_span.start()
    try:
        yield bracket
    except Exception as e:
        bracket.error = f"{type(e).__name__}: {e}"
        root_span.set_error(type(e).__name__, data={"message": str(e)})
        logger.error("Error during graph execution: %s", e)
        raise
    finally:
        if bracket.error is None:
            emit_task_end(verbose_hooks, graph, task_name, task_id, success=True)
        else:
            emit_task_end(
                verbose_hooks,
                graph,
                task_name,
                task_id,
                success=False,
                error=bracket.error,
            )
        root_span.finish()


def _resolve_error_handler(
    exc: Exception,
    handlers: dict[type[Exception], Any],
) -> Any | None:
    """Return the most-derived matching handler for ``exc``, or ``None``.

    Walks ``type(exc).__mro__`` in order (most-derived first) and returns
    the first handler whose key is a superclass (or exact match) of the
    exception's type.  Returns ``None`` when no entry matches.
    """
    for cls in type(exc).__mro__:
        if cls in handlers:
            return handlers[cls]
    return None


class _HookEventEmitter(RunHooks[Any]):
    """Wraps a :class:`~troopai.adk.run.stream.RunResultStreaming` to fan hook
    lifecycle moments into the event stream.

    Installed by :meth:`Runner._run_streamed_impl` when
    :attr:`~troopai.adk.run.config.RunConfig.include_hook_events` is ``True``.
    Emits a :class:`~troopai.adk.run.stream.HookLifecycleEvent` to the queue
    at each tool and guardrail hook call site.  All other hook methods
    delegate to the no-op base.
    """

    def __init__(self, result: RunResultStreaming) -> None:
        self._result = result

    async def _emit(
        self,
        kind: HookEventKind,
        agent: Any,
        payload: dict[str, Any],
    ) -> None:
        await self._result.put_event(HookLifecycleEvent(kind=kind, agent_name=agent.name, payload=payload))

    @override
    async def on_tool_start(self, context: Any, agent: Any, tool_name: str, tool_input: dict[str, Any]) -> None:
        del context
        await self._emit(HookEventKind.TOOL_START, agent, {"tool_name": tool_name, "tool_input": tool_input})

    @override
    async def on_tool_end(self, context: Any, agent: Any, tool_name: str, tool_output: Any) -> None:
        del context
        await self._emit(HookEventKind.TOOL_END, agent, {"tool_name": tool_name, "tool_output": tool_output})

    @override
    async def on_input_guardrail_start(self, context: Any, agent: Any, guardrail_name: str) -> None:
        del context
        await self._emit(HookEventKind.GUARDRAIL_INPUT_START, agent, {"guardrail_name": guardrail_name})

    @override
    async def on_input_guardrail_end(self, context: Any, agent: Any, result: Any) -> None:
        del context
        await self._emit(HookEventKind.GUARDRAIL_INPUT_END, agent, {"guardrail_result": result})

    @override
    async def on_output_guardrail_start(self, context: Any, agent: Any, guardrail_name: str) -> None:
        del context
        await self._emit(HookEventKind.GUARDRAIL_OUTPUT_START, agent, {"guardrail_name": guardrail_name})

    @override
    async def on_output_guardrail_end(self, context: Any, agent: Any, result: Any) -> None:
        del context
        await self._emit(HookEventKind.GUARDRAIL_OUTPUT_END, agent, {"guardrail_result": result})


class Runner:
    """Executes agents with guardrails, context, and hooks.

    The Runner is responsible for:
    1. Creating and managing the RunContext
    2. Running input guardrails (blocking and parallel)
    3. Executing the agent loop (LLM calls, tools, handoffs)
    4. Running output guardrails
    5. Tracking usage and calling hooks

    Supports two methods with an optional ``stream`` parameter:
    - run(): Synchronous blocking (for scripts and simple apps)
    - arun(): Async non-blocking (for async applications)

    Both accept ``stream=True`` to enable streaming with real-time events,
    mirroring the ``LLM.complete()`` / ``LLM.acomplete()`` pattern.

    Example (sync)::

        result = Runner.run(agent, "Hello!")

    Example (async)::

        result = await Runner.arun(agent, "Hello!")

    Example (streaming)::

        result = Runner.run(agent, "Hello!", stream=True)
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                logger.info(event.data)
    """

    @classmethod
    def configure(
        cls,
        run_config: RunConfig | None = None,
        *,
        context: TContext | None = None,
    ) -> RunnerProfile:
        """Create an immutable reusable runner profile.

        ``RunnerProfile`` stores target-agnostic defaults. Bind it to an
        executable primitive with ``.agent(...)``, ``.swarm(...)``,
        ``.graph(...)``, ``.task(...)``, ``.pipeline(...)``,
        ``.task_group(...)``, or ``.flow(...)``.
        """
        from troopai.adk.run.profile import RunnerProfile

        return RunnerProfile(run_config=run_config, context=context)

    # -- arun(): async entry point ------------------------------------------

    @overload
    @classmethod
    async def arun(
        cls,
        agent: Agent,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[False] = False,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> RunResult: ...

    @overload
    @classmethod
    async def arun(
        cls,
        agent: Agent,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[True],
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> RunResultStreaming: ...

    @classmethod
    async def arun(
        cls,
        agent: Agent,
        user_prompt: UserPrompt | RunState,
        *,
        stream: bool = False,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> RunResult | RunResultStreaming:
        """Run an agent asynchronously, with optional streaming.

        This is the main async entry point for executing an agent. It handles
        the full execution lifecycle including guardrails, LLM calls,
        tool execution, and handoffs. Supports resuming from a RunState
        for Human-in-the-Loop (HITL) workflows.

        Mirrors the ``LLM.acomplete(stream=...)`` pattern.

        Args:
            agent: The agent to run.
            user_prompt: User prompt (string or message list), or RunState to resume.
            stream: If True, return RunResultStreaming with event iterator.
            context: Optional user context to pass through execution.
            hooks: Optional lifecycle hooks.
            max_turns: Maximum turns in the agent loop (default 10).
            run_config: Optional execution configuration.
            session: Optional session for conversation persistence.
            memory: Optional memory configuration for knowledge injection/extraction.

        Returns:
            RunResult when stream=False; RunResultStreaming when stream=True.
            If HITL is triggered, result.requires_action will be True
            and deferred_requests will contain tools needing approval.

        Raises:
            AgentInputGuardrailTripwireTriggered: If an input guardrail fails.
            AgentOutputGuardrailTripwireTriggered: If an output guardrail fails.
            MaxTurnsExceeded: If max_turns is reached without final output.
        """
        if stream:
            return cls._run_streamed(
                agent=agent,
                user_prompt=user_prompt,
                context=context,
                hooks=hooks,
                max_turns=max_turns,
                run_config=run_config,
                session=session,
                memory=memory,
            )

        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)

        # Check if resuming from state
        if isinstance(user_prompt, RunState):
            return await resume_from_state(
                agent=agent,
                state=user_prompt,
                hooks=hooks,
                max_turns=max_turns,
                config=config,
                context=context,
            )

        # Create run context (before session load so hooks have context)
        lifecycle = _build_run_lifecycle(context, hooks, config)
        run_context: RunContext[TContext] = lifecycle.run_context
        ctx_wrapper: RunContext[TContext] = lifecycle.ctx_wrapper
        hooks = lifecycle.hooks

        # --- Bracketing state (initialised before opening any bracket) ----
        # These are set up front so the finally can tear down whatever opened,
        # even if a setup step below (sandbox open, session load, memory,
        # start hooks, span build) raises. The 📋 Task panel is emitted now —
        # paired with emit_task_end in finally — so the panel stays balanced
        # regardless of where setup fails.
        task_id = str(uuid.uuid4())
        task_name = _derive_task_name(user_prompt)
        task_error: str | None = None
        _root_span: Span[AgentSpanData] | None = None
        original_user_prompt = user_prompt
        effective_input: UserPrompt = user_prompt
        from troopai.adk.verbose.hooks import emit_task_end, emit_task_start

        emit_task_start(hooks, agent, task_name, task_id)

        # --- Sandbox lifecycle bracket ---------------------------------
        # When the agent is a SandboxAgent OR RunConfig.sandbox is non-None,
        # open the sandbox session for the duration of this arun call.
        # Opened INSIDE the try so a failure to open it (or in any later setup
        # step) still runs the finally that closes it via _sandbox_stack.aclose()
        # — otherwise a partially-opened sandbox session leaks. After the bracket
        # opens, ``agent`` is rebound to a CLONE that carries the capability
        # tools so the agent loop's ``build_tools`` sees them automatically.
        _sandbox_stack = contextlib.AsyncExitStack()
        await _sandbox_stack.__aenter__()

        try:
            await _maybe_open_sandbox_bracket(
                stack=_sandbox_stack,
                agent=agent,
                config=config,
                run_context=run_context,
                hooks=hooks,
            )
            agent = _maybe_clone_agent_with_capability_tools(
                agent=agent,
                run_context=run_context,
            )

            # --- Reset deferred-tool revealed sets ------------------------
            # Sequential ``await Runner.arun()`` calls from the same coroutine
            # share the same asyncio context, so a ContextVar.set() in run #1
            # is still visible at the start of run #2.  Reset every search
            # tool's revealed set here so each run starts with a clean slate.
            # (Concurrent tasks are already isolated by the context copy that
            # ``asyncio.create_task`` performs at scheduling time.)
            from troopai.adk.tools.tool_search import reset_revealed_sets

            reset_revealed_sets(agent.tools)

            # --- Session: load history --------------------------------------
            if session is not None and isinstance(user_prompt, str):
                limit = None
                if session.settings is not None:
                    limit = session.settings.limit
                events = await session.get(limit=limit)
                if len(events) > 0:
                    await hooks.on_session_load(ctx_wrapper, session, events)
                    history = [e.content for e in events]
                    user_msg: LLMInputEasyMessage = {"role": "user", "content": user_prompt}
                    effective_input = [*history, user_msg]

            # --- Memory: inject relevant memories ---------------------------
            if memory is not None and memory.inject:
                effective_input = await _inject_memories(effective_input, memory)

            # Call hooks
            await hooks.on_agent_start(ctx_wrapper, agent)
            if agent.hooks is not None:
                await agent.hooks.on_start(ctx_wrapper, agent)

            # Root agent span for the entire run. Children (generation,
            # function, handoff, guardrail) attach via the contextvars chain.
            _root_span = agent_span(
                name=agent.name,
                tools=[getattr(t, "name", str(t)) for t in agent.tools],
                handoffs=_handoff_names_for_span(agent),
                output_type=_output_type_name_for_span(agent),
                metadata=config.tracing_metadata,
                tenant_id=ctx_wrapper.tenant_id,
                disabled=not (config.tracing_enabled or config.metrics_enabled),
            )
            _root_span.start()

            # Run blocking input guardrails before agent starts
            blocking_results = await run_blocking_input_guardrails(
                agent,
                effective_input,
                ctx_wrapper,
                hooks,
                config.guardrails.input,
                tracing_enabled=config.tracing_enabled,
                metrics_enabled=config.metrics_enabled,
            )

            # Start parallel input guardrails concurrently with agent loop
            parallel_task = asyncio.create_task(
                run_parallel_input_guardrails(
                    agent,
                    effective_input,
                    ctx_wrapper,
                    hooks,
                    config.guardrails.input,
                    tracing_enabled=config.tracing_enabled,
                    metrics_enabled=config.metrics_enabled,
                )
            )

            # Run agent loop and parallel guardrails together.
            # If the agent loop raises, cancel the parallel task to avoid
            # leaked coroutines.
            try:
                result = await run_agent_loop(
                    agent=agent,
                    user_prompt=effective_input,
                    context=run_context,
                    ctx_wrapper=ctx_wrapper,
                    hooks=hooks,
                    max_turns=max_turns,
                    config=config,
                )

                # Await parallel guardrails (raises if tripwire triggered)
                parallel_results = await parallel_task
            except BaseException:
                if not parallel_task.done():
                    parallel_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await parallel_task
                elif not parallel_task.cancelled():
                    # Task finished while the agent loop was raising.
                    # If it completed with a guardrail tripwire, surface that
                    # instead of the agent-loop exception — the security concern
                    # takes priority and the caller sees the right error.
                    task_exc = parallel_task.exception()
                    if task_exc is not None:
                        if isinstance(task_exc, AgentInputGuardrailTripwireTriggered):
                            raise task_exc
                        logger.warning(
                            "parallel input guardrail raised %s while agent loop was also raising; "
                            "agent-loop exception takes precedence",
                            type(task_exc).__name__,
                        )
                raise

            # Results are in input order (asyncio.gather preserves it)
            result.guardrail_results.input = tuple(blocking_results + parallel_results)

            # Output guardrails run on the *final* agent after handoffs.
            # RunResult.last_agent tracks the agent that produced the output;
            # falls back to the starting agent if no handoff occurred.
            if not result.requires_action:
                output_agent = result.last_agent or agent
                output_results = await run_output_guardrails(
                    output_agent,
                    result.final_output,
                    ctx_wrapper,
                    hooks,
                    config.guardrails.output,
                    on_transform=lambda replacement: apply_output_transform(result, replacement),
                    tracing_enabled=config.tracing_enabled,
                    metrics_enabled=config.metrics_enabled,
                )
                result.guardrail_results.output = tuple(output_results)

            result.guardrail_audit = ctx_wrapper.collect_guardrail_audit()

            # Call hooks
            await hooks.on_agent_end(ctx_wrapper, agent, result)
            if agent.hooks is not None:
                await agent.hooks.on_end(ctx_wrapper, agent, result.final_output)

            # --- Build events from results --------------------------------
            from troopai.adk.session.session_event import SessionEvent, create_session_event

            events_to_save: list[SessionEvent] = []
            if isinstance(original_user_prompt, str):
                user_event_msg: LLMInputEasyMessage = {"role": "user", "content": original_user_prompt}
                events_to_save.append(
                    create_session_event(
                        author="user",
                        content=user_event_msg,
                    )
                )
            for item in result.new_items:
                events_to_save.append(
                    create_session_event(
                        author=_infer_author(item),
                        content=item.to_param(),
                    )
                )

            # --- Session: save new events ---------------------------------
            if session is not None and len(events_to_save) > 0:
                await session.add(events_to_save)
                try:
                    await session.save_state()
                except Exception:
                    logger.warning("session.save_state() failed; state delta may not be persisted", exc_info=True)
                await hooks.on_session_save(ctx_wrapper, session, events_to_save)

            # --- Memory: extract from conversation -----------------------
            if memory is not None and memory.auto_extract and memory.extractor is not None and len(events_to_save) > 0:
                await memory.memory.add_events(
                    events_to_save,
                    namespace=memory.namespace,
                    extractor=memory.extractor,
                    session_id=session.id if session is not None else None,
                    agent_name=agent.name,
                )

            sandbox_handle = getattr(run_context, "_sandbox_handle", None)
            if sandbox_handle is not None:
                observability = sandbox_handle.observability
                if observability is not None:
                    result.sandbox_usage = observability.usage

            return result

        except (AgentInputGuardrailTripwireTriggered, AgentOutputGuardrailTripwireTriggered) as e:
            task_error = f"{type(e).__name__}: {e}"
            if _root_span is not None:
                _root_span.set_error(type(e).__name__, data={"message": str(e)})
            raise
        except Exception as e:
            task_error = f"{type(e).__name__}: {e}"
            if _root_span is not None:
                _root_span.set_error(type(e).__name__, data={"message": str(e)})
            if config.error_handlers is not None:
                handler = _resolve_error_handler(e, config.error_handlers)
                if handler is not None:
                    logger.warning(
                        "Error handler recovering from %s: %s",
                        type(e).__name__,
                        e,
                    )
                    raw = handler(e)
                    fallback = await raw if inspect.isawaitable(raw) else raw
                    recovered: RunResult = RunResult(
                        final_output=fallback,
                        user_prompt=effective_input,
                        context=run_context,
                        recovered=True,
                    )
                    await hooks.on_agent_end(ctx_wrapper, agent, recovered)
                    if agent.hooks is not None:
                        await agent.hooks.on_end(ctx_wrapper, agent, recovered.final_output)
                    task_error = None
                    return recovered
            logger.error("Error during agent execution: %s", e)
            raise
        finally:
            if task_error is None:
                emit_task_end(hooks, agent, task_name, task_id, success=True)
            else:
                emit_task_end(hooks, agent, task_name, task_id, success=False, error=task_error)
                # Sweep any generic verbose tree blocks left open when the run
                # ended by exception; a clean run closes its own blocks.
                _sweep_verbose_panels(hooks)
            if _root_span is not None:
                _root_span.finish()
            await _dispose_agent_toolsets(agent)
            # Close the sandbox bracket LAST so the session outlives
            # everything that may have referenced it during the run.
            await _sandbox_stack.aclose()

    # -- run(): sync entry point ---------------------------------------------

    @overload
    @classmethod
    def run(
        cls,
        agent: Agent,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[False] = False,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> RunResult: ...

    @overload
    @classmethod
    def run(
        cls,
        agent: Agent,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[True],
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> RunResultStreaming: ...

    @classmethod
    def run(
        cls,
        agent: Agent,
        user_prompt: UserPrompt | RunState,
        *,
        stream: bool = False,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> RunResult | RunResultStreaming:
        """Run an agent synchronously, with optional streaming.

        This is the sync entry point. Blocks until execution completes
        (unless ``stream=True``, which returns immediately with a
        RunResultStreaming).

        Mirrors the ``LLM.complete(stream=...)`` pattern.

        Args:
            agent: The agent to run.
            user_prompt: User prompt (string or message list), or RunState to resume.
            stream: If True, return RunResultStreaming with event iterator.
            context: Optional user context to pass through execution.
            hooks: Optional lifecycle hooks.
            max_turns: Maximum turns in the agent loop (default 10).
            run_config: Optional execution configuration.
            session: Optional session for conversation persistence.
            memory: Optional memory configuration for knowledge injection/extraction.

        Returns:
            RunResult when stream=False; RunResultStreaming when stream=True.
        """
        if stream:
            return cls._run_streamed(
                agent=agent,
                user_prompt=user_prompt,
                context=context,
                hooks=hooks,
                max_turns=max_turns,
                run_config=run_config,
                session=session,
                memory=memory,
            )

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context - run in thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    cls.arun(
                        agent,
                        user_prompt,
                        context=context,
                        hooks=hooks,
                        max_turns=max_turns,
                        run_config=run_config,
                        session=session,
                        memory=memory,
                    ),
                )
                return future.result()
        else:
            # No running loop - use asyncio.run
            return asyncio.run(
                cls.arun(
                    agent,
                    user_prompt,
                    context=context,
                    hooks=hooks,
                    max_turns=max_turns,
                    run_config=run_config,
                    session=session,
                    memory=memory,
                )
            )

    # -- Swarm entry points -------------------------------------------------

    @classmethod
    async def arun_swarm(
        cls,
        swarm: Swarm,
        user_prompt: UserPrompt,
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        checkpointer: SwarmCheckpointer | None = None,
    ) -> SwarmRunResult:
        """Execute a swarm asynchronously.

        Orchestrates :class:`~troopai.adk.swarms.swarm.Swarm` execution
        end-to-end. Delegates to :func:`run_swarm_loop` for the driver
        loop, which in turn delegates to :func:`run_agent_loop` for each
        member turn. Swarms do not resume from
        :class:`~troopai.adk.run.state.RunState` — pause/resume
        semantics go through
        :class:`~troopai.adk.swarms.result.SwarmRunResult.state`
        (serializable via ``to_json()``) instead.

        Session handling: when ``session`` is supplied *and*
        ``user_prompt`` is a string, the session's persisted history is
        loaded and prepended to the swarm's opening turn, mirroring
        :meth:`Runner.arun`. Subsequent turns use the swarm's own
        :class:`~troopai.adk.swarms.shared_context_strategy.SharedContextStrategy`.

        Args:
            swarm: The swarm configuration (roster, policy, termination,
                budgets).
            user_prompt: Input passed to the entry agent (string or
                Layer 1 :class:`LLMInputContentItem` list).
            context: Optional user context passed to every member.
            hooks: Optional :class:`RunHooks`. Swarm-level
                :class:`~troopai.adk.swarms.hooks.SwarmHooks` attached to
                ``swarm.hooks`` fire in addition.
            run_config: Optional :class:`RunConfig`. ``max_total_turns``
                is the absolute LLM-call safety net; swarm-specific
                budgets live on :attr:`Swarm.config`.
            session: Optional :class:`~troopai.adk.types.session.SessionStore` for conversation
                persistence.
            checkpointer: Optional :class:`SwarmCheckpointer`. When
                supplied, auto-saves the run state after each completed
                turn and on interrupt via the swarm hook registry. The
                checkpointer's own ``thread_id`` (set at construction)
                is used as the logical run key.

        Returns:
            A :class:`SwarmRunResult` with the terminal output and
            serializable final state.

        Raises:
            MaxTurnsExceeded: From the inner runner when
                ``config.max_total_turns`` is exceeded.

        The ``SwarmConfig.max_handoffs`` / ``max_total_tokens`` hard
        guards do NOT raise — they end the run cleanly with
        ``stop_reason.kind == "max_handoffs" | "max_total_tokens"``.
        """
        # Deferred import: swarm_loop pulls in troopai.adk.swarms.* which
        # transitively imports from troopai.adk.handoffs.handoff. Hoisting
        # this import to module top creates a circular dep through
        # tools.function_tool -> run.context -> run.__init__ -> run.runner.
        from troopai.adk.run.swarm_loop import run_swarm_loop
        from troopai.adk.session.session_event import (
            SessionEvent,
            create_session_event,
        )

        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)

        # Create run context wrapper threaded through the driver and every
        # inner run_agent_loop call so usage accumulates.
        lifecycle = _build_run_lifecycle(context, hooks, config)
        ctx_wrapper: RunContext[TContext] = lifecycle.ctx_wrapper
        resolved_hooks = lifecycle.hooks

        # --- Verbose: 📋 Task panel brackets the whole swarm run.
        task_id = str(uuid.uuid4())
        task_name = _derive_task_name(user_prompt)
        from troopai.adk.verbose.hooks import emit_task_end, emit_task_start

        emit_task_start(resolved_hooks, swarm.entry, task_name, task_id)
        task_error: str | None = None

        # Session: load history before the opening turn.
        original_user_prompt = user_prompt
        effective_input: UserPrompt = user_prompt
        if session is not None and isinstance(user_prompt, str):
            limit = None
            if session.settings is not None:
                limit = session.settings.limit
            events = await session.get(limit=limit)
            if len(events) > 0:
                await resolved_hooks.on_session_load(ctx_wrapper, session, events)
                history = [e.content for e in events]
                user_msg: LLMInputEasyMessage = {
                    "role": "user",
                    "content": user_prompt,
                }
                effective_input = [*history, user_msg]

        # Root swarm span for the whole swarm run — children (per-turn
        # spans plus their per-member generation / function / handoff
        # spans) attach via contextvars.
        swarm_id_for_run = str(uuid.uuid4())
        _root_span = swarm_span(
            swarm_id=swarm_id_for_run,
            entry=swarm.entry.name,
            disabled=not (config.tracing_enabled or config.metrics_enabled),
        )
        _root_span.start()

        try:
            result = await run_swarm_loop(
                swarm=swarm,
                user_prompt=effective_input,
                ctx_wrapper=ctx_wrapper,
                hooks=resolved_hooks,
                config=config,
                swarm_id=swarm_id_for_run,
                checkpointer=checkpointer,
            )
            if (config.tracing_enabled or config.metrics_enabled) and result.state is not None:
                # Mirror the _stamp_turn_span pattern: only stamp the span
                # payload when span tracing OR metrics is enabled (i.e.
                # whenever the span was actually created); the disabled branch
                # returns a NoOpSpan whose ``data`` is the typed
                # SwarmSpanData (no ``.data`` dict to write into).
                payload = cast(CustomSpanData, _root_span.data).data
                payload["status"] = result.stop_reason.kind
                payload["turns_total"] = result.state.total_turns

            # Session: save new events from the full swarm run.
            if session is not None:
                events_to_save: list[SessionEvent] = []
                if isinstance(original_user_prompt, str):
                    user_event_msg: LLMInputEasyMessage = {
                        "role": "user",
                        "content": original_user_prompt,
                    }
                    events_to_save.append(
                        create_session_event(
                            author="user",
                            content=user_event_msg,
                        )
                    )
                for item in result.new_items:
                    events_to_save.append(
                        create_session_event(
                            author=_infer_author(item),
                            content=item.to_param(),
                        )
                    )
                if len(events_to_save) > 0:
                    await session.add(events_to_save)
                    try:
                        await session.save_state()
                    except Exception:
                        logger.warning("session.save_state() failed; state delta may not be persisted", exc_info=True)
                    await resolved_hooks.on_session_save(
                        ctx_wrapper,
                        session,
                        events_to_save,
                    )

            return result

        except Exception as e:
            task_error = f"{type(e).__name__}: {e}"
            _root_span.set_error(type(e).__name__, data={"message": str(e)})
            logger.error("Error during swarm execution: %s", e)
            raise
        finally:
            if task_error is None:
                emit_task_end(resolved_hooks, swarm.entry, task_name, task_id, success=True)
            else:
                emit_task_end(
                    resolved_hooks,
                    swarm.entry,
                    task_name,
                    task_id,
                    success=False,
                    error=task_error,
                )
            _root_span.finish()
            # Dispose toolsets for every member agent so MCP connections
            # opened by auto_connect=True during run_agent_loop are closed.
            # Each member may hold independent server subprocesses/HTTP
            # connections; leaving them open leaks N×T connections for a
            # swarm with N members and T total turns.
            for member_agent in swarm.members:
                await _dispose_agent_toolsets(member_agent)

    @classmethod
    def run_swarm(
        cls,
        swarm: Swarm,
        user_prompt: UserPrompt,
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        checkpointer: SwarmCheckpointer | None = None,
    ) -> SwarmRunResult:
        """Execute a swarm synchronously. Blocks until the run completes.

        Sync wrapper around :meth:`arun_swarm`. Uses the same strategy as
        :meth:`Runner.run`: when invoked inside a running event loop, the
        call is offloaded to a worker thread via
        :class:`concurrent.futures.ThreadPoolExecutor`; otherwise
        :func:`asyncio.run` drives the coroutine.

        See :meth:`arun_swarm` for argument and return semantics.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    cls.arun_swarm(
                        swarm,
                        user_prompt,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        session=session,
                        checkpointer=checkpointer,
                    ),
                )
                return future.result()

        return asyncio.run(
            cls.arun_swarm(
                swarm,
                user_prompt,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                checkpointer=checkpointer,
            )
        )

    @classmethod
    async def arun_swarm_streamed(
        cls,
        swarm: Swarm[TContext],
        user_prompt: UserPrompt,
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        initial_state: SwarmState[TContext] | None = None,
        resume: SwarmResume | None = None,
        checkpointer: SwarmCheckpointer | None = None,
    ) -> SwarmRunResultStreaming[TContext]:
        """Execute a swarm with real-time event streaming.

        Returns a :class:`SwarmRunResultStreaming` immediately; events
        are produced in the background and consumed via
        :meth:`SwarmRunResultStreaming.stream_events`. Cancellation is
        available via :meth:`SwarmRunResultStreaming.cancel`.

        Args:
            swarm: The compiled :class:`Swarm` to execute.
            user_prompt: Input passed to the entry member.
            context: Optional user context threaded to every member.
            hooks: Optional :class:`RunHooks`.
            run_config: Optional :class:`RunConfig`.
            session: Unused — accepted for signature parity with
                :meth:`arun_swarm`.
            initial_state: Optional restored :class:`SwarmState` for
                resume. Threaded into ``run_swarm_loop_streamed``; the
                same deep-resume splice from ``swarm_resume.py`` fires
                inside the streamed loop when ``resume`` is also supplied.

                **Note:** on a checkpoint missing ``swarm_id`` (older
                persisted payload), the runner regenerates a fresh UUID
                and **mutates** ``initial_state.swarm_id`` in place so
                the returned ``SwarmRunResultStreaming.state.swarm_id``
                matches the new root span's id. Callers that hold a
                reference to ``initial_state`` after this call will see
                the regenerated id rather than ``None``.
            resume: Optional :class:`SwarmResume` carrying per-member
                replies. Applied at the splice point inside
                ``run_swarm_loop_streamed`` exactly as in the synchronous
                ``arun_swarm_from_checkpoint``.
            checkpointer: Optional :class:`SwarmCheckpointer`. When
                supplied, auto-saves the run state after each completed
                turn and on interrupt via the swarm hook registry. The
                checkpointer's own ``thread_id`` (set at construction)
                is used as the logical run key.

        Returns:
            :class:`SwarmRunResultStreaming` whose ``stream_events()``
            yields events until the run completes or is cancelled.
        """
        # Local imports — deferred to avoid circular dependencies
        # between runner.py and the swarm modules.
        from troopai.adk.run.swarm_loop_streamed import run_swarm_loop_streamed
        from troopai.adk.swarms.result import SwarmRunResultStreaming

        del session  # parity-only; accepted for signature parity with arun_swarm.

        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)
        swarm_id_for_run, regenerated = _resolve_swarm_id(initial_state)
        if regenerated and initial_state is not None:
            logger.warning(
                "arun_swarm_streamed: loaded checkpoint has no swarm_id "
                "(field absent from the persisted payload); regenerating. "
                "Trace correlation across this resume boundary will not "
                "work — the new root span starts a fresh troopai.swarm.id."
            )
            initial_state.swarm_id = swarm_id_for_run

        ctx_wrapper, resolved_hooks = _build_swarm_run_context(context, hooks, config)

        result: SwarmRunResultStreaming[TContext] = SwarmRunResultStreaming(
            user_prompt=user_prompt,
        )

        async def _driver() -> None:
            _root_span = swarm_span(
                swarm_id=swarm_id_for_run,
                entry=swarm.entry.name,
                disabled=not (config.tracing_enabled or config.metrics_enabled),
            )
            _root_span.start()
            try:
                await run_swarm_loop_streamed(
                    swarm=swarm,
                    user_prompt=user_prompt,
                    ctx_wrapper=ctx_wrapper,
                    hooks=resolved_hooks,
                    config=config,
                    result=result,
                    initial_state=initial_state,
                    swarm_resume=resume,
                    swarm_id=swarm_id_for_run,
                    checkpointer=checkpointer,
                )
                if (config.tracing_enabled or config.metrics_enabled) and result.state is not None:
                    # Only stamp the span payload when span tracing OR metrics
                    # is enabled (i.e. whenever the span was actually created).
                    payload = cast(CustomSpanData, _root_span.data).data
                    payload["status"] = result.stop_reason.kind if result.stop_reason is not None else "completed"
                    payload["turns_total"] = result.state.total_turns
            except Exception as exc:
                _root_span.set_error(type(exc).__name__, data={"message": str(exc)})
                result.set_exception(exc)
                raise  # propagate to the task so consumer sees it via task.exception()
            finally:
                _root_span.finish()
                # Dispose toolsets for every member agent so MCP connections
                # opened by auto_connect=True during streaming turns are closed.
                for member_agent in swarm.members:
                    await _dispose_agent_toolsets(member_agent)
                # Safety: run_swarm_loop_streamed calls complete() in its own
                # finally; this idempotent call covers any path where the loop
                # exits without doing so (mock, early return, etc.).
                await result.complete()

        try:
            task = asyncio.get_running_loop().create_task(_driver())
            result.set_run_task(task)
        except RuntimeError:
            result.set_deferred_run_impl(_driver)

        return result

    # -- Graph entry points -------------------------------------------------

    @classmethod
    async def _run_graph_bracketed(
        cls,
        *,
        graph: Graph[Any],
        user_prompt: UserPrompt,
        run_context: RunContext[Any],
        config: RunConfig,
        hooks: list[GraphHooks[Any] | HookProvider] | None,
        thread_id: str | None,
        initial_state: GraphState[Any] | None,
        resume: GraphResume | None = None,
    ) -> GraphRunResult[Any]:
        """Run :func:`run_graph_loop` wrapped in the verbose Task panel
        and OTel root span, mapping a FAILED graph result to a failed
        panel close. Shared by :meth:`arun_graph` and
        :meth:`arun_graph_from_checkpoint`."""
        from troopai.adk.graphs.result import GraphRunStatus
        from troopai.adk.run.graph_loop import run_graph_loop

        async with _graph_run_bracket(graph=graph, user_prompt=user_prompt, config=config) as bracket:
            result = await run_graph_loop(
                graph=graph,
                user_prompt=user_prompt,
                context=run_context,
                config=config,
                hooks=hooks,
                thread_id=thread_id,
                initial_state=initial_state,
                resume=resume,
            )
            if result.status == GraphRunStatus.FAILED:
                bracket.mark_failed(result.error if result.error is not None else "graph failed")
            return result

    @classmethod
    async def arun_graph(
        cls,
        graph: Graph[Any],
        user_prompt: UserPrompt,
        *,
        context: TContext | None = None,
        hooks: list[GraphHooks[Any] | HookProvider] | None = None,
        run_config: RunConfig | None = None,
        thread_id: str | None = None,
    ) -> GraphRunResult[Any]:
        """Execute a :class:`Graph` asynchronously.

        Orchestrates end-to-end graph execution. Delegates to
        :func:`run_graph_loop` for the BSP superstep driver, which in
        turn delegates per-node execution to :class:`Executable.invoke`
        — so an :class:`~troopai.adk.agents.agent.Agent` node runs through
        :func:`run_agent_loop`, a :class:`~troopai.adk.swarms.swarm.Swarm`
        node runs through :func:`run_swarm_loop`, and a nested
        :class:`Graph` node runs through this same driver (one level
        deeper).

        Graphs do NOT auto-load session history. Each node with an
        attached session manages its own persistence; the graph layer
        is pure orchestration over executables.

        Args:
            graph: The compiled graph to execute.
            user_prompt: Input passed to the entry node (string or
                Layer 1 :class:`LLMInputContentItem` list).
            context: Optional user context threaded to every node.
            hooks: Optional list of :class:`GraphHooks` and/or
                :class:`HookProvider`\\ s. Providers register via
                :meth:`HookProvider.register`; plain :class:`GraphHooks`
                are added directly. :class:`Checkpointer`\\ s are
                :class:`HookProvider`\\ s — attach them here.
            run_config: Optional :class:`RunConfig`. ``tracing_enabled``
                / ``metrics_enabled`` / ``tracing_metadata`` flow through to the
                ``graph:<id>`` root span; per-node config is threaded
                to the inner :func:`run_agent_loop` / :func:`run_swarm_loop`
                calls.
            thread_id: Optional checkpointer thread id. When omitted
                and a :class:`Checkpointer` is in ``hooks``, the driver
                auto-generates a ``thread-XXXX`` id.

        Returns:
            A :class:`GraphRunResult` with the terminal output, status,
            per-node usage attribution, and the full
            :class:`GraphState`.

        Raises:
            MaxTurnsExceeded: From inner agent loops when
                ``config.max_total_turns`` is exceeded inside a node.
            Exception: Any exception raised by a node with
                :attr:`GraphConfig.fail_fast` enabled; siblings are
                cancelled and the error surfaces on the
                :class:`GraphRunResult.error` field (and is re-raised).
        """
        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)
        run_context: RunContext[Any] = RunContext.make(context, tenant_id=config.tenant_id)
        return await cls._run_graph_bracketed(
            graph=graph,
            user_prompt=user_prompt,
            run_context=run_context,
            config=config,
            hooks=hooks,
            thread_id=thread_id,
            initial_state=None,
        )

    @classmethod
    def run_graph(
        cls,
        graph: Graph[Any],
        user_prompt: UserPrompt,
        *,
        context: TContext | None = None,
        hooks: list[GraphHooks[Any] | HookProvider] | None = None,
        run_config: RunConfig | None = None,
        thread_id: str | None = None,
    ) -> GraphRunResult[Any]:
        """Execute a graph synchronously. Blocks until the run completes.

        Sync wrapper around :meth:`arun_graph`. Uses the same strategy as
        :meth:`Runner.run`: when invoked inside a running event loop,
        the call is offloaded to a worker thread via
        :class:`concurrent.futures.ThreadPoolExecutor`; otherwise
        :func:`asyncio.run` drives the coroutine.

        See :meth:`arun_graph` for argument and return semantics.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    cls.arun_graph(
                        graph,
                        user_prompt,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        thread_id=thread_id,
                    ),
                )
                return future.result()

        return asyncio.run(
            cls.arun_graph(
                graph,
                user_prompt,
                context=context,
                hooks=hooks,
                run_config=run_config,
                thread_id=thread_id,
            )
        )

    @classmethod
    async def arun_swarm_from_checkpoint(
        cls,
        swarm: Swarm,
        *,
        checkpointer: SwarmCheckpointer,
        thread_id: str,
        user_prompt: UserPrompt = "",
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        resume: SwarmResume | None = None,
    ) -> SwarmRunResult:
        """Resume a swarm run from a persisted checkpoint.

        Loads :class:`SwarmState` via ``checkpointer.load``, rehydrates
        the state against ``swarm``, clears any
        :attr:`SwarmState.pending_interrupts` parked at suspend time,
        and re-enters :func:`run_swarm_loop` with the carried-over
        ``total_turns`` / ``shared_history`` / ``per_agent_scratch``.
        The checkpointer continues auto-saving after each turn during
        the resumed run via the swarm hook registry.

        Args:
            swarm: Compiled swarm the checkpoint was produced from
                (matching member names).
            checkpointer: The :class:`SwarmCheckpointer` holding the run.
                Also passed into the resumed loop so auto-saving continues
                for the duration of the resumed run.
            thread_id: Logical run key to resume.
            user_prompt: Optional input passed to the next member turn.
                On resume the swarm carries shared history forward, so
                this is usually empty.
            context: Optional user context threaded to every member.
            hooks: Optional :class:`RunHooks`.
            run_config: Optional :class:`RunConfig`.
            resume: Optional :class:`SwarmResume` carrying per-member
                replies. Threaded into :func:`run_swarm_loop`'s
                splice at step 7: a nested-agent-defer reply is applied
                via :meth:`AgentExecutable.resume_from_snapshot`, and a
                pure-HITL reply is seeded onto the run context for the
                parked member's tool to consume on its re-fire. When
                ``None`` the loop falls back to the clear-and-restart
                code path and re-runs the parked turn from scratch.

        Note:
            The ``checkpointer`` MUST have been constructed with the same
            ``thread_id`` as the ``thread_id`` argument passed here. The load
            uses the ``thread_id`` argument to find the checkpoint, but
            auto-saves during the resumed run write under the checkpointer's
            own ``thread_id`` (set at construction). A mismatch would load
            from one key and save to a different one.

        Returns:
            A :class:`SwarmRunResult` from resume to halt.

        Raises:
            ValueError: No checkpoint for ``thread_id``.
        """
        # Deferred import: see :meth:`arun_swarm` for the circular-dep
        # rationale that keeps this lookup local to the call site.
        from troopai.adk.run.swarm_loop import run_swarm_loop
        from troopai.adk.swarms.state import SwarmState, SwarmStateDict

        # ``resume`` is threaded into ``run_swarm_loop`` below — the
        # step-7 splice applies typed replies to parked members.

        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)
        checkpoint = await checkpointer.load(thread_id, swarm)
        if checkpoint is None:
            raise ValueError(
                f"arun_swarm_from_checkpoint: no checkpoint for thread_id={thread_id!r}. Nothing to resume."
            )

        loaded_state = SwarmState.from_dict(
            cast("SwarmStateDict", checkpoint.state),
            swarm,
        )
        run_context: RunContext[TContext] = RunContext.make(context, tenant_id=config.tenant_id)
        ctx_wrapper: RunContext[TContext] = RunContext.from_run_context(run_context)
        resolved_hooks = wrap_hooks_with_verbose(hooks, config)
        _activate_mcp_run_hooks_bridge(resolved_hooks, ctx_wrapper)

        if loaded_state.swarm_id is not None:
            swarm_id_for_run = loaded_state.swarm_id
        else:
            swarm_id_for_run = str(uuid.uuid4())
            logger.warning(
                "arun_swarm_from_checkpoint: loaded checkpoint has no "
                "swarm_id; regenerating. Trace correlation across this "
                "resume boundary will not work — the new root span "
                "starts a fresh troopai.swarm.id."
            )
            loaded_state.swarm_id = swarm_id_for_run

        _root_span = swarm_span(
            swarm_id=swarm_id_for_run,
            entry=swarm.entry.name,
            disabled=not (config.tracing_enabled or config.metrics_enabled),
        )
        _root_span.start()

        try:
            result = await run_swarm_loop(
                swarm=swarm,
                user_prompt=user_prompt,
                ctx_wrapper=ctx_wrapper,
                hooks=resolved_hooks,
                config=config,
                initial_state=loaded_state,
                swarm_resume=resume,
                swarm_id=swarm_id_for_run,
                checkpointer=checkpointer,
            )
            if (config.tracing_enabled or config.metrics_enabled) and result.state is not None:
                # Mirror the _stamp_turn_span pattern: only stamp the span
                # payload when span tracing OR metrics is enabled (i.e.
                # whenever the span was actually created).
                payload = cast(CustomSpanData, _root_span.data).data
                payload["status"] = result.stop_reason.kind
                payload["turns_total"] = result.state.total_turns
            return result
        except Exception as e:
            _root_span.set_error(type(e).__name__, data={"message": str(e)})
            logger.error("Error during swarm resume: %s", e)
            raise
        finally:
            _root_span.finish()

    @classmethod
    async def arun_graph_from_checkpoint(
        cls,
        graph: Graph[Any],
        *,
        checkpointer: Checkpointer,
        thread_id: str,
        user_prompt: UserPrompt | None = None,
        context: TContext | None = None,
        hooks: list[GraphHooks[Any] | HookProvider] | None = None,
        run_config: RunConfig | None = None,
        resume: GraphResume | None = None,
    ) -> GraphRunResult[Any]:
        """Resume a graph run from a persisted checkpoint.

        Loads :class:`GraphState` via ``checkpointer.load``; only nodes
        with unconsumed upstream output re-fire. Budgets are cumulative.

        Args:
            graph: Compiled graph the checkpoint was produced from
                (matching id / node ids). Id mismatch raises in
                ``checkpointer.load``.
            checkpointer: The :class:`Checkpointer` holding the run.
            thread_id: Logical run key to resume.
            user_prompt: Entry-node input; used only if the entry node
                re-fires (uncommon on resume).
            context: Optional user context threaded to every node.
            hooks: Hooks/providers; ``checkpointer`` is appended
                automatically if absent.
            run_config: Optional :class:`RunConfig`.
            resume: Optional human replies for pending interrupt nodes.
                Each entry in ``replies`` or ``rejected`` is injected
                into the matching node's input on the first superstep
                where that node is ready, then cleared from
                :attr:`GraphState.pending_interrupts`.

        Returns:
            A :class:`GraphRunResult` from resume to halt.

        Raises:
            ValueError: No checkpoint for ``thread_id``, or the stored
                graph id does not match ``graph``.
        """
        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)
        state = await checkpointer.load(thread_id, graph)
        if state is None:
            raise ValueError(
                f"arun_graph_from_checkpoint: no checkpoint for thread_id={thread_id!r} "
                f"on graph id={graph.id!r}. Nothing to resume."
            )

        effective_hooks: list[GraphHooks[Any] | HookProvider]
        effective_hooks = list(hooks) if hooks is not None else []
        if checkpointer not in effective_hooks:
            effective_hooks.append(checkpointer)

        run_context: RunContext[Any] = RunContext.make(context, tenant_id=config.tenant_id)
        prompt: UserPrompt = user_prompt if user_prompt is not None else ""
        return await cls._run_graph_bracketed(
            graph=graph,
            user_prompt=prompt,
            run_context=run_context,
            config=config,
            hooks=effective_hooks,
            thread_id=thread_id,
            initial_state=state,
            resume=resume,
        )

    @classmethod
    def run_graph_from_checkpoint(
        cls,
        graph: Graph[Any],
        *,
        checkpointer: Checkpointer,
        thread_id: str,
        user_prompt: UserPrompt | None = None,
        context: TContext | None = None,
        hooks: list[GraphHooks[Any] | HookProvider] | None = None,
        run_config: RunConfig | None = None,
        resume: GraphResume | None = None,
    ) -> GraphRunResult[Any]:
        """Synchronous wrapper around :meth:`arun_graph_from_checkpoint`.

        Same running-loop / ``ThreadPoolExecutor`` strategy as
        :meth:`run_graph`.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    cls.arun_graph_from_checkpoint(
                        graph,
                        checkpointer=checkpointer,
                        thread_id=thread_id,
                        user_prompt=user_prompt,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        resume=resume,
                    ),
                )
                return future.result()

        return asyncio.run(
            cls.arun_graph_from_checkpoint(
                graph,
                checkpointer=checkpointer,
                thread_id=thread_id,
                user_prompt=user_prompt,
                context=context,
                hooks=hooks,
                run_config=run_config,
                resume=resume,
            )
        )

    @classmethod
    async def arun_graph_streamed(
        cls,
        graph: Graph[Any],
        user_prompt: UserPrompt,
        *,
        context: TContext | None = None,
        hooks: list[GraphHooks[Any] | HookProvider] | None = None,
        run_config: RunConfig | None = None,
        thread_id: str | None = None,
        initial_state: GraphState[Any] | None = None,
        resume: GraphResume | None = None,
    ) -> GraphRunResultStreaming:
        """Execute a graph with real-time event streaming.

        Returns a :class:`GraphRunResultStreaming` immediately. Events are
        produced in the background and consumed via
        :meth:`GraphRunResultStreaming.stream_events`. Cancellation is
        available via :meth:`GraphRunResultStreaming.cancel`.

        Args:
            graph: The compiled graph to execute.
            user_prompt: Input passed to the entry node.
            context: Optional user context threaded to every node.
            hooks: Optional list of :class:`GraphHooks` and/or
                :class:`HookProvider` instances.
            run_config: Optional :class:`RunConfig`.
            thread_id: Optional checkpointer thread id.
            initial_state: Optional restored :class:`GraphState` for resume.
                Passed through to :func:`run_graph_loop_streamed` so that
                :func:`_seed_barriers_from_checkpoint` can reconstruct
                barriers for selective node re-fire.
            resume: Optional human replies for pending interrupt nodes.
                Threaded to :func:`run_graph_loop_streamed` which injects
                each reply into the matching node's :class:`ExecutableInput`
                metadata on the first superstep where that node is ready.

        Returns:
            :class:`GraphRunResultStreaming` whose ``stream_events()`` yields
            :class:`GraphStreamEvent` instances until the run completes or
            is cancelled.
        """
        from troopai.adk.graphs.result import GraphRunResultStreaming, GraphRunStatus
        from troopai.adk.run.graph_loop import run_graph_loop_streamed

        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)
        run_context: RunContext[Any] = RunContext.make(context, tenant_id=config.tenant_id)
        result: GraphRunResultStreaming = GraphRunResultStreaming(user_prompt=user_prompt)

        async def _driver() -> None:
            async with _graph_run_bracket(graph=graph, user_prompt=user_prompt, config=config) as bracket:
                # run_graph_loop_streamed populates result fields, stores any
                # exception, and calls result.complete() in its own finally —
                # do NOT call result.complete()/set_exception() here.
                await run_graph_loop_streamed(
                    graph=graph,
                    user_prompt=user_prompt,
                    context=run_context,
                    config=config,
                    result=result,
                    hooks=hooks,
                    thread_id=thread_id,
                    initial_state=initial_state,
                    resume=resume,
                )
                if result.status == GraphRunStatus.FAILED:
                    bracket.mark_failed(result.error if result.error is not None else "graph failed")

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_driver())
            result.set_run_task(task)
        except RuntimeError:
            # No running loop — store for lazy creation in stream_events().
            result.set_deferred_run_impl(_driver)

        return result

    # -- Task entry points --------------------------------------------------

    @classmethod
    async def arun_task(
        cls,
        task: Task[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> TaskOutput:
        """Execute a :class:`Task` asynchronously.

        Additive convenience over :meth:`Runner.arun` for developers
        who want a named, documented unit of work with explicit
        per-call overrides (output schema, guardrails, budgets) and
        first-class ``on_task_*`` lifecycle hooks. Every classic
        ``Runner.arun(...)`` call continues to work unchanged.

        The runner builds a transient effective ``Agent`` (via
        ``dataclasses.replace`` when :attr:`Task.output_schema` is set)
        and a transient :class:`RunConfig` whose
        :attr:`RunConfig.guardrails` extends the caller-supplied
        config with the task's input/output guardrails — run-scope
        guardrails run first, task-scope second. :attr:`Task.max_turns`
        is passed as the ``max_turns`` kwarg to the inner ``arun``,
        NOT folded into the transient ``RunConfig`` (consistent with
        ``Runner.arun``'s signature).

        Fires :meth:`RunHooks.on_task_start` before the inner ``arun``
        and :meth:`RunHooks.on_task_end` after — in both success and
        exception paths. The existing verbose Task panel continues to
        render via ``emit_task_start`` / ``emit_task_end`` inside the
        inner ``arun`` path.

        Args:
            task: The task to execute.
            context: Optional user context. Threaded into the inner
                ``arun`` and into the synthetic :class:`RunContext`
                passed to the task hooks.
            hooks: Optional :class:`RunHooks`. Receives the new
                ``on_task_start`` / ``on_task_end`` callbacks in
                addition to all existing run-level events.
            run_config: Optional base :class:`RunConfig`. The runner
                builds a transient extension that adds the task's
                guardrails and (when set) ``usage_limits``.
            session: Optional :class:`~troopai.adk.types.session.SessionStore` for conversation
                persistence — flows through unchanged.
            memory: Optional :class:`MemoryConfig` for memory
                injection / extraction — flows through unchanged.

        Returns:
            A :class:`TaskOutput` whose :attr:`TaskOutput.error` is
            ``None`` on success and a stringified exception on failure.
            On failure, :meth:`on_task_end` fires with the error-set
            :class:`TaskOutput` before the exception is re-raised.

        Raises:
            Exception: Any exception raised by the inner ``arun``
                (after :meth:`on_task_end` has fired with the failure
                :class:`TaskOutput`). :class:`BaseException` subclasses
                like ``KeyboardInterrupt`` / ``asyncio.CancelledError``
                propagate untouched and bypass the hook side-effects —
                cooperative cancellation is preserved.
        """
        from troopai.adk.agents.agent import Agent
        from troopai.adk.graphs.graph import Graph
        from troopai.adk.swarms.swarm import Swarm
        from troopai.adk.tasks.task_output import TaskOutput

        effective_target = _effective_task_target(task)
        effective_config = _build_effective_task_config(task, _snapshot_run_config(run_config))
        task_id, task_name = _resolve_task_identity(task)

        pre_ctx: RunContext[TContext] = RunContext.make(context, tenant_id=effective_config.tenant_id)
        resolved_hooks: RunHooks[TContext] = hooks if hooks is not None else RunHooks()

        try:
            await resolved_hooks.on_task_start(pre_ctx, effective_target, task)
            if isinstance(effective_target, Agent):
                output = await _run_task_as_agent(
                    cls,
                    effective_target,
                    task,
                    task_id,
                    task_name,
                    context,
                    hooks,
                    effective_config,
                    session,
                    memory,
                )
            elif isinstance(effective_target, Swarm):
                output = await _run_task_as_swarm(
                    cls,
                    effective_target,
                    task,
                    task_id,
                    task_name,
                    context,
                    hooks,
                    effective_config,
                    session,
                )
            elif isinstance(effective_target, Graph):
                output = await _run_task_as_graph(
                    cls,
                    effective_target,
                    task,
                    task_id,
                    task_name,
                    context,
                    hooks,
                    effective_config,
                    session,
                    memory,
                )
            else:
                raise TypeError(
                    f"Task.agent must be Agent | Swarm | Graph, got {type(effective_target).__name__}",
                )
        except Exception as e:
            err_output = TaskOutput(
                task_id=task_id,
                task_name=task_name,
                error=_format_task_error(e),
                metadata=dict(task.metadata),
            )
            await resolved_hooks.on_task_end(pre_ctx, effective_target, task, output=err_output)
            logger.error("Task '%s' (%s) failed: %s", task_name, task_id, e)
            raise

        await resolved_hooks.on_task_end(pre_ctx, effective_target, task, output=output)
        return output

    @classmethod
    async def arun_task_streamed(
        cls,
        task: Task[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> RunResultStreaming:
        """Execute a :class:`Task` with real-time event streaming.

        Streams events from a single-task execution. Lets developers
        observe a Task's run incrementally (raw LLM tokens, tool calls,
        tool outputs) without losing the Task abstraction or its
        ``on_task_*`` lifecycle hooks.

        Fires :meth:`RunHooks.on_task_start` synchronously before
        returning — the returned :class:`RunResultStreaming` is
        observable only after that callback has completed. Fires
        :meth:`RunHooks.on_task_end` from the background impl's
        ``finally`` arm with a :class:`TaskOutput` built from
        ``result.final_output`` / ``result.new_items`` /
        ``result.context.usage`` — so the hook also fires on early
        cancellation or error paths.

        The return value is a raw :class:`RunResultStreaming` (NOT a
        Task-specific wrapper). Consume events with
        ``async for event in result.stream_events()``. Cancellation
        semantics match :meth:`RunResultStreaming.cancel` exactly.

        Args:
            task: The task to execute.
            context: Optional user context. Threaded into the inner
                streamed impl and into the synthetic
                :class:`RunContext` passed to the task hooks.
            hooks: Optional :class:`RunHooks`. Receives the new
                ``on_task_start`` / ``on_task_end`` callbacks in
                addition to all existing run-level events.
            run_config: Optional base :class:`RunConfig`. The runner
                builds a transient extension that adds the task's
                guardrails and (when set) ``usage_limits``.
            session: Optional :class:`~troopai.adk.types.session.SessionStore` for conversation
                persistence — flows through unchanged.
            memory: Optional :class:`MemoryConfig` for memory
                injection / extraction — flows through unchanged.

        Returns:
            A :class:`RunResultStreaming` whose ``stream_events()``
            async iterator yields the run's events in real time and
            whose ``final_output`` populates after the stream drains.

        Raises:
            Exception: Any exception raised during execution surfaces
                via :meth:`RunResultStreaming.stream_events` (the
                stream re-raises the stored exception after the event
                queue drains). :meth:`on_task_end` has already fired
                with the failure :class:`TaskOutput` by that point.
        """
        from dataclasses import replace

        from troopai.adk.agents.agent import Agent
        from troopai.adk.tasks.task_output import TaskOutput

        if not isinstance(task.agent, Agent):
            raise TypeError(
                "Runner.arun_task_streamed only supports Task targets of "
                "type Agent. For Swarm / Graph targets, use "
                "Runner.arun_task (non-streamed). Streamed Swarm / Graph "
                "is tracked as a separate follow-up.",
            )
        # The isinstance guard above narrows `task.agent` to Agent.
        # Build the effective agent inline (mirroring the Agent branch
        # of _effective_task_target) so the type narrowing survives
        # `python -O` builds — bare `assert` is stripped.
        effective_agent: Agent[TContext]
        if task.output_schema is None:
            effective_agent = task.agent
        else:
            effective_agent = replace(task.agent, output_schema=task.output_schema)
        effective_config = _build_effective_task_config(task, _snapshot_run_config(run_config))
        # This path calls _run_streamed_impl directly, bypassing _run_streamed,
        # so validate the budget config here to match every other entry point.
        validate_budget_config(effective_config.tenant_budget, effective_config.cost_ledger)
        task_id, task_name = _resolve_task_identity(task)
        max_turns = task.max_turns if task.max_turns is not None else DEFAULT_MAX_TURNS

        pre_ctx: RunContext[TContext] = RunContext.make(context, tenant_id=effective_config.tenant_id)
        resolved_hooks: RunHooks[TContext] = hooks if hooks is not None else RunHooks()

        run_context: RunContext[TContext] = RunContext.make(context, tenant_id=effective_config.tenant_id)
        result = RunResultStreaming(
            current_agent=effective_agent,
            current_turn=0,
            max_turns=max_turns,
            user_prompt=task.description,
            context=run_context,
        )

        def _build_output(*, final: Any = None, items: tuple[Any, ...] = (), err: str | None = None) -> TaskOutput:
            return TaskOutput(
                task_id=task_id,
                task_name=task_name,
                final_output=final,
                new_items=items,
                usage=run_context.usage,
                error=err,
                metadata=dict(task.metadata),
            )

        async def run_impl() -> None:
            from troopai.adk.verbose.hooks import emit_task_end, emit_task_start

            nonlocal effective_agent
            task_output: TaskOutput | None = None
            verbose_hooks_chain: RunHooks[TContext] | None = None
            # Sandbox lifecycle bracket — mirrors the arun() bracket so that
            # SandboxAgent task runs with stream=True open and close their
            # sessions correctly and have capability tools injected.
            _sandbox_stack = contextlib.AsyncExitStack()
            await _sandbox_stack.__aenter__()
            try:
                verbose_hooks_chain = wrap_hooks_with_verbose(hooks, effective_config)
                emit_task_start(verbose_hooks_chain, effective_agent, task_name, task_id)
                _activate_mcp_run_hooks_bridge(
                    verbose_hooks_chain,
                    RunContext.from_run_context(run_context),
                )

                # Sandbox bracket: open session and inject capability tools.
                await _maybe_open_sandbox_bracket(
                    stack=_sandbox_stack,
                    agent=effective_agent,
                    config=effective_config,
                    run_context=run_context,
                    hooks=verbose_hooks_chain,
                )
                effective_agent = _maybe_clone_agent_with_capability_tools(
                    agent=effective_agent,
                    run_context=run_context,
                )

                effective_input = await _streamed_load_session_history(
                    session,
                    task.description,
                    verbose_hooks_chain,
                    run_context,
                )
                if memory is not None and memory.inject:
                    effective_input = await _inject_memories(effective_input, memory)
                await cls._run_streamed_impl(
                    agent=effective_agent,
                    user_prompt=effective_input,
                    result=result,
                    hooks=verbose_hooks_chain,
                    config=effective_config,
                )
                await _streamed_persist_events(
                    result,
                    session,
                    memory,
                    effective_agent,
                    verbose_hooks_chain,
                    run_context,
                    task.description,
                )
                task_output = _build_output(
                    final=result.final_output,
                    items=tuple(result.new_items),
                )
            except asyncio.CancelledError as e:
                task_output = _build_output(err="task cancelled before completion")
                # A developer-issued cancel(mode="immediate") is a clean,
                # requested stop; surface the cancellation to the consumer
                # only when it came from outside (cancel_mode not IMMEDIATE),
                # so an external teardown is not mistaken for clean completion.
                if result.cancel_mode != CancelMode.IMMEDIATE:
                    result.set_exception(e)
                raise
            except Exception as e:
                task_output = _build_output(err=_format_task_error(e))
                result.set_exception(e)
            finally:
                # A None `task_output` means cancel arrived before either
                # arm built one — synthesise so on_task_end always fires.
                if task_output is None:
                    task_output = _build_output(err="task cancelled before completion")
                await resolved_hooks.on_task_end(
                    pre_ctx,
                    effective_agent,
                    task,
                    output=task_output,
                )
                if verbose_hooks_chain is not None:
                    emit_task_end(
                        verbose_hooks_chain,
                        effective_agent,
                        task_name,
                        task_id,
                        success=task_output.error is None,
                        error=task_output.error,
                    )
                await result.complete()
                # Close sandbox bracket LAST so session outlives everything
                # that may have referenced it during the run.
                await _sandbox_stack.aclose()

        # Fire on_task_start synchronously before scheduling the producer
        # task, so the returned handle is observable only after the callback
        # completes (the documented contract). on_task_end still fires from
        # run_impl's finally on every path.
        await resolved_hooks.on_task_start(pre_ctx, effective_agent, task)
        try:
            loop = asyncio.get_running_loop()
            result.set_run_task(loop.create_task(run_impl()))
        except RuntimeError:
            result.set_deferred_run_impl(run_impl)

        return result

    @classmethod
    def run_task(
        cls,
        task: Task[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> TaskOutput:
        """Execute a :class:`Task` synchronously.

        Sync wrapper around :meth:`arun_task`. Uses the same event-loop
        strategy as :meth:`Runner.run`: when invoked inside a running
        loop, offloads to a worker thread; otherwise drives the
        coroutine with :func:`asyncio.run`.

        See :meth:`arun_task` for argument and return semantics.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    cls.arun_task(
                        task,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        session=session,
                        memory=memory,
                    ),
                )
                return future.result()

        return asyncio.run(
            cls.arun_task(
                task,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                memory=memory,
            )
        )

    @classmethod
    async def arun_task_pipeline(
        cls,
        task_pipeline: TaskPipeline[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> TaskPipelineResult[TContext]:
        """Execute a :class:`TaskPipeline` asynchronously.

        Iterates the pipeline's tasks in order, calling
        :meth:`arun_task` for each. The pipeline maintains an
        accumulating :class:`RunContext` whose ``usage`` field is
        summed from each executed task — :attr:`TaskPipelineResult.context.usage`
        equals the cumulative LLM usage across all non-skipped tasks.

        Conditional skip: when :attr:`Task.skip_if` is supplied and
        returns ``True`` against the tuple of prior outputs (including
        previously skipped ones), the runner inserts a
        ``TaskOutput(skipped=True, …)`` slot and continues — slots are
        NEVER silently dropped, so positional indexing matches the
        input pipeline.

        Each task's :attr:`Task.description` is fed verbatim as the
        agent's user prompt — the pipeline does NOT transform prompts
        at runtime. To forward an upstream task's output into a
        downstream task's prompt, call :meth:`Runner.arun_task` for
        the upstream first, then construct the downstream
        :class:`Task` with a description that embeds the prior
        output, then call :meth:`Runner.arun_task` again. The
        pipeline is the right abstraction when you want sequential
        execution with conditional skip and usage aggregation; it is
        NOT the abstraction for runtime prompt rewriting.

        Error handling: if a task raises, the runner captures the
        exception into a ``TaskOutput(error=…)`` slot and HALTS the
        pipeline — no retries, no subsequent tasks. The partial
        :class:`TaskPipelineResult` is returned (NOT raised) so the
        developer always sees the work that did complete.

        Args:
            task_pipeline: The :class:`TaskPipeline` to execute.
            context: Optional user context shared across every task.
            hooks: Optional :class:`RunHooks` shared across every task.
            run_config: Optional base :class:`RunConfig` shared across
                every task. Each task's overrides extend it.
            session: Optional :class:`~troopai.adk.types.session.SessionStore`. Shared across every
                task — events from each task append to the same
                session.
            memory: Optional :class:`MemoryConfig`. Shared across
                every task. Note: when ``memory.inject=True``,
                memories are re-injected for each task — the
                developer can disable injection per-pipeline if
                undesired.

        Returns:
            A :class:`TaskPipelineResult` whose ``task_outputs`` tuple
            has one slot per pipeline task (skipped tasks present
            with ``skipped=True``). ``final_output`` is the last
            non-skipped task's ``final_output``, or ``None`` if every
            task was skipped or the pipeline halted before producing
            output.
        """
        from troopai.adk.tasks.task_pipeline import TaskPipelineResult

        pipeline_ctx: RunContext[TContext] = RunContext.make(
            context, tenant_id=_snapshot_run_config(run_config).tenant_id
        )

        if any(t.depends_on is not None and len(t.depends_on) > 0 for t in task_pipeline.tasks):
            task_outputs_list, final_output = await _arun_task_pipeline_dag(
                cls,
                task_pipeline,
                pipeline_ctx,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                memory=memory,
                completed_task_ids=frozenset(),
                preloaded_outputs=(),
            )
        else:
            task_outputs_list, final_output = await _arun_task_pipeline_sequential(
                cls,
                task_pipeline,
                pipeline_ctx,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                memory=memory,
            )

        return TaskPipelineResult(
            task_outputs=tuple(task_outputs_list),
            final_output=final_output,
            context=pipeline_ctx,
        )

    @classmethod
    async def arun_task_pipeline_from_state(
        cls,
        task_pipeline: TaskPipeline[TContext],
        state: TaskPipelineState,
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> TaskPipelineResult[TContext]:
        """Resume a :class:`TaskPipeline` from a persisted state.

        Continues execution from ``state.resume_index`` onwards. The
        slots in ``state.slots`` cover the tasks that completed before
        the checkpoint — they are returned verbatim in the result's
        ``task_outputs`` tuple. Tasks at and after the resume index
        run via :meth:`arun_task` exactly as the fresh pipeline path
        does, with the prior slots threaded into each task's
        ``skip_if`` evaluation.

        Contract: the developer is responsible for reconstructing the
        same :class:`TaskPipeline` definition (same number of tasks,
        same agent identities) on the resuming side. The framework
        does not serialize :attr:`Task.agent`, :attr:`Task.skip_if`,
        or :attr:`Task.metadata`; the resume relies on the
        reconstruction to provide them. Mismatched pipelines produce
        undefined behaviour (the wrong agent for the wrong slot
        position).

        Usage cumulation: usage from prior slots (in
        ``state.slots``) is folded into the resumed
        ``RunContext.usage`` so the final result's cumulative usage
        reflects the WHOLE pipeline run, not just the resumed tail.

        Args:
            task_pipeline: The reconstructed pipeline definition.
                Length MUST be at least ``state.resume_index +
                len(remaining_tasks)`` — typically equal to the
                original pipeline's length.
            state: The :class:`TaskPipelineState` to resume from.
                Required-field presence is enforced by
                :meth:`TaskPipelineState.from_json` (loud raise on a
                truncated payload); this method trusts the state
                object passed in.
            context: Optional user context shared across the
                remaining tasks.
            hooks: Optional :class:`RunHooks` shared across the
                remaining tasks. The pre-state tasks did NOT fire
                their hooks during this resume (they fired in the
                originating run).
            run_config: Optional base :class:`RunConfig`.
            session: Optional shared :class:`~troopai.adk.types.session.SessionStore`.
            memory: Optional shared :class:`MemoryConfig`.

        Returns:
            A :class:`TaskPipelineResult` whose ``task_outputs``
            concatenates the recorded prior slots with the freshly
            executed slots. ``final_output`` is the last non-skipped
            task's final output across the whole pipeline.

        Raises:
            ValueError: When ``state.resume_index`` exceeds the
                pipeline's task count — the developer reconstructed
                a shorter pipeline than the original.
        """
        from dataclasses import replace

        from troopai.adk.tasks.task_pipeline import TaskPipelineResult

        if state.resume_index > len(task_pipeline.tasks):
            raise ValueError(
                f"TaskPipelineState.resume_index ({state.resume_index}) "
                f"exceeds the reconstructed pipeline length "
                f"({len(task_pipeline.tasks)}). The resuming side must "
                f"supply at least as many tasks as the originating run.",
            )

        pipeline_ctx: RunContext[TContext] = RunContext.make(
            context, tenant_id=_snapshot_run_config(run_config).tenant_id
        )
        for slot in state.slots:
            if slot.usage is not None:
                pipeline_ctx.usage = pipeline_ctx.usage + slot.usage

        is_dag = any(t.depends_on is not None and len(t.depends_on) > 0 for t in task_pipeline.tasks)

        if is_dag:
            task_outputs_list, final_output = await _arun_task_pipeline_dag(
                cls,
                task_pipeline,
                pipeline_ctx,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                memory=memory,
                completed_task_ids=frozenset(state.completed_task_ids),
                preloaded_outputs=tuple(state.slots),
            )
        else:
            if state.resume_index < len(state.slots):
                raise ValueError(
                    f"TaskPipelineState.resume_index ({state.resume_index}) is less than the "
                    f"number of already-completed slots ({len(state.slots)}). "
                    "Reload the correct state."
                )
            task_outputs_list = list(state.slots)
            final_output = None
            for slot in task_outputs_list:
                if not slot.skipped and slot.error is None:
                    final_output = slot.final_output

            for task in task_pipeline.tasks[state.resume_index :]:
                prior_outputs = tuple(task_outputs_list)
                slot_id, slot_name = _resolve_task_identity(task)

                if task.skip_if is not None:
                    try:
                        should_skip = task.skip_if(prior_outputs)
                    except Exception as e:
                        task_outputs_list.append(
                            _build_task_error_slot(slot_id, slot_name, e, task.metadata),
                        )
                        logger.warning(
                            "Resumed pipeline halted at '%s' (%s): skip_if raised %s",
                            slot_name,
                            slot_id,
                            e,
                        )
                        break
                    if should_skip:
                        task_outputs_list.append(
                            _build_task_skip_slot(slot_id, slot_name, task.metadata),
                        )
                        continue

                effective_task = replace(task, task_id=slot_id, name=slot_name)
                try:
                    output = await cls.arun_task(
                        effective_task,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        session=session,
                        memory=memory,
                    )
                except Exception as e:
                    task_outputs_list.append(
                        _build_task_error_slot(slot_id, slot_name, e, effective_task.metadata),
                    )
                    logger.warning(
                        "Resumed pipeline halted at '%s' (%s): %s",
                        slot_name,
                        slot_id,
                        e,
                    )
                    break

                task_outputs_list.append(output)
                if not output.skipped and output.error is None:
                    final_output = output.final_output
                if output.usage is not None:
                    pipeline_ctx.usage = pipeline_ctx.usage + output.usage

        return TaskPipelineResult(
            task_outputs=tuple(task_outputs_list),
            final_output=final_output,
            context=pipeline_ctx,
        )

    @classmethod
    def run_task_pipeline(
        cls,
        task_pipeline: TaskPipeline[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> TaskPipelineResult[TContext]:
        """Execute a :class:`TaskPipeline` synchronously.

        Sync wrapper around :meth:`arun_task_pipeline`. Same event-loop
        strategy as :meth:`Runner.run`.

        See :meth:`arun_task_pipeline` for argument and return semantics.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    cls.arun_task_pipeline(
                        task_pipeline,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        session=session,
                        memory=memory,
                    ),
                )
                return future.result()

        return asyncio.run(
            cls.arun_task_pipeline(
                task_pipeline,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                memory=memory,
            )
        )

    @classmethod
    def arun_task_pipeline_streamed(
        cls,
        task_pipeline: TaskPipeline[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> AsyncIterator[tuple[int, RunResultStreaming | None]]:
        """Stream a :class:`TaskPipeline` task-by-task.

        Returns an async iterator that yields one
        ``(task_index, RunResultStreaming | None)`` pair per pipeline
        slot, in input order. The caller drives both loops:

        ::

            streaming = Runner.arun_task_pipeline_streamed(pipeline)
            async for task_index, task_stream in streaming:
                if task_stream is None:
                    logger.info("task %d skipped", task_index)
                    continue
                async for event in task_stream.stream_events():
                    process(event)

        Skip slots yield ``(index, None)`` so positional indexing
        matches :attr:`TaskPipeline.tasks` exactly — silent skips
        would lose observability of skip_if firings during a stream.

        Per-task semantics mirror :meth:`Runner.arun_task_streamed`:
        ``on_task_start`` fires synchronously before each inner stream
        is yielded; ``on_task_end`` fires from the background impl's
        ``finally`` arm. The pipeline does NOT aggregate inner outputs
        into a :class:`TaskPipelineResult` — consumers read
        ``task_stream.final_output`` / ``task_stream.usage`` per task.
        An aggregated streamed-result type is tracked as a follow-up.

        Pipeline halt on inner failure is the **caller's**
        responsibility: ``break`` out of the outer iterator when a
        task's ``stream_events()`` re-raises an exception. The
        pipeline iterator does NOT inspect inner-stream state — that
        would require touching :class:`RunResultStreaming` private
        attributes across module boundaries.

        Cancellation:

        * The consumer can call ``task_stream.cancel(mode=...)`` on
          any inner stream to halt that task. The pipeline iterator
          continues to the next task on the consumer's next iteration.
        * The consumer can ``break`` out of the outer iterator to
          stop the pipeline at the current task boundary. Any
          subsequent tasks never start.

        Args:
            task_pipeline: The pipeline to stream.
            context: Optional user context. Threaded into every task's
                inner streaming impl.
            hooks: Optional :class:`RunHooks`. Receives
                ``on_task_start`` / ``on_task_end`` per non-skipped
                task. Hooks observers see the same lifecycle they get
                from :meth:`arun_task_streamed`.
            run_config: Optional base :class:`RunConfig` shared by
                every task; per-task overrides still apply.
            session: Optional shared :class:`~troopai.adk.types.session.SessionStore`. Events from
                every task accumulate in the same store.
            memory: Optional :class:`MemoryConfig`. Memory injection /
                extraction runs per-task identically to the
                non-streamed pipeline path.

        Returns:
            An async iterator of ``(task_index, inner_stream)`` pairs.
            ``inner_stream`` is ``None`` for skip slots.
        """
        return _stream_task_pipeline_impl(
            cls,
            task_pipeline,
            context,
            hooks,
            run_config,
            session,
            memory,
        )

    @classmethod
    async def arun_task_group(
        cls,
        task_group: TaskGroup[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> TaskGroupResult[TContext]:
        """Execute a :class:`TaskGroup` asynchronously (parallel fan-out).

        Schedules every task in the group concurrently under
        :func:`asyncio.gather` semantics, optionally bounded by an
        :class:`asyncio.Semaphore` when
        :attr:`TaskGroup.max_concurrent` is set. Each task runs through
        :meth:`Runner.arun_task` unchanged — so per-task hooks,
        guardrails, output schemas, ``max_turns``, and
        ``usage_limits`` continue to apply.

        The :class:`TaskGroup.error_policy` controls failure handling:
        ``"collect_all"`` (default) runs every task to completion and
        surfaces failures via :attr:`TaskOutput.error` slots;
        ``"halt_on_first"`` cancels still-running siblings on the
        first failure (cancelled slots carry an explanatory
        ``error`` field so positional indexing stays stable).

        Concurrency contract:

        - ``on_task_start`` / ``on_task_end`` callbacks fire
          **concurrently** across tasks. Hooks holding shared mutable
          state MUST lock — the framework does not synchronise.
        - Shared :class:`~troopai.adk.types.session.SessionStore`: events from concurrent tasks
          interleave. If event ordering matters, supply per-task
          sessions instead.
        - LLM client cancellation is provider-dependent. The cancel
          on ``halt_on_first`` is best-effort — in-flight provider
          HTTP calls may finish even though their slot is recorded
          as cancelled.

        Args:
            task_group: The :class:`TaskGroup` to execute.
            context: Optional user context threaded through every
                task. Each task receives its own :class:`RunContext`
                derived from this value (no shared mutation between
                tasks).
            hooks: Optional :class:`RunHooks`. Receives
                ``on_task_start`` / ``on_task_end`` for every task in
                the group, fired concurrently — see contract above.
            run_config: Optional base :class:`RunConfig` shared by
                every task. Each task's own per-task overrides
                (``output_schema``, guardrails, ``usage_limits``)
                still apply on top.
            session: Optional shared :class:`~troopai.adk.types.session.SessionStore`. When set, all
                tasks write to the same store; events interleave.
            memory: Optional shared :class:`MemoryConfig`. Memory
                injection / extraction runs per-task identically to
                the non-group path.

        Returns:
            A :class:`TaskGroupResult` whose
            :attr:`task_outputs` preserves the input order from
            :attr:`TaskGroup.tasks`. Cumulative LLM usage across every
            task that produced one is aggregated into
            :attr:`TaskGroupResult.context.usage` via the commutative
            :meth:`LLMUsage.__add__`.
        """
        from troopai.adk.tasks.task_group import TaskGroupResult

        pipeline_ctx: RunContext[TContext] = RunContext.make(
            context, tenant_id=_snapshot_run_config(run_config).tenant_id
        )
        concurrency = task_group.max_concurrent if task_group.max_concurrent is not None else len(task_group.tasks)
        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(task: Task[TContext]) -> TaskOutput:
            async with semaphore:
                return await cls.arun_task(
                    task,
                    context=context,
                    hooks=hooks,
                    run_config=run_config,
                    session=session,
                    memory=memory,
                )

        aio_tasks = [asyncio.create_task(_run_one(t)) for t in task_group.tasks]

        if task_group.error_policy == "halt_on_first":
            try:
                await asyncio.gather(*aio_tasks)
            except BaseException as exc:
                # First failure — cancel still-running siblings, then
                # drain every task with return_exceptions=True so all
                # cancellations settle before we read results. The
                # single gather avoids the race window of a
                # per-task `if not done: await` pattern (a task could
                # finish between the check and the await), and the
                # broad BaseException catches asyncio.CancelledError —
                # which inherits from BaseException, not Exception, so
                # a parent-scope cancel would otherwise leak pending
                # tasks. Parent-scope cancels are re-raised after
                # drain so cancellation propagates correctly.
                cancelled_by_parent = isinstance(exc, asyncio.CancelledError)
                for t in aio_tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*aio_tasks, return_exceptions=True)
                if cancelled_by_parent:
                    raise
        else:
            await asyncio.gather(*aio_tasks, return_exceptions=True)

        outputs = _collect_task_group_outputs(task_group.tasks, aio_tasks)
        for output in outputs:
            if output.usage is not None:
                pipeline_ctx.usage = pipeline_ctx.usage + output.usage

        return TaskGroupResult(
            task_outputs=tuple(outputs),
            context=pipeline_ctx,
            metadata=dict(task_group.metadata),
        )

    @classmethod
    def run_task_group(
        cls,
        task_group: TaskGroup[TContext],
        *,
        context: TContext | None = None,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
    ) -> TaskGroupResult[TContext]:
        """Execute a :class:`TaskGroup` synchronously.

        Sync wrapper around :meth:`arun_task_group`. Same event-loop
        strategy as :meth:`Runner.run`: when invoked inside a running
        loop, offloads to a worker thread; otherwise drives the
        coroutine with :func:`asyncio.run`.

        See :meth:`arun_task_group` for argument and return semantics.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    cls.arun_task_group(
                        task_group,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        session=session,
                        memory=memory,
                    ),
                )
                return future.result()

        return asyncio.run(
            cls.arun_task_group(
                task_group,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                memory=memory,
            )
        )

    # -- _run_streamed(): internal streaming implementation -----------------

    @classmethod
    def _run_streamed(
        cls,
        agent: Agent,
        user_prompt: UserPrompt | RunState,
        *,
        context: TContext | None = None,
        shared_run_context: RunContext[TContext] | None = None,
        hooks: RunHooks[TContext] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        run_config: RunConfig | None = None,
        session: SessionStore | None = None,
        memory: MemoryConfig | None = None,
        initial_messages: list[Any] | None = None,
        extra_tools: list[Any] | None = None,
        swarm_tool_names: set[str] | None = None,
        dispose_toolsets: bool = True,
    ) -> RunResultStreaming:
        """Internal streaming implementation shared by run() and arun().

        Returns immediately with a RunResultStreaming object that can be
        used to iterate over events as they are generated. Supports
        resuming from a RunState for Human-in-the-Loop (HITL) workflows.

        Args:
            shared_run_context: Optional caller-supplied ``RunContext`` to
                execute on instead of minting a fresh one. The streamed
                swarm driver threads its per-run ``ctx_wrapper`` here so
                every member turn accumulates cost / usage onto the same
                context — the per-run dollar budget and usage limits then
                accrue cumulatively across the swarm (matching the sync
                swarm) instead of resetting each turn. ``None`` (default)
                preserves the standalone behaviour of a fresh context.
            initial_messages: Pre-built messages to pass as the initial
                context to ``run_agent_loop_streamed``.  Used by the
                swarm driver (Steps 4-5 of the sync loop) so that
                ``SharedContextStrategy`` and per-member scratch are
                applied correctly on non-first swarm turns.
            extra_tools: Additional tools to inject for this turn (e.g.
                ``transfer_to_<name>`` / ``swarm_done`` tools built by
                ``swarm.policy.build_step_tools``).
            swarm_tool_names: Names of the swarm-injected tools so that
                ``resolve_swarm_yield_step`` can recognise swarm-routed
                tool calls inside the streamed loop.
            dispose_toolsets: When ``True`` (default), dispose the agent's
                toolsets in the impl's ``finally``. The swarm driver passes
                ``False`` so a member's toolsets survive across turns; the
                driver disposes every member once at end of run.
        """
        config = _snapshot_run_config(run_config)
        validate_budget_config(config.tenant_budget, config.cost_ledger)

        # Check if resuming from state
        if isinstance(user_prompt, RunState):
            if shared_run_context is not None:
                # The resume path mints its own RunContext, so a caller-supplied
                # shared context would be silently dropped. Since that context is
                # the sole carrier of per-run dollar cost, dropping it would let a
                # resumed member turn escape the per-run budget — fail closed on
                # this un-implemented combination rather than open that cost-cap
                # gap. Thread shared_run_context through resume_from_state_streamed
                # before enabling it.
                raise ValueError(
                    "shared_run_context is not supported when resuming from a "
                    "RunState; thread it through resume_from_state_streamed "
                    "before enabling this combination."
                )
            return resume_from_state_streamed(
                agent=agent,
                state=user_prompt,
                hooks=hooks,
                max_turns=max_turns,
                config=config,
                context=context,
            )

        # --- Session: load history deferred to the async impl -------------
        # ``_run_streamed`` is sync but session loading is async, so the
        # actual history merge happens inside ``_run_streamed_impl``.
        effective_input: UserPrompt = user_prompt

        # Create run context — or reuse a caller-supplied one so a streamed
        # swarm's member turns share the driver's RunContext (per-run dollar
        # budget + usage limits accumulate cumulatively, matching the sync
        # swarm) instead of resetting on a fresh context each turn.
        run_context: RunContext[TContext] = (
            shared_run_context
            if shared_run_context is not None
            else RunContext.make(context, tenant_id=config.tenant_id)
        )

        # Create streaming result
        result = RunResultStreaming(
            current_agent=agent,
            current_turn=0,
            max_turns=max_turns,
            user_prompt=user_prompt,
            context=run_context,
        )

        # Start execution in background task
        async def run_impl():
            from troopai.adk.verbose.hooks import emit_task_end, emit_task_start

            task_id = str(uuid.uuid4())
            task_name = _derive_task_name(user_prompt)
            task_error: str | None = None
            resolved_hooks: RunHooks[TContext] | None = None
            # Sandbox lifecycle bracket — mirrors the arun() bracket so that
            # SandboxAgent runs with stream=True open and close their sessions
            # correctly and have capability tools injected.
            _sandbox_stack = contextlib.AsyncExitStack()
            await _sandbox_stack.__aenter__()
            # ``effective_agent`` is the (possibly cloned) agent used for this
            # streamed run — may gain capability tools when a sandbox is active.
            effective_agent = agent
            try:
                nonlocal effective_input

                # Session: load history inside async context
                resolved_hooks = wrap_hooks_with_verbose(hooks, config)
                emit_task_start(resolved_hooks, effective_agent, task_name, task_id)
                _activate_mcp_run_hooks_bridge(resolved_hooks, RunContext.from_run_context(run_context))

                # Sandbox bracket: open session and inject capability tools.
                await _maybe_open_sandbox_bracket(
                    stack=_sandbox_stack,
                    agent=effective_agent,
                    config=config,
                    run_context=run_context,
                    hooks=resolved_hooks,
                )
                effective_agent = _maybe_clone_agent_with_capability_tools(
                    agent=effective_agent,
                    run_context=run_context,
                )

                if session is not None and isinstance(user_prompt, str):
                    limit = None
                    if session.settings is not None:
                        limit = session.settings.limit
                    events = await session.get(limit=limit)
                    if len(events) > 0:
                        ctx_w: RunContext[TContext] = RunContext.from_run_context(run_context)
                        await resolved_hooks.on_session_load(ctx_w, session, events)
                        history = [e.content for e in events]
                        user_msg_s: LLMInputEasyMessage = {"role": "user", "content": user_prompt}
                        effective_input = [*history, user_msg_s]

                # Memory: inject relevant memories
                if memory is not None and memory.inject:
                    effective_input = await _inject_memories(effective_input, memory)

                await cls._run_streamed_impl(
                    agent=effective_agent,
                    user_prompt=effective_input,
                    result=result,
                    hooks=resolved_hooks,
                    config=config,
                    initial_messages=initial_messages,
                    extra_tools=extra_tools,
                    swarm_tool_names=swarm_tool_names,
                    dispose_toolsets=dispose_toolsets,
                )

                if result.recovered:
                    # An error handler produced the final output. Skip
                    # session/memory persistence: result.new_items holds a
                    # truncated turn fragment, and persisting it would seed
                    # the next turn with a half-formed exchange. Matches the
                    # non-streaming recovery contract.
                    return

                # Build events from results
                from troopai.adk.session.session_event import SessionEvent, create_session_event

                events_to_save: list[SessionEvent] = []
                if isinstance(user_prompt, str):
                    user_event_msg_s: LLMInputEasyMessage = {"role": "user", "content": user_prompt}
                    events_to_save.append(
                        create_session_event(
                            author="user",
                            content=user_event_msg_s,
                        )
                    )
                for item in result.new_items:
                    events_to_save.append(
                        create_session_event(
                            author=_infer_author(item),
                            content=item.to_param(),
                        )
                    )

                # Session: save new events
                if session is not None and len(events_to_save) > 0:
                    await session.add(events_to_save)
                    try:
                        await session.save_state()
                    except Exception:
                        logger.warning("session.save_state() failed; state delta may not be persisted", exc_info=True)
                    ctx_w = RunContext.from_run_context(run_context)
                    await resolved_hooks.on_session_save(ctx_w, session, events_to_save)

                # Memory: extract from conversation
                if (
                    memory is not None
                    and memory.auto_extract
                    and memory.extractor is not None
                    and len(events_to_save) > 0
                ):
                    await memory.memory.add_events(
                        events_to_save,
                        namespace=memory.namespace,
                        extractor=memory.extractor,
                        session_id=session.id if session is not None else None,
                        agent_name=effective_agent.name,
                    )

            except asyncio.CancelledError as e:
                # Propagate external cancellation to the stream consumer.
                # Without this handler, CancelledError (a BaseException since
                # Python 3.8) bypasses result.set_exception(), the finally
                # still calls result.complete(), and the consumer sees a false
                # clean completion instead of the cancellation.
                task_error = f"{type(e).__name__}: {e}"
                # A developer-issued cancel(mode="immediate") is a clean,
                # requested stop; store the exception only when the cancel
                # came from outside (cancel_mode not IMMEDIATE), so an external
                # teardown is not mistaken for clean completion and an
                # immediate cancel does not surface a spurious error to the
                # consumer (mirrors the task-based streamed twin).
                if result.cancel_mode != CancelMode.IMMEDIATE:
                    result.set_exception(e)
                raise
            except Exception as e:
                task_error = f"{type(e).__name__}: {e}"
                # Emit a structured GUARDRAIL_TRIGGERED event before storing
                # the exception so stream consumers can pattern-match on it
                # (RunItemType.GUARDRAIL_TRIGGERED) rather than relying solely
                # on catching the exception.
                if isinstance(
                    e,
                    (AgentInputGuardrailTripwireTriggered, AgentOutputGuardrailTripwireTriggered),
                ):
                    from troopai.adk.run.stream import RunItemStreamEvent, RunItemType

                    await result.put_event(
                        RunItemStreamEvent(
                            name=RunItemType.GUARDRAIL_TRIGGERED,
                            item={"reason": str(e)},
                        )
                    )
                result.set_exception(e)
            finally:
                if resolved_hooks is not None:
                    if task_error is None:
                        emit_task_end(resolved_hooks, effective_agent, task_name, task_id, success=True)
                    else:
                        emit_task_end(
                            resolved_hooks,
                            effective_agent,
                            task_name,
                            task_id,
                            success=False,
                            error=task_error,
                        )
                        # Sweep generic verbose tree blocks left open when the
                        # run ended by exception. Gated on task_error so a clean
                        # turn never sweeps — a streamed swarm shares one hooks
                        # chain across member turns, and a healthy turn's blocks
                        # must survive for the turns that follow.
                        _sweep_verbose_panels(resolved_hooks)
                await result.complete()
                # Close sandbox bracket LAST so session outlives everything
                # that may have referenced it during the run.
                await _sandbox_stack.aclose()

        # --- Reset deferred-tool revealed sets ----------------------------
        # Parallel resets apply to the non-streaming ``arun()`` path too;
        # here we reset BEFORE scheduling the background task so that the
        # context snapshot ``create_task`` takes starts with an empty
        # revealed set.  Sequential ``_run_streamed`` calls made from the
        # same coroutine would otherwise carry over reveals from run #1
        # into run #2 because ``create_task`` copies the *current* context.
        from troopai.adk.tools.tool_search import reset_revealed_sets

        reset_revealed_sets(agent.tools)

        # Schedule the task
        try:
            loop = asyncio.get_running_loop()
            result.set_run_task(loop.create_task(run_impl()))
        except RuntimeError:
            # No running loop — store for lazy creation in stream_events()
            result.set_deferred_run_impl(run_impl)

        return result

    @classmethod
    async def arun_flow(
        cls,
        flow: Flow[Any],
        *,
        config: FlowConfig | None = None,
        context: Any = None,
    ) -> FlowRunResult[Any]:
        """Execute a :class:`Flow` asynchronously.

        Constructs a :class:`FlowExecutor` for the flow, attaches a
        :class:`RunContext` so step bodies that opt into shared usage
        tracking (``await Runner.arun(..., context=self.run_context)``
        from inside a step) accumulate into the same ``LLMUsage``,
        and drives the executor to completion.

        The framework does NOT auto-inject the run context or any
        ``RunConfig`` into inner ``Runner.arun`` calls — step bodies
        pass ``context=`` / ``run_config=`` explicitly when they want
        sharing.

        Args:
            flow: The :class:`Flow` instance to execute. MUST have a
                non-``None`` ``__flow_registry__`` (i.e., be a concrete
                subclass with at least one ``@flow_start`` method).
            config: Optional :class:`FlowConfig` carrying ``max_steps``,
                error policy, and fan-out cap. ``None`` uses defaults
                (``max_steps=100``, ``error_policy="halt"``,
                ``max_listeners_per_step=20``).
            context: Optional developer-supplied ``TContext`` value
                stored on the flow's ``run_context``. Step bodies that
                read ``self.run_context.context`` see this value.

        Returns:
            A :class:`FlowRunResult` whose ``status`` reflects the
            terminating condition: ``"completed"`` for clean termination,
            ``"failed"`` for an unrecoverable step exception,
            ``"halted_max_steps"`` on cap overflow.

        Raises:
            FlowDefinitionError: When the flow class is abstract (no
                ``@flow_start`` declared) or when the wired step graph
                violates a structural rule.
        """
        from troopai.adk.flows.config import FlowConfig
        from troopai.adk.flows.executor import FlowExecutor

        effective_config = config if config is not None else FlowConfig()
        run_ctx: RunContext[Any] = RunContext.make(context)
        flow.run_context = run_ctx
        executor: FlowExecutor[Any] = FlowExecutor(flow, config=effective_config)
        return await executor.run()

    @classmethod
    def run_flow(
        cls,
        flow: Flow[Any],
        *,
        config: FlowConfig | None = None,
        context: Any = None,
    ) -> FlowRunResult[Any]:
        """Execute a :class:`Flow` synchronously.

        Sync wrapper around :meth:`arun_flow`. Same event-loop strategy
        as :meth:`Runner.run` / :meth:`Runner.run_task_pipeline`: when
        invoked inside a running loop, offloads to a worker thread;
        otherwise drives the coroutine with :func:`asyncio.run`.

        See :meth:`arun_flow` for argument and return semantics.

        Args:
            flow: The :class:`Flow` instance to execute.
            config: Optional :class:`FlowConfig`. See :meth:`arun_flow`.
            context: Optional developer-supplied ``TContext``.

        Returns:
            The :class:`FlowRunResult` from the underlying async run.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    cls.arun_flow(flow, config=config, context=context),
                )
                return future.result()

        return asyncio.run(cls.arun_flow(flow, config=config, context=context))

    @classmethod
    async def arun_flow_from_checkpoint(
        cls,
        flow: Flow[Any],
        checkpoint: FlowCheckpoint,
        *,
        config: FlowConfig | None = None,
        context: Any = None,
        agent_resolutions: Mapping[str, str] | None = None,
    ) -> FlowRunResult[Any]:
        """Resume a Flow run from a serialized :class:`FlowCheckpoint`.

        See ``docs/flows/flows.md`` for the full resumption contract.
        Summary:

        - The developer reconstructs ``flow`` with the same class as
          the originating run and the state rehydrated from
          ``checkpoint.state_data``.
        - Step-level approvals already live on ``checkpoint.decisions``
          (recorded via :meth:`FlowCheckpoint.approve` /
          :meth:`FlowCheckpoint.reject`); the runner forwards them to
          the flow before the executor starts.
        - Agent-bridge deferrals (caused by inner agents whose tools
          required approval) are resumed via ``agent_resolutions`` —
          a mapping from :attr:`FlowDeferredStep.defer_key` to the
          :class:`RunState` JSON the consumer recorded decisions on
          (via :meth:`RunState.approve` / :meth:`RunState.reject` then
          :meth:`RunState.to_dict` ``+`` ``json.dumps``).

        Security: ``checkpoint.pending_steps`` is treated as untrusted
        input by :meth:`FlowExecutor._invoke_step` — every name is
        validated against the Flow's registry before invocation, so a
        tampered checkpoint cannot trigger arbitrary methods on the
        Flow subclass. Callers loading checkpoints from shared / public
        stores SHOULD still authenticate / sign the JSON.

        Args:
            flow: The :class:`Flow` instance to resume.
            checkpoint: The :class:`FlowCheckpoint` produced by an
                earlier run.
            config: Optional :class:`FlowConfig` for the resumed run.
                Defaults to a fresh :class:`FlowConfig` if not
                supplied; bounds reset at resume.
            context: Optional developer-supplied ``TContext`` value.
            agent_resolutions: Optional mapping
                ``defer_key → RunState JSON`` used by
                :func:`arun_flow_agent` on the agent-bridge resume
                path. Empty / ``None`` for runs that did not defer
                via an inner agent.

        Returns:
            A :class:`FlowRunResult` representing the run from resume
            to completion (or another halt).

        Raises:
            FlowDefinitionError: When the flow's registry references
                step names that do not match the checkpoint's
                ``completed_steps`` / ``pending_steps``.
        """
        from troopai.adk.flows.config import FlowConfig
        from troopai.adk.flows.executor import FlowExecutor

        effective_config = config if config is not None else FlowConfig()
        run_ctx: RunContext[Any] = RunContext.make(context)
        flow.run_context = run_ctx
        flow.set_pending_approvals(checkpoint.decisions)
        flow.set_pending_agent_resolutions(dict(agent_resolutions) if agent_resolutions is not None else {})

        executor: FlowExecutor[Any] = FlowExecutor(flow, config=effective_config)
        _seed_executor_from_checkpoint(executor, checkpoint)
        return await executor.run()

    @classmethod
    async def arun_flow_from_id(
        cls,
        flow: Flow[Any],
        checkpoint_id: str,
        backend: FlowWorkerBackend,
        *,
        config: FlowConfig | None = None,
        context: Any = None,
        agent_resolutions: Mapping[str, str] | None = None,
    ) -> FlowRunResult[Any]:
        """Resume a Flow run by looking up a checkpoint from a backend by id.

        Convenience wrapper over :meth:`arun_flow_from_checkpoint` for
        callers that hold only the string ``checkpoint_id`` (i.e. the
        :attr:`~troopai.adk.flows.checkpoint.FlowCheckpoint.flow_id`) rather
        than the full :class:`~troopai.adk.flows.checkpoint.FlowCheckpoint`
        object.

        The call loads the checkpoint via
        :meth:`~troopai.adk.flows.worker_backend.FlowWorkerBackend.load_checkpoint_by_id`,
        then delegates to :meth:`arun_flow_from_checkpoint`.  Raises
        :class:`~troopai.adk.flows.exceptions.FlowCheckpointNotFoundError`
        when ``checkpoint_id`` is not found in the backend so the caller
        can distinguish "id not found" from "flow is complete".

        Args:
            flow: The :class:`~troopai.adk.flows.flow.Flow` instance to
                resume.  MUST use the same class as the originating run.
            checkpoint_id: The :attr:`~troopai.adk.flows.checkpoint.FlowCheckpoint.flow_id`
                stored in the backend.
            backend: A :class:`~troopai.adk.flows.worker_backend.FlowWorkerBackend`
                that persisted the checkpoint.
            config: Optional :class:`~troopai.adk.flows.config.FlowConfig`.
            context: Optional developer-supplied ``TContext`` value.
            agent_resolutions: Optional agent-bridge resolution map
                (see :meth:`arun_flow_from_checkpoint`).

        Returns:
            A :class:`~troopai.adk.flows.result.FlowRunResult` from
            resume to completion (or another halt).

        Raises:
            FlowCheckpointNotFoundError: When ``checkpoint_id`` is not
                found in ``backend``.
        """
        from troopai.adk.flows.exceptions import FlowCheckpointNotFoundError

        checkpoint = await backend.load_checkpoint_by_id(checkpoint_id)
        if checkpoint is None:
            raise FlowCheckpointNotFoundError(checkpoint_id)
        return await cls.arun_flow_from_checkpoint(
            flow,
            checkpoint,
            config=config,
            context=context,
            agent_resolutions=agent_resolutions,
        )

    @classmethod
    async def arun_flow_distributed(
        cls,
        flow: Flow[Any],
        backend: FlowWorkerBackend,
        *,
        worker_id: str | None = None,
        config: FlowConfig | None = None,
        context: Any = None,
        agent_resolutions: Mapping[str, str] | None = None,
    ) -> FlowRunResult[Any]:
        """Run one batch of a flow under a shared :class:`FlowWorkerBackend`.

        Cross-process distribution entry point. The worker claims the
        flow's next batch via ``backend.claim_batch``, runs every step
        in that batch (in parallel via :func:`asyncio.gather` within
        the worker process), writes the resulting checkpoint via
        ``backend.release_batch``, and returns a :class:`FlowRunResult`.

        A subsequent call by the same or another worker resumes from
        the persisted checkpoint and processes the next batch. The
        loop terminates when ``backend.load_checkpoint`` returns a
        checkpoint with no ``pending_steps`` — the caller observes
        ``status="completed"``.

        Single-batch semantics — the caller drives the polling
        loop. Tests and one-shot integrations call this directly;
        a daemon wrapper around it is the natural extension when a
        long-running worker is desired.

        Args:
            flow: The :class:`Flow` instance to run one batch of.
                MUST have the same registry as the originating flow.
            backend: A :class:`FlowWorkerBackend` instance.
            worker_id: Opaque identifier for this worker. ``None``
                defaults to ``f"worker-{uuid4().hex[:8]}"``.
            config: Optional :class:`FlowConfig`.
            context: Optional ``TContext``.
            agent_resolutions: Optional agent-bridge resolution map
                (see :meth:`arun_flow_from_checkpoint`).

        Returns:
            A :class:`FlowRunResult` for the executor that ran the
            claimed batch. Status ``"completed"`` when no further
            batches remain; ``"deferred"`` when a HITL gate fired;
            otherwise the partial run terminates on a per-batch
            boundary and the developer schedules the next call.
        """
        from troopai.adk.flows.checkpoint import FlowCheckpoint
        from troopai.adk.flows.config import FlowConfig

        effective_config = config if config is not None else FlowConfig()
        resolved_worker_id = worker_id if worker_id is not None else f"worker-{uuid.uuid4().hex[:8]}"
        existing = await backend.load_checkpoint(flow.flow_id)
        if existing is None:
            # Cold start — synthesise the initial checkpoint so workers
            # have a stable hand-off shape from the very first claim.
            # ``starts`` is a frozenset; sort it for cross-process
            # determinism so two workers cold-starting the same flow
            # never disagree on the start ordering.
            existing = FlowCheckpoint(
                flow_id=flow.flow_id,
                completed_steps=(),
                pending_steps=tuple(sorted(flow.get_registry().starts)),
                and_gate_arrivals={},
                consumed_gates=(),
                state_data=_encode_flow_state(flow),
            )
            await backend.save_checkpoint(existing)
        batch_id = len(existing.completed_steps)
        logger.info(
            "arun_flow_distributed: flow=%s batch=%d worker=%s claim attempt",
            flow.flow_id,
            batch_id,
            resolved_worker_id,
        )
        claimed = await backend.claim_batch(flow.flow_id, batch_id, resolved_worker_id)
        if not claimed:
            logger.info(
                "arun_flow_distributed: flow=%s batch=%d worker=%s lost claim",
                flow.flow_id,
                batch_id,
                resolved_worker_id,
            )
            return _build_lost_claim_result(flow, batch_id, resolved_worker_id)
        # CRITICAL: a claim leaked past TTL would block all workers
        # for ttl_seconds. Always release in finally — even when the
        # inner run raises (programmer errors, cancellation, etc.).
        try:
            result = await cls.arun_flow_from_checkpoint(
                flow,
                existing,
                config=effective_config,
                context=context,
                agent_resolutions=agent_resolutions,
            )
        except BaseException:
            logger.exception(
                "arun_flow_distributed: flow=%s batch=%d worker=%s aborted by exception; "
                "releasing claim with the original checkpoint preserved.",
                flow.flow_id,
                batch_id,
                resolved_worker_id,
            )
            await backend.release_batch(
                flow.flow_id,
                batch_id,
                resolved_worker_id,
                existing,
            )
            raise
        if result.checkpoint is not None:
            await backend.release_batch(
                flow.flow_id,
                batch_id,
                resolved_worker_id,
                result.checkpoint,
            )
        else:
            # Terminal status (completed / failed) — persist a final
            # checkpoint reflecting completed_steps for downstream
            # observability, then release the claim.
            final = FlowCheckpoint(
                flow_id=flow.flow_id,
                completed_steps=result.completed_steps,
                pending_steps=(),
                and_gate_arrivals={},
                consumed_gates=(),
                state_data=_encode_flow_state(flow),
            )
            await backend.release_batch(
                flow.flow_id,
                batch_id,
                resolved_worker_id,
                final,
            )
        return result

    @classmethod
    async def arun_flow_for_each(
        cls,
        flow_factory: Callable[[Any], Flow[Any]],
        initial_states: Sequence[Any],
        *,
        concurrency: int = 1,
        config: FlowConfig | None = None,
        context: Any = None,
    ) -> tuple[FlowRunResult[Any], ...]:
        """Fan out a Flow run over a batch of initial states.

        See ``docs/flows/flows.md`` for the full discussion of batch
        semantics, cancellation, and aliasing. Summary:

        - One fresh :class:`Flow` per input via ``flow_factory(state)``;
          results aligned to input order.
        - Per-item :class:`TroopAIError` failures land on
          ``FlowRunResult.status == "failed"`` rather than being raised
          — programmer errors (``NameError``, ``AttributeError``,
          ``TypeError``) and cancellation-class exceptions
          (``CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit``)
          propagate; the batch is all-or-nothing under those.
        - ``concurrency=1`` (default) is the cost-conservative
          sequential path. ``concurrency>=2`` caps fan-out via
          :class:`asyncio.Semaphore`.
        - Caller-supplied state instances MAY be partially mutated by
          the factory before a failure surfaces — callers should not
          retry a failed item against the same state reference without
          first inspecting it.

        Args:
            flow_factory: Callable that builds one :class:`Flow` from
                one input state. Called once per element of
                ``initial_states`` immediately before that item's run.
            initial_states: Sequence of state values; empty ⇒ ``()``.
            concurrency: Maximum concurrent runs (``>= 1``).
            config: Optional :class:`FlowConfig`. Same instance applied
                to every per-item executor; do not mutate from steps.
            context: Optional ``TContext`` attached to every per-item
                ``RunContext``.

        Returns:
            Tuple of :class:`FlowRunResult` aligned to
            ``initial_states``. Per-item failures surface via
            ``result.status`` / ``result.error``; cancellation /
            programmer errors propagate.

        Raises:
            ValueError: When ``concurrency < 1``.
        """
        _validate_batch_inputs(concurrency)
        if len(initial_states) == 0:
            return ()
        if concurrency == 1:
            return await cls._run_flow_batch_sequential(
                flow_factory,
                initial_states,
                config=config,
                context=context,
            )
        return await cls._run_flow_batch_concurrent(
            flow_factory,
            initial_states,
            concurrency=concurrency,
            config=config,
            context=context,
        )

    @classmethod
    async def _run_flow_batch_sequential(
        cls,
        flow_factory: Callable[[Any], Flow[Any]],
        initial_states: Sequence[Any],
        *,
        config: FlowConfig | None,
        context: Any,
    ) -> tuple[FlowRunResult[Any], ...]:
        """Sequential branch — one ``await arun_flow`` per element, in order."""
        results: list[FlowRunResult[Any]] = []
        for state in initial_states:
            results.append(
                await cls._run_one_in_batch(flow_factory, state, config=config, context=context),
            )
        return tuple(results)

    @classmethod
    async def _run_flow_batch_concurrent(
        cls,
        flow_factory: Callable[[Any], Flow[Any]],
        initial_states: Sequence[Any],
        *,
        concurrency: int,
        config: FlowConfig | None,
        context: Any,
    ) -> tuple[FlowRunResult[Any], ...]:
        """Bounded-parallel branch — ``asyncio.gather`` capped by a semaphore."""
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(state: Any) -> FlowRunResult[Any]:
            async with sem:
                return await cls._run_one_in_batch(
                    flow_factory,
                    state,
                    config=config,
                    context=context,
                )

        # `_run_one_in_batch` absorbs every per-item TroopAIError into a
        # FlowRunResult; any exception escaping the helper is a
        # programmer error (NameError, AttributeError, ...) or a
        # cancellation — both of which MUST propagate immediately.
        # return_exceptions=True would silently merge them into the
        # return tuple as raw exception objects, breaking the
        # type contract.
        gathered = await asyncio.gather(
            *(_bounded(state) for state in initial_states),
            return_exceptions=False,
        )
        return tuple(gathered)

    @classmethod
    async def _run_one_in_batch(
        cls,
        flow_factory: Callable[[Any], Flow[Any]],
        state: Any,
        *,
        config: FlowConfig | None,
        context: Any,
    ) -> FlowRunResult[Any]:
        """Run one input through ``arun_flow`` and capture framework errors.

        Step-level exceptions are already routed through
        :attr:`FlowConfig.error_policy` inside the executor — they
        surface as :class:`FlowRunResult` with ``status="failed"``
        without ever escaping. This guard catches framework-level
        :class:`TroopAIError` subclasses that can still bubble out
        (e.g. :class:`FlowDefinitionError` from ``encode_state`` on
        an unsupported state type, or a future :class:`TroopAIError`
        raised by the framework boundary).

        Programmer errors (``NameError``, ``AttributeError``,
        ``TypeError``) and cancellation-class exceptions are
        intentionally NOT caught — they indicate a real bug in the
        flow factory / framework, or operator-initiated shutdown.
        Catching them would silently swallow real bugs into per-item
        ``status="failed"`` strings.

        On framework failure, the full traceback is logged via
        ``logger.exception`` so the configured logging handler
        records it in addition to the synthesised error string on the
        returned ``FlowRunResult``. The result also preserves
        ``cumulative_usage`` from the flow's :class:`RunContext` when
        available — partial token spend is not silently dropped from
        batch totals.
        """
        from troopai.adk.exceptions.exceptions import TroopAIError
        from troopai.adk.flows.exceptions import FlowAgentDeferred
        from troopai.adk.flows.result import FlowRunResult
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        try:
            flow = flow_factory(state)
        except TroopAIError as exc:
            logger.exception("Runner.arun_flow_for_each: flow_factory raised")
            return FlowRunResult(
                final_state=state,
                flow_id=f"flow-factory-failed-{uuid.uuid4().hex[:8]}",
                status="failed",
                completed_steps=(),
                cumulative_usage=LLMUsage(),
                error=f"flow_factory raised: {type(exc).__name__}: {exc}",
            )
        try:
            return await cls.arun_flow(flow, config=config, context=context)
        except FlowAgentDeferred:
            # Internal flow-control signal — MUST never be swallowed
            # as a per-item failure. The executor catches it inside
            # ``_process_batch_results``; if it escapes that far the
            # surrounding routing has regressed and we want the
            # exception to surface immediately.
            raise
        except TroopAIError as exc:
            logger.exception(
                "Runner.arun_flow_for_each: arun_flow raised for flow_id=%s",
                flow.flow_id,
            )
            run_ctx = getattr(flow, "run_context", None)
            partial_usage = run_ctx.usage if run_ctx is not None else LLMUsage()
            return FlowRunResult(
                final_state=flow.state,
                flow_id=flow.flow_id,
                status="failed",
                completed_steps=(),
                cumulative_usage=partial_usage,
                error=f"{type(exc).__name__}: {exc}",
            )

    @classmethod
    def arun_flow_streamed(
        cls,
        flow: Flow[Any],
        *,
        config: FlowConfig | None = None,
        context: Any = None,
    ) -> FlowRunResultStreaming[Any]:
        """Execute a :class:`Flow` asynchronously with event streaming.

        Returns a :class:`FlowRunResultStreaming` immediately. Consumers
        iterate via ``async for event in result.stream_events():`` to
        receive :class:`FlowEvent` instances as steps progress. The
        final state and status populate on the result instance once the
        stream ends.

        A background asyncio task (:func:`_drive_flow_stream`) drives the
        :class:`FlowExecutor` whose ``on_event`` callback pushes events
        into the result's queue. ``final_state`` / ``status`` populate
        when the stream ends.

        Args:
            flow: The :class:`Flow` instance to execute.
            config: Optional :class:`FlowConfig`. See :meth:`arun_flow`.
            context: Optional developer-supplied ``TContext``.

        Returns:
            A :class:`FlowRunResultStreaming` whose ``stream_events()``
            yields :class:`FlowEvent` instances as the run progresses.
        """
        from troopai.adk.flows.config import FlowConfig
        from troopai.adk.flows.executor import FlowExecutor
        from troopai.adk.flows.result import FlowRunResultStreaming

        effective_config = config if config is not None else FlowConfig()
        run_ctx: RunContext[Any] = RunContext.make(context)
        flow.run_context = run_ctx

        result: FlowRunResultStreaming[Any] = FlowRunResultStreaming(flow_id=flow.flow_id)

        executor: FlowExecutor[Any] = FlowExecutor(
            flow,
            config=effective_config,
            on_event=result.push_event,
        )
        try:
            loop = asyncio.get_running_loop()
            result.set_producer_task(loop.create_task(_drive_flow_stream(executor, result)))
        except RuntimeError:
            # No running loop — store the producer for lazy creation on the
            # first stream_events() call (which runs inside a loop). Without
            # this, no producer is ever scheduled and stream_events() blocks
            # forever on an empty queue.
            result.set_deferred_run_impl(lambda: _drive_flow_stream(executor, result))

        return result

    @classmethod
    async def _run_streamed_impl(
        cls,
        agent: Agent,
        user_prompt: UserPrompt,
        result: RunResultStreaming,
        hooks: RunHooks[TContext],
        config: RunConfig,
        initial_messages: list[Any] | None = None,
        extra_tools: list[Any] | None = None,
        swarm_tool_names: set[str] | None = None,
        dispose_toolsets: bool = True,
    ) -> None:
        """Internal implementation of streamed execution.

        ``dispose_toolsets`` defaults to ``True`` so a standalone streamed
        run releases its toolset connections in the ``finally`` arm. The
        swarm driver passes ``False``: it revisits the same member across
        turns and disposes every member once in its own ``finally``, so
        disposing per turn would leave an MCP member with an empty toolset
        on its second and later turns.
        """
        if result.context is None:
            raise ValueError(
                "_run_streamed_impl: result.context must not be None — "
                "pass context=run_context when constructing RunResultStreaming"
            )
        ctx_wrapper = RunContext.from_run_context(result.context)

        if config.include_hook_events:
            hooks = compose_run_hooks(_HookEventEmitter(result), hooks)

        # Call hooks
        await hooks.on_agent_start(ctx_wrapper, agent)
        if agent.hooks is not None:
            await agent.hooks.on_start(ctx_wrapper, agent)

        # Root agent span for the streamed run (mirrors the non-streaming
        # path). Opened here inside the producer task so child spans from
        # the loop attach via contextvars.
        _root_span = agent_span(
            name=agent.name,
            tools=[getattr(t, "name", str(t)) for t in agent.tools],
            handoffs=_handoff_names_for_span(agent),
            output_type=_output_type_name_for_span(agent),
            metadata=config.tracing_metadata,
            tenant_id=ctx_wrapper.tenant_id,
            disabled=not (config.tracing_enabled or config.metrics_enabled),
        )
        _root_span.start()

        try:
            # Run blocking input guardrails before agent starts
            blocking_results = await run_blocking_input_guardrails(
                agent,
                user_prompt,
                ctx_wrapper,
                hooks,
                config.guardrails.input,
                tracing_enabled=config.tracing_enabled,
                metrics_enabled=config.metrics_enabled,
            )
            result.guardrail_results.input = tuple(blocking_results)

            # Start parallel input guardrails concurrently with agent loop
            result.set_input_guardrails_task(
                asyncio.create_task(
                    run_parallel_input_guardrails(
                        agent,
                        user_prompt,
                        ctx_wrapper,
                        hooks,
                        config.guardrails.input,
                        tracing_enabled=config.tracing_enabled,
                        metrics_enabled=config.metrics_enabled,
                    )
                )
            )

            # Execute agent loop with streaming.
            # If the loop raises, _cleanup() cancels _input_guardrails_task
            # via RunResultStreaming._cleanup().
            await run_agent_loop_streamed(
                agent=agent,
                user_prompt=user_prompt,
                result=result,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                config=config,
                initial_messages=initial_messages,
                extra_tools=extra_tools,
                swarm_tool_names=swarm_tool_names,
            )

            # Await parallel guardrails (raises if tripwire triggered)
            guardrails_task = result.get_input_guardrails_task()
            if guardrails_task is not None:
                parallel_results = await guardrails_task
                result.guardrail_results.input = (*result.guardrail_results.input, *parallel_results)
                result.clear_input_guardrails_task()

            # Output guardrails run on the *current* agent after handoffs.
            # RunResultStreaming.current_agent is always set (initialized to
            # the starting agent, updated on each handoff by the loop).
            if not result.requires_action:
                output_agent = result.current_agent
                output_results = await run_output_guardrails(
                    output_agent,
                    result.final_output,
                    ctx_wrapper,
                    hooks,
                    config.guardrails.output,
                    on_transform=lambda replacement: apply_output_transform(result, replacement),
                    tracing_enabled=config.tracing_enabled,
                    metrics_enabled=config.metrics_enabled,
                )
                result.guardrail_results.output = tuple(output_results)

            result.guardrail_audit = ctx_wrapper.collect_guardrail_audit()

            # Call hooks
            await hooks.on_agent_end(ctx_wrapper, agent, result)
            if agent.hooks is not None:
                await agent.hooks.on_end(ctx_wrapper, agent, result.final_output)

        except (AgentInputGuardrailTripwireTriggered, AgentOutputGuardrailTripwireTriggered) as e:
            _root_span.set_error(type(e).__name__, data={"message": str(e)})
            raise
        except Exception as e:
            _root_span.set_error(type(e).__name__, data={"message": str(e)})
            # An input-guardrail tripwire takes precedence over error-handler
            # recovery: if the parallel guardrail task already tripped, surface
            # that verdict instead of quietly "recovering" a guardrail-blocked
            # run because the loop happened to raise first (mirrors the
            # non-streamed path's parallel-guardrail await).
            tripwire = await _pending_streamed_input_tripwire(result)
            if tripwire is not None:
                _root_span.set_error(type(tripwire).__name__, data={"message": str(tripwire)})
                raise tripwire from e
            if config.error_handlers is not None:
                handler = _resolve_error_handler(e, config.error_handlers)
                if handler is not None:
                    logger.warning(
                        "Error handler recovering from %s: %s",
                        type(e).__name__,
                        e,
                    )
                    raw = handler(e)
                    fallback = await raw if inspect.isawaitable(raw) else raw
                    result.final_output = fallback
                    result.recovered = True
                    await hooks.on_agent_end(ctx_wrapper, agent, result)
                    if agent.hooks is not None:
                        await agent.hooks.on_end(ctx_wrapper, agent, result.final_output)
                    return  # recovered — don't raise; persistence is skipped by run_impl
            logger.error("Error during streamed execution: %s", e)
            raise
        finally:
            _root_span.finish()
            if dispose_toolsets:
                await _dispose_agent_toolsets(agent)


async def _pending_streamed_input_tripwire(
    result: RunResultStreaming,
) -> AgentInputGuardrailTripwireTriggered | None:
    """Return a completed parallel input-guardrail tripwire, if any.

    Mirrors the non-streamed parallel-guardrail handling: if the task already
    finished with an ``AgentInputGuardrailTripwireTriggered``, return it so the
    caller can give it precedence over error-handler recovery. A still-running
    task is cancelled and its verdict abandoned (as on the non-streamed path);
    a non-tripwire task failure never masks the loop error. The task is cleared
    either way so the stream consumer's cleanup does not re-cancel it.
    """
    task = result.get_input_guardrails_task()
    if task is None:
        return None
    try:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return None
        if task.cancelled():
            return None
        task_exc = task.exception()
        if isinstance(task_exc, AgentInputGuardrailTripwireTriggered):
            return task_exc
        return None
    finally:
        result.clear_input_guardrails_task()


# =====================================================================
# Session helpers (module-level, used by Runner)
# =====================================================================


async def _streamed_load_session_history(
    session: SessionStore | None,
    user_prompt: UserPrompt,
    hooks: RunHooks[Any],
    run_context: RunContext[Any],
) -> UserPrompt:
    """Load session history and prepend it to the user prompt.

    Mirrors the session-load path on the non-streamed `arun` and the
    swarm path: when a `Session` is attached AND the user prompt is a
    string, persisted history is fetched and merged ahead of the new
    user message. When either condition fails, the original prompt
    flows through unchanged.

    Returns the effective input to feed to the streaming impl.
    """
    if session is None or not isinstance(user_prompt, str):
        return user_prompt
    limit = session.settings.limit if session.settings is not None else None
    events = await session.get(limit=limit)
    if len(events) == 0:
        return user_prompt
    ctx_w: RunContext[Any] = RunContext.from_run_context(run_context)
    await hooks.on_session_load(ctx_w, session, events)
    history = [e.content for e in events]
    user_msg: LLMInputEasyMessage = {"role": "user", "content": user_prompt}
    return [*history, user_msg]


async def _streamed_persist_events(
    result: RunResultStreaming,
    session: SessionStore | None,
    memory: MemoryConfig | None,
    agent: Agent,
    hooks: RunHooks[Any],
    run_context: RunContext[Any],
    user_prompt: UserPrompt,
) -> None:
    """Persist new events to the session + extract memories.

    Builds the user-message + agent-output events from the drained
    `RunResultStreaming` and appends them to the session (when set).
    When `MemoryConfig.auto_extract` is on and an extractor is wired,
    delegates extraction to the memory store. No-op when neither
    session nor memory is configured.
    """
    from troopai.adk.session.session_event import SessionEvent, create_session_event

    events_to_save: list[SessionEvent] = []
    if isinstance(user_prompt, str):
        user_event: LLMInputEasyMessage = {"role": "user", "content": user_prompt}
        events_to_save.append(create_session_event(author="user", content=user_event))
    for item in result.new_items:
        events_to_save.append(
            create_session_event(author=_infer_author(item), content=item.to_param()),
        )
    if session is not None and len(events_to_save) > 0:
        await session.add(events_to_save)
        ctx_w: RunContext[Any] = RunContext.from_run_context(run_context)
        await hooks.on_session_save(ctx_w, session, events_to_save)
    if memory is not None and memory.auto_extract and memory.extractor is not None and len(events_to_save) > 0:
        await memory.memory.add_events(
            events_to_save,
            namespace=memory.namespace,
            extractor=memory.extractor,
            session_id=session.id if session is not None else None,
            agent_name=agent.name,
        )


def _handoff_names_for_span(agent: Agent) -> list[str] | None:
    """Flatten ``agent.handoffs`` into a plain list of target names for tracing.

    Supports both handoff strategies without importing their types:
    - ``list[Agent | Handoff]`` → names extracted via attribute lookup
    - ``HandoffRoute`` → the route's destination agent names
    - ``None`` → returns ``None`` (no handoffs configured)
    """
    handoffs = agent.handoffs
    if handoffs is None:
        return None
    if isinstance(handoffs, list):
        return [str(getattr(h, "agent_name", None) or getattr(h, "name", str(h))) for h in handoffs]
    # HandoffRoute: walk destination agents via get_destinations() if present
    destinations = getattr(handoffs, "destinations", None)
    if destinations is None:
        return None
    try:
        return [getattr(d, "name", str(d)) for d in destinations]
    except Exception:
        return None


def _output_type_name_for_span(agent: Agent) -> str | None:
    """Best-effort friendly name for ``agent.output_schema`` in tracing spans."""
    schema = getattr(agent, "output_schema", None)
    if schema is None:
        return None
    return getattr(schema, "__name__", None) or type(schema).__name__


def _infer_author(item: object) -> str:
    """Infer the session event author from a RunItem's type."""
    from troopai.adk.types.items.items import (
        SystemItem,
        ToolCallOutputItem,
        UserItem,
    )

    if isinstance(item, UserItem):
        return "user"
    if isinstance(item, SystemItem):
        return "system"
    if isinstance(item, ToolCallOutputItem):
        return "tool"
    # All other items (MessageOutputItem, ToolCallItem, ReasoningItem, etc.)
    return "assistant"


# =====================================================================
# Memory helpers (module-level, used by Runner)
# =====================================================================


_TASK_NAME_MAX_LEN = 80
"""Cap the task-panel name at 80 chars. Longer prompts get truncated
with an ellipsis suffix so the bordered panel stays one line."""

_TASK_ERROR_MAX_LEN = 500
"""Cap the stringified exception captured in ``TaskOutput.error`` to
limit credential / response-body leakage from provider exceptions
(``litellm.exceptions.AuthenticationError``, ``httpx.HTTPStatusError``,
etc.). The full exception still propagates via ``raise`` in
``arun_task`` and via ``logger.error`` so debug fidelity is preserved
elsewhere; only the developer-facing summary is truncated."""


async def _arun_task_pipeline_sequential(
    runner_cls: type[Runner],
    task_pipeline: TaskPipeline[TContext],
    pipeline_ctx: RunContext[TContext],
    *,
    context: TContext | None,
    hooks: RunHooks[TContext] | None,
    run_config: RunConfig | None,
    session: SessionStore | None,
    memory: MemoryConfig | None,
) -> tuple[list[TaskOutput], Any]:
    """Run pipeline tasks in declaration order, halting on first error.

    The original :class:`TaskPipeline` execution path. Used when no
    task declares :attr:`Task.depends_on`. Skip / error / usage
    handling unchanged from the prior implementation.
    """
    from dataclasses import replace

    task_outputs_list: list[TaskOutput] = []
    final_output: Any = None

    for task in task_pipeline.tasks:
        prior_outputs = tuple(task_outputs_list)
        slot_id, slot_name = _resolve_task_identity(task)

        if task.skip_if is not None:
            try:
                should_skip = task.skip_if(prior_outputs)
            except Exception as e:
                task_outputs_list.append(_build_task_error_slot(slot_id, slot_name, e, task.metadata))
                logger.warning("Pipeline halted at '%s' (%s): skip_if raised %s", slot_name, slot_id, e)
                return task_outputs_list, final_output
            if should_skip:
                task_outputs_list.append(_build_task_skip_slot(slot_id, slot_name, task.metadata))
                continue

        effective_task = replace(task, task_id=slot_id, name=slot_name)
        try:
            output = await runner_cls.arun_task(
                effective_task,
                context=context,
                hooks=hooks,
                run_config=run_config,
                session=session,
                memory=memory,
            )
        except Exception as e:
            task_outputs_list.append(_build_task_error_slot(slot_id, slot_name, e, effective_task.metadata))
            logger.warning("Pipeline halted at '%s' (%s): %s", slot_name, slot_id, e)
            return task_outputs_list, final_output

        task_outputs_list.append(output)
        if not output.skipped and output.error is None:
            final_output = output.final_output
        if output.usage is not None:
            pipeline_ctx.usage = pipeline_ctx.usage + output.usage

    return task_outputs_list, final_output


async def _arun_task_pipeline_dag(
    runner_cls: type[Runner],
    task_pipeline: TaskPipeline[TContext],
    pipeline_ctx: RunContext[TContext],
    *,
    context: TContext | None,
    hooks: RunHooks[TContext] | None,
    run_config: RunConfig | None,
    session: SessionStore | None,
    memory: MemoryConfig | None,
    completed_task_ids: frozenset[str],
    preloaded_outputs: tuple[TaskOutput, ...],
) -> tuple[list[TaskOutput], Any]:
    """Run pipeline tasks by topological level, gathered per depth.

    Each level's tasks run concurrently via ``asyncio.gather`` once
    every dependency in earlier levels has settled. Halt semantics:
    when any task in a level errors, its siblings in the SAME level
    are allowed to finish (predictable usage accounting beats partial
    cancellation, since HTTP-bound siblings may be mid-request and
    ``Session.add`` is not idempotent); no later level fires.

    ``skip_if`` receives the in-completion-order tuple of prior
    :class:`TaskOutput` results (skipped + errored slots included).

    The ``completed_task_ids`` set lets the resume path skip tasks
    that finished before a checkpoint — they appear in
    ``preloaded_outputs`` and are placed at their declaration-order
    positions in the returned list. An ID in ``completed_task_ids``
    that doesn't correspond to a task in the pipeline raises
    :class:`TaskPipelineDefinitionError` so a renamed task doesn't
    silently re-run.

    Output ordering: the returned list is sorted to match
    ``task_pipeline.tasks`` declaration order, so positional indexing
    on the result is stable whether or not the pipeline is DAG-shaped.
    """
    from troopai.adk.tasks.topology import TaskPipelineDefinitionError

    known_ids = {t.task_id for t in task_pipeline.tasks if t.task_id is not None}
    unknown_completed = completed_task_ids - known_ids
    if len(unknown_completed) > 0:
        raise TaskPipelineDefinitionError(
            f"Resume state references unknown task_id(s): {sorted(unknown_completed)!r}. "
            f"Pipeline definition has drifted from the checkpoint.",
        )

    by_id_outputs: dict[str, TaskOutput] = {}
    by_id_items: dict[str, tuple[Any, ...]] = {}
    final_output: Any = None
    for slot in preloaded_outputs:
        by_id_outputs[slot.task_id] = slot
        by_id_items[slot.task_id] = tuple(slot.new_items)
        if not slot.skipped and slot.error is None:
            final_output = slot.final_output

    # Resolve each task's identity ONCE, keyed by object identity. A task that
    # declared no task_id gets a fresh runner UUID that cannot be re-derived
    # later, so remembering it here lets output ordering map every task -- id'd
    # or not -- back to its TaskOutput instead of silently dropping the no-id
    # ones (which run as the final DAG level).
    identity_by_task: dict[int, tuple[str, str]] = {id(t): _resolve_task_identity(t) for t in task_pipeline.tasks}

    halted = False

    for level in task_pipeline.topological_levels():
        if halted:
            break
        ready = tuple(t for t in level if t.task_id is None or t.task_id not in completed_task_ids)
        if len(ready) == 0:
            continue

        prior_outputs = _completion_order(task_pipeline, by_id_outputs, identity_by_task)
        coros = []
        identities: list[tuple[str, str]] = []
        for task in ready:
            slot_id, slot_name = identity_by_task[id(task)]
            identities.append((slot_id, slot_name))
            coros.append(
                _run_one_task_in_level(
                    runner_cls,
                    task,
                    slot_id,
                    slot_name,
                    prior_outputs,
                    by_id_outputs,
                    by_id_items,
                    context=context,
                    hooks=hooks,
                    run_config=run_config,
                    session=session,
                    memory=memory,
                )
            )

        level_results = await asyncio.gather(*coros, return_exceptions=False)
        for (slot_id, slot_name), output in zip(identities, level_results, strict=True):
            by_id_outputs[output.task_id] = output
            by_id_items[output.task_id] = tuple(output.new_items)
            if output.usage is not None:
                pipeline_ctx.usage = pipeline_ctx.usage + output.usage
            if not output.skipped and output.error is None:
                final_output = output.final_output
            if output.error is not None:
                logger.warning("DAG pipeline halts after level: '%s' (%s) failed", slot_name, slot_id)
                halted = True

    return _declaration_order(task_pipeline, by_id_outputs, identity_by_task), final_output


def _completion_order(
    pipeline: TaskPipeline[Any],
    by_id: dict[str, TaskOutput],
    identity_by_task: dict[int, tuple[str, str]],
) -> tuple[TaskOutput, ...]:
    """Return the outputs seen so far in declaration order.

    Same shape as :func:`_declaration_order` but typed as a tuple for
    pass-through to ``skip_if`` predicates.
    """
    return tuple(_declaration_order(pipeline, by_id, identity_by_task))


def _declaration_order(
    pipeline: TaskPipeline[Any],
    by_id: dict[str, TaskOutput],
    identity_by_task: dict[int, tuple[str, str]],
) -> list[TaskOutput]:
    """Order the accumulated outputs to match the pipeline declaration.

    Every task is matched through ``identity_by_task`` — the per-execution
    map from task object to its resolved ``(slot_id, name)`` — so a task that
    declared no ``task_id`` (and ran under a generated UUID as the final DAG
    level) is placed at its declaration position too, not silently dropped.
    Tasks not yet completed, or absent from ``by_id``, are skipped; finished
    slots are appended in declaration order so positional indexing is stable
    for both id'd and no-id tasks.
    """
    ordered: list[TaskOutput] = []
    for task in pipeline.tasks:
        slot_id = identity_by_task[id(task)][0]
        if slot_id in by_id:
            ordered.append(by_id[slot_id])
    return ordered


async def _run_one_task_in_level(
    runner_cls: type[Runner],
    task: Task,
    slot_id: str,
    slot_name: str,
    prior_outputs: tuple[TaskOutput, ...],
    prior_outputs_by_id: dict[str, TaskOutput],
    prior_items_by_id: dict[str, tuple[Any, ...]],
    *,
    context: Any,
    hooks: RunHooks[Any] | None,
    run_config: RunConfig | None,
    session: SessionStore | None,
    memory: MemoryConfig | None,
) -> TaskOutput:
    """Run a single task inside a DAG level, returning an error / skip slot on failure.

    Wraps :func:`Runner.arun_task` so exceptions surface as
    ``TaskOutput.error`` slots rather than propagating up through
    ``asyncio.gather`` — siblings in the same level are allowed to
    finish; the DAG executor halts AFTER the level when any task in
    it errored.
    """
    from dataclasses import replace

    if task.skip_if is not None:
        try:
            if task.skip_if(prior_outputs):
                return _build_task_skip_slot(slot_id, slot_name, task.metadata)
        except Exception as e:
            logger.warning("Task '%s' (%s) skip_if raised: %s", slot_name, slot_id, e, exc_info=True)
            return _build_task_error_slot(slot_id, slot_name, e, task.metadata)

    composed_description = _compose_task_prompt(task, prior_outputs_by_id, prior_items_by_id)
    effective_task = replace(
        task,
        task_id=slot_id,
        name=slot_name,
        description=composed_description,
    )
    try:
        return await runner_cls.arun_task(
            effective_task,
            context=context,
            hooks=hooks,
            run_config=run_config,
            session=session,
            memory=memory,
        )
    except BaseException as e:
        # CancelledError must propagate: when the level's gather is cancelled
        # (external cancel / timeout), swallowing it into an error slot would
        # defeat the cancellation and let siblings keep running. Only a genuine
        # task *failure* (an Exception) becomes an error slot so siblings finish.
        if isinstance(e, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise
        logger.warning("Task '%s' (%s) failed: %s", slot_name, slot_id, e, exc_info=True)
        return _build_task_error_slot(slot_id, slot_name, e, effective_task.metadata)


def _compose_task_prompt(
    task: Task,
    prior_outputs_by_id: dict[str, TaskOutput],
    prior_items_by_id: dict[str, tuple[Any, ...]],
) -> Any:
    """Build the effective user prompt for ``task`` given prior outputs.

    For each upstream in ``task.depends_on``, the runner builds a
    :class:`TaskInputData` snapshot of the upstream's completion and
    calls ``task.input_filter``. The filter sets
    :attr:`TaskInputData.forwarded` — the :class:`RunItem` subset to
    flow into this task's input. The runner converts the items to
    Layer-1 ``LLMInputContentItem`` via :meth:`RunItem.to_param` and
    prepends them to the message(s) derived from ``task.description``.

    When ``task.input_filter`` is ``None`` or no upstream contributes
    forwarded items, returns ``task.description`` unchanged (the
    cost-conservative default — no prompt rewriting).

    Args:
        task: The downstream task whose prompt is being composed.
        prior_outputs_by_id: Map of completed upstream ``task_id`` →
            :class:`TaskOutput`.
        prior_items_by_id: Map of completed upstream ``task_id`` →
            tuple of :class:`RunItem` produced during that task's run
            (``RunResult.new_items``).

    Returns:
        Either ``task.description`` (no forwarding) or a
        ``list[LLMInputContentItem]`` with forwarded messages first,
        followed by the description messages.
    """
    from troopai.adk.tasks.task import Task, TaskDependency
    from troopai.adk.tasks.task_input_data import TaskInputData

    if task.depends_on is None:
        return task.description

    forwarded_items: list[Any] = []
    for entry in task.depends_on:
        if isinstance(entry, TaskDependency):
            inner = entry.task
            ref_id = inner.task_id if isinstance(inner, Task) else inner
            filter_fn = entry.input_filter
        else:
            # Bare Task or task_id string — pure ordering, no filter.
            continue
        if filter_fn is None or ref_id is None or ref_id not in prior_outputs_by_id:
            continue
        data = TaskInputData(
            task_id=ref_id,
            output=prior_outputs_by_id[ref_id],
            items=prior_items_by_id.get(ref_id, ()),
        )
        filtered = filter_fn(data)
        if filtered.forwarded is None or len(filtered.forwarded) == 0:
            continue
        forwarded_items.extend(filtered.forwarded)

    if len(forwarded_items) == 0:
        return task.description

    forwarded_params = [item.to_param() for item in forwarded_items]
    description_messages = _description_to_messages(task.description)
    return [*forwarded_params, *description_messages]


def _description_to_messages(description: Any) -> list[Any]:
    """Coerce a Task.description into a list of Layer-1 message items.

    ``Task.description`` is ``UserPrompt = str | list[LLMInputContentItem]``.
    Strings wrap into a single user message; lists pass through.
    """
    if isinstance(description, str):
        return [{"role": "user", "content": description}]
    return list(description)


def _resolve_task_identity(task: Task) -> tuple[str, str]:
    """Return the ``(task_id, task_name)`` pair for a task.

    Generates a fresh ``str(uuid.uuid4())`` (full 36-char UUID) when
    ``task.task_id`` is ``None``; falls back to
    ``_derive_task_name(task.description)`` when ``task.name`` is
    ``None``. Called exactly once per task at the pipeline level so
    the same identity flows into the inner ``arun_task`` AND any
    pipeline-level error slot — the two never diverge.

    The verbose Task panel truncates the display to the first 8 chars
    inside the renderer; the full UUID propagates intact through
    hooks, tracing, session events, and :class:`TaskOutput.task_id`.
    """
    task_id = task.task_id if task.task_id is not None else str(uuid.uuid4())
    task_name = task.name if task.name is not None else _derive_task_name(task.description)
    return task_id, task_name


async def _run_task_as_agent(
    runner_cls: type[Runner],
    target: Agent,
    task: Task,
    task_id: str,
    task_name: str,
    context: Any,
    hooks: RunHooks[Any] | None,
    effective_config: RunConfig,
    session: SessionStore | None,
    memory: MemoryConfig | None,
) -> TaskOutput:
    """Dispatch a Task whose target is an :class:`Agent` via :meth:`Runner.arun`.

    Projects the resulting :class:`RunResult` into a
    :class:`TaskOutput` shape (carrying the task identity, final
    output, item trail, and run-context usage).
    """
    from troopai.adk.tasks.task_output import TaskOutput

    result = await runner_cls.arun(
        target,
        task.description,
        context=context,
        hooks=hooks,
        max_turns=task.max_turns if task.max_turns is not None else DEFAULT_MAX_TURNS,
        run_config=effective_config,
        session=session,
        memory=memory,
    )
    return TaskOutput(
        task_id=task_id,
        task_name=task_name,
        final_output=result.final_output,
        new_items=tuple(result.new_items),
        usage=result.context.usage,
        metadata=dict(task.metadata),
    )


async def _run_task_as_swarm(
    runner_cls: type[Runner],
    target: Swarm,
    task: Task,
    task_id: str,
    task_name: str,
    context: Any,
    hooks: RunHooks[Any] | None,
    effective_config: RunConfig,
    session: SessionStore | None,
) -> TaskOutput:
    """Dispatch a Task whose target is a :class:`Swarm`.

    ``Task.memory`` is NOT forwarded because :meth:`Runner.arun_swarm`
    does not accept ``memory`` — swarm-level memory wiring lives on
    member agents directly. ``hooks`` flow through unchanged (the
    swarm path accepts :class:`RunHooks`).

    Projects :class:`SwarmRunResult` to :class:`TaskOutput` by reading
    ``final_output`` and the cumulative usage on the swarm's shared
    :class:`RunContext`.
    """
    from troopai.adk.tasks.task_output import TaskOutput

    result = await runner_cls.arun_swarm(
        target,
        task.description,
        context=context,
        hooks=hooks,
        run_config=effective_config,
        session=session,
    )
    usage = result.context.usage if result.context is not None else None
    return TaskOutput(
        task_id=task_id,
        task_name=task_name,
        final_output=result.final_output,
        new_items=tuple(result.new_items),
        usage=usage,
        metadata=dict(task.metadata),
    )


async def _run_task_as_graph(
    runner_cls: type[Runner],
    target: Graph,
    task: Task,
    task_id: str,
    task_name: str,
    context: Any,
    hooks: RunHooks[Any] | None,
    effective_config: RunConfig,
    session: SessionStore | None,
    memory: MemoryConfig | None,
) -> TaskOutput:
    """Dispatch a Task whose target is a :class:`Graph`.

    ``RunHooks`` are NOT propagated into the graph run because
    :meth:`Runner.arun_graph` takes ``list[GraphHooks | HookProvider]``
    rather than ``RunHooks``. The outer ``on_task_start`` /
    ``on_task_end`` still fire from :meth:`Runner.arun_task`; per-node
    hooks must attach to the :class:`Graph` directly. A
    ``GraphHookProvider`` adapter is a tracked follow-up.

    ``Task.max_turns``, ``Task.usage_limits``, ``session`` and
    ``memory`` have no graph-level analogue on
    :meth:`Runner.arun_graph` (the graph layer's budgets live on
    :class:`GraphConfig` and per-node configs). Supplying any of them
    on a Graph-targeted task emits a single ``logger.warning`` so the
    disconnect is visible, then the values are dropped on the floor.
    """
    from troopai.adk.graphs.result import GraphRunStatus
    from troopai.adk.tasks.task_output import TaskOutput

    if hooks is not None:
        logger.warning(
            "Task '%s' has hooks but Task.agent is a Graph — RunHooks are "
            "not propagated into the graph run. Attach GraphHooks to the "
            "Graph directly.",
            task_name,
        )
    if task.max_turns is not None:
        logger.warning(
            "Task '%s' sets max_turns=%s but Task.agent is a Graph — "
            "max_turns has no graph-level analogue. Use GraphConfig "
            "budgets (max_supersteps / max_total_tokens) instead.",
            task_name,
            task.max_turns,
        )
    if task.usage_limits is not None:
        logger.warning(
            "Task '%s' sets usage_limits but Task.agent is a Graph — "
            "usage_limits is not honoured on the graph path. Use "
            "GraphConfig.max_total_tokens instead.",
            task_name,
        )
    if session is not None:
        logger.warning(
            "Task '%s' has session but Task.agent is a Graph — session "
            "is not threaded into the graph run. Attach per-node session "
            "wiring inside the graph nodes themselves.",
            task_name,
        )
    if memory is not None:
        logger.warning(
            "Task '%s' has memory but Task.agent is a Graph — memory is "
            "not threaded into the graph run. Wire memory per-node "
            "inside the graph instead.",
            task_name,
        )

    result = await runner_cls.arun_graph(
        target,
        task.description,
        context=context,
        run_config=effective_config,
    )
    error_str = result.error if result.status == GraphRunStatus.FAILED else None
    return TaskOutput(
        task_id=task_id,
        task_name=task_name,
        final_output=result.final_output if error_str is None else None,
        new_items=tuple(result.new_items),
        usage=result.cumulative_usage,
        error=error_str,
        metadata=dict(task.metadata),
    )


def _effective_task_target(task: Task) -> Agent | Swarm | Graph:
    """Return ``task.agent`` with optional ``output_schema`` override applied.

    Identity preserved for Swarm / Graph targets — ``output_schema``
    is validated at Task construction to require an Agent, so Swarm /
    Graph branches never enter the replace path.

    For Agent targets: when :attr:`Task.output_schema` is ``None``,
    returns ``task.agent`` unchanged. When set, builds a transient
    agent via ``dataclasses.replace`` — the original agent definition
    is untouched. Tracing's ``_output_type_name_for_span`` and
    ``llm_calls.resolve_output_schema`` both read
    ``agent.output_schema``, so they pick up the override naturally.
    """
    from dataclasses import replace

    from troopai.adk.agents.agent import Agent

    if task.output_schema is None or not isinstance(task.agent, Agent):
        return task.agent
    return replace(task.agent, output_schema=task.output_schema)


def _build_effective_task_config(task: Task, base: RunConfig) -> RunConfig:
    """Build the transient ``RunConfig`` for a task invocation.

    Returns the caller's ``base`` config unchanged when the task adds
    nothing (no guardrails, no usage_limits override). Otherwise
    extends ``base.guardrails`` by APPENDING task-level guardrails
    after run-scope ones (run-scope first, task-scope second per
    ``RunConfig.guardrails`` contract; no de-duplication) and applies
    the task's ``usage_limits`` when set (override semantic — explicit
    opt-in by the developer).
    """
    if len(task.guardrails.input) == 0 and len(task.guardrails.output) == 0 and task.usage_limits is None:
        return base
    from dataclasses import replace

    from troopai.adk.agents.agent_guardrails import AgentGuardrails

    merged = AgentGuardrails(
        input=[*base.guardrails.input, *task.guardrails.input],
        output=[*base.guardrails.output, *task.guardrails.output],
    )
    return replace(
        base,
        guardrails=merged,
        usage_limits=task.usage_limits if task.usage_limits is not None else base.usage_limits,
    )


def _format_task_error(exc: BaseException) -> str:
    """Stringify an exception for ``TaskOutput.error`` with a length cap.

    Prevents leaking provider response bodies / credentials embedded
    in long exception messages into hook callbacks, logs, or session
    persistence. The cap is :data:`_TASK_ERROR_MAX_LEN`.
    """
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= _TASK_ERROR_MAX_LEN:
        return text
    return text[: _TASK_ERROR_MAX_LEN - 3] + "..."


def _build_task_error_slot(
    task_id: str,
    task_name: str,
    exc: BaseException,
    metadata: dict[str, Any],
) -> TaskOutput:
    """Construct an error-set :class:`TaskOutput` slot for a halted pipeline task.

    Centralises the truncation + metadata-copy contract so both
    pipeline error paths (predicate failure + inner-task failure)
    produce identical-shape slots.
    """
    from troopai.adk.tasks.task_output import TaskOutput

    return TaskOutput(
        task_id=task_id,
        task_name=task_name,
        error=_format_task_error(exc),
        metadata=dict(metadata),
    )


def _evaluate_stream_skip(
    task: Task[Any],
    slot_id: str,
    slot_name: str,
    prior_outputs: list[TaskOutput],
) -> tuple[bool, TaskOutput | None]:
    """Evaluate ``task.skip_if`` for the streamed pipeline path.

    Returns ``(should_skip, error_slot)``. When ``skip_if`` raises,
    ``error_slot`` is the recorded error TaskOutput and the caller
    halts the pipeline. When ``skip_if`` returns ``True``,
    ``should_skip`` is ``True`` and ``error_slot`` is ``None``.
    ``(False, None)`` means proceed normally.
    """
    if task.skip_if is None:
        return False, None
    try:
        should_skip = task.skip_if(prior_outputs)
    except Exception as e:
        logger.warning(
            "Pipeline streaming halted at '%s' (%s): skip_if raised %s",
            slot_name,
            slot_id,
            e,
        )
        return False, _build_task_error_slot(slot_id, slot_name, e, task.metadata)
    return should_skip, None


async def _stream_task_pipeline_impl(
    runner_cls: type[Runner],
    task_pipeline: TaskPipeline[Any],
    context: Any,
    hooks: RunHooks[Any] | None,
    run_config: RunConfig | None,
    session: SessionStore | None,
    memory: MemoryConfig | None,
) -> AsyncIterator[tuple[int, RunResultStreaming | None]]:
    """Async generator backing :meth:`Runner.arun_task_pipeline_streamed`.

    Iterates the pipeline tasks in order. Skip slots and error-stop
    slots both yield ``(index, None)``; executable slots yield
    ``(index, RunResultStreaming)``. An ``(index, None)`` yield followed
    by generator exhaustion (no further yields) signals an error-stop —
    the consumer can inspect the final prior_outputs entry to retrieve
    the error detail. A ``(index, None)`` yield followed by additional
    yields is a skipped task.

    ``skip_if`` evaluation walks the prior :class:`TaskOutput` slots —
    the generator records a ``streaming_placeholder=True`` slot for each
    non-skipped task after yielding it, because the consumer-driven inner
    stream may not have completed by the time the next ``skip_if`` fires.
    The placeholder flag is the discriminator: consumers' ``skip_if``
    callables can branch on ``prior_outputs[n].streaming_placeholder``
    to detect "real completion vs awaiting drain". The pipeline does NOT
    aggregate per-task usage; consumers read ``task_stream.usage`` per
    yield.
    """
    from troopai.adk.tasks.task_output import TaskOutput

    prior_outputs: list[TaskOutput] = []
    for index, task in enumerate(task_pipeline.tasks):
        slot_id, slot_name = _resolve_task_identity(task)
        should_skip, error_slot = _evaluate_stream_skip(task, slot_id, slot_name, prior_outputs)
        if error_slot is not None:
            prior_outputs.append(error_slot)
            yield (index, None)
            break
        if should_skip:
            prior_outputs.append(_build_task_skip_slot(slot_id, slot_name, task.metadata))
            yield (index, None)
            continue

        task_stream = await runner_cls.arun_task_streamed(
            task,
            context=context,
            hooks=hooks,
            run_config=run_config,
            session=session,
            memory=memory,
        )
        yield (index, task_stream)
        prior_outputs.append(
            TaskOutput(
                task_id=slot_id,
                task_name=slot_name,
                streaming_placeholder=True,
                metadata=dict(task.metadata),
            ),
        )


def _collect_task_group_outputs(
    tasks: tuple[Task[Any], ...],
    aio_tasks: list[asyncio.Task[TaskOutput]],
) -> list[TaskOutput]:
    """Build the per-task :class:`TaskOutput` slots for a TaskGroup run.

    Walks the asyncio task list in input order. Resolved tasks carry
    their :class:`TaskOutput` directly. Cancelled tasks (under
    ``halt_on_first``) surface as :class:`TaskOutput` slots with an
    explanatory ``error`` field so positional indexing stays stable.
    Tasks that raised pass through the existing
    :func:`_build_task_error_slot` helper so error formatting matches
    the pipeline path.
    """
    from troopai.adk.tasks.task_output import TaskOutput

    outputs: list[TaskOutput] = []
    for task, aio_t in zip(tasks, aio_tasks, strict=True):
        slot_id = task.task_id if task.task_id is not None else str(uuid.uuid4())
        slot_name = task.name if task.name is not None else _derive_task_name(task.description)
        if aio_t.cancelled():
            outputs.append(
                TaskOutput(
                    task_id=slot_id,
                    task_name=slot_name,
                    error="cancelled by halt_on_first policy",
                    metadata=dict(task.metadata),
                ),
            )
            continue
        try:
            outputs.append(aio_t.result())
        except asyncio.CancelledError:
            # The task carried a CancelledError even though
            # `aio_t.cancelled()` was False — happens when the
            # cancellation arrives but the result accessor races the
            # cancelled-flag transition. Surface as a cancel slot so
            # positional indexing stays stable.
            outputs.append(
                TaskOutput(
                    task_id=slot_id,
                    task_name=slot_name,
                    error="cancelled by halt_on_first policy",
                    metadata=dict(task.metadata),
                ),
            )
        except Exception as e:
            outputs.append(_build_task_error_slot(slot_id, slot_name, e, task.metadata))
    return outputs


def _build_task_skip_slot(
    task_id: str,
    task_name: str,
    metadata: dict[str, Any],
) -> TaskOutput:
    """Construct a ``skipped=True`` :class:`TaskOutput` slot.

    Skipped tasks remain in :attr:`TaskPipelineResult.task_outputs`
    so positional indexing stays stable across the pipeline.
    """
    from troopai.adk.tasks.task_output import TaskOutput

    return TaskOutput(
        task_id=task_id,
        task_name=task_name,
        skipped=True,
        metadata=dict(metadata),
    )


def _derive_task_name(user_prompt: UserPrompt) -> str:
    """Derive a short task name from the user prompt for the 📋 Task panel.

    String prompts are truncated to :data:`_TASK_NAME_MAX_LEN` characters
    with an ellipsis suffix. Structured prompts (message lists) fall
    through to a generic label so we never call ``str()`` on a large
    list and inflate the panel.
    """
    if isinstance(user_prompt, str):
        if len(user_prompt) <= _TASK_NAME_MAX_LEN:
            return user_prompt
        return user_prompt[: _TASK_NAME_MAX_LEN - 3] + "..."
    return "(structured input)"


def _extract_query(user_prompt: UserPrompt) -> str:
    """Extract a search query from the user prompt.

    Uses the last user message text as the query for memory search.
    """
    if isinstance(user_prompt, str):
        return user_prompt

    # list of messages — find the last user message
    if isinstance(user_prompt, list):
        for msg in reversed(user_prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content

    return ""


async def _inject_memories(
    effective_input: UserPrompt,
    memory_config: MemoryConfig,
) -> UserPrompt:
    """Search memory and inject results into the input.

    Returns the (possibly modified) effective_input.
    """
    from troopai.adk.memory.memory_config import MemoryInjectionPosition

    query = _extract_query(effective_input)
    if len(query) == 0:
        return effective_input

    results = await memory_config.memory.search(
        query,
        namespace=memory_config.namespace,
        limit=memory_config.inject_limit,
    )
    if len(results) == 0:
        return effective_input

    # Format memories as text
    lines = ["[Relevant memories from previous interactions]"]
    for r in results:
        lines.append(f"- {r.entry.content}")
    memory_text = "\n".join(lines)

    logger.info("Injecting %d memories into input", len(results))

    if memory_config.inject_position == MemoryInjectionPosition.DEVELOPER_MESSAGE:
        # Insert as a developer message before user input
        dev_msg: LLMInputEasyMessage = {"role": "developer", "content": memory_text}
        if isinstance(effective_input, str):
            user_input_msg: LLMInputEasyMessage = {"role": "user", "content": effective_input}
            return [dev_msg, user_input_msg]
        elif isinstance(effective_input, list):
            return [dev_msg, *list(effective_input)]

    elif memory_config.inject_position == MemoryInjectionPosition.SYSTEM_SUFFIX:
        # Append to the first system message, or prepend a new one
        if isinstance(effective_input, str):
            sys_msg: LLMInputEasyMessage = {"role": "system", "content": memory_text}
            user_input_msg_2: LLMInputEasyMessage = {"role": "user", "content": effective_input}
            return [sys_msg, user_input_msg_2]
        elif isinstance(effective_input, list):
            new_input: list[LLMInputContentItem] = list(effective_input)
            for i, msg in enumerate(new_input):
                if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
                    existing = msg.get("content", "")
                    # Preserve structured (list-of-parts) content by appending a
                    # text part; str()-coercing a parts list would flatten it to
                    # a repr and corrupt the message. Scalar text concatenates.
                    # Spread-and-override yields a ``dict[str, Any]``; the ``Any``
                    # local keeps the TypedDict assignment un-widened.
                    if isinstance(existing, list):
                        appended_part: LLMInputText = {"type": "input_text", "text": memory_text}
                        merged: Any = {**msg, "content": [*existing, appended_part]}
                    else:
                        merged = {**msg, "content": str(existing) + "\n\n" + memory_text}
                    new_input[i] = merged
                    return new_input
            # No system message found — prepend one
            prepended: LLMInputEasyMessage = {"role": "system", "content": memory_text}
            return [prepended, *new_input]

    # MemoryInjectionPosition only has DEVELOPER_MESSAGE and SYSTEM_SUFFIX
    # today, and every branch above returns. Both checkers treat the fall-
    # through as unreachable; pyright still demands a terminal return, so
    # raise here to stay total if the enum ever grows.
    raise AssertionError(f"Unhandled inject_position: {memory_config.inject_position!r}")


async def _drive_flow_stream(executor: Any, result: Any) -> None:
    """Drive a :class:`FlowExecutor` in the background of a streamed run.

    Module-level helper used by :meth:`Runner.arun_flow_streamed` so the
    main classmethod stays compact. Pulls the executor to completion,
    then writes the final state /
    status / usage onto ``result``. Always calls ``result.complete()``
    in ``finally`` so the consumer's ``stream_events()`` exits cleanly.

    Captures both expected step exceptions (already surfaced as
    ``status="failed"`` from the executor) and unexpected
    executor-level exceptions (programming bugs in the transition
    table build, for example). Critical exceptions
    (``CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit``)
    propagate untouched so async cancellation works correctly.

    Args:
        executor: The :class:`FlowExecutor` to drive.
        result: The :class:`FlowRunResultStreaming` to finalize.
    """
    try:
        final = await executor.run()
        result.final_state = final.final_state
        result.status = final.status
        result.completed_steps = final.completed_steps
        result.cumulative_usage = final.cumulative_usage
        # Mirror per-step usage attribution onto the streamed result so a
        # streamed flow's per-step breakdown matches the non-streamed
        # FlowRunResult (built by FlowExecutor._build_result). Without this
        # the streamed result's per_step_usage stayed empty.
        result.per_step_usage = final.per_step_usage
        result.error = final.error
        result.guardrail_audit = final.guardrail_audit
        # Mirror the deferral payload onto the streamed result. Without this,
        # result.deferred_steps stays empty, so result.requires_action (defined
        # as len(deferred_steps) > 0) is False even when the flow deferred — and
        # with no checkpoint, streamed HITL would be unrecoverable.
        result.deferred_steps = final.deferred_steps
        result.checkpoint = final.checkpoint
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        result.status = "failed"
        result.error = f"executor crashed: {type(exc).__name__}: {exc}"
        logger.error("Flow executor crashed in arun_flow_streamed", exc_info=exc)
    finally:
        result.complete()


def _encode_flow_state(flow: Any) -> str:
    """Encode ``flow.state`` to JSON for cold-start checkpoint synthesis.

    Routes through :func:`troopai.adk.flows.executor.encode_state` so
    the supported state shapes (Pydantic ``BaseModel`` /
    ``@dataclass``) and the loud-failure semantics for everything else
    stay in one place.
    """
    from troopai.adk.flows.executor import encode_state

    return encode_state(flow.state)


def _build_lost_claim_result(flow: Any, batch_id: int, worker_id: str) -> FlowRunResult[Any]:
    """Return a :class:`FlowRunResult` describing a lost claim attempt.

    Lands when another worker already holds ``batch_id`` within its
    TTL window. The caller treats the result as a non-error
    "skip" signal: schedule the next call, the other worker will
    release its batch shortly.
    """
    from troopai.adk.flows.result import FlowRunResult
    from troopai.adk.types.tokens.llm_usage import LLMUsage

    return FlowRunResult(
        final_state=flow.state,
        flow_id=flow.flow_id,
        status="failed",
        completed_steps=(),
        cumulative_usage=LLMUsage(),
        error=f"FlowWorkerBackend: batch {batch_id} already claimed by another worker (this worker: {worker_id!r})",
    )


def _validate_batch_inputs(concurrency: int) -> None:
    """Boundary check for :meth:`Runner.arun_flow_for_each`.

    Raises :class:`ValueError` when ``concurrency < 1``. Extracted into a
    module-level helper so the public method stays under the function-length
    cap.
    """
    if concurrency < 1:
        raise ValueError(
            f"Runner.arun_flow_for_each: concurrency must be >= 1, got {concurrency}.",
        )


def _seed_executor_from_checkpoint(executor: Any, checkpoint: Any) -> None:
    """Seed a fresh :class:`FlowExecutor` from a :class:`FlowCheckpoint`.

    Module-level helper used by :meth:`Runner.arun_flow_from_checkpoint`
    so the main classmethod stays compact. Mutates ``executor`` in
    place by writing the
    checkpoint's recorded state into the executor's internal fields,
    and replaces the table's ``starts`` tuple with the checkpoint's
    pending queue.

    The ``step_count`` is reset to 0 — the resumed run's
    :attr:`FlowConfig.max_steps` cap is independent of the prior run's
    step count. Names from ``pending_steps`` flow through
    :meth:`FlowExecutor._invoke_step` which validates each name against
    the Flow's registry before invocation; a tampered checkpoint
    cannot trigger arbitrary methods on the Flow subclass.

    Args:
        executor: The newly-constructed :class:`FlowExecutor` to seed.
        checkpoint: The :class:`FlowCheckpoint` providing the recorded
            state.
    """
    from dataclasses import replace

    executor.completed_steps = list(checkpoint.completed_steps)
    executor.step_count = 0
    executor.and_arrivals = {gate_id: set(triggers) for gate_id, triggers in checkpoint.and_gate_arrivals.items()}
    executor.consumed_gates = set(checkpoint.consumed_gates)
    executor.table = replace(executor.table, starts=checkpoint.pending_steps)
    # Re-seed pending_triggers from the persisted FlowDeferredStep.triggers so
    # the resumed FlowStepContext carries the same `triggers` tuple the
    # cold-start invocation would have seen. Without this, a gate callable
    # that inspects `ctx.triggers` would flip behaviour between cold start
    # and resume.
    for deferred in checkpoint.deferred_steps:
        executor.pending_triggers[deferred.step_name] = list(deferred.triggers)
    # Also restore pending_triggers for non-deferred pending steps.  These are
    # steps that were scheduled by sibling completions in the same batch as the
    # deferral but are not themselves deferred.  Their trigger events are
    # serialized in pending_step_triggers (added in the same fix round).
    # Older checkpoints that lack this field produce an empty dict, so the
    # resume degrades gracefully to the previous behaviour.
    for step_name, triggers in checkpoint.pending_step_triggers.items():
        executor.pending_triggers[step_name] = list(triggers)


# --- Sandbox bracket helper --------------------------------------------


async def _maybe_open_sandbox_bracket(
    *,
    stack: contextlib.AsyncExitStack,
    agent: Any,
    config: RunConfig,
    run_context: RunContext[Any],
    hooks: Any = None,
) -> None:
    """Open a sandbox lifecycle context when this run is sandboxed.

    Detects the sandbox path two ways:

    1. ``isinstance(agent, SandboxAgent)`` — the typed entry point.
    2. ``config.sandbox is not None`` — explicit run-level config.

    When detected, opens ``sandbox_run_context`` into ``stack`` so the
    session is acquired before the agent loop and released after
    everything else cleans up. The lifecycle handle is published on
    ``run_context._sandbox_handle`` so capability tools can find it.

    No-op when neither trigger applies — preserves the standard
    no-sandbox path byte-for-byte.
    """
    from troopai.adk.sandbox.agent import SandboxAgent
    from troopai.adk.sandbox.runner_integration.lifecycle import (
        sandbox_run_context,
    )

    sandbox_config = config.sandbox
    capabilities: list[Any] = []
    run_as: Any = None
    concurrency_guard: Any = None

    if isinstance(agent, SandboxAgent):
        capabilities = list(agent.capabilities)
        run_as = agent.run_as
        concurrency_guard = agent.get_concurrency_guard()
        # If the agent declares default_manifest and the run config
        # has no manifest, fall back to the agent's default.
        if sandbox_config is not None and sandbox_config.manifest is None and agent.default_manifest is not None:
            sandbox_config = dataclasses.replace(
                sandbox_config,
                manifest=agent.default_manifest,
            )

    if sandbox_config is None:
        return

    handle = await stack.enter_async_context(
        sandbox_run_context(
            config=sandbox_config,
            capabilities=capabilities,
            run_as=run_as,
            concurrency_guard=concurrency_guard,
            agent_name=agent.name,
            agent=agent,
            run_context=run_context,
            hooks=hooks,
            tracing_enabled=config.tracing_enabled,
        ),
    )
    # Publish on the RunContext so capability tools + observers can
    # find the live session. Object setattr bypasses the dataclass
    # frozen contract for this internal-attribute write.
    object.__setattr__(run_context, "_sandbox_handle", handle)


def _maybe_clone_agent_with_capability_tools(
    *,
    agent: Any,
    run_context: RunContext[Any],
) -> Any:
    """Return a clone of ``agent`` with sandbox capability tools merged.

    When ``run_context._sandbox_handle`` was populated by
    ``_maybe_open_sandbox_bracket``, every bound capability's
    ``tools()`` output is appended to ``agent.tools`` on a clone.
    The clone leaves the original agent untouched — important because
    the same Agent instance may participate in multiple concurrent
    runs and capability tools are session-scoped.

    Returns ``agent`` unchanged when no sandbox handle is present.
    """
    handle = getattr(run_context, "_sandbox_handle", None)
    if handle is None:
        return agent
    # Gather every bound capability's tools.
    cap_tools: list[Any] = []
    for cap in handle.capabilities:
        cap_tools.extend(cap.tools())
    if len(cap_tools) == 0:
        return agent
    # Clone via dataclasses.replace, extending the tools list.
    merged_tools = [*(agent.tools or []), *cap_tools]
    return dataclasses.replace(agent, tools=merged_tools)
