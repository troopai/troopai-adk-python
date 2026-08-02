"""Run hooks for lifecycle callbacks during agent execution.

Hooks allow users to observe and react to events during agent execution,
such as agent start/end, tool calls, handoffs, and guardrail results.

The base-class methods in this file are interface contracts — they exist
so that subclasses can override them. The no-op base implementations use
``del`` on each parameter to signal "intentionally unused" to linters and
type checkers (ruff ARG002 and pyright ``is not accessed``).

Example:
    class MyHooks(RunHooks):
        async def on_agent_start(self, context, agent):
            logger.info(f"Starting: {agent.name}")

        async def on_agent_end(self, context, agent, result):
            logger.info(f"Finished: {agent.name}")
            logger.info(f"Tokens used: {context.usage.total_tokens}")

    result = await Runner.arun(agent, "Hello", hooks=MyHooks())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar, override

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.agents.agent_guardrails import (
        AgentInputGuardrailResult,
        AgentOutputGuardrailResult,
    )
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.session.session_event import SessionEvent
    from troopai.adk.session.state import State
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.tasks.task import Task
    from troopai.adk.tasks.task_output import TaskOutput
    from troopai.adk.tools.tool_guardrails import (
        ToolInputGuardrailResult,
        ToolOutputGuardrailResult,
    )
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.run.run_result import RunResult
    from troopai.adk.types.sandbox.exec_result import ExecResult
    from troopai.adk.types.sandbox.snapshot import SnapshotMetadata
    from troopai.adk.types.sandbox.usage import SandboxUsage
    from troopai.adk.types.session import SessionStore


TContext = TypeVar("TContext")


class RunHooks[TContext]:
    """Base class for run lifecycle hooks.

    Override methods to observe events during agent execution.
    All methods are async and optional - only override the ones you need.

    Type Parameters:
        TContext: The type of the user-provided context.

    Example:
        class LoggingHooks(RunHooks):
            async def on_agent_start(self, context, agent):
                logger.info(f"Agent {agent.name} starting")

            async def on_tool_call(self, context, agent, tool_name, tool_input):
                logger.info(f"Tool call: {tool_name}")
    """

    # --- Task Hooks ---
    #
    # Task lifecycle brackets ``Runner.arun_task`` / ``Runner.arun_task_pipeline``
    # invocations. Fires once per :class:`Task` execution — for a
    # pipeline, the runner fires ``on_task_start`` / ``on_task_end``
    # for every non-skipped task in order. Skipped tasks (predicate
    # returned True) emit neither hook — the verbose renderer surfaces
    # the skip via its panel, but lifecycle observers see only executed
    # tasks. The hooks fire in addition to the existing verbose
    # ``emit_task_start`` / ``emit_task_end`` rendering, never instead.

    async def on_task_start(
        self,
        context: RunContext[TContext],
        agent: Agent | Swarm | Graph,
        task: Task,
    ) -> None:
        """Called when a :class:`Task` starts execution.

        Fires once at the top of :meth:`Runner.arun_task`, before any
        ``on_agent_start`` for the task's agent. In a pipeline, fires
        once per non-skipped task.

        Args:
            context: The run context wrapper.
            agent: The execution target. May be an :class:`Agent`, a
                :class:`Swarm`, or a :class:`Graph` — matches the
                widened :attr:`Task.agent` field. Hooks observing
                multiple target types should branch via
                :func:`isinstance`.
            task: The :class:`Task` about to run, carrying its
                description, overrides, and metadata.
        """
        del context, agent, task

    async def on_task_end(
        self,
        context: RunContext[TContext],
        agent: Agent | Swarm | Graph,
        task: Task,
        *,
        output: TaskOutput,
    ) -> None:
        """Called when a :class:`Task` finishes execution.

        Fires in the ``finally`` arm of :meth:`Runner.arun_task` — so
        observers see both success and failure paths. The
        :class:`TaskOutput` carries ``error`` set when the task raised.

        Args:
            context: The run context wrapper.
            agent: The execution target. Mirrors the value passed to
                :meth:`on_task_start` — Agent, Swarm, or Graph.
            task: The :class:`Task` that just finished.
            output: The resulting :class:`TaskOutput` — inspect
                ``output.error`` to discriminate success from failure.
        """
        del context, agent, task, output

    async def on_agent_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
    ) -> None:
        """Called when an agent starts execution.

        Args:
            context: The run context wrapper.
            agent: The agent that is starting.
        """
        del context, agent

    async def on_agent_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        result: RunResult | RunResultStreaming,
    ) -> None:
        """Called when an agent finishes execution.

        Args:
            context: The run context wrapper.
            agent: The agent that finished.
            result: The result of the agent run.
        """
        del context, agent, result

    async def on_llm_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        messages: list[LLMInputContentItem],
    ) -> None:
        """Called before an LLM call is made.

        Args:
            context: The run context wrapper.
            agent: The agent making the call.
            messages: The messages being sent to the LLM.
        """
        del context, agent, messages

    async def on_llm_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        response: Any,
    ) -> None:
        """Called after an LLM call completes.

        Args:
            context: The run context wrapper.
            agent: The agent that made the call.
            response: The LLM response.
        """
        del context, agent, response

    async def on_tool_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Called before a tool is executed.

        Args:
            context: The run context wrapper.
            agent: The agent executing the tool.
            tool_name: The name of the tool.
            tool_input: The input to the tool.
        """
        del context, agent, tool_name, tool_input

    async def on_tool_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        tool_output: Any,
    ) -> None:
        """Called after a tool finishes execution.

        Args:
            context: The run context wrapper.
            agent: The agent that executed the tool.
            tool_name: The name of the tool.
            tool_output: The output from the tool.
        """
        del context, agent, tool_name, tool_output

    async def on_handoff(
        self,
        context: RunContext[TContext],
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        """Called when a handoff occurs between agents.

        Args:
            context: The run context wrapper.
            from_agent: The agent handing off.
            to_agent: The agent receiving the handoff.
        """
        del context, from_agent, to_agent

    async def on_input_guardrail_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        guardrail_name: str,
    ) -> None:
        """Called before an input guardrail runs.

        Args:
            context: The run context wrapper.
            agent: The agent being guarded.
            guardrail_name: The name of the guardrail.
        """
        del context, agent, guardrail_name

    async def on_input_guardrail_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        result: AgentInputGuardrailResult,
    ) -> None:
        """Called after an input guardrail completes.

        Args:
            context: The run context wrapper.
            agent: The agent being guarded.
            result: The guardrail result.
        """
        del context, agent, result

    async def on_output_guardrail_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        guardrail_name: str,
    ) -> None:
        """Called before an output guardrail runs.

        Args:
            context: The run context wrapper.
            agent: The agent being guarded.
            guardrail_name: The name of the guardrail.
        """
        del context, agent, guardrail_name

    async def on_output_guardrail_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        result: AgentOutputGuardrailResult,
    ) -> None:
        """Called after an output guardrail completes.

        Args:
            context: The run context wrapper.
            agent: The agent being guarded.
            result: The guardrail result.
        """
        del context, agent, result

    # --- Tool Guardrail Hooks ---
    #
    # Tool-level guardrails run *inside* a single tool invocation —
    # distinct from agent-level guardrails which run around the whole
    # agent turn. These four hooks are additive and optional: the
    # default no-op keeps every existing ``RunHooks`` subclass working
    # without changes, while verbose renderers can over-ride them to
    # surface the ADK-first-class tool-guardrail layer CrewAI lacks.

    async def on_tool_input_guardrail_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        guardrail_name: str,
    ) -> None:
        """Called before a per-tool input guardrail runs.

        Args:
            context: The run context wrapper.
            agent: The agent that owns the tool.
            tool_name: The name of the tool being guarded.
            guardrail_name: The name of the input guardrail.
        """
        del context, agent, tool_name, guardrail_name

    async def on_tool_input_guardrail_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        result: ToolInputGuardrailResult,
    ) -> None:
        """Called after a per-tool input guardrail completes.

        Args:
            context: The run context wrapper.
            agent: The agent that owns the tool.
            tool_name: The name of the guarded tool.
            result: The tool guardrail result (with ``output`` carrying
                the allow / reject / raise behaviour).
        """
        del context, agent, tool_name, result

    async def on_tool_output_guardrail_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        guardrail_name: str,
    ) -> None:
        """Called before a per-tool output guardrail runs.

        Args:
            context: The run context wrapper.
            agent: The agent that owns the tool.
            tool_name: The name of the guarded tool.
            guardrail_name: The name of the output guardrail.
        """
        del context, agent, tool_name, guardrail_name

    async def on_tool_output_guardrail_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        result: ToolOutputGuardrailResult,
    ) -> None:
        """Called after a per-tool output guardrail completes.

        Args:
            context: The run context wrapper.
            agent: The agent that owns the tool.
            tool_name: The name of the guarded tool.
            result: The tool guardrail result.
        """
        del context, agent, tool_name, result

    # --- Skill Hooks ---

    async def on_skill_activated(
        self,
        context: RunContext[TContext],
        agent: Agent,
        skill_name: str,
    ) -> None:
        """Called when a skill is activated (LAZY mode only).

        Fired when a skill's tool is called for the first time,
        triggering instruction injection into the system prompt.

        Args:
            context: The run context wrapper.
            agent: The agent whose skill was activated.
            skill_name: The name of the activated skill.
        """
        del context, agent, skill_name

    # --- Session Hooks ---

    async def on_session_load(
        self,
        context: RunContext[TContext],
        session: SessionStore,
        events: list[SessionEvent],
    ) -> None:
        """Called after session history is loaded from storage.

        Fires after ``session.get()`` returns events and before
        the history is used as LLM input.

        Args:
            context: The run context wrapper.
            session: The session that was loaded.
            events: The loaded events.
        """
        del context, session, events

    async def on_session_save(
        self,
        context: RunContext[TContext],
        session: SessionStore,
        events: list[SessionEvent],
    ) -> None:
        """Called after new events are saved to session storage.

        Fires after ``session.add()`` persists events.

        Args:
            context: The run context wrapper.
            session: The session that was saved to.
            events: The events that were saved.
        """
        del context, session, events

    async def on_state_save(
        self,
        context: RunContext[TContext],
        session: SessionStore,
        state: State,
    ) -> None:
        """Called after session state is persisted.

        Fires after ``session.save_state()`` commits state changes.

        Args:
            context: The run context wrapper.
            session: The session whose state was saved.
            state: The state object after persistence.
        """
        del context, session, state

    async def on_mcp_connect(
        self,
        context: RunContext[TContext],
        server_name: str,
    ) -> None:
        """Called before an MCP server's transport opens.

        Fires from ``MCPServerWithClientSession.connect`` so verbose
        output can show the connection attempt before any latency.

        Args:
            context: The run context wrapper.
            server_name: Name of the MCP server about to connect.
        """
        del context, server_name

    async def on_mcp_connected(
        self,
        context: RunContext[TContext],
        server_name: str,
    ) -> None:
        """Called after an MCP server's session has initialised.

        Args:
            context: The run context wrapper.
            server_name: Name of the MCP server now usable.
        """
        del context, server_name

    async def on_mcp_error(
        self,
        context: RunContext[TContext],
        server_name: str,
        error: BaseException,
    ) -> None:
        """Called when an MCP server's transport / session raises.

        Fires for connect failures and unexpected runtime errors —
        not for ``isError=True`` tool results, which flow through the
        regular tool-error path.

        Args:
            context: The run context wrapper.
            server_name: Name of the MCP server that errored.
            error: The exception raised.
        """
        del context, server_name, error

    # --- Sandbox lifecycle ----------------------------------------------------

    async def on_sandbox_start(
        self,
        context: RunContext[TContext],
        agent: Agent[TContext],
        session: Any,
    ) -> None:
        """Called when a sandbox session is acquired for the run.

        Fires once per ``Runner.arun`` when the run involves a
        ``SandboxAgent`` (or ``RunConfig.sandbox`` is set). The session
        argument is typed ``Any`` because ``BaseSandboxSession`` lives
        in the sandbox client package; importing it here would create
        a hooks→clients dependency that the framework keeps loose.

        Args:
            context: The run context wrapper.
            agent: The (sandbox) agent that owns the session.
            session: The live ``BaseSandboxSession``.
        """
        del context, agent, session

    async def on_sandbox_stop(
        self,
        context: RunContext[TContext],
        agent: Agent[TContext],
        session: Any,
        usage: SandboxUsage,
    ) -> None:
        """Called when a sandbox session is released at the end of a run.

        Args:
            context: The run context wrapper.
            agent: The (sandbox) agent that owned the session.
            session: The ``BaseSandboxSession`` that is shutting down.
            usage: Cumulative resource usage for the session.
        """
        del context, agent, session, usage

    async def on_sandbox_exec_start(
        self,
        context: RunContext[TContext],
        agent: Agent[TContext],
        command: str,
    ) -> None:
        """Called before a command runs inside the sandbox.

        Fires once per non-PTY command invocation. PTY interactions
        stream their own events and do NOT fire this hook.

        Args:
            context: The run context wrapper.
            agent: The (sandbox) agent owning the session.
            command: The command about to run (truncated for tracing).
        """
        del context, agent, command

    async def on_sandbox_exec_end(
        self,
        context: RunContext[TContext],
        agent: Agent[TContext],
        command: str,
        result: ExecResult,
    ) -> None:
        """Called after a command finishes inside the sandbox.

        Non-zero ``result.exit_code`` is surfaced via this hook (NOT
        as an exception) — the hook is observation only.

        Args:
            context: The run context wrapper.
            agent: The (sandbox) agent owning the session.
            command: The command that ran (truncated for tracing).
            result: Captured stdout / stderr / exit code / duration.
        """
        del context, agent, command, result

    async def on_sandbox_snapshot(
        self,
        context: RunContext[TContext],
        agent: Agent[TContext],
        metadata: SnapshotMetadata,
    ) -> None:
        """Called when the runtime persists a workspace snapshot.

        Fires from ``session.stop`` when the configured ``SnapshotStore``
        successfully writes a snapshot.

        Args:
            context: The run context wrapper.
            agent: The (sandbox) agent owning the session.
            metadata: Descriptive record of the persisted snapshot.
        """
        del context, agent, metadata

    async def on_sandbox_error(
        self,
        context: RunContext[TContext],
        agent: Agent[TContext],
        error: BaseException,
    ) -> None:
        """Called when the sandbox session raises an unexpected error.

        Fires for backend transport failures, manifest materialization
        errors, snapshot persistence failures — anything in the
        ``SandboxRuntimeError`` / ``SandboxArtifactError`` families.
        ``SandboxConfigurationError`` raises before the session is
        acquired and surfaces through normal error propagation.

        Args:
            context: The run context wrapper.
            agent: The (sandbox) agent that owned the session.
            error: The exception raised.
        """
        del context, agent, error


class CompositeRunHooks(RunHooks[Any]):
    """Fan-out composite over a sequence of ``RunHooks``.

    Internal helper used by the runner to layer framework-installed
    hooks (e.g. :class:`~troopai.adk.verbose.VerboseHooks`) on top of
    a user-provided hooks instance. All member hooks receive every
    callback in registration order; exceptions from one member do
    not short-circuit the others (they are re-raised after all have
    run).

    Not part of the public API. Users compose their own behaviour by
    overriding :class:`RunHooks` directly.
    """

    def __init__(self, members: list[RunHooks[Any]]) -> None:
        """Initialise the composite with an ordered list of hook instances.

        ``None`` entries in ``members`` are silently dropped so callers may
        pass optional hooks without pre-filtering.

        Args:
            members: Hook instances to fan out to, in invocation order.
                ``None`` values are discarded.
        """
        self.members = [m for m in members if m is not None]

    async def _fanout(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Invoke ``method`` on every member, logging all errors, raising the first.

        Every member is called regardless of whether an earlier one raised, so
        one broken hook cannot silently suppress the others. EACH member
        exception is logged at ERROR with the offending member + method (so a
        later member's failure never vanishes), then the first captured
        exception is re-raised so the caller still observes a failure.

        Only ``Exception`` is caught: ``KeyboardInterrupt`` / ``SystemExit`` /
        ``asyncio.CancelledError`` (all ``BaseException``, not ``Exception``)
        propagate immediately, so process teardown and cancellation are never
        delayed or swallowed by a hook.

        Args:
            method: Name of the ``RunHooks`` method to call on each member.
            *args: Positional arguments forwarded to the method.
            **kwargs: Keyword arguments forwarded to the method.
        """
        first_error: Exception | None = None
        for m in self.members:
            try:
                await getattr(m, method)(*args, **kwargs)
            except Exception as exc:
                logger.error(
                    "RunHooks member %s.%s raised: %s",
                    type(m).__name__,
                    method,
                    exc,
                )
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    @override
    async def on_task_start(self, context: Any, agent: Any, task: Any) -> None:
        await self._fanout("on_task_start", context, agent, task)

    @override
    async def on_task_end(self, context: Any, agent: Any, task: Any, *, output: Any) -> None:
        await self._fanout("on_task_end", context, agent, task, output=output)

    @override
    async def on_agent_start(self, context: Any, agent: Any) -> None:
        await self._fanout("on_agent_start", context, agent)

    @override
    async def on_agent_end(self, context: Any, agent: Any, result: Any) -> None:
        await self._fanout("on_agent_end", context, agent, result)

    @override
    async def on_llm_start(self, context: Any, agent: Any, messages: Any) -> None:
        await self._fanout("on_llm_start", context, agent, messages)

    @override
    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        await self._fanout("on_llm_end", context, agent, response)

    @override
    async def on_tool_start(self, context: Any, agent: Any, tool_name: str, tool_input: Any) -> None:
        await self._fanout("on_tool_start", context, agent, tool_name, tool_input)

    @override
    async def on_tool_end(self, context: Any, agent: Any, tool_name: str, tool_output: Any) -> None:
        await self._fanout("on_tool_end", context, agent, tool_name, tool_output)

    @override
    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        await self._fanout("on_handoff", context, from_agent, to_agent)

    @override
    async def on_input_guardrail_start(self, context: Any, agent: Any, guardrail_name: str) -> None:
        await self._fanout("on_input_guardrail_start", context, agent, guardrail_name)

    @override
    async def on_input_guardrail_end(self, context: Any, agent: Any, result: Any) -> None:
        await self._fanout("on_input_guardrail_end", context, agent, result)

    @override
    async def on_output_guardrail_start(self, context: Any, agent: Any, guardrail_name: str) -> None:
        await self._fanout("on_output_guardrail_start", context, agent, guardrail_name)

    @override
    async def on_output_guardrail_end(self, context: Any, agent: Any, result: Any) -> None:
        await self._fanout("on_output_guardrail_end", context, agent, result)

    @override
    async def on_tool_input_guardrail_start(
        self,
        context: Any,
        agent: Any,
        tool_name: str,
        guardrail_name: str,
    ) -> None:
        await self._fanout(
            "on_tool_input_guardrail_start",
            context,
            agent,
            tool_name,
            guardrail_name,
        )

    @override
    async def on_tool_input_guardrail_end(
        self,
        context: Any,
        agent: Any,
        tool_name: str,
        result: Any,
    ) -> None:
        await self._fanout(
            "on_tool_input_guardrail_end",
            context,
            agent,
            tool_name,
            result,
        )

    @override
    async def on_tool_output_guardrail_start(
        self,
        context: Any,
        agent: Any,
        tool_name: str,
        guardrail_name: str,
    ) -> None:
        await self._fanout(
            "on_tool_output_guardrail_start",
            context,
            agent,
            tool_name,
            guardrail_name,
        )

    @override
    async def on_tool_output_guardrail_end(
        self,
        context: Any,
        agent: Any,
        tool_name: str,
        result: Any,
    ) -> None:
        await self._fanout(
            "on_tool_output_guardrail_end",
            context,
            agent,
            tool_name,
            result,
        )

    @override
    async def on_skill_activated(self, context: Any, agent: Any, skill_name: str) -> None:
        await self._fanout("on_skill_activated", context, agent, skill_name)

    @override
    async def on_session_load(self, context: Any, session: Any, events: Any) -> None:
        await self._fanout("on_session_load", context, session, events)

    @override
    async def on_session_save(self, context: Any, session: Any, events: Any) -> None:
        await self._fanout("on_session_save", context, session, events)

    @override
    async def on_state_save(self, context: Any, session: Any, state: Any) -> None:
        await self._fanout("on_state_save", context, session, state)

    @override
    async def on_mcp_connect(self, context: Any, server_name: str) -> None:
        await self._fanout("on_mcp_connect", context, server_name)

    @override
    async def on_mcp_connected(self, context: Any, server_name: str) -> None:
        await self._fanout("on_mcp_connected", context, server_name)

    @override
    async def on_mcp_error(self, context: Any, server_name: str, error: BaseException) -> None:
        await self._fanout("on_mcp_error", context, server_name, error)

    @override
    async def on_sandbox_start(self, context: Any, agent: Any, session: Any) -> None:
        await self._fanout("on_sandbox_start", context, agent, session)

    @override
    async def on_sandbox_stop(self, context: Any, agent: Any, session: Any, usage: Any) -> None:
        await self._fanout("on_sandbox_stop", context, agent, session, usage)

    @override
    async def on_sandbox_exec_start(self, context: Any, agent: Any, command: str) -> None:
        await self._fanout("on_sandbox_exec_start", context, agent, command)

    @override
    async def on_sandbox_exec_end(self, context: Any, agent: Any, command: str, result: Any) -> None:
        await self._fanout("on_sandbox_exec_end", context, agent, command, result)

    @override
    async def on_sandbox_snapshot(self, context: Any, agent: Any, metadata: Any) -> None:
        await self._fanout("on_sandbox_snapshot", context, agent, metadata)

    @override
    async def on_sandbox_error(self, context: Any, agent: Any, error: BaseException) -> None:
        await self._fanout("on_sandbox_error", context, agent, error)


def compose_run_hooks(*members: RunHooks[Any] | None) -> RunHooks[Any]:
    """Compose multiple ``RunHooks`` into a single fan-out instance.

    ``None`` entries are silently dropped. Returns a no-op
    :class:`RunHooks` when every input is ``None``. Returns the sole
    member directly (no wrapping) when only one is provided — keeps
    the hot path cheap when verbose is disabled.

    Args:
        *members: Hook instances to compose. ``None`` values are ignored.

    Returns:
        A no-op :class:`RunHooks` when all inputs are ``None``, the single
        non-``None`` member unchanged when exactly one is provided, or a
        :class:`CompositeRunHooks` fan-out over all non-``None`` members.
    """
    filtered = [m for m in members if m is not None]
    if len(filtered) == 0:
        return RunHooks()
    if len(filtered) == 1:
        return filtered[0]
    return CompositeRunHooks(filtered)


class AgentHooks[TContext]:
    """Base class for per-agent lifecycle hooks.

    Attached to an individual ``Agent`` via ``Agent.hooks``. Fires
    alongside ``RunHooks`` at matching lifecycle points, but scoped to
    the specific agent instance — useful in multi-agent swarms where
    each agent wants its own observability, metrics, or side effects
    without cluttering the run-level hooks with per-agent conditionals.

    All methods are async and optional — override only what you need.
    A ``None`` return is ignored.

    Naming differs from ``RunHooks`` deliberately:

    - ``RunHooks.on_agent_start(ctx, agent)`` — run-level "agent X started"
    - ``AgentHooks.on_start(ctx, agent)`` — self-referential "I started"

    On handoffs, ``on_handoff`` fires on the **incoming** agent (the
    one being handed off TO), with ``source`` as the outgoing agent —
    this is the moment the new agent becomes active.

    AgentHooks intentionally excludes guardrail, skill, and session
    hooks — those are run-level concerns.

    Type Parameters:
        TContext: The type of the user-provided context.

    Example:
        class MetricsHooks(AgentHooks):
            async def on_start(self, context, agent):
                metrics.increment(f"agent.{agent.name}.starts")

            async def on_tool_end(self, context, agent, tool_name, tool_output):
                metrics.histogram(f"agent.{agent.name}.tool.{tool_name}", 1)

        agent = Agent(name="Router", system_prompt="...", hooks=MetricsHooks())
    """

    async def on_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
    ) -> None:
        """Called when this agent starts execution.

        Fires immediately after ``RunHooks.on_agent_start`` for the
        same turn.

        Args:
            context: The run context wrapper.
            agent: This agent (the one whose hooks fired).
        """
        del context, agent

    async def on_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        output: Any,
    ) -> None:
        """Called when this agent finishes execution.

        Fires immediately after ``RunHooks.on_agent_end`` for the same
        turn. ``output`` is the agent's final output (string or
        structured type when ``output_schema`` is set).

        Args:
            context: The run context wrapper.
            agent: This agent.
            output: The agent's final output.
        """
        del context, agent, output

    async def on_handoff(
        self,
        context: RunContext[TContext],
        agent: Agent,
        source: Agent,
    ) -> None:
        """Called when this agent is handed off TO from another agent.

        Fires on the **incoming** agent (``agent`` is this one), with
        ``source`` as the outgoing agent. This is the moment this
        agent becomes the active agent in the run loop.

        Args:
            context: The run context wrapper.
            agent: This agent (the incoming/new active agent).
            source: The outgoing agent that handed off.
        """
        del context, agent, source

    async def on_llm_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        messages: list[LLMInputContentItem],
    ) -> None:
        """Called before an LLM call made by this agent.

        Args:
            context: The run context wrapper.
            agent: This agent.
            messages: The messages being sent to the LLM (after all
                pre-call transformations — history processors, context
                management, ``call_model_input_filter``).
        """
        del context, agent, messages

    async def on_llm_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        response: Any,
    ) -> None:
        """Called after an LLM call made by this agent completes.

        Args:
            context: The run context wrapper.
            agent: This agent.
            response: The raw LLM response.
        """
        del context, agent, response

    async def on_tool_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Called before a tool executed by this agent runs.

        Args:
            context: The run context wrapper.
            agent: This agent.
            tool_name: The name of the tool.
            tool_input: The parsed input to the tool.
        """
        del context, agent, tool_name, tool_input

    async def on_tool_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool_name: str,
        tool_output: Any,
    ) -> None:
        """Called after a tool executed by this agent finishes.

        Args:
            context: The run context wrapper.
            agent: This agent.
            tool_name: The name of the tool.
            tool_output: The output from the tool.
        """
        del context, agent, tool_name, tool_output
