"""Result types for agent execution.

This module provides the RunResult class which represents the outcome of
an agent run, including support for Human-in-the-Loop (HITL) interruptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from troopai.adk.agents.agent_guardrails import AgentGuardrailResults
from troopai.adk.types.run.guardrail_audit import GuardrailAuditRecord

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.state import RunState
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.swarms.yield_signal import SwarmYieldSignal
    from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.sandbox.usage import SandboxUsage

T = TypeVar("T")


def _make_default_context() -> Any:
    """Factory for the default ``RunContext`` used when a caller omits one.

    Lazy-imports ``RunContext`` to avoid a circular import with
    ``troopai.adk.run.context`` (which pulls in pieces of this module).
    """
    from troopai.adk.run.context import RunContext

    return RunContext(context=None)


@dataclass
class RunResult[T]:
    """Result of a completed (or interrupted) agent run.

    This class represents the outcome of running an agent. It contains the
    final output (if the run completed), all messages generated during execution,
    and supports HITL interruptions via deferred_requests.

    Attributes:
        final_output: The final output from the agent, or None if interrupted.
        user_prompt: The original user prompt passed to the run.
        new_items: New items generated during this run (assistant messages, tool results).
        context: The run context with usage tracking.
        last_agent: The last agent that was active.
        recovered: Whether an error handler produced the final output
            after the run raised (persistence is skipped on that path).
        deferred_requests: Tools captured for approval/external execution.
        state: Serializable state for resuming interrupted runs.

    Example:
        # Normal completion
        result = await Runner.arun(agent, "Hello!")
        logger.info(result.final_output)

        # Interrupted for approval
        result = await Runner.arun(agent, "Delete user 123")
        if result.requires_action:
            for req in list(result.deferred_requests.approvals):
                if await confirm(f"Approve {req.tool_name}?"):
                    result.state.approve(req)
                else:
                    result.state.reject(req, "Denied")
            result = await Runner.arun(agent, result.state)
    """

    final_output: Any
    """The final output from the agent, or None if interrupted for approval."""

    user_prompt: UserPrompt
    """The original user prompt passed to the run."""

    new_items: list[RunItem] = field(default_factory=list)
    """Layer 3 RunItems generated during this run (messages, tool calls, results)."""

    context: RunContext[T] = field(default_factory=lambda: _make_default_context())
    """The run context with usage tracking."""

    last_agent: Agent[T] | None = None
    """The last agent that was active."""

    recovered: bool = False
    """``True`` when an error handler produced ``final_output`` after the run
    raised. Recovered runs skip session/memory persistence, and ``new_items``
    reflects only the partial progress made before the error."""

    deferred_requests: DeferredToolRequests | None = None
    """Tools captured for approval/external execution. None if run completed."""

    state: RunState | None = None
    """Serializable state for resuming interrupted runs.

    Populated when the run is interrupted for HITL approval. Contains the
    full conversation history, deferred requests, and context needed to
    resume execution after the user approves or rejects the deferred tools.

    Example:
        result = await Runner.arun(agent, "Delete user 123")
        if result.requires_action:
            for req in list(result.deferred_requests.approvals):
                if await confirm(f"Approve {req.tool_name}?"):
                    result.state.approve(req)
                else:
                    result.state.reject(req, "Denied")
            result = await Runner.arun(agent, result.state)
    """

    guardrail_results: AgentGuardrailResults = field(default_factory=AgentGuardrailResults)
    """Per-phase agent-level guardrail audit trail for this run.

    A single :class:`~troopai.adk.agents.agent_guardrails.AgentGuardrailResults`
    config object holds:

    - ``guardrail_results.input``: results from every input guardrail
      that ran (blocking + parallel).
    - ``guardrail_results.output``: results from every output guardrail
      that ran.

    Populated after guardrail execution completes. The slots are
    immutable tuples once set — use for auditing, debugging, and
    inspecting guardrail decisions.
    """

    guardrail_audit: tuple[GuardrailAuditRecord, ...] = field(default_factory=tuple)
    """Per-action guardrail audit trail across every level (agent, tool, flow).

    Each record captures one verdict as hashes — never raw payloads. Empty when
    no guardrail ran. Drained from the run context once the run completes.
    """

    swarm_yield: SwarmYieldSignal | None = None
    """Set only by the swarm driver when an agent turn yielded control.

    This field is populated by ``run_agent_loop`` when the current
    agent's turn ended with a ``swarm_done`` or ``transfer_to_<name>``
    tool call AND the turn was dispatched with a non-empty
    ``swarm_tool_names`` set. Non-swarm callers never observe a
    non-None value here — it defaults to ``None`` for every
    ``Runner.arun()`` path.

    The swarm driver (``run_swarm_loop``) inspects this field after
    each inner turn to decide whether to advance the current agent
    (``SwarmHandoff``) or terminate the run (``SwarmDone``). See
    ``troopai.adk.run.swarm_loop``.
    """

    sandbox_usage: SandboxUsage | None = None
    """Aggregate sandbox resource + cost usage for the run (``None`` when
    no sandbox session ran). Populated by the Runner from the sandbox
    lifecycle handle; ``billed_cost_usd`` is filled at teardown when
    live-cost capture is enabled."""

    @property
    def requires_action(self) -> bool:
        """True if human approval or external action is needed.

        Returns:
            True if there are deferred requests pending, False otherwise.
        """
        return self.deferred_requests is not None and (
            len(self.deferred_requests.approvals) > 0 or len(self.deferred_requests.calls) > 0
        )

    @property
    def interruptions(self) -> list[DeferredToolCall]:
        """Tool calls awaiting human approval, as a flat list.

        Convenience property matching OpenAI's ``result.interruptions``
        pattern.  Returns an empty list when the run completed normally.
        """
        if self.deferred_requests is None:
            return []
        return list(self.deferred_requests.approvals)

    @property
    def last_response_id(self) -> str | None:
        """The ``response_id`` of the most recent LLM response in this run.

        Walks ``new_items`` in reverse and returns the ``id`` of the
        latest :class:`MessageOutputItem` — the field is populated from
        :attr:`LLMResponse.response_id` in
        :meth:`ItemHelpers.response_to_run_items`. Returns ``None`` when
        no message output item is present (e.g. a purely tool-call turn
        that deferred or was interrupted before any text output landed).

        Useful for provider-native response chaining (OpenAI Responses
        API, Anthropic prompt caching audit) and for correlating a run
        with provider-side logs.
        """
        from troopai.adk.types.items.items import MessageOutputItem

        for item in reversed(self.new_items):
            if isinstance(item, MessageOutputItem) and item.id is not None:
                return item.id
        return None

    def release_agents(self, *, release_new_items: bool = True) -> None:
        """Release strong references to agents and (optionally) run items.

        Long-running processes holding many completed ``RunResult``
        instances can pin significant agent graphs in memory —
        system prompts, tool closures, handoff targets, etc. This
        method nulls out ``last_agent`` and (by default) clears
        ``new_items`` so those objects become garbage-collectable
        while the cheap metadata on the result (final_output, usage)
        stays around.

        Args:
            release_new_items: If ``True`` (default), also clears
                ``new_items``. Pass ``False`` to keep the conversation
                history but still drop the agent reference.

        Note:
            The result is mutated in place. After calling this method,
            :meth:`to_input_list` and :attr:`last_response_id` will
            return empty/None values when ``release_new_items=True``.
        """
        self.last_agent = None
        if release_new_items:
            self.new_items = []

    def to_input_list(self) -> list[Any]:
        """Convert to input list for continued conversation.

        Converts Layer 3 RunItems to Layer 1 params for passing as
        input to a subsequent Runner.arun() call.

        Returns:
            A list of content items that can be passed as input to continue,
            keeping reasoning blocks, tool calls, and tool outputs in order.
            This is the shape required for multi-turn tool-use flows against
            Anthropic and OpenAI.
        """
        from troopai.adk.types.items.items import ItemHelpers

        user_items: list[Any]
        if isinstance(self.user_prompt, str):
            user_items = [{"role": "user", "content": self.user_prompt}]
        else:
            user_items = list(self.user_prompt)
        return user_items + ItemHelpers.run_items_to_params(self.new_items)

    def final_output_as(self, output_type: type[T]) -> T:
        """Cast the final output to the expected type.

        This is useful when using structured output with output_type
        on the agent.

        Args:
            output_type: The expected type of the final output.

        Returns:
            The final output cast to the specified type.

        Raises:
            TypeError: If the output cannot be cast to the specified type.
            ValueError: If the run was interrupted and has no final output.
        """
        if self.final_output is None:
            raise ValueError("Run was interrupted and has no final output. Handle deferred_requests first.")
        if isinstance(self.final_output, output_type):
            return self.final_output
        raise TypeError(f"Final output is {type(self.final_output).__name__}, expected {output_type.__name__}")
