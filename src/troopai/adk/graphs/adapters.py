"""Adapters — thin bridges that let ``Agent`` / ``Swarm`` / plain callables
act as :class:`Executable`\\ s inside a graph without inheriting from it.

Why thin wrappers and not a mixin on ``Agent`` / ``Swarm`` themselves?

- **"Agent = config" is load-bearing.** Adding an ``invoke`` method to
  ``Agent`` would re-introduce the ``agent.run()`` pattern the project
  explicitly forbids: ``Agent`` is configuration-only, ``Runner``
  executes.
- **Composition without subclass surgery.** Graphs compose three
  primitives today (``Agent``, ``Swarm``, ``Graph``) plus arbitrary
  callables. Each needs its own adapter; none needs to know about the
  others. The :class:`Executable` ABC is the only shared contract.

Three adapters in this file:

- :class:`AgentExecutable` — wraps an :class:`Agent`, forwards to
  :meth:`Runner.arun`. Produces a :class:`NodeResult` whose
  ``new_items`` are the agent run's produced items and whose ``usage``
  is the inner :class:`RunContext.usage` delta.

- :class:`SwarmExecutable` — wraps a :class:`Swarm`, forwards to
  :meth:`Runner.arun_swarm`. Same conversion.

- :class:`CallableExecutable` — wraps a plain Python callable for
  trivial transformation nodes (routing predicates, format adapters,
  deterministic post-processing). Zero LLM cost. Detects the
  callable's arity at wrap time so users can write the simplest
  signature that suits their task.

Design note on context threading. :meth:`Runner.arun` creates its own
inner :class:`RunContext` and does not mutate the caller's. The
adapter returns the inner context's ``usage`` on the
:class:`NodeResult`; the graph loop aggregates it into the outer
``RunContext`` by calling :meth:`GraphState.record`. Mirror of how
``run/swarm_loop.py`` threads usage upward through
``SwarmState.cumulative_usage``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, assert_never, override

from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.interrupt import (
    InterruptException,
    NestedAgentApproval,
    NestedAgentInterrupt,
    NestedAgentRejection,
    NestedAgentReply,
    NestedAgentResumeError,
)
from troopai.adk.orchestration.executable import (
    Executable,
    ExecutableInput,
    NodeResult,
)
from troopai.adk.run.state import RunState
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.run.run_result import RunResult


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _extract_text_from_content(
    content: list[LLMInputContentItem],
) -> str:
    """Best-effort ``str`` view of a Layer 1 content list.

    Concatenates text payloads from user-role messages. Rationale: a
    :class:`CallableExecutable` usually wants a plain ``str`` — the
    graph merge strategies (:class:`Merge`) produce either a ``str``
    wrapped into a single message or a list of Layer 1 items built
    from upstream ``final_text`` values. Either shape yields readable
    text here.

    Skips non-message items (tool calls, reasoning blocks) so a
    callable wired after an agent node doesn't see leaked provider
    structure. Returns an empty string if no text-shaped content
    exists.
    """
    texts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "message" or item_type is None:
                # Easy messages omit "type"; strict messages use "message".
                inner = item.get("content")
                if isinstance(inner, str):
                    if len(inner) > 0:
                        texts.append(inner)
                elif isinstance(inner, list):
                    for part in inner:
                        if isinstance(part, dict):
                            text_val = part.get("text")
                            if isinstance(text_val, str) and len(text_val) > 0:
                                texts.append(text_val)
    return "\n".join(texts)


def _content_to_user_prompt(
    content: list[LLMInputContentItem],
) -> str | list[LLMInputContentItem]:
    """Shape :class:`ExecutableInput.content` into a :data:`UserPrompt`.

    :meth:`Runner.arun` accepts either a raw string or a
    ``list[LLMInputContentItem]``. When the content is empty we return
    an empty string so the inner runner starts from a clean slate; the
    agent's system prompt still drives the first turn.
    """
    if len(content) == 0:
        return ""
    return content


# ---------------------------------------------------------------------
# Agent node result builder
# ---------------------------------------------------------------------


def _run_agent_node_result(
    agent_name: str,
    final_output: Any,
    new_items: list[RunItem],
    inner_usage: LLMUsage,
    last_agent_name: str | None,
) -> NodeResult[Any]:
    """Build a :class:`NodeResult` from the fields produced by an agent run.

    Single source of truth for converting an agent execution's terminal
    fields into a :class:`NodeResult`. Called by both
    :meth:`AgentExecutable.invoke` (non-streaming) and
    :meth:`AgentExecutable.stream_async` (streaming) so the two paths
    produce byte-identical results for the same inputs.

    Args:
        agent_name: Name of the wrapped agent.
        final_output: Terminal value from the agent run.
        new_items: Layer 3 items produced during the run.
        inner_usage: Cumulative :class:`LLMUsage` from the inner run context.
        last_agent_name: Name of the last active agent (after handoffs), or
            ``None`` when unavailable.

    Returns:
        A fully populated :class:`NodeResult`.
    """
    final_text = final_output if isinstance(final_output, str) else None
    return NodeResult(
        output=final_output,
        new_items=list(new_items),
        usage=inner_usage,
        final_text=final_text,
        metadata={
            "adapter": "agent",
            "agent_name": agent_name,
            "last_agent_name": last_agent_name,
        },
    )


# ---------------------------------------------------------------------
# AgentExecutable
# ---------------------------------------------------------------------


@dataclass
class AgentExecutable[TContext](Executable[TContext]):
    """Wrap an :class:`Agent` so it can sit inside a graph node.

    Calling :meth:`invoke` delegates to :meth:`Runner.arun` and
    converts the resulting :class:`RunResult` into a
    :class:`NodeResult`. The agent is NOT mutated — it stays pure
    configuration.

    Attributes:
        agent: The :class:`Agent` to run.
        max_turns: Optional per-node ``max_turns`` override. When
            ``None`` the agent's default from :meth:`Runner.arun`
            applies.
    """

    agent: Agent[TContext]
    """The wrapped agent. Kept as-is — no runtime mutation."""

    max_turns: int | None = None
    """Optional override. ``None`` uses :meth:`Runner.arun`'s default."""

    @override
    async def invoke(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> NodeResult[TContext]:
        """Run the wrapped agent once and package its ``RunResult``.

        Threads ``context.context`` (the user's TContext) to the inner
        runner. The inner :class:`RunContext` created by
        :meth:`Runner.arun` is independent of ``context``; its usage
        delta surfaces on :attr:`NodeResult.usage` for the graph loop
        to aggregate.

        Args:
            input: The :class:`ExecutableInput` envelope from the graph
                loop. ``input.content`` is normalised to a
                :data:`UserPrompt` and forwarded to the agent.
            context: The outer :class:`RunContext`. Its ``context``
                field (user's ``TContext``) is threaded to the inner
                runner.
            config: :class:`RunConfig` threaded from the graph run.

        Returns:
            A :class:`NodeResult` wrapping the agent's terminal output,
            produced items, and inner usage.

        Raises:
            InterruptException: When the agent run defers a tool for human
                approval (HITL) — either the top-level run returns a
                ``RunResult`` with ``requires_action`` True, or a nested
                sub-agent raises
                :class:`~troopai.adk.exceptions.AgentToolDeferral` — and the
                BSP loop has seeded the required side-channel metadata.
            RuntimeError: When the BSP loop has not seeded the required
                ``__interrupt_node_id__`` or ``__nested_agent_snapshots__``
                metadata keys before a deferral occurs, or when a top-level
                deferral is missing its resumable ``state``.
        """
        # Local import — Runner is in run/ which imports graphs elsewhere.
        from troopai.adk.run.runner import Runner

        user_prompt = _content_to_user_prompt(input.content)

        logger.debug(
            "AgentExecutable.invoke: agent=%s from_node=%s edge_label=%s",
            self.agent.name,
            input.from_node,
            input.edge_label,
        )

        try:
            if self.max_turns is None:
                result = await Runner.arun(
                    self.agent,
                    user_prompt,
                    context=context.context,
                    run_config=config,
                )
            else:
                result = await Runner.arun(
                    self.agent,
                    user_prompt,
                    context=context.context,
                    run_config=config,
                    max_turns=self.max_turns,
                )
        except AgentToolDeferral as exc:
            raise self._lift_deferral_to_interrupt(exc, input.metadata) from exc

        # Top-level HITL: ``Runner.arun`` does NOT raise — it returns a
        # ``RunResult`` with ``requires_action`` True and ``deferred_requests``
        # populated. Mirror ``stream_async``: synthesise the deferral and route
        # through the same lift so the node suspends instead of recording a
        # ``final_output=None`` result that the graph loop treats as complete.
        if result.deferred_requests is not None and len(result.deferred_requests.approvals) > 0:
            if result.state is None:
                raise RuntimeError(
                    f"AgentExecutable: non-streaming run for agent "
                    f"{self.agent.name!r} populated deferred_requests but not "
                    f"state — cannot lift to NestedAgentInterrupt."
                )
            deferring_agent_name = result.last_agent.name if result.last_agent is not None else self.agent.name
            synthesised = AgentToolDeferral(
                agent_name=deferring_agent_name,
                deferred_requests=result.deferred_requests,
                state=result.state,
            )
            raise self._lift_deferral_to_interrupt(synthesised, input.metadata)

        # RunResult.context is a required non-None field, so usage is always present.
        inner_usage = result.context.usage
        last_agent_name = result.last_agent.name if result.last_agent is not None else None

        return _run_agent_node_result(
            agent_name=self.agent.name,
            final_output=result.final_output,
            new_items=list(result.new_items),
            inner_usage=inner_usage,
            last_agent_name=last_agent_name,
        )

    def _lift_deferral_to_interrupt(
        self,
        exc: AgentToolDeferral,
        metadata: dict[str, Any],
    ) -> InterruptException:
        """Translate an ``AgentToolDeferral`` into the graph-level interrupt.

        Reads the reserved metadata keys ``__interrupt_node_id__`` and
        ``__nested_agent_snapshots__`` (seeded by the BSP loop), validates
        the side-channel is present, deposits the deferral's ``RunState``,
        logs, and returns the ``InterruptException`` for the caller to
        ``raise ... from exc``. Splitting the channel checks, deposit,
        logging, and exception build into a single ordered helper keeps
        ``AgentExecutable.invoke`` within the project's per-function length
        limit and enforces validate -> mutate -> log -> raise ordering.

        Raises:
            RuntimeError: When the BSP loop has not seeded either reserved
                metadata key, or when ``__nested_agent_snapshots__`` is
                present but not a ``dict``. These are programmer-error
                conditions — the bridge cannot lift a deferral without a
                place to deposit the sub-agent ``RunState``, and silently
                accepting the loss would defeat the HITL contract.
        """
        if "__interrupt_node_id__" not in metadata:
            raise RuntimeError(
                f"AgentExecutable: BSP loop did not seed "
                f"'__interrupt_node_id__' on ExecutableInput.metadata for "
                f"agent {self.agent.name!r}; cannot lift AgentToolDeferral "
                f"to NestedAgentInterrupt."
            )
        node_id = metadata["__interrupt_node_id__"]
        if not isinstance(node_id, str) or len(node_id) == 0:
            raise RuntimeError(
                f"AgentExecutable: '__interrupt_node_id__' on "
                f"ExecutableInput.metadata for agent {self.agent.name!r} "
                f"is not a non-empty string (got {type(node_id).__name__})."
            )
        if "__nested_agent_snapshots__" not in metadata:
            raise RuntimeError(
                f"AgentExecutable: BSP loop did not seed "
                f"'__nested_agent_snapshots__' side-channel for node "
                f"{node_id!r}; refusing to lift deferral without a place "
                f"to deposit the sub-agent RunState (resume would deadlock)."
            )
        snapshots = metadata["__nested_agent_snapshots__"]
        if not isinstance(snapshots, dict):
            raise TypeError(
                f"AgentExecutable: '__nested_agent_snapshots__' must be a "
                f"dict, got {type(snapshots).__name__} for node {node_id!r}."
            )

        # Validate the interrupt payload BEFORE mutating shared state or
        # emitting any log line. NestedAgentInterrupt.from_deferral raises
        # ValueError on a zero-approvals deferral.
        interrupt = NestedAgentInterrupt.from_deferral(node_id=node_id, deferral=exc)
        snapshots[node_id] = exc.state
        # Log both names because a handoff inside the inner agent makes them
        # differ — entry_agent is what the graph node wraps; deferring_agent
        # is who actually raised the deferral.
        logger.info(
            "AgentExecutable: node=%s entry_agent=%s deferring_agent=%s "
            "deferred %d tool call(s); lifting to NestedAgentInterrupt",
            node_id,
            self.agent.name,
            exc.agent_name,
            len(exc.deferred_requests.approvals),
        )
        return InterruptException(interrupt)

    async def _arun_streamed(
        self,
        user_prompt: str | list[LLMInputContentItem],
        context: RunContext[TContext],
        config: RunConfig,
    ) -> RunResultStreaming:
        """Call :meth:`Runner.arun` with ``stream=True`` honouring ``max_turns``.

        Shared with :meth:`stream_async`. Splitting on the ``max_turns is None``
        branch here keeps :meth:`stream_async` under the project's per-function
        length limit without duplicating the ``Runner.arun`` invocation. The
        ``max_turns`` default is centralized in :mod:`troopai.adk.run.config`;
        the adapter must NOT hardcode a fallback that would silently shadow it.

        Args:
            user_prompt: Normalised user input forwarded to :meth:`Runner.arun`.
            context: The outer :class:`RunContext`. Its ``context`` field is
                threaded to the inner runner.
            config: :class:`RunConfig` threaded from the graph run.

        Returns:
            A :class:`RunResultStreaming` ready for iteration via
            :meth:`stream_events`.
        """
        from troopai.adk.run.runner import Runner

        if self.max_turns is None:
            return await Runner.arun(
                self.agent,
                user_prompt,
                stream=True,
                context=context.context,
                run_config=config,
            )
        return await Runner.arun(
            self.agent,
            user_prompt,
            stream=True,
            context=context.context,
            run_config=config,
            max_turns=self.max_turns,
        )

    @override
    async def stream_async(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent interior events then yield a terminal result event.

        Forwards every inner event as
        ``{"type": "agent_event", "event": ev}``. A HITL deferral never
        raises out of ``stream_events()`` — the streaming runner absorbs
        the :class:`AgentToolDeferral` and stores
        ``deferred_requests`` + ``state`` on the result. This bridge
        inspects those fields AFTER iteration: when populated it
        synthesises the deferral and routes through
        :meth:`_lift_deferral_to_interrupt`, suppressing the terminal
        event. Otherwise the terminal
        ``{"type": "result", "result": NodeResult}`` fires, built from
        the same helper as :meth:`invoke` for byte-identical results.

        Args:
            input: The :class:`ExecutableInput` envelope from the graph
                loop.
            context: The outer :class:`RunContext`.
            config: :class:`RunConfig` threaded from the graph run.

        Raises:
            InterruptException: When the streaming run's deferred
                requests are populated after iteration completes,
                indicating a HITL tool deferral.
            RuntimeError: When ``deferred_requests`` is set but
                ``state`` is missing on the streaming result.
        """
        user_prompt = _content_to_user_prompt(input.content)

        logger.debug(
            "AgentExecutable.stream_async: agent=%s from_node=%s edge_label=%s",
            self.agent.name,
            input.from_node,
            input.edge_label,
        )

        streamed = await self._arun_streamed(user_prompt, context, config)

        async for ev in streamed.stream_events():
            yield {"type": "agent_event", "event": ev}

        if streamed.deferred_requests is not None and len(streamed.deferred_requests.approvals) > 0:
            if streamed.state is None:
                raise RuntimeError(
                    f"AgentExecutable: streaming run for agent "
                    f"{self.agent.name!r} populated deferred_requests but not "
                    f"state — cannot lift to NestedAgentInterrupt."
                )
            synthesised = AgentToolDeferral(
                agent_name=streamed.current_agent.name,
                deferred_requests=streamed.deferred_requests,
                state=streamed.state,
            )
            raise self._lift_deferral_to_interrupt(synthesised, input.metadata)

        # No deferral — emit terminal result built from the same helper as invoke.
        if streamed.context is not None:
            inner_usage = streamed.context.usage
        else:
            logger.warning(
                "AgentExecutable: inner RunContext is None for agent %s; recording zero usage.",
                self.agent.name,
            )
            inner_usage = LLMUsage()
        yield {
            "type": "result",
            "result": _run_agent_node_result(
                agent_name=self.agent.name,
                final_output=streamed.final_output,
                new_items=list(streamed.new_items),
                inner_usage=inner_usage,
                last_agent_name=streamed.current_agent.name,
            ),
        }

    def _apply_nested_reply(
        self,
        *,
        snapshot: RunState,
        reply: NestedAgentReply,
        node_id: str,
    ) -> None:
        """Validate ``reply`` against ``snapshot`` then apply each decision.

        Fail-fast on an unknown or duplicate ``tool_call_id`` — raises
        :class:`NestedAgentResumeError` BEFORE any mutation so the
        caller can retry against the unmutated snapshot. Membership is
        checked before duplicates so the error pinpoints the more
        actionable failure first. Closed-union dispatch on
        ``NestedAgentDecision``: a missing variant raises rather than
        silently no-op'ing.
        """
        pending_by_id = {c.tool_call_id: c for c in snapshot.deferred_tool_requests.approvals}
        seen_ids: set[str] = set()
        for decision in reply.decisions:
            if decision.tool_call_id not in pending_by_id:
                raise NestedAgentResumeError(
                    node_id=node_id,
                    detail=(
                        f"decision targets tool_call_id {decision.tool_call_id!r} "
                        f"not in snapshot deferred approvals "
                        f"(pending: {sorted(pending_by_id.keys())})"
                    ),
                )
            if decision.tool_call_id in seen_ids:
                raise NestedAgentResumeError(
                    node_id=node_id,
                    detail=f"duplicate decision for tool_call_id {decision.tool_call_id!r}",
                )
            seen_ids.add(decision.tool_call_id)

        for decision in reply.decisions:
            tool_call = pending_by_id[decision.tool_call_id]
            if isinstance(decision, NestedAgentApproval):
                snapshot.approve(tool_call, approver_id=decision.approver_id, reason=decision.reason)
            elif isinstance(decision, NestedAgentRejection):
                snapshot.reject(
                    tool_call,
                    message=decision.message,
                    approver_id=decision.approver_id,
                    reason=decision.reason,
                )
            else:
                # ``NestedAgentDecision = NestedAgentApproval | NestedAgentRejection``
                # is a closed union. ``assert_never`` gives mypy/pyright the
                # exhaustiveness check at type-check time — adding a third
                # variant without updating this dispatch site fails the gate,
                # matching the pattern in ``run/tools_executor.py``.
                assert_never(decision)

    def _handle_re_deferral(
        self,
        result: RunResult[Any],
        node_id: str,
        nested_agent_snapshots: dict[str, RunState],
    ) -> InterruptException | None:
        """Lift a re-deferred ``RunResult`` to a fresh ``NestedAgentInterrupt``.

        Returns ``None`` when the resumed run completed (no re-deferral) so
        the caller falls through to terminal packaging. When the resumed
        agent deferred again, deposits the fresh ``RunState`` keyed by the
        same ``node_id`` so the BSP loop's next checkpoint picks it up, and
        returns the ``InterruptException`` carrying a new
        ``NestedAgentInterrupt`` for the caller to ``raise``.

        Raises:
            RuntimeError: When ``result.requires_action`` is True but
                ``result.state`` or ``result.deferred_requests`` is missing —
                the bridge cannot lift a second interrupt without both.
        """
        if not result.requires_action:
            return None
        if result.state is None or result.deferred_requests is None:
            raise RuntimeError(
                f"AgentExecutable.resume_from_snapshot: node={node_id!r} "
                f"agent={self.agent.name!r} re-deferred but RunResult is missing "
                f"state or deferred_requests — cannot lift second "
                f"NestedAgentInterrupt."
            )
        deferring_agent_name = result.last_agent.name if result.last_agent is not None else self.agent.name
        re_defer = AgentToolDeferral(
            agent_name=deferring_agent_name,
            deferred_requests=result.deferred_requests,
            state=result.state,
        )
        nested_agent_snapshots[node_id] = re_defer.state
        logger.info(
            "AgentExecutable.resume_from_snapshot: node=%s agent=%s re-deferred "
            "%d tool call(s); lifting fresh NestedAgentInterrupt",
            node_id,
            deferring_agent_name,
            len(re_defer.deferred_requests.approvals),
        )
        return InterruptException(NestedAgentInterrupt.from_deferral(node_id=node_id, deferral=re_defer))

    def _log_decisions_applied(self, node_id: str, reply: NestedAgentReply) -> None:
        """Log how many decisions the resume just applied to the snapshot.

        Branches on the empty-decisions case so the log line distinguishes a
        deliberate no-op resume (caller chose to let every pending call
        re-defer) from a normal multi-decision resume — both flow through
        the same code path but mean different things operationally.
        """
        if len(reply.decisions) == 0:
            logger.info(
                "AgentExecutable.resume_from_snapshot: node=%s agent=%s applying "
                "0 decisions (no-op resume — sub-agent will re-defer all "
                "pending approvals)",
                node_id,
                self.agent.name,
            )
        else:
            logger.info(
                "AgentExecutable.resume_from_snapshot: node=%s agent=%s applied %d decision(s)",
                node_id,
                self.agent.name,
                len(reply.decisions),
            )

    async def _arun_from_snapshot(
        self,
        snapshot: RunState,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> RunResult[Any]:
        """Call :meth:`Runner.arun` from a ``RunState`` honouring ``max_turns``.

        Mirror of :meth:`_arun_streamed` for the non-streaming resume path —
        splitting on the ``max_turns is None`` branch here keeps the caller
        within the project's per-function length limit without duplicating
        the ``Runner.arun`` invocation.

        Args:
            snapshot: The paused :class:`~troopai.adk.run.state.RunState` to
                resume from (after decisions have been applied).
            context: The outer :class:`RunContext`. Its ``context`` field is
                threaded to the inner runner.
            config: :class:`RunConfig` threaded from the graph run.

        Returns:
            The terminal :class:`RunResult` from the resumed agent run.
        """
        from troopai.adk.run.runner import Runner

        if self.max_turns is None:
            return await Runner.arun(self.agent, snapshot, context=context.context, run_config=config)
        return await Runner.arun(
            self.agent,
            snapshot,
            context=context.context,
            run_config=config,
            max_turns=self.max_turns,
        )

    async def resume_from_snapshot(
        self,
        *,
        snapshot: RunState,
        reply: NestedAgentReply,
        context: RunContext[TContext],
        config: RunConfig,
        node_id: str,
        nested_agent_snapshots: dict[str, RunState],
    ) -> NodeResult[TContext]:
        """Resume the wrapped agent from a paused ``RunState`` with a typed reply.

        Validates ``reply`` against the snapshot's deferred approvals
        (fail-fast — :class:`NestedAgentResumeError` on unknown or duplicate
        ``tool_call_id``, snapshot untouched), applies each decision via
        ``RunState.approve`` / ``RunState.reject``, then re-enters the agent
        loop via ``Runner.arun(agent, state, ...)``. ``node_id`` threads
        through any raised :class:`NestedAgentResumeError` so the caller's
        error handler knows which ``GraphResume.replies`` entry was bad.

        When the resumed agent defers AGAIN, deposits the fresh snapshot in
        ``nested_agent_snapshots`` keyed by the same ``node_id`` (the dict
        the BSP loop owns on ``GraphState``) and lifts a fresh
        :class:`NestedAgentInterrupt` so the loop pauses again for the next
        round of human decisions.

        Args:
            snapshot: The paused :class:`~troopai.adk.run.state.RunState`
                from ``GraphState.nested_agent_snapshots[node_id]``.
            reply: The human-supplied :class:`NestedAgentReply` carrying
                approve/reject decisions for each deferred tool call.
            context: The outer :class:`RunContext`. Its ``context`` field
                is threaded to the inner runner.
            config: :class:`RunConfig` threaded from the graph run.
            node_id: Id of the graph node being resumed. Used for error
                messages and snapshot keying.
            nested_agent_snapshots: The mutable snapshot dict from
                ``GraphState.nested_agent_snapshots``. Updated in-place
                when the agent re-defers.

        Returns:
            A :class:`NodeResult` wrapping the agent's terminal output
            once it completes without further deferral.

        Raises:
            ValueError: When ``node_id`` is not a non-empty string.
            NestedAgentResumeError: When ``reply`` references an unknown
                or duplicate ``tool_call_id``.
            InterruptException: When the resumed agent defers again — a
                fresh :class:`NestedAgentInterrupt` is lifted and the
                caller should pause the run again.
            RuntimeError: When the re-deferred run is missing ``state``
                or ``deferred_requests``.
        """
        if not isinstance(node_id, str) or len(node_id) == 0:
            raise ValueError(
                f"AgentExecutable.resume_from_snapshot: node_id must be a "
                f"non-empty str, got {type(node_id).__name__}({node_id!r})"
            )

        self._apply_nested_reply(snapshot=snapshot, reply=reply, node_id=node_id)
        self._log_decisions_applied(node_id, reply)

        result = await self._arun_from_snapshot(snapshot, context, config)

        re_defer_exc = self._handle_re_deferral(result, node_id, nested_agent_snapshots)
        if re_defer_exc is not None:
            raise re_defer_exc

        # RunResult.context is a required non-None field, so usage is always present.
        inner_usage = result.context.usage
        last_agent_name = result.last_agent.name if result.last_agent is not None else None
        return _run_agent_node_result(
            agent_name=self.agent.name,
            final_output=result.final_output,
            new_items=list(result.new_items),
            inner_usage=inner_usage,
            last_agent_name=last_agent_name,
        )


# ---------------------------------------------------------------------
# SwarmExecutable
# ---------------------------------------------------------------------


@dataclass
class SwarmExecutable[TContext](Executable[TContext]):
    """Wrap a :class:`Swarm` so it can sit inside a graph node.

    Delegates to :meth:`Runner.arun_swarm`; the swarm's own
    termination conditions and budgets still apply. A
    :class:`SwarmRunResult` is produced and flattened into a
    :class:`NodeResult` — the full ``SwarmRunResult`` is preserved on
    :attr:`NodeResult.output` so downstream edge predicates can
    inspect fields like ``stop_reason`` when routing.

    Attributes:
        swarm: The :class:`Swarm` to run.
    """

    swarm: Swarm[TContext]
    """The wrapped swarm."""

    @override
    async def invoke(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> NodeResult[TContext]:
        """Run the wrapped swarm once and package its ``SwarmRunResult``.

        Args:
            input: The :class:`ExecutableInput` envelope from the graph
                loop. ``input.content`` is normalised to a
                :data:`UserPrompt` and forwarded to the swarm.
            context: The outer :class:`RunContext`. Its ``context``
                field is threaded to the inner runner.
            config: :class:`RunConfig` threaded from the graph run.

        Returns:
            A :class:`NodeResult` whose :attr:`~NodeResult.output` is
            the full ``SwarmRunResult``, allowing downstream edge
            predicates to inspect fields such as ``stop_reason``.
        """
        from troopai.adk.run.runner import Runner

        user_prompt = _content_to_user_prompt(input.content)

        logger.debug(
            "SwarmExecutable.invoke: swarm_entry=%s from_node=%s",
            self.swarm.entry.name,
            input.from_node,
        )

        result = await Runner.arun_swarm(
            self.swarm,
            user_prompt,
            context=context.context,
            run_config=config,
        )

        final_output = result.final_output
        final_text = final_output if isinstance(final_output, str) else None

        if result.context is not None:
            inner_usage = result.context.usage
        else:
            logger.warning(
                "SwarmExecutable: inner RunContext is None for swarm entry %s; recording zero usage.",
                self.swarm.entry.name,
            )
            inner_usage = LLMUsage()

        return NodeResult(
            output=result,
            new_items=list(result.new_items),
            usage=inner_usage,
            final_text=final_text,
            metadata={
                "adapter": "swarm",
                "entry_agent": self.swarm.entry.name,
                "total_turns": result.total_turns,
                "handoff_count": result.handoff_count,
            },
        )


# ---------------------------------------------------------------------
# CallableExecutable
# ---------------------------------------------------------------------


# Accepted user-callable shapes (all can be sync or async):
#
#   () -> Any
#   (text: str) -> Any
#   (text: str, context: RunContext[TContext]) -> Any
#   (input: ExecutableInput, context: RunContext[TContext]) -> Any
#
# Detected at wrap time via inspect.signature — a malformed callable
# raises ValueError up-front instead of crashing mid-run.
CallableNodeFn = Callable[..., Any]
"""Any user-supplied callable wrapped by :class:`CallableExecutable`."""


@dataclass
class CallableExecutable[TContext](Executable[TContext]):
    """Wrap a plain Python callable as a graph-composable node.

    Zero LLM cost — :attr:`NodeResult.usage` is an empty
    :class:`LLMUsage`. Use for routing predicates that emit a label,
    text formatters, deterministic post-processing between two agent
    nodes, or bridging to an external API.

    The callable's arity is detected at wrap time via
    :func:`inspect.signature`. Supported signatures:

    - ``() -> Any`` — pure producer (ignores upstream).
    - ``(text: str) -> Any`` — text transformer.
    - ``(text: str, context: RunContext) -> Any`` — text + context.
    - ``(input: ExecutableInput, context: RunContext) -> Any`` —
      full-control hook for advanced use.

    The return value becomes :attr:`NodeResult.output`. When it is a
    ``str`` the same value is mirrored on :attr:`NodeResult.final_text`
    so downstream merge strategies can read it without introspection.

    Attributes:
        fn: The user callable. Sync or async.
        passes_full_input: Computed at construction — ``True`` iff the
            detected signature takes the full :class:`ExecutableInput`
            (both parameters) instead of a plain ``str``. Internal;
            users don't set this.
        arity: Computed at construction — parameter count of ``fn``.
    """

    fn: CallableNodeFn
    """The user callable. Shape detected at wrap time."""

    passes_full_input: bool = False
    """``True`` when the detected signature takes :class:`ExecutableInput`."""

    arity: int = 0
    """Detected positional-parameter count of ``fn``."""

    def __post_init__(self) -> None:
        """Detect ``fn``'s shape and cache ``arity`` / ``passes_full_input``.

        Raises:
            ValueError: When ``fn`` is not callable or takes more than
                two positional parameters (ambiguous signature).
        """
        if not callable(self.fn):
            raise ValueError(f"CallableExecutable.fn must be callable, got {type(self.fn).__name__!r}")

        try:
            sig = inspect.signature(self.fn)
        except (TypeError, ValueError) as exc:
            # Builtins and C-extensions can refuse introspection. Treat
            # as single-arg text transformer — the most common case.
            logger.debug(
                "CallableExecutable: could not introspect %r (%s); assuming single-arg (text) signature.",
                self.fn,
                exc,
            )
            object.__setattr__(self, "arity", 1)
            object.__setattr__(self, "passes_full_input", False)
            return

        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        arity = len(positional)
        if arity > 2:
            raise ValueError(
                f"CallableExecutable.fn takes {arity} positional parameters; "
                "at most 2 are supported. Accepted shapes: () | (text) | "
                "(text, context) | (input, context)."
            )

        # Heuristic: when arity == 2 AND the first parameter is annotated
        # as ExecutableInput, pass the full envelope. Otherwise pass text.
        passes_full_input = False
        if arity == 2:
            first_annotation = positional[0].annotation
            if first_annotation is ExecutableInput or (
                isinstance(first_annotation, str) and "ExecutableInput" in first_annotation
            ):
                passes_full_input = True

        object.__setattr__(self, "arity", arity)
        object.__setattr__(self, "passes_full_input", passes_full_input)

    @override
    async def invoke(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> NodeResult[TContext]:
        """Call :attr:`fn` with the detected signature and wrap the result.

        Args:
            input: The :class:`ExecutableInput` envelope from the graph
                loop. Text is extracted via :func:`_extract_text_from_content`
                for single- and double-argument callables.
            context: The outer :class:`RunContext`. Passed through to
                the callable when its arity is 2 (or it accepts the full
                :class:`ExecutableInput`).
            config: :class:`RunConfig` threaded from the graph run.
                Not passed to the callable; present for ABC conformance.

        Returns:
            A zero-usage :class:`NodeResult` whose
            :attr:`~NodeResult.output` is the callable's return value.
            When the return value is a ``str`` it is also mirrored on
            :attr:`~NodeResult.final_text`.
        """
        # Local import to avoid circular dependency at module import time.
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        if self.arity == 0:
            raw = self.fn()
        elif self.arity == 1:
            text = _extract_text_from_content(input.content)
            raw = self.fn(text)
        elif self.passes_full_input:
            raw = self.fn(input, context)
        else:
            text = _extract_text_from_content(input.content)
            raw = self.fn(text, context)

        if inspect.isawaitable(raw):
            output = await raw
        else:
            output = raw

        logger.debug(
            "CallableExecutable.invoke: fn=%s from_node=%s -> output_type=%s",
            getattr(self.fn, "__name__", repr(self.fn)),
            input.from_node,
            type(output).__name__,
        )

        final_text = output if isinstance(output, str) else None

        return NodeResult(
            output=output,
            new_items=[],
            usage=LLMUsage(),
            final_text=final_text,
            metadata={
                "adapter": "callable",
                "fn_name": getattr(self.fn, "__name__", repr(self.fn)),
            },
        )


# ---------------------------------------------------------------------
# Convenience: auto-wrap
# ---------------------------------------------------------------------


def to_executable(obj: Any) -> Executable[Any]:
    """Coerce a graph-composable object into an :class:`Executable`.

    Type dispatch:

    =======================  ========================================
    Input                    Returned adapter
    =======================  ========================================
    :class:`Executable`      returned as-is (including
                             :class:`Graph`, nested)
    :class:`Agent`           :class:`AgentExecutable` wrapper
    :class:`Swarm`           :class:`SwarmExecutable` wrapper
    callable                 :class:`CallableExecutable` wrapper
    anything else            ``TypeError``
    =======================  ========================================

    Used by :meth:`GraphBuilder.node` so callers can write::

        builder.node("triage", triage_agent)  # Agent
        builder.node("research", research_swarm)  # Swarm
        builder.node("legal", legal_subgraph)  # Graph
        builder.node("reformat", reformat_fn)  # callable

    without thinking about adapter boilerplate.

    Args:
        obj: The object to coerce.

    Returns:
        An :class:`Executable`. Already-Executables are returned
        unchanged so nested graphs compose without an extra wrapper.

    Raises:
        TypeError: When ``obj`` is none of the supported shapes.
    """
    # Local imports — graphs/adapters.py is loaded early in the
    # graphs package; importing Agent/Swarm at module top would pull
    # the whole swarms and agents modules into every graph import.
    from troopai.adk.agents.agent import Agent
    from troopai.adk.swarms.swarm import Swarm

    if isinstance(obj, Executable):
        return obj
    # A2AAgent is checked BEFORE the general ``Agent`` branch even
    # though A2AAgent is a ``BaseAgent`` (not an ``Agent``) — guarding
    # explicitly keeps the dispatch order stable if A2AAgent is ever
    # changed to extend ``Agent``. The import is wrapped in try /
    # except so the optional ``[a2a]`` extra not being installed
    # doesn't crash the graphs package.
    try:
        from troopai.adk.a2a.a2a_agent import A2AAgent
        from troopai.adk.a2a.adapters import A2AExecutableAdapter
    except ImportError as exc:
        if getattr(exc, "name", None) != "a2a":
            raise
        # Optional extra not installed — skip A2A dispatch and fall
        # through to the local-Agent / Swarm / callable branches.
    else:
        if isinstance(obj, A2AAgent):
            return A2AExecutableAdapter(agent=obj)
    if isinstance(obj, Agent):
        return AgentExecutable(agent=obj)
    if isinstance(obj, Swarm):
        return SwarmExecutable(swarm=obj)
    # Flow adapter — lazy import keeps the flows module out of the
    # graphs eager-import path (flows imports run/runner.py which
    # imports graphs/adapters.py — circular at module load time).
    from troopai.adk.flows.executable import FlowExecutable
    from troopai.adk.flows.flow import Flow

    if isinstance(obj, Flow):
        return FlowExecutable(flow=obj)
    if callable(obj):
        return CallableExecutable(fn=obj)
    raise TypeError(
        f"Cannot wrap {type(obj).__name__!r} as an Executable. Accepted: Executable, Agent, A2AAgent, Swarm, or callable."
    )


__all__ = [
    "AgentExecutable",
    "CallableExecutable",
    "CallableNodeFn",
    "SwarmExecutable",
    "to_executable",
]
