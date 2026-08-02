"""Deep-resume helpers for the swarm driver loop.

Two helpers dispatch on the kind of parked interrupt:

- :func:`run_resumed_nested_turn` — nested-agent-defer path. Wraps the
  parked member in an :class:`AgentExecutable` and calls
  :meth:`AgentExecutable.resume_from_snapshot`, then converts the
  returned :class:`NodeResult` into a :class:`RunResult`-shaped value
  that :func:`run_swarm_loop`'s per-turn post-processing block can
  consume uniformly.
- :func:`run_resumed_hitl_turn` — pure-HITL path. Seeds the caller's
  reply onto the run context's HITL resume slot (consumed via
  :func:`troopai.adk.swarms.interrupt.request_human_input_in_swarm`)
  and re-fires the member via :func:`run_agent_loop` exactly as a
  fresh turn would.

Both helpers pop the parked entries from :class:`SwarmState` on
entry and rely on the swarm loop's existing
:class:`InterruptException` / :class:`AgentToolDeferral` handlers to
re-park on re-deferral.

The split into a sibling module (rather than private underscored
functions inside :mod:`troopai.adk.run.swarm_loop`) keeps the helpers
unit-testable in isolation — :func:`run_swarm_loop` itself remains
hard to unit-test because of its breadth, but the resume primitives
have well-defined inputs and outputs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.graphs.adapters import AgentExecutable
from troopai.adk.graphs.interrupt import NestedAgentReply, NestedAgentResumeError
from troopai.adk.run.loop import run_agent_loop
from troopai.adk.types.run.run_result import RunResult

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.swarms.interrupt import SwarmResume
    from troopai.adk.swarms.state import SwarmState
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.input import LLMInputContentItem


logger = logging.getLogger(__name__)


async def run_resumed_nested_turn(
    *,
    member: Agent[Any],
    swarm_resume: SwarmResume,
    state: SwarmState[Any],
    ctx_wrapper: RunContext[Any],
    config: RunConfig,
) -> RunResult[Any]:
    """Resume a swarm member parked on a nested-agent-defer interrupt.

    Validates the caller-supplied reply against the parked snapshot,
    pops both parked entries from ``state``, wraps the member in an
    :class:`AgentExecutable`, calls
    :meth:`AgentExecutable.resume_from_snapshot`, folds the resumed
    run's usage into ``ctx_wrapper`` (load-bearing — see the
    discussion in the design spec) and synthesizes a :class:`RunResult`
    for the swarm loop's step-8 history-accumulation block.

    On re-deferral, :meth:`AgentExecutable.resume_from_snapshot` deposits
    the fresh snapshot into ``state.nested_agent_snapshots`` and raises
    :class:`InterruptException`. This helper does NOT catch it — re-park
    is handled by :func:`run_swarm_loop`'s existing
    ``except InterruptException`` clause.

    Args:
        member: The parked swarm member to resume.
        swarm_resume: The caller-supplied resume payload; the
            ``replies[member.name]`` entry MUST be a
            :class:`NestedAgentReply`.
        state: The live :class:`SwarmState`. Modified in place:
            ``pending_interrupts[member.name]`` and
            ``nested_agent_snapshots[member.name]`` are popped on
            entry. On re-deferral, the snapshot dict is repopulated
            by ``resume_from_snapshot`` before the exception propagates.
        ctx_wrapper: The shared run context. ``ctx_wrapper.usage`` is
            advanced by the resumed run's token consumption so the
            swarm loop's per-member delta computation captures it.
        config: The active :class:`RunConfig`.

    Returns:
        A :class:`RunResult` with the resumed turn's items, the agent's
        final output, ``swarm_yield=None`` (the resumed run does not
        receive swarm-mode tools), and ``last_agent=member``. The
        swarm loop's step-9 dispatch routes via ``policy.select_next``
        on subsequent turns.

    Raises:
        ValueError: When ``swarm_resume.replies`` has no entry for
            ``member.name`` or the entry is not a
            :class:`NestedAgentReply`. The parked snapshot is left
            untouched so a corrected retry succeeds.
        NestedAgentResumeError: Propagated from
            :meth:`AgentExecutable.resume_from_snapshot` on invalid
            ``tool_call_id`` in the reply.
        InterruptException: On re-deferral (caught upstream by
            :func:`run_swarm_loop`).
    """
    if member.name not in swarm_resume.replies:
        raise ValueError(
            f"swarm deep resume: no reply provided for parked member "
            f"{member.name!r} (nested-agent-defer path requires a "
            f"NestedAgentReply entry in SwarmResume.replies)"
        )
    reply = swarm_resume.replies[member.name]
    if not isinstance(reply, NestedAgentReply):
        raise ValueError(
            f"swarm deep resume: reply for member {member.name!r} must "
            f"be NestedAgentReply for the nested-agent-defer path, got "
            f"{type(reply).__name__}"
        )

    parked_interrupt = state.pending_interrupts.get(member.name)
    snapshot = state.nested_agent_snapshots.pop(member.name)
    state.pending_interrupts.pop(member.name, None)

    logger.info(
        "swarm_resume: nested-defer path for member=%s; applying %d decision(s)",
        member.name,
        len(reply.decisions),
    )

    executable = AgentExecutable(agent=member)
    try:
        node_result = await executable.resume_from_snapshot(
            snapshot=snapshot,
            reply=reply,
            context=ctx_wrapper,
            config=config,
            node_id=member.name,
            nested_agent_snapshots=state.nested_agent_snapshots,
        )
        # Inner resume returned successfully — only now should the
        # resume counter advance. Bumping before the call would
        # permanently inflate the counter on a retriable
        # NestedAgentResumeError validation failure (the except
        # clause below restores the snapshot for a corrected retry).
        state.resume_counts[member.name] = state.resume_counts.get(member.name, 0) + 1
    except NestedAgentResumeError:
        # The reply failed validation against the snapshot (bad
        # tool_call_id). resume_from_snapshot never ran the inner
        # agent and never re-deposited, so the parked state was left
        # in nothing's hands. Restore so a corrected retry succeeds.
        state.nested_agent_snapshots[member.name] = snapshot
        if parked_interrupt is not None:
            state.pending_interrupts[member.name] = parked_interrupt
        raise

    ctx_wrapper.usage = ctx_wrapper.usage + node_result.usage

    logger.info(
        "swarm_resume: nested-defer completed for member=%s; %d new item(s)",
        member.name,
        len(node_result.new_items),
    )

    return RunResult(
        final_output=node_result.output,
        user_prompt="",
        new_items=list(node_result.new_items),
        context=ctx_wrapper,
        last_agent=member,
        swarm_yield=None,
    )


async def run_resumed_hitl_turn(
    *,
    member: Agent[Any],
    swarm_resume: SwarmResume,
    state: SwarmState[Any],
    ctx_wrapper: RunContext[Any],
    config: RunConfig,
    hooks: RunHooks[Any],
    user_prompt: UserPrompt,
    is_first_turn: bool,
    turn_messages: list[LLMInputContentItem],
    extra_tools: list[FunctionTool] | None,
    swarm_tool_names: set[str] | None,
    max_turns: int,
) -> RunResult[Any]:
    """Resume a swarm member parked on a pure-HITL interrupt.

    Seeds the caller-supplied reply onto ``ctx_wrapper`` (consumed by
    :func:`request_human_input_in_swarm` inside the member's tool
    body) and re-fires the member via :func:`run_agent_loop` exactly
    as a fresh turn would. Pops the parked interrupt from ``state``
    on entry.

    Missing-reply detection uses key presence in
    ``swarm_resume.replies`` so an explicit ``None`` reply (abstain)
    is distinguishable from "no reply supplied". On re-park (the
    member's tool raises again because the reply was insufficient or
    a multi-stage HITL fired), the existing
    ``InterruptException`` handler in :func:`run_swarm_loop`
    re-captures the fresh interrupt.

    Args:
        member: The parked swarm member to resume.
        swarm_resume: The caller-supplied resume payload.
        state: The live :class:`SwarmState`. ``pending_interrupts[
            member.name]`` is popped on entry.
        ctx_wrapper: The shared run context. The reply is seeded onto
            its private slot and consumed during the member's run.
        config: The active :class:`RunConfig`.
        hooks: Forwarded to :func:`run_agent_loop`.
        user_prompt: The original user prompt (passed through for
            first-turn semantics if applicable).
        is_first_turn: Pass-through flag controlling the inner
            runner's first-turn message construction.
        turn_messages: The prepared initial messages for the resumed
            turn.
        extra_tools: Policy-injected tools for this turn.
        swarm_tool_names: Names of the swarm tools the inner loop
            should recognize as yield signals.
        max_turns: Inner per-member turn cap.

    Returns:
        The :class:`RunResult` produced by the inner agent run.

    Raises:
        ValueError: When ``swarm_resume.replies`` does not have a key
            for ``member.name`` (the parked interrupt is left intact
            so a corrected retry succeeds).
    """
    if member.name not in swarm_resume.replies:
        raise ValueError(
            f"swarm deep resume: no reply provided for parked member "
            f"{member.name!r} (pure-HITL path; expected an entry in "
            f"SwarmResume.replies — pass None for an abstain answer)"
        )

    reply = swarm_resume.replies[member.name]
    state.pending_interrupts.pop(member.name, None)

    logger.info(
        "swarm_resume: HITL-pure path for member=%s; seeding reply onto run context",
        member.name,
    )

    ctx_wrapper.seed_swarm_resume_reply(reply)
    try:
        result = await run_agent_loop(
            agent=member,
            user_prompt=user_prompt if is_first_turn else "",
            context=ctx_wrapper,
            ctx_wrapper=ctx_wrapper,
            hooks=hooks,
            max_turns=max_turns,
            config=config,
            initial_messages=turn_messages,
            initial_new_items=None,
            extra_tools=extra_tools if extra_tools is not None and len(extra_tools) > 0 else None,
            swarm_tool_names=swarm_tool_names if swarm_tool_names is not None and len(swarm_tool_names) > 0 else None,
        )
        # Inner run completed successfully — advance the resume
        # counter only after the re-fired member returned without
        # re-parking. An ``InterruptException`` from a re-park inside
        # the inner loop would otherwise permanently inflate the
        # counter for a retriable suspend.
        state.resume_counts[member.name] = state.resume_counts.get(member.name, 0) + 1
    finally:
        # Defensive: clear the slot even if the tool didn't consume
        # the reply (e.g. the LLM took a different path). Prevents
        # leaking the reply into a subsequent turn.
        ctx_wrapper.clear_swarm_resume_reply()

    logger.info(
        "swarm_resume: HITL-pure completed for member=%s",
        member.name,
    )

    return result


__all__ = ["run_resumed_hitl_turn", "run_resumed_nested_turn"]
