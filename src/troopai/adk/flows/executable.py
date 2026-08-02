"""FlowExecutable — wraps a :class:`Flow` as an :class:`Executable` for Graph nesting.

Lets a :class:`Flow` sit inside a :class:`Graph` node uniformly with
:class:`Agent` / :class:`Swarm` / callables. Mirrors the
:class:`AgentExecutable` / :class:`SwarmExecutable` precedent in
``graphs/adapters.py`` — single-class adapter forwarding to
:meth:`Runner.arun_flow` and flattening the :class:`FlowRunResult`
into a :class:`NodeResult` the graph loop can route.

The adapter does NOT mutate the Flow. The Flow's typed state is
preserved on :attr:`NodeResult.output` so downstream edge predicates
can inspect fields when routing — same composition pattern as
:class:`SwarmExecutable` preserving the full
:class:`SwarmRunResult` for `stop_reason`-based routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from troopai.adk.exceptions import UserError
from troopai.adk.orchestration.executable import (
    Executable,
    ExecutableInput,
    NodeResult,
)

if TYPE_CHECKING:
    from troopai.adk.flows.config import FlowConfig
    from troopai.adk.flows.flow import Flow
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext


@dataclass
class FlowExecutable[TContext](Executable[TContext]):
    """Wrap a :class:`Flow` so it can sit inside a graph node.

    Calling :meth:`invoke` delegates to
    :meth:`troopai.adk.run.runner.Runner.arun_flow` and converts the
    resulting :class:`FlowRunResult` into a :class:`NodeResult`. The
    Flow's typed state surfaces on :attr:`NodeResult.output` so
    downstream edges can route on it.

    Attributes:
        flow: The :class:`Flow` to run. Treated as configuration — NOT
            mutated by :meth:`invoke`.
        config: Optional :class:`FlowConfig` carrying ``max_steps``,
            error policy, etc. ``None`` uses the
            :class:`FlowConfig` defaults.
    """

    flow: Flow[Any]
    """The wrapped Flow. Kept as-is — no runtime mutation."""

    config: FlowConfig | None = None
    """Optional :class:`FlowConfig`. ``None`` uses defaults."""

    @override
    async def invoke(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> NodeResult[TContext]:
        """Run the wrapped Flow once and package its :class:`FlowRunResult`.

        Threads ``context.context`` (the user's ``TContext``) into the
        inner :meth:`Runner.arun_flow` call. The inner Flow's
        :attr:`run_context` is independent of ``context`` (the executor
        creates its own); usage / new_items are forwarded onto the
        :class:`NodeResult` for the graph loop to aggregate.

        Args:
            input: Upstream :class:`ExecutableInput`. The Flow itself
                does NOT consume the input messages — Flow step bodies
                build their own prompts from ``self.state``. Accepted
                for forward compatibility but currently not consumed
                by the Flow.
            context: Outer graph's :class:`RunContext`. ``context.context``
                threads through to the Flow's inner runs; ``context.usage``
                accumulates the Flow's cumulative usage delta.
            config: Outer :class:`RunConfig`. Accepted for
                :class:`~troopai.adk.orchestration.executable.Executable`
                interface conformance and deliberately unused — the
                wrapped Flow's own :class:`FlowConfig` governs the run,
                so inner agent runs inside step bodies do NOT inherit
                the outer graph's guardrails / hooks / model overrides.

        Returns:
            A :class:`NodeResult` whose ``output`` is the Flow's final
            typed state, ``new_items`` is empty (Flows produce items
            on inner agent runs that already record into the run
            context), ``usage`` is the cumulative LLM usage delta
            from the run, and ``metadata`` carries the flow id +
            status for observability.
        """
        del input, config
        from troopai.adk.run.runner import Runner

        result = await Runner.arun_flow(
            self.flow,
            config=self.config,
            context=context.context,
        )

        if result.status == "failed":
            raise UserError(
                f"FlowExecutable: nested Flow {result.flow_id!r} failed — {result.error or 'no error detail'}",
            )
        if result.status in ("halted_max_steps", "halted_max_tokens"):
            # A halted run did NOT complete — it hit a step / token cap before
            # terminating. Surfacing it as a normal NodeResult would route the
            # graph downstream on partial, mid-flow state as though the flow had
            # finished. Treat it as a node failure instead.
            raise UserError(
                f"FlowExecutable: nested Flow {result.flow_id!r} halted before completion "
                f"(status={result.status!r}); raise its FlowConfig.max_steps / max_total_tokens "
                f"or restructure the flow so it terminates.",
            )

        metadata: dict[str, Any] = {
            "adapter": "flow",
            "flow_id": result.flow_id,
            "status": result.status,
            "completed_steps": list(result.completed_steps),
        }
        if result.status == "deferred":
            # Flows-in-graphs HITL: deferral is the caller's responsibility.
            # The checkpoint is surfaced in metadata so graph edge predicates
            # can detect it and route to an approval node. There is no live-inject
            # channel — the developer resumes via Runner.arun_flow_from_checkpoint.
            metadata["checkpoint"] = result.checkpoint
            metadata["deferred_steps"] = list(result.deferred_steps)

        return NodeResult(
            output=result.final_state,
            new_items=list(result.new_items),
            usage=result.cumulative_usage,
            final_text=None,
            metadata=metadata,
        )
