"""Production-overview Flow example — Agents + Tasks + TaskGroup + error routing.

Demonstrates a realistic Flow you might deploy in production:

1. **Intake**: ``@flow_start`` captures the user's question into typed state.
2. **Classification**: a classifier ``Agent`` produces a route label.
3. **Routing**: ``@flow_router`` dispatches to one of three specialist branches.
4. **Specialist branches**: each ``@flow_listen("<label>")`` wraps a
   specialist agent in a :class:`Task` for declarative budget /
   guardrail / metadata control.
5. **Parallel sub-tasks**: the "research" branch uses a
   :class:`TaskGroup` to fan-out two background lookups concurrently.
6. **Synthesis**: a fan-in via ``@flow_listen(branch_a | branch_b | branch_c)``
   collects whichever branch ran.
7. **Error routing**: ``error_policy="route_to_error_handler"`` plus
   ``@flow_listen("__error__")`` recovers from a step failure.
8. **Streaming**: the run is driven through ``arun_flow_streamed`` so
   the operator dashboard sees ``flow.step_start`` / ``flow.step_end`` /
   ``flow.route_evaluated`` events live.
9. **Usage tracking**: step bodies share ``self.run_context`` so
   :attr:`FlowRunResult.cumulative_usage` reflects the total LLM spend.

Production patterns illustrated:

- **No hidden behavior** — every cross-step data flow is an explicit
  ``self.state.<field> = ...`` assignment. The framework never
  injects, infers, or auto-routes.
- **Cost-conservative defaults** — ``FlowConfig(max_steps=20)`` caps
  runaway step counts.
- **Composition with existing primitives** — Flow steps invoke
  ``Runner.arun`` (Agent), ``Runner.arun_task`` (Task), and
  ``Runner.arun_task_group`` (TaskGroup) inline.
- **Recoverable failure** — a single step exception routes to an
  error handler rather than aborting the whole flow.

Requires an LLM API key (``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``)
loaded via ``load_dotenv()`` from a project ``.env`` file. Without
one, the underlying provider call raises an authentication error.

Run::

    python examples/flows/research_with_agents.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import sys

from pydantic import BaseModel

from troopai.adk import (
    Agent,
    Flow,
    FlowConfig,
    Runner,
    Task,
    TaskGroup,
    flow_listen,
    flow_router,
    flow_start,
)
from troopai.adk.flows.events import (
    FlowRouteEvaluatedEvent,
    FlowStepEndEvent,
    FlowStepErrorEvent,
    FlowStepStartEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Typed shared state — mutated by every step, surfaced on the result.
# ---------------------------------------------------------------------


class TriageState(BaseModel):
    """Typed state for the production triage flow.

    The state IS the audit trail: every step's contribution is written
    here, the final state is what the consumer reads.
    """

    question: str = ""
    """The user's input question. Seeded by :meth:`TriageFlow.receive`."""

    classification: str = ""
    """Classifier verdict. One of ``'math'`` / ``'coding'`` / ``'research'``."""

    math_answer: str = ""
    """Populated when the math branch fires."""

    coding_answer: str = ""
    """Populated when the coding branch fires."""

    research_findings: list[str] = []
    """Populated by the research branch's parallel TaskGroup."""

    research_synthesis: str = ""
    """Populated by the research branch after the TaskGroup completes."""

    final_answer: str = ""
    """Synthesized final answer surfaced to the caller."""

    error_log: list[str] = []
    """Populated by the ``__error__`` handler on recoverable failures."""


# ---------------------------------------------------------------------
# Specialist agents — pure configuration. Runner executes them.
# ---------------------------------------------------------------------


classifier_agent = Agent(
    name="classifier",
    system_prompt=(
        "Classify the user's question into exactly one of: "
        "'math', 'coding', 'research'. Output ONLY the single label "
        "with no other text."
    ),
)

math_agent = Agent(
    name="math",
    system_prompt=(
        "You are a math tutor. Answer the user's math question concisely "
        "(2-3 sentences) and show the key reasoning step."
    ),
)

coding_agent = Agent(
    name="coding",
    system_prompt=(
        "You are a senior software engineer. Answer with a concrete code snippet and a 1-2 sentence explanation."
    ),
)

# Two research sub-specialists run concurrently inside the research branch.
research_facts_agent = Agent(
    name="research_facts",
    system_prompt="List 2-3 verified facts relevant to the user's question. One per line.",
)

research_context_agent = Agent(
    name="research_context",
    system_prompt="Provide 2-3 sentences of historical or contextual background.",
)

research_synth_agent = Agent(
    name="research_synth",
    system_prompt=(
        "Synthesize the provided facts and context into a single 3-4 sentence answer. Be neutral and factual."
    ),
)


# ---------------------------------------------------------------------
# The Flow — orchestration topology.
# ---------------------------------------------------------------------


class TriageFlow(Flow[TriageState]):
    """Production triage flow with classifier → router → specialist branches."""

    @flow_start
    async def receive(self) -> None:
        """Entry. Treats ``self.state.question`` as already seeded by caller."""
        if len(self.state.question) == 0:
            self.state.question = "Explain the time complexity of mergesort."
        logger.info("receive: question=%s", self.state.question)

    @flow_listen(receive)
    async def classify(self) -> None:
        """Run the classifier; write a normalized label into state."""
        result = await Runner.arun(
            classifier_agent,
            self.state.question,
            context=self.run_context.context,
        )
        verdict = str(result.final_output).strip().lower()
        if verdict not in {"math", "coding", "research"}:
            logger.warning("classifier returned %r; falling back to 'research'", verdict)
            verdict = "research"
        self.state.classification = verdict
        logger.info("classify: verdict=%s", verdict)

    @flow_router(classify)
    async def route(self) -> str:
        """Return the classification as the route label.

        The framework dispatches to the matching ``@flow_listen("...")`` method.
        Plain ``@flow_listen`` methods returning strings do NOT auto-route —
        only ``@flow_router`` returns drive dispatch (other step return values are ignored).
        """
        return self.state.classification

    # --- Specialist branch A: math ----------------------------------

    @flow_listen("math")
    async def answer_math(self) -> None:
        """Math branch — single Task with name + metadata for audit."""
        task = Task(
            description=self.state.question,
            agent=math_agent,
            name="math-task",
            metadata={"branch": "math", "kind": "answer"},
        )
        output = await Runner.arun_task(task, context=self.run_context.context)
        self.state.math_answer = str(output.final_output)
        logger.info("answer_math: done")

    # --- Specialist branch B: coding --------------------------------

    @flow_listen("coding")
    async def answer_coding(self) -> None:
        """Coding branch — single Task with code-snippet expectation."""
        task = Task(
            description=self.state.question,
            agent=coding_agent,
            name="coding-task",
            metadata={"branch": "coding", "kind": "answer"},
        )
        output = await Runner.arun_task(task, context=self.run_context.context)
        self.state.coding_answer = str(output.final_output)
        logger.info("answer_coding: done")

    # --- Specialist branch C: research (parallel TaskGroup) ---------

    @flow_listen("research")
    async def answer_research(self) -> None:
        """Research branch — parallel TaskGroup + serial synthesis.

        Demonstrates composing :class:`TaskGroup` (parallel) under a
        Flow step, then doing a follow-up agent run inline.
        """
        group = TaskGroup(
            tasks=(
                Task(
                    description=self.state.question,
                    agent=research_facts_agent,
                    name="facts",
                ),
                Task(
                    description=self.state.question,
                    agent=research_context_agent,
                    name="context",
                ),
            ),
            metadata={"branch": "research", "kind": "parallel_lookups"},
        )
        group_result = await Runner.arun_task_group(
            group,
            context=self.run_context.context,
        )
        self.state.research_findings = [str(out.final_output) for out in group_result.task_outputs]
        # Follow-up synthesis using the gathered material
        combined = "\n\n".join(self.state.research_findings)
        synth = await Runner.arun(
            research_synth_agent,
            f"QUESTION: {self.state.question}\n\nMATERIAL:\n{combined}",
            context=self.run_context.context,
        )
        self.state.research_synthesis = str(synth.final_output)
        logger.info("answer_research: done")

    # --- Fan-in: synthesize whichever branch ran --------------------

    @flow_listen(answer_math | answer_coding | answer_research)
    async def synthesize(self) -> None:
        """OR-gate fan-in. Fires ONCE after whichever branch ran.

        Built fluently via ``|`` operator chaining. Single-fire
        semantic: only one branch fires per run (the router dispatches
        to exactly one label), so the OR gate fires this synthesizer
        exactly once.
        """
        chosen = self.state.classification
        picked = {
            "math": self.state.math_answer,
            "coding": self.state.coding_answer,
            "research": self.state.research_synthesis,
        }.get(chosen, "")
        self.state.final_answer = f"[{chosen.upper()}] {picked}"
        logger.info("synthesize: final answer length=%d chars", len(picked))

    # --- Error handler: routes recoverable failures here ------------

    @flow_listen("__error__")
    async def on_step_error(self) -> None:
        """Recover from any step failure under ``error_policy="route_to_error_handler"``.

        The framework fires this listener when a step raises. The
        flow continues from this handler rather than halting — useful
        for production resilience where a transient failure in one
        branch shouldn't abort the whole pipeline.
        """
        self.state.error_log.append(f"step failed (classification={self.state.classification!r}); flow recovered")
        # Provide a degraded answer so downstream consumers always have something
        self.state.final_answer = (
            f"[DEGRADED] Could not answer due to a transient error. Classification was {self.state.classification!r}."
        )
        logger.warning("on_step_error: handled, final_answer is degraded")


# ---------------------------------------------------------------------
# Run helpers — sync, streamed, and configured.
# ---------------------------------------------------------------------


async def run_with_streaming(question: str) -> None:
    """Drive the flow via ``arun_flow_streamed`` to demonstrate live observability.

    Production wiring: hook this iteration up to your operator dashboard,
    OpenTelemetry span emitter, or progress UI. Each event is structured
    and discriminated by ``event.type``.
    """
    flow = TriageFlow(initial_state=TriageState(question=question))
    config = FlowConfig(
        max_steps=20,
        max_total_tokens=200_000,
        error_policy="route_to_error_handler",
    )
    streaming = Runner.arun_flow_streamed(flow, config=config)

    async for event in streaming.stream_events():
        # isinstance narrows the union; pyright/mypy validate field access per-arm.
        if isinstance(event, FlowStepStartEvent):
            logger.info("→ start: %s (step %d)", event.step_name, event.step_count)
        elif isinstance(event, FlowStepEndEvent):
            logger.info("← end:   %s", event.step_name)
        elif isinstance(event, FlowRouteEvaluatedEvent):
            logger.info(
                "↳ route: %s returned %r → fired %s",
                event.router_step,
                event.route_label,
                event.triggered_steps,
            )
        elif isinstance(event, FlowStepErrorEvent):
            logger.warning(
                "✗ error: %s raised %s: %s",
                event.step_name,
                event.error_type,
                event.error_message,
            )

    logger.info("FINAL status=%s", streaming.status)
    logger.info("FINAL classification=%s", flow.state.classification)
    logger.info("FINAL answer=%s", flow.state.final_answer)
    logger.info("FINAL errors=%s", flow.state.error_log)


async def main() -> int:
    """Entry. Runs the streamed flow with a default question."""
    await run_with_streaming("What's the time complexity of mergesort?")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
