"""Agent middleware — composable interceptors around per-agent contributions.

An ``AgentMiddleware`` wraps the work one agent does inside a run — its
turn batch from when it starts until it hands off to another agent or
produces a final output. Cross-cutting concerns (per-agent latency
accounting, audit logs scoped to one agent's contribution, distributed
tracing spans framed around an agent's tenure, retry-the-whole-block
policies) can run without modifying every agent definition.

Middleware lives at one registration point today:

- **Agent-global** — ``Agent(middleware=Middleware(agents=[Audit(), Latency()]))``
  wraps every per-agent block. The ``Middleware`` config dataclass
  holds typed lists per layer; the ``agents`` slot drives the chain
  composed inside the agent loop.

The chain re-fires on every handoff / swarm transition: in a 3-agent
handoff chain, the chain is invoked three times (once per agent's
contribution). This is intentional and gives users meaningful
per-agent observability that is not reachable from ``RunHooks`` alone
(``on_agent_start`` / ``on_agent_end`` frame the whole run, not each
agent's contribution).

## Why a separate mechanism

Agent-level guardrails (``AgentInputGuardrail`` / ``AgentOutputGuardrail``)
are typed verdicts with ``allow`` / ``reject_content`` /
``raise_exception`` semantics. They cannot share state between pre and
post (separate functions), cannot transform messages going into the
block, cannot short-circuit inner middleware, and operate on the
agent's whole output rather than per-block. ``RunHooks.on_agent_start``
/ ``on_agent_end`` are observation callbacks — they frame the whole
run and cannot modify messages, results, or control flow.

Middleware fills the gap: a single function that wraps the per-agent
block from before to after, owns shared state across pre and post
(timer start, span context), can mutate messages going into the
terminal, can transform the outcome coming back, and can short-circuit
by not calling ``next()``. Microsoft Agent Framework's
``AgentMiddleware`` is the direct inspiration; the value-return shape
matches the sibling ``ToolMiddleware`` Protocol so this codebase keeps
one mental model across all three layers (function / agent / LLM).

## Scope: non-streaming and streaming runs

The chain composes inside both :func:`run_agent_loop` and
:func:`run_agent_loop_streamed`. Each driver wraps the per-block
terminal (``run_agent_block`` non-streaming,
``run_agent_block_streamed`` streaming) with the same chain-builder
(``compose_agent_middleware``). One mental model regardless of
stream mode.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from troopai.adk.exceptions import TroopAIError

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.run.run_result import RunResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome carrier — flows through the chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBlockOutcome:
    """The result of one agent's contribution to a run.

    A per-agent block ends in exactly one of two ways:

    1. **final** — the agent produced a final output. The run is done;
       the outer iteration loop returns ``result`` to the caller.
    2. **handoff** — the agent (or a swarm transition) handed off to
       another agent. The outer iteration loop reads
       ``handoff_target`` and ``next_messages``, resets per-agent
       state, and re-enters the middleware chain with the new agent.

    Middleware sees the outcome on its way back up the chain and may
    inspect or transform it (via ``dataclasses.replace`` on this
    frozen dataclass).

    Attributes:
        kind: Discriminator — ``"final"`` or ``"handoff"``.
        result: The completed ``RunResult``. Set when ``kind="final"``;
            ``None`` for handoff outcomes.
        handoff_target: The agent to run next. Set when
            ``kind="handoff"``; ``None`` for final outcomes.
        next_messages: The message list to pass to the next agent's
            block. Set when ``kind="handoff"``; ``None`` for final
            outcomes.
        next_context_end: ``context_end`` index for the next agent's
            block (where the prior context ends and the new output
            begins). Set when ``kind="handoff"``; ``None`` for final
            outcomes.
        turn: The number of turns this block consumed.
        total_turns_consumed: Cumulative turn count across the whole
            run after this block, used by the outer iteration loop to
            enforce ``RunConfig.max_total_turns``.
    """

    kind: Literal["final", "handoff"]
    """Discriminator — ``"final"`` or ``"handoff"``."""

    result: RunResult[Any] | None = None
    """The completed ``RunResult``. Set when ``kind="final"``."""

    handoff_target: Agent | None = None
    """The agent to run next. Set when ``kind="handoff"``."""

    next_messages: list[LLMInputContentItem] | None = None
    """The message list to pass to the next agent's block.

    Set when ``kind="handoff"``.
    """

    next_context_end: int | None = None
    """The ``context_end`` index for the next agent's block.

    Set when ``kind="handoff"``. Marks where the new agent's prior
    context ends and its output begins, so the next block's temporal
    slicing is correct.
    """

    turn: int = 0
    """The number of turns this block consumed."""

    total_turns_consumed: int = 0
    """Cumulative turn count across the whole run after this block."""


# ---------------------------------------------------------------------------
# Protocol + chain composition
# ---------------------------------------------------------------------------


AgentMiddlewareNext = Callable[
    ["Agent", "list[LLMInputContentItem]"],
    Awaitable[AgentBlockOutcome],
]
"""The ``next`` callable passed to an ``AgentMiddleware``.

Invokes the rest of the chain (inner middleware, then the per-agent
block terminal). A middleware that does not call ``next`` short-circuits
the chain — useful for circuit-breaker patterns that want to return a
stored outcome without re-running the agent.
"""


@runtime_checkable
class AgentMiddleware(Protocol):
    """Plumbing-only interceptor around a single agent's contribution.

    Implementations wrap one per-agent block (the work the agent does
    from the start of its tenure until it produces a final output or
    hands off). Receive the active agent, the message list as the
    block sees it, the run context, and a ``next`` callable. Return
    the ``AgentBlockOutcome`` either by invoking ``next`` and possibly
    transforming the outcome, or by constructing one directly
    (short-circuit).

    Plumbing-only contract
    ----------------------
    Middleware MUST encode cross-cutting infrastructure:

      - Per-agent latency / audit logs
      - Distributed-tracing span open/close framed around the block
      - Retry-with-backoff for transient block failures
      - Cross-cutting message injection (correlation IDs, tenant tags
        derived from ``RunContext``)
      - Caching of full block outcomes when the input is identical

    Middleware MUST NOT encode policy or verdict. Anything that
    answers *"should this agent's input be allowed?"* or *"is this
    output acceptable?"* belongs in a typed verdict surface, not here:

      - PII / jailbreak / content filter on agent input ->
        ``AgentInputGuardrail``, which returns typed verdicts
        ``allow`` / ``reject_content`` / ``raise_exception``.
      - PII / content filter on agent output -> ``AgentOutputGuardrail``
        (with ``remediation`` for self-correcting retry).
      - RBAC (deny by user role) -> ``RunConfig.can_use_tool`` permission
        callback or ``FunctionTool.enabled`` callable.
      - Approval gates -> ``FunctionTool.requires_approval`` (HITL
        deferral integrated with ``RunState`` resumption).
      - Rate limiting at the agent-turn scope ->
        ``RunConfig.usage_limits`` (typed token / request budget).

    The decision rule: if the answer to *"should this block proceed
    with this input/produce this output?"* is involved, it is a
    guardrail. If only the surrounding plumbing (log lines, metric
    records, retry loop, span boundaries, correlation IDs, cache
    hits) changes, it is middleware.

    Blast-radius note
    -----------------
    A short-circuit at this scope skips the entire LLM call AND every
    tool call inside one agent's contribution — wider than
    ``ToolMiddleware``'s single-tool blast radius. Use the
    short-circuit path with care; circuit breakers are a legitimate
    use case, but anything verdict-shaped belongs in a guardrail.

    Registration
    ------------
    Implementations are registered via ``Agent.middleware.agents``
    for agent-global wrapping. The chain re-fires on every handoff /
    swarm transition: in a 3-agent flow, the chain is invoked three
    times.

    Example: an agent-block timing middleware::

        class TimeAgentBlock:
            async def __call__(self, agent, messages, context, next):
                start = time.monotonic()
                outcome = await next(agent, messages)
                logger.info(
                    "agent '%s' took %.3fs",
                    agent.name,
                    time.monotonic() - start,
                )
                return outcome
    """

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        context: RunContext[Any],
        next: AgentMiddlewareNext,
    ) -> AgentBlockOutcome: ...


class AgentMiddlewareTermination(TroopAIError):
    """Short-circuit signal raised by middleware to stop the chain.

    Carries the outcome the loop should treat as the agent block's
    result. Useful for circuit-breaker / cache-hit / replay middleware
    that wants to surface a synthetic handoff / final outcome without
    invoking the agent's actual block.

    ``compose_agent_middleware`` catches this exception **at the
    raising middleware's frame** and returns the carried outcome as
    a normal value through the rest of the chain. Outer middleware
    therefore sees the carried outcome as a regular ``await next(…)``
    return — post-call code (timer stop, span close, metrics
    ``record_outcome(success=True)``) runs cleanly. This is the
    typical request-pipeline shape; without it, outer
    metrics/tracing middleware would mis-classify a cache hit as a
    ``BaseException`` failure.

    Note: agent-level guardrails (``AgentInputGuardrail`` /
    ``AgentOutputGuardrail``) do not run when the block is
    short-circuited by termination. A cached ``RunResult`` carried
    in ``outcome.result`` reflects guardrail verdicts from the
    original run that produced it, not from the current context.
    Use the typed verdict surfaces (input/output guardrails,
    ``RunConfig.guardrails``) for the agent-block scope to keep an
    audit trail.
    """

    def __init__(self, outcome: AgentBlockOutcome) -> None:
        super().__init__("AgentMiddleware short-circuit")
        self.outcome = outcome


def compose_agent_middleware(
    middleware: list[AgentMiddleware],
    terminal: AgentMiddlewareNext,
    *,
    context: RunContext[Any],
) -> AgentMiddlewareNext:
    """Compose a middleware list around a terminal handler.

    The first middleware in the list runs first (outermost); the
    last middleware runs last (innermost, just before ``terminal``).
    This matches the intuition of a request pipeline: the entry of
    the request walks the chain top-to-bottom; the response walks
    back bottom-to-top.

    Empty list returns ``terminal`` unchanged (zero-overhead path).

    Args:
        middleware: Outer-to-inner ordered list of middleware.
        terminal: The function that runs after the entire chain has
            been traversed (the per-agent block).
        context: The ``RunContext`` to thread through every middleware
            call. Bound here so the returned ``next`` callable matches
            the simple ``(agent, messages) -> outcome`` shape callers
            expect.

    Returns:
        A ``next`` callable that, when invoked, runs the full chain
        followed by ``terminal``.
    """
    if len(middleware) == 0:
        return terminal

    chain: AgentMiddlewareNext = terminal
    # Walk the list right-to-left so the first item ends up wrapping
    # all later items (outermost). Each iteration captures the current
    # ``chain`` in a closure as the next-link for ``mw``. The
    # try/except converts ``AgentMiddlewareTermination`` raised by
    # ``_mw`` (or any inner link) into a normal return value, so outer
    # middleware sees a clean ``await next(…)`` result and runs its
    # post-call code (timer stop, metrics ``success=True``) cleanly.
    for mw in reversed(middleware):
        next_link = chain

        async def wrapped(
            agent: Agent,
            messages: list[LLMInputContentItem],
            *,
            _mw: AgentMiddleware = mw,
            _next: AgentMiddlewareNext = next_link,
            _ctx: RunContext[Any] = context,
        ) -> AgentBlockOutcome:
            try:
                return await _mw(agent, messages, _ctx, _next)
            except AgentMiddlewareTermination as term:
                return term.outcome

        chain = wrapped
    return chain


# ---------------------------------------------------------------------------
# Standard middleware shipped out of the box
# ---------------------------------------------------------------------------


@dataclass
class AgentLoggingMiddleware:
    """Single-purpose baseline that logs the start and end of each block.

    Replaces the ad-hoc ``logger.info`` lines users would otherwise
    sprinkle around per-agent observability. Uses the project's
    standard ``logger = logging.getLogger(__name__)`` pattern.

    Attributes:
        level: Log level for both start and end records. Defaults to
            ``logging.INFO``.
        log_messages: When ``True``, the message count is included in
            the start log record. Default ``False`` to keep log lines
            terse.
    """

    level: int = logging.INFO
    """Log level for both start and end records."""

    log_messages: bool = False
    """Include the message count in the start record."""

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        context: RunContext[Any],
        next: AgentMiddlewareNext,
    ) -> AgentBlockOutcome:
        del context
        if self.log_messages:
            logger.log(
                self.level,
                "agent '%s' starting (messages=%d)",
                agent.name,
                len(messages),
            )
        else:
            logger.log(self.level, "agent '%s' starting", agent.name)
        outcome = await next(agent, messages)
        logger.log(
            self.level,
            "agent '%s' completed (kind=%s, turns=%d)",
            agent.name,
            outcome.kind,
            outcome.turn,
        )
        return outcome


@runtime_checkable
class AgentMetricsRecorder(Protocol):
    """Protocol for ``AgentMetricsMiddleware``'s metrics sink.

    The framework does not bundle a metrics implementation —
    OpenTelemetry, StatsD, Prometheus, or a custom in-memory recorder
    can all conform to this two-method shape. Plug in whichever
    matches your observability stack.
    """

    def record_duration(self, agent_name: str, duration_seconds: float) -> None:
        """Record a per-agent block's wall-clock duration."""

    def record_outcome(self, agent_name: str, *, success: bool) -> None:
        """Record whether a per-agent block completed cleanly or raised."""


@dataclass
class AgentMetricsMiddleware:
    """Records per-agent latency and success/failure to a recorder.

    The recorder is a small Protocol with two methods —
    ``record_duration`` and ``record_outcome`` — so users can plug
    in StatsD, OpenTelemetry, Prometheus, or an in-memory test
    recorder without the framework taking a dependency on any
    specific metrics library.

    Attributes:
        recorder: The metrics sink. See :class:`AgentMetricsRecorder`.
    """

    recorder: AgentMetricsRecorder
    """The metrics sink that receives duration and outcome records."""

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        context: RunContext[Any],
        next: AgentMiddlewareNext,
    ) -> AgentBlockOutcome:
        del context
        start = time.monotonic()
        try:
            outcome = await next(agent, messages)
        except BaseException:
            self.recorder.record_duration(agent.name, time.monotonic() - start)
            self.recorder.record_outcome(agent.name, success=False)
            raise
        self.recorder.record_duration(agent.name, time.monotonic() - start)
        self.recorder.record_outcome(agent.name, success=True)
        return outcome
