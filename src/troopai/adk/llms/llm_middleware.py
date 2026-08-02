"""LLM middleware — composable interceptors around each ``LLM.acomplete()`` call.

An ``LLMMiddleware`` wraps one call from the runner to the underlying
``LLM`` implementation. Cross-cutting concerns (per-call latency
accounting, prompt-cache hit-rate metrics, retry-with-backoff on
transient provider errors, deterministic-replay caching, cross-cutting
metadata injection on the request) can run without subclassing the
``LLM`` ABC.

Middleware lives at one registration point today:

- **Agent-global** — ``Agent(middleware=Middleware(llms=[Cache(), Latency()]))``
  wraps every LLM call the runner issues for that agent. The
  ``Middleware`` config dataclass holds typed lists per layer; the
  ``llms`` slot drives the chain composed inside ``call_llm`` from
  ``troopai.adk.run.llm_calls``.

The chain fires on every LLM call the agent makes. In a 5-turn agent
loop, it is invoked five times.

## Why a separate mechanism

Today users can subclass the ``LLM`` ABC to wrap calls — but that
forces one wrapper per provider and does not compose. A formal
middleware chain at the runner's call boundary lets multiple wrappers
stack cleanly, share state across pre- and post-call, and short-
circuit by not calling ``next``. Microsoft Agent Framework's
``ChatMiddleware`` is the direct inspiration; the value-return shape
matches the sibling ``ToolMiddleware`` and ``AgentMiddleware``
Protocols so this codebase keeps one mental model across all three
layers (function / agent / LLM).

## Scope: non-streaming calls

This Protocol returns ``LLMResponse`` and applies to
``LLM.acomplete(stream=False)``. Streaming calls
(``LLM.acomplete(stream=True)``) return
``AsyncIterator[LLMStreamEvent]`` and have a sibling Protocol —
:class:`~troopai.adk.llms.llm_stream_middleware.LLMStreamMiddleware`
— registered via ``Agent.middleware.stream_llms``. Two non-overlapping
return types mean two Protocols rather than a polymorphic union.

Use :func:`~troopai.adk.llms.llm_stream_middleware.make_logging_middlewares`
to register one paired logger across both LLM paths in one call.

When ``Agent.middleware.llms`` is non-empty AND the runner issues a
streaming call (``Runner.arun(stream=True)``) without registering a
sibling streaming middleware, ``call_llm_streamed`` emits one
``logger.warning`` line per call so the user notices the path
mismatch.

## Provider agnosticism

The Protocol takes ``LLMConfig`` and returns ``LLMResponse`` — both
are framework Layer 1 types defined under ``src/troopai/adk/types/``.
No provider-specific (litellm, anthropic, openai) types appear in
this module. A middleware authored once works against any ``LLM``
subclass.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from troopai.adk.exceptions import TroopAIError

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.responses.llm_response import LLMResponse


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol + chain composition
# ---------------------------------------------------------------------------


LLMMiddlewareNext = Callable[
    ["list[LLMInputContentItem]", "LLMConfig | None"],
    Awaitable["LLMResponse"],
]
"""The ``next`` callable passed to an ``LLMMiddleware``.

Invokes the rest of the chain (inner middleware, then the actual
``LLM.acomplete()`` call). A middleware that does not call ``next``
short-circuits the chain — useful for caching / deterministic-replay
middleware that returns a stored response without invoking the
provider.
"""


@runtime_checkable
class LLMMiddleware(Protocol):
    """Plumbing-only interceptor around a single LLM call.

    Implementations wrap one ``LLM.acomplete(...)`` call from the
    runner. Receive the active agent, the message list as the call
    sees it, the resolved ``LLMConfig`` (or ``None`` if the agent
    relies on defaults), the run context, and a ``next`` callable.
    Return the ``LLMResponse`` either by invoking ``next`` and
    possibly transforming the response, or by constructing one
    directly (short-circuit).

    Plumbing-only contract
    ----------------------
    Middleware MUST encode cross-cutting infrastructure:

      - Logging start/end of calls (model name, token deltas)
      - Latency / retry / cache-hit metrics
      - Distributed-tracing span open/close around each call
      - Retry-with-backoff on transient provider failures
      - Deterministic-replay caching (record/replay test mode)
      - Cross-cutting request metadata (request_id, tenant_id) injected
        via ``LLMConfig`` extras

    Middleware MUST NOT encode policy or verdict. Anything that
    answers *"should this prompt go to the LLM?"* or *"is this
    response acceptable?"* belongs in a typed verdict surface, not
    here:

      - Prompt-injection / PII detection on outgoing messages ->
        ``AgentInputGuardrail`` (or ``RunConfig.guardrails.input``).
      - Content filter / refusal-shaping on the response ->
        ``AgentOutputGuardrail`` (with ``remediation`` for retry).
      - Cost / token-budget enforcement -> ``RunConfig.usage_limits``
        (checked after each LLM response in the runner).
      - Tool-choice override / forced tool use ->
        ``LLMConfig.tool_choice`` and the runner's
        ``tool_choice_override`` plumbing.

    Composition with ``RunHooks``
    -----------------------------
    The order around one LLM call is::

        RunHooks.on_llm_start
          LLMMiddleware.__call__   (outermost middleware)
            …
              LLMMiddleware.__call__   (innermost middleware)
                LLM.acomplete         (terminal)
              return
            …
          return
        RunHooks.on_llm_end

    Hooks observe; middleware wraps. Both can be active at once.

    Blast-radius note
    -----------------
    A short-circuit at this scope skips one LLM call. The agent loop
    proceeds with the synthetic response on the next turn. This is
    narrower than ``AgentMiddleware`` (one block) and wider than
    ``ToolMiddleware`` (one tool call). Use short-circuit for caching
    / replay / circuit-breaker only — never for content policy.

    Registration
    ------------
    Implementations are registered via ``Agent.middleware.llms`` for
    agent-global wrapping. The chain fires on every LLM call the
    runner issues for that agent.

    Example: a per-call timing middleware::

        class TimeLLM:
            async def __call__(self, agent, messages, llm_config, context, next):
                start = time.monotonic()
                response = await next(messages, llm_config)
                logger.info(
                    "model '%s' took %.3fs",
                    response.model,
                    time.monotonic() - start,
                )
                return response
    """

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMMiddlewareNext,
    ) -> LLMResponse: ...


class LLMMiddlewareTermination(TroopAIError):
    """Short-circuit signal raised by middleware to stop the chain.

    Carries the response the runner should treat as the LLM call's
    output. Useful for caching / deterministic-replay middleware that
    wants to surface a stored response without invoking the provider.

    ``compose_llm_middleware`` catches this exception **at the
    raising middleware's frame** and returns the carried response as
    a normal value through the rest of the chain. Outer middleware
    therefore sees the carried response as a regular ``await next(…)``
    return — post-call code (timer stop, span close, metrics
    ``record_outcome(success=True)``) runs cleanly. This is the
    typical request-pipeline shape; without it, outer
    metrics/tracing middleware would mis-classify a cache hit as a
    ``BaseException`` failure.

    Synthetic-response audit note
    -----------------------------
    Because ``LLMResponse`` is a non-frozen dataclass, a middleware
    returning a crafted response (with arbitrary
    ``LLMResponseFunctionToolCall`` parts) will cause the runner to
    dispatch those tool calls exactly as if the provider had emitted
    them. This is the design point that makes caching and
    deterministic replay possible — but it is a real authority the
    middleware author is taking on. Treat synthetic-response
    middleware (cache, replay) as a security-relevant surface.
    """

    def __init__(self, response: LLMResponse) -> None:
        super().__init__("LLMMiddleware short-circuit")
        self.response = response


def compose_llm_middleware(
    middleware: list[LLMMiddleware],
    terminal: LLMMiddlewareNext,
    *,
    agent: Agent,
    context: RunContext[Any],
) -> LLMMiddlewareNext:
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
            been traversed (the actual ``llm.acomplete(...)`` call).
        agent: The active agent. Bound here so the returned ``next``
            callable matches the simple ``(messages, llm_config) ->
            response`` shape callers expect.
        context: The ``RunContext`` to thread through every middleware
            call.

    Returns:
        A ``next`` callable that, when invoked, runs the full chain
        followed by ``terminal``.
    """
    if len(middleware) == 0:
        return terminal

    chain: LLMMiddlewareNext = terminal
    # Walk the list right-to-left so the first item ends up wrapping
    # all later items (outermost). Each iteration captures the current
    # ``chain`` in a closure as the next-link for ``mw``. The
    # try/except converts ``LLMMiddlewareTermination`` raised by
    # ``_mw`` (or any inner link) into a normal return value, so outer
    # middleware sees a clean ``await next(…)`` result and runs its
    # post-call code (timer stop, metrics ``success=True``) cleanly.
    for mw in reversed(middleware):
        next_link = chain

        async def wrapped(
            messages: list[LLMInputContentItem],
            llm_config: LLMConfig | None,
            *,
            _mw: LLMMiddleware = mw,
            _next: LLMMiddlewareNext = next_link,
            _agent: Agent = agent,
            _ctx: RunContext[Any] = context,
        ) -> LLMResponse:
            try:
                return await _mw(_agent, messages, llm_config, _ctx, _next)
            except LLMMiddlewareTermination as term:
                return term.response

        chain = wrapped
    return chain


# ---------------------------------------------------------------------------
# Standard middleware shipped out of the box
# ---------------------------------------------------------------------------


@dataclass
class LLMLoggingMiddleware:
    """Single-purpose baseline that logs the start and end of each LLM call.

    Replaces the ad-hoc ``logger.info`` lines users would otherwise
    sprinkle around per-call observability. Uses the project's
    standard ``logger = logging.getLogger(__name__)`` pattern.

    Attributes:
        level: Log level for both start and end records. Defaults to
            ``logging.INFO``.
        log_token_delta: When ``True``, the response's
            ``usage.total_tokens`` is logged at completion. Default
            ``True`` because token counts are the most-asked-for
            datum on this surface; pass ``False`` to silence.
    """

    level: int = logging.INFO
    """Log level for both start and end records."""

    log_token_delta: bool = True
    """Include ``response.usage.total_tokens`` in the end record."""

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMMiddlewareNext,
    ) -> LLMResponse:
        del context
        model_hint = agent.llm if isinstance(agent.llm, str) else None
        if model_hint is not None:
            logger.log(
                self.level,
                "llm call starting (agent='%s', model='%s', messages=%d)",
                agent.name,
                model_hint,
                len(messages),
            )
        else:
            logger.log(
                self.level,
                "llm call starting (agent='%s', messages=%d)",
                agent.name,
                len(messages),
            )
        response = await next(messages, llm_config)
        if self.log_token_delta and response.usage is not None:
            logger.log(
                self.level,
                "llm call completed (agent='%s', model='%s', total_tokens=%d)",
                agent.name,
                response.model,
                response.usage.total_tokens,
            )
        else:
            logger.log(
                self.level,
                "llm call completed (agent='%s', model='%s')",
                agent.name,
                response.model,
            )
        return response


@runtime_checkable
class LLMMetricsRecorder(Protocol):
    """Protocol for ``LLMMetricsMiddleware``'s metrics sink.

    The framework does not bundle a metrics implementation —
    OpenTelemetry, StatsD, Prometheus, or a custom in-memory recorder
    can all conform to this two-method shape. Plug in whichever
    matches your observability stack.
    """

    def record_duration(self, model: str, duration_seconds: float) -> None:
        """Record a single LLM call's wall-clock duration."""

    def record_outcome(self, model: str, *, success: bool) -> None:
        """Record whether a single LLM call succeeded or raised."""


@dataclass
class LLMMetricsMiddleware:
    """Records per-call latency and success/failure to a recorder.

    The recorder is a small Protocol with two methods —
    ``record_duration`` and ``record_outcome`` — so users can plug
    in StatsD, OpenTelemetry, Prometheus, or an in-memory test
    recorder without the framework taking a dependency on any
    specific metrics library.

    Attributes:
        recorder: The metrics sink. See :class:`LLMMetricsRecorder`.
    """

    recorder: LLMMetricsRecorder
    """The metrics sink that receives duration and outcome records."""

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMMiddlewareNext,
    ) -> LLMResponse:
        del context
        # Best-effort model identifier for the failure path — we don't
        # have ``response.model`` until ``next`` returns successfully,
        # so derive a stable label from ``agent.llm`` when it's a
        # string. Falls back to ``"unknown"`` for callable / instance
        # ``agent.llm`` values.
        failure_label = agent.llm if isinstance(agent.llm, str) else "unknown"
        start = time.monotonic()
        try:
            response = await next(messages, llm_config)
        except BaseException:
            self.recorder.record_duration(failure_label, time.monotonic() - start)
            self.recorder.record_outcome(failure_label, success=False)
            raise
        self.recorder.record_duration(response.model, time.monotonic() - start)
        self.recorder.record_outcome(response.model, success=True)
        return response
