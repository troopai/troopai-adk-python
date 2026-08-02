# LLM Middleware — Composable Interceptors Around Each `LLM.acomplete()` Call

An `LLMMiddleware` wraps one call from the runner to the underlying
`LLM` implementation. Cross-cutting concerns scoped to *one LLM
invocation* (per-call latency, prompt-cache hit-rate metrics,
retry-with-backoff on transient provider errors,
deterministic-replay caching, cross-cutting metadata injection) can
run without subclassing the `LLM` ABC.

The shape is `(agent, messages, llm_config, ctx, next) -> LLMResponse`.
Implementations can:

- Run code **before** the LLM call (timer start, span open)
- Pass control to the rest of the chain via `next(messages, llm_config)`
- Run code **after** the call (timer stop, span close, response
  inspection)
- **Mutate `messages` or `llm_config`** before passing them to the
  next link
- **Transform the response** before returning it (the response is a
  non-frozen `@dataclass` — mutate or `dataclasses.replace`)
- **Short-circuit** by returning a `LLMResponse` directly without
  calling `next` (or by raising `LLMMiddlewareTermination`)

The chain fires on **every LLM call** the runner issues for the
agent. In a 5-turn agent loop, it is invoked five times.

## Why a separate mechanism

Today users can subclass the `LLM` ABC to wrap calls — but that
forces one wrapper per provider (a custom subclass of `LiteLLM`,
`AnthropicLLM`, etc.) and does not compose. A formal middleware
chain at the runner's call boundary lets multiple wrappers stack
cleanly, share state across pre- and post-call, and short-circuit
by not calling `next`.

## Quickstart

```python
from troopai.adk.agents import Agent, Middleware
from troopai.adk.llms.llm_middleware import LLMLoggingMiddleware

agent = Agent(
    name="Researcher",
    system_prompt="...",
    middleware=Middleware(
        llms=[LLMLoggingMiddleware()],
    ),
)
```

Each entry of `Middleware.llms` runs once at the start and once at
the end of every LLM call. Composition is **outer-to-inner**.

## Authoring a custom middleware

```python
import logging
import time

from troopai.adk.types.responses.llm_response import LLMResponse


class TimeLLMCall:
    """Time each LLM call and log the duration with the model name."""

    async def __call__(self, agent, messages, llm_config, context, next) -> LLMResponse:
        start = time.monotonic()
        response = await next(messages, llm_config)
        logging.getLogger(__name__).info(
            "model '%s' took %.3fs (input=%d output=%d)",
            response.model,
            time.monotonic() - start,
            response.usage.input_tokens if response.usage else -1,
            response.usage.output_tokens if response.usage else -1,
        )
        return response
```

The Protocol is `runtime_checkable`, so any class with the matching
async `__call__` shape passes `isinstance(obj, LLMMiddleware)`. No
inheritance is required.

## Provider-agnostic types

The Protocol uses framework Layer 1 types only:

- `messages: list[LLMInputContentItem]` — provider-agnostic Layer 1
  input items.
- `llm_config: LLMConfig | None` — the resolved config for the call.
- Returns `LLMResponse` — the framework's Layer 1 response container.

No litellm or other provider SDK types appear. A middleware authored
once works against any `LLM` subclass (LiteLLM, AnthropicLLM, etc.).

## Chain semantics

Same outer-to-inner / inner-to-outer trace as the other two
middleware layers. The order around one LLM call is:

```
RunHooks.on_llm_start
  LLMMiddleware.__call__   (outermost middleware)
    …
      LLMMiddleware.__call__   (innermost middleware)
        LLM.acomplete         (terminal)
      return
    …
  return
RunHooks.on_llm_end
```

Hooks observe; middleware wraps. Both can be active at once.

## Short-circuiting

Two ways to short-circuit:

```python
# 1. Cache hit — return without calling next.
class ResponseCache:
    def __init__(self, cache): self.cache = cache
    async def __call__(self, agent, messages, llm_config, context, next):
        key = self._key(agent, messages)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        response = await next(messages, llm_config)
        self.cache[key] = response
        return response
```

```python
# 2. Deterministic replay — raise the typed termination.
from troopai.adk.llms.llm_middleware import LLMMiddlewareTermination

class ReplayMiddleware:
    def __init__(self, replay_log): self.log = replay_log
    async def __call__(self, agent, messages, llm_config, context, next):
        key = self._key(agent, messages)
        if key in self.log:
            raise LLMMiddlewareTermination(self.log[key])
        return await next(messages, llm_config)
```

## Standard middleware shipped

| Class | Behaviour |
|---|---|
| `LLMLoggingMiddleware` | Logs `llm call starting (agent='X', model='m', messages=N)` and `llm call completed (agent='X', model='m', total_tokens=N)` at INFO. |
| `LLMMetricsMiddleware` | Calls `record_duration(model, seconds)` and `record_outcome(model, success=…)` on a user-supplied `LLMMetricsRecorder`. |

Bring-your-own metrics sink (`LLMMetricsRecorder` is a Protocol —
StatsD, OpenTelemetry, Prometheus, or an in-memory test sink all
conform structurally).

## When NOT to write an `LLMMiddleware`

Middleware is **plumbing only**. Policy / verdict belongs in
guardrails or typed surfaces.

| If you want to… | Use this instead |
|---|---|
| Detect prompt injection / PII in outgoing messages | `AgentInputGuardrail` |
| Filter / mask PII in the response text | `AgentOutputGuardrail` (with `remediation` for retry) |
| Enforce token / request budgets | `RunConfig.usage_limits` (checked after each LLM response) |
| Force a tool choice / forbid tools | `LLMConfig.tool_choice` and the runner's `tool_choice_override` plumbing |
| Switch model on errors | `LLMConfig.fallbacks` (litellm-native) or `LLMConfig.retry_policy` |

## Blast radius

A short-circuit at this scope skips one LLM call. The agent loop
proceeds with the synthetic response on the next turn. Narrower
than `AgentMiddleware` (one block) and wider than `ToolMiddleware`
(one tool call). Use short-circuit for caching / replay /
circuit-breaker only — never for content policy.

## Streaming calls — sibling Protocol

`LLMMiddleware` returns `LLMResponse` and applies to non-streaming
`LLM.acomplete()` calls. Streaming calls (`stream=True`) return
`AsyncIterator[LLMStreamEvent]` and have a sibling Protocol,
`LLMStreamMiddleware`, registered via `Agent.middleware.stream_llms`.
Two Protocols (rather than a polymorphic union) keep the type
checker friendly across both paths.

```python
from troopai.adk.agents.middleware import Middleware
from troopai.adk.llms.llm_middleware import LLMLoggingMiddleware
from troopai.adk.llms.llm_stream_middleware import (
    LLMStreamLoggingMiddleware,
    make_logging_middlewares,
)

# Register independently per path:
mw = Middleware(
    llms=[LLMLoggingMiddleware()],
    stream_llms=[LLMStreamLoggingMiddleware()],
)

# Or share one logger across both via the factory:
ns, st = make_logging_middlewares()
mw = Middleware(llms=[ns], stream_llms=[st])
```

If `llms` is configured WITHOUT a sibling `stream_llms` and the
runner issues a streaming call, `call_llm_streamed` emits one
`logger.warning` line per call — the user notices the path mismatch
and registers via `stream_llms` instead.

The plumbing-only contract applies unchanged across both paths.
The forbidden-vs-allowed table doesn't gain new rows; the same
verdict-vs-plumbing split holds regardless of streaming mode.

## See also

- `examples/llms/llm_middleware/` — runnable examples.
- `src/troopai/adk/llms/llm_middleware.py` — Protocol definition,
  shipped middleware, and the chain composition helpers.
- `docs/tools/middleware.md` — `ToolMiddleware` (sibling layer);
  contains the canonical forbidden-vs-allowed table.
- `docs/agents/agent_middleware.md` — `AgentMiddleware` (sibling
  layer).
