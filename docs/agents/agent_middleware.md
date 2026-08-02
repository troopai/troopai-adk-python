# Agent Middleware — Composable Interceptors Around a Per-Agent Block

An `AgentMiddleware` wraps the work one agent does inside a run —
its turn batch from when it starts until it hands off to another
agent or produces a final output. Cross-cutting concerns scoped to
*one agent's contribution* (per-agent latency, audit logs, retry
the whole block, distributed-tracing spans framed around an agent's
tenure) can run without modifying every agent definition.

The shape is `(agent, messages, ctx, next) -> AgentBlockOutcome`.
Implementations can:

- Run code **before** the agent's block (timer start, span open)
- Pass control to the rest of the chain via `next(agent, messages)`
- Run code **after** the block (timer stop, span close, outcome
  inspection)
- **Mutate `messages`** before passing them to the next link
- **Transform the outcome** before returning it (via
  `dataclasses.replace` on the frozen `AgentBlockOutcome`)
- **Short-circuit** by raising `AgentMiddlewareTermination`
  carrying a synthetic outcome

The chain re-fires on **every handoff / swarm transition**: in a
3-agent handoff chain, the chain is invoked three times (once per
agent's contribution). This is intentional — `RunHooks.on_agent_start`
and `RunHooks.on_agent_end` frame the *whole run*, not each agent.
Middleware fills that observability gap.

## Why a separate mechanism

Agent-level guardrails (`AgentInputGuardrail`, `AgentOutputGuardrail`)
are typed verdicts (`allow` / `reject_content` / `raise_exception`).
They cannot share state between pre and post (separate functions),
cannot transform messages going into the block, cannot short-circuit
inner middleware, and operate on the whole agent's output rather
than per-block. `RunHooks` are observation callbacks — they cannot
modify messages, results, or control flow.

`AgentMiddleware` fills the gap: a single function that wraps the
per-agent block from before to after, owns shared state across pre
and post, can mutate messages going into the terminal, can transform
the outcome coming back, and can short-circuit by not calling
`next()`.

## Quickstart

```python
from troopai.adk.agents import Agent, Middleware
from troopai.adk.run.agent_middleware import AgentLoggingMiddleware

agent = Agent(
    name="Researcher",
    system_prompt="...",
    middleware=Middleware(
        agents=[AgentLoggingMiddleware()],
    ),
)
```

Each entry of `Middleware.agents` runs once at the start and once at
the end of every per-agent block. The list is composed
**outer-to-inner**: the first entry runs first (outermost), the last
entry runs last (innermost, just before the actual agent block).

## Authoring a custom middleware

```python
import logging
import time

from troopai.adk.run.agent_middleware import AgentBlockOutcome


class TimeAgentBlock:
    """Time each per-agent block and log the duration."""

    async def __call__(self, agent, messages, context, next) -> AgentBlockOutcome:
        start = time.monotonic()
        outcome = await next(agent, messages)
        elapsed = time.monotonic() - start
        logging.getLogger(__name__).info(
            "agent '%s' took %.3fs (kind=%s, turns=%d)",
            agent.name,
            elapsed,
            outcome.kind,
            outcome.turn,
        )
        return outcome
```

The Protocol is `runtime_checkable`, so any class with the matching
async `__call__` shape passes `isinstance(obj, AgentMiddleware)`. No
inheritance is required.

## `AgentBlockOutcome` shape

`AgentMiddleware` returns `AgentBlockOutcome` — a frozen dataclass
with a `kind: Literal["final", "handoff"]` discriminator:

| Field | Set when | Meaning |
|---|---|---|
| `kind` | always | `"final"` or `"handoff"` |
| `result` | `final` | The completed `RunResult` |
| `handoff_target` | `handoff` | The agent for the next block |
| `next_messages` | `handoff` | The message list for the next block |
| `next_context_end` | `handoff` | Where the next block's prior context ends |
| `turn` | always | Turns this block consumed |
| `total_turns_consumed` | always | Cumulative turn count |

Middleware that wants to mutate the outcome uses
`dataclasses.replace(outcome, ...)` (the dataclass is frozen).

## Chain semantics

A three-middleware chain `[A, B, C]` produces this trace:

```
+A   (outermost enters)
  +B
    +C
      run_agent_block(...)   (the actual block runs)
    -C
  -B
-A   (outermost exits)
```

Both pre-call and post-call code are inside the same Python frame,
so middleware can keep state on `self` or in the local scope between
the two halves (timers, span context, request IDs).

## Mutating messages

```python
class InjectCorrelationId:
    """Prepend a developer message carrying a correlation ID from
    the user context. Visible to the agent's LLM calls inside the
    block, but not to other agents in a handoff chain."""

    async def __call__(self, agent, messages, context, next):
        cid = getattr(context.context, "correlation_id", None)
        if cid is not None:
            messages = [
                {"role": "developer", "content": f"correlation_id={cid}"},
                *messages,
            ]
        return await next(agent, messages)
```

## Short-circuiting

Two ways to short-circuit:

```python
# 1. Don't call next at all — return a synthetic outcome.
class CacheHitMiddleware:
    def __init__(self, cache): self.cache = cache
    async def __call__(self, agent, messages, context, next):
        cached = self.cache.get(agent.name)
        if cached is not None:
            return cached  # Skip the block entirely.
        outcome = await next(agent, messages)
        self.cache[agent.name] = outcome
        return outcome
```

```python
# 2. Raise AgentMiddlewareTermination — outer middleware unwinds.
from troopai.adk.run.agent_middleware import AgentMiddlewareTermination

class CircuitBreaker:
    async def __call__(self, agent, messages, context, next):
        if self._is_open(agent.name):
            from dataclasses import replace
            from troopai.adk.run.agent_middleware import AgentBlockOutcome
            raise AgentMiddlewareTermination(
                AgentBlockOutcome(kind="final", result=self._fallback_result()),
            )
        return await next(agent, messages)
```

The two are NOT equivalent: returning bypasses inner middleware but
still runs outer middleware's post-call code. Raising
`AgentMiddlewareTermination` unwinds directly to the loop, skipping
all surrounding middleware.

## Standard middleware shipped

| Class | Behaviour |
|---|---|
| `AgentLoggingMiddleware` | Logs `agent 'X' starting` / `agent 'X' completed (kind=…, turns=N)` at INFO. |
| `AgentMetricsMiddleware` | Calls `record_duration(agent_name, seconds)` and `record_outcome(agent_name, success=…)` on a user-supplied `AgentMetricsRecorder`. |

Bring-your-own metrics sink (`AgentMetricsRecorder` is a Protocol —
StatsD, OpenTelemetry, Prometheus, or an in-memory test sink all
conform structurally).

## When NOT to write an `AgentMiddleware`

Middleware is **plumbing only**. Anything that decides *whether* an
agent's input should be allowed or *what to do* about an
unacceptable output belongs in a typed verdict surface, not here.

| If you want to… | Use this instead |
|---|---|
| Block the agent on PII / jailbreak in the input | `AgentInputGuardrail` (typed `raise_exception` verdict) |
| Filter / mask PII from the agent's output | `AgentOutputGuardrail` (typed `reject_content` verdict, with `remediation` for retry) |
| Deny by user role | `RunConfig.can_use_tool` callback or `FunctionTool.enabled` |
| Rate limit total tokens / requests | `RunConfig.usage_limits` |
| Approval gate before a sensitive tool fires | `FunctionTool.requires_approval` (HITL deferral) |

The line: *if the answer to "should this block proceed with this
input / produce this output?" is involved, it is a guardrail. If
only the surrounding plumbing changes, it is middleware.* This is
the same line `ToolMiddleware` draws — see `docs/tools/middleware.md`
for the full forbidden-vs-allowed table that applies unchanged at
agent scope.

## Blast radius

A short-circuit at this scope skips the entire LLM call AND every
tool call inside one agent's contribution — wider than
`ToolMiddleware`'s single-tool blast radius. Use the short-circuit
path with care; circuit breakers and replay caches are legitimate,
but anything verdict-shaped belongs in a guardrail.

## Streaming support

The chain composes inside both `run_agent_loop` (non-streaming) and
`run_agent_loop_streamed` (streaming). The streaming driver wraps
its per-block terminal — `run_agent_block_streamed` — with
`compose_agent_middleware`, just like the non-streaming driver wraps
`run_agent_block`. The chain re-fires on every handoff /
swarm-yield transition under `Runner.arun(stream=True)` exactly as
it does on the non-streaming path. One mental model regardless of
stream mode.

The streaming-only termination pathways (interruption, swarm yield,
cancel, final output) all surface as
`AgentBlockOutcome(kind="final")` because the streaming driver only
needs the binary "continue (handoff) or stop" decision; per-pathway
state has already been mutated onto the `RunResultStreaming`
object inside the block.

## See also

- `examples/agents/agent_middleware/` — runnable examples.
- `src/troopai/adk/run/agent_middleware.py` — Protocol definition,
  shipped middleware, and the chain composition helpers.
- `docs/tools/middleware.md` — `ToolMiddleware` (sibling layer);
  contains the canonical forbidden-vs-allowed table.
- `docs/llms/llm_middleware.md` — `LLMMiddleware` (sibling layer).
