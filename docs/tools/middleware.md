# Tool Middleware — Composable Interceptors Around Tool Execution

A `ToolMiddleware` wraps the call to `tool.on_invoke(...)` so
cross-cutting concerns — logging, metrics, distributed-tracing
spans, retries, agent-global arg injection — can run without
modifying every tool individually.

The shape is `(ctx, tool, args, next) -> result`. Implementations
can:

- Run code **before** the tool (timer start, span open, args validation)
- Pass control to the rest of the chain via `next(ctx, tool, args)`
- Run code **after** the tool (timer stop, span close, result transformation)
- **Mutate `args`** before passing them to the next link
- **Transform the result** before returning it to the outer middleware
- **Short-circuit** by returning a result without calling `next` (or by raising `ToolMiddlewareTermination`)

## Why a separate mechanism

Tool guardrails (`tool.input_guardrails`, `tool.output_guardrails`)
are **per-tool typed verdicts** with `.allow()` / `.reject_content()`
/ `.raise_exception()` semantics. They cannot share state between
pre and post (separate functions), cannot transform args, cannot
short-circuit inner middleware, and must be re-declared on every
tool to apply broadly. `RunHooks.on_tool_start` / `on_tool_end` are
observation callbacks — they cannot modify args, results, or
control flow.

| Concern | Right tool |
|---|---|
| "Reject input containing profanity" | Guardrail (per-tool verdict) |
| "PII-scrub the text result" | Guardrail (per-tool verdict) |
| "Time every call and emit a metric" | Middleware (agent-global, shared state pre/post) |
| "Open an OpenTelemetry span around the tool" | Middleware (the span wraps the call) |
| "Inject `request_id` into every tool's args" | Middleware (mutates args, agent-global) |
| "Retry transient failures with backoff" | Middleware (the retry loop wraps the call) |

Pydantic-AI ships both per-tool validators and `WrapperToolset`;
Microsoft Agent Framework ships `FunctionMiddleware`,
`AgentMiddleware`, and `ChatMiddleware` as three separate Protocols
with their own typed contexts. We follow that precedent.

## Comparison with other frameworks

Different frameworks answer the same question differently. The matrix
below shows where each one places safety verdicts (PII detection,
content filtering, approval gates) versus plumbing (logging, metrics,
retries):

| Framework | Where do safety verdicts live? | Where does plumbing live? |
|---|---|---|
| **LangChain** | `AgentMiddleware` subclasses such as `PIIMiddleware(strategy="block")` and `HumanInTheLoopMiddleware` — middleware *is* the safety surface | The same `AgentMiddleware` umbrella — one polymorphic class with hooks (`before_model`, `wrap_tool_call`, …) dispatched per subclass |
| **Microsoft Agent Framework** | Per-function validators (separate from middleware) | `FunctionMiddleware` / `AgentMiddleware` / `ChatMiddleware` — three separate Protocols, one per scope |
| **Pydantic-AI** | Per-tool input/output validators | `WrapperToolset.call_tool` overrides at the toolset boundary |
| **TroopAI ADK** (this framework) | `AgentInputGuardrail` / `AgentOutputGuardrail` (agent-level) + `ToolInputGuardrail` / `ToolOutputGuardrail` (per-tool), all returning typed verdicts (`allow` / `reject_content` / `raise_exception`) | `ToolMiddleware` Protocol — function-scope only today; `Middleware.agents` and `Middleware.llms` slots reserved for future Protocols, each with its own typed context |

Three notes that explain the TroopAI ADK position:

1. **Why the umbrella shape is rejected.** LangChain's
   `PIIMiddleware(strategy="block")` collapses verdict and plumbing
   into one polymorphic surface. The cost is that the verdict becomes
   implicit in the middleware's behaviour rather than an explicit
   typed return — a reader cannot tell from the signature whether a
   middleware halts execution, redacts content, or merely observes.
   The TroopAI ADK keeps `Guardrail` as the typed-verdict surface and
   `ToolMiddleware` as plumbing only; both are loadable by the same
   review tooling, but they do not share a registration list.
2. **Why the three-Protocol split is preferred.** Microsoft's
   `FunctionMiddleware` / `AgentMiddleware` / `ChatMiddleware` is the
   model. Each layer carries its own typed context, so a turn-scope
   middleware does not have to pretend it sees a `ToolContext`. The
   `Middleware` config dataclass on `Agent` mirrors that split with
   plural slot names (`tools`, `agents`, `llms`).
3. **What this means in practice.** Reach for a guardrail when the
   answer is "verdict on a single decision point". Reach for
   middleware when the answer is "wrap the call with shared state,
   transforms, or short-circuits". The two surfaces compose: the
   executor runs `input guardrail → middleware chain → tool →
   middleware unwind → output guardrail`, so they cannot conflict.

## Quickstart

```python
from troopai.adk import Agent, Middleware
from troopai.adk.tools import ToolLoggingMiddleware, ToolMetricsMiddleware


class MyMetrics:
    def record_duration(self, tool_name, duration_seconds):
        # Send to StatsD/Prometheus/OpenTelemetry/etc.
        ...

    def record_outcome(self, tool_name, *, success):
        ...


agent = Agent(
    name="Service",
    system_prompt="...",
    tools=[search, summarise, lookup],
    middleware=Middleware(
        tools=[
            ToolLoggingMiddleware(),
            ToolMetricsMiddleware(recorder=MyMetrics()),
        ],
    ),
)
```

`Agent.middleware` is a single typed config object holding per-layer
middleware lists. Today only the `tools` slot is wired into the run
loop; the `agents` (turn-scope) and `llms` (LLM-call) slots are
reserved for future Protocols.

Every call to `search`, `summarise`, or `lookup` flows through both
middleware in the listed order (logging outer, metrics inner).

## Authoring a custom middleware

```python
import time
from troopai.adk.tools import ToolMiddleware


class TimingMiddleware:
    """Time every tool call and accumulate per-tool totals."""

    def __init__(self):
        self.totals: dict[str, float] = {}

    async def __call__(self, ctx, tool, args, next):
        start = time.monotonic()
        result = await next(ctx, tool, args)
        duration = time.monotonic() - start
        self.totals[tool.name] = self.totals.get(tool.name, 0.0) + duration
        return result
```

The `ToolMiddleware` Protocol is `runtime_checkable`, so
`isinstance(my_middleware, ToolMiddleware)` works for tests and
introspection.

## Chain semantics

The list is composed **outer-to-inner**: the first middleware runs
first (outermost), the last runs last (innermost). Each middleware
can call `next(...)` exactly once or zero times.

```
[A, B, C]  →  A.pre → B.pre → C.pre → tool → C.post → B.post → A.post
```

## Mutating args

A middleware can mutate the `args` dict in-place (Python passes
dicts by reference) before calling `next`. The tool sees the
mutated args:

```python
class InjectRequestID:
    async def __call__(self, ctx, tool, args, next):
        args["request_id"] = ctx.context.get("request_id", "anonymous")
        return await next(ctx, tool, args)
```

## Short-circuiting

A middleware that does not call `next` short-circuits the chain.
Useful for circuit-breaker / cache patterns:

```python
from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult


class CacheMiddleware:
    def __init__(self):
        self.cache: dict[tuple, str] = {}

    async def __call__(self, ctx, tool, args, next):
        key = (tool.name, tuple(sorted(args.items())))
        if key in self.cache:
            return FunctionToolCallResult(
                type="function_call_output",
                call_id=ctx.tool_call_id or "",
                output=self.cache[key],
            )
        result = await next(ctx, tool, args)
        self.cache[key] = result.output
        return result
```

For deeper chains where you want to skip multiple middleware
layers, raise `ToolMiddlewareTermination` with the desired result —
the executor unwinds directly without invoking outer middleware:

```python
from troopai.adk.tools import ToolMiddlewareTermination


class CircuitBreaker:
    async def __call__(self, ctx, tool, args, next):
        if self.tripped(tool.name):
            raise ToolMiddlewareTermination(
                FunctionToolCallResult(
                    type="function_call_output",
                    call_id=ctx.tool_call_id or "",
                    output=f"[breaker tripped for {tool.name}]",
                )
            )
        return await next(ctx, tool, args)
```

## Toolset-scoped middleware

Wrap a toolset with `WrapperToolset` to apply middleware only to
that toolset's tools:

```python
from troopai.adk import Middleware
from troopai.adk.tools import WrapperToolset, FunctionToolset

db_tools = FunctionToolset(tools=[query, insert]).prefixed("db")
db_with_audit = WrapperToolset(
    wrapped=db_tools,
    middleware=[AuditMiddleware()],   # toolset-scoped (a list[ToolMiddleware])
)

agent = Agent(
    name="Service",
    system_prompt="...",
    tools=[ad_hoc_tool, db_with_audit],
    middleware=Middleware(tools=[ToolLoggingMiddleware()]),   # agent-global
)
```

Composition order: agent-global middleware **wraps** the
toolset-scoped middleware. So a call to `db_query` runs through:

```
ToolLoggingMiddleware.pre → AuditMiddleware.pre → query → AuditMiddleware.post → ToolLoggingMiddleware.post
```

`ad_hoc_tool` only runs through `ToolLoggingMiddleware` (no toolset
wrapping).

## Standard middleware shipped

| Class | Purpose |
|---|---|
| `ToolLoggingMiddleware` | Log start/end of each call. Optional `log_args` and `log_result` flags |
| `ToolMetricsMiddleware` | Record duration + success/failure to a `ToolMetricsRecorder` Protocol — bring your own (StatsD, OpenTelemetry, in-memory) |

Both are reference implementations meant to be copied or extended.
The framework deliberately does not ship a full observability
stack — choosing the metrics backend is a deployment concern.

## When NOT to write a middleware

Middleware is **plumbing**. Anything that decides *whether a call
should proceed*, *whether content is acceptable*, or *what verdict
to apply on detection* belongs elsewhere. Common confusions and
their right-place-to-go:

| Tempted to write… | Use this instead | Why |
|---|---|---|
| `PIIBlockMiddleware` (halt on detection) | `ToolInputGuardrail` returning `raise_exception()` | Verdict is typed; framework records it on `RunResult.guardrail_results.input` for audit |
| `PIIRejectMiddleware` (ask LLM to retry without PII) | `ToolInputGuardrail` returning `reject_content("Please remove PII from your input.")` | Typed rejection-with-feedback; framework swaps the result with the message and the LLM re-asks |
| `PIIMaskOutputMiddleware` (scrub PII from result text) | `ToolOutputGuardrail` returning `reject_content(masked_text)` | Output guardrail operates on the result; the framework replaces the result the LLM sees |
| `JailbreakMiddleware` | `AgentInputGuardrail` (agent-level) with `tripwire_triggered=True` | Runs before the agent loop; saves tokens when blocking mode is selected |
| `PromptInjectionDetectionMiddleware` | `ToolInputGuardrail` returning `raise_exception()` (or agent-level `AgentInputGuardrail`) | Detection is a verdict; recording it as a guardrail produces an audit entry that observability tooling can pivot on |
| `SecretsRedactMiddleware` (strip secrets from tool output) | `ToolOutputGuardrail` returning `reject_content(...)` | Mutating outputs without a verdict trail is unauditable; the guardrail surface records every redaction decision |
| `RBACMiddleware` (block tool call by user role) | `FunctionTool.enabled = lambda ctx: ctx.context["role"] == "admin"` or `RunConfig.can_use_tool` | Permission denial is a verdict; both surfaces produce typed denials integrated with the tool list |
| `ContentFilterMiddleware` | `AgentOutputGuardrail` with `remediation` | Built-in retry on trip — middleware has no remediation hook |
| `OutputSchemaCheckMiddleware` (per-tool result shape) | `ToolOutputGuardrail` validating shape (or `Agent.output_schema` for the final agent output) | Schema verdicts compose; middleware cannot remediate |
| `SchemaValidationMiddleware` (per-tool input) | `tool.schema_enforcement = SchemaEnforcement.STRICT` | Framework-level enforcement before invocation |
| `RateLimitMiddleware` | `tool.rate_limit = ToolRateLimit(rpm=...)` | Sliding-window enforcement built into the tool dataclass |
| `ApprovalMiddleware` | `tool.requires_approval = True` | HITL deferral integrated with `RunState` resumption |
| `ToolLoggingMiddleware(redact_pii=True)` (suppress fields by classification) | Compose: a `ToolInputGuardrail` runs first, then a plain `ToolLoggingMiddleware` logs cleared calls | The classification is a verdict — extract it; let the middleware log only after the guardrail clears the call |

Allowed plumbing patterns — these all wrap the call without
encoding a verdict:

| Allowed | Why it is plumbing |
|---|---|
| `ToolLoggingMiddleware`, `ToolMetricsMiddleware` | Observation only; no decision. Note: `log_args=True` should be paired with pre-logging sanitisation if any tool may receive sensitive input |
| `TracingMiddleware` (your own subclass) | Wraps the call in a span; no decision |
| `RetryWithBackoffMiddleware` | Wraps `next()` in a retry loop; the *call attempt* changes, not the policy |
| `RequestIDInjectionMiddleware` | Mutates `args` to carry context; framework-agnostic concern |
| `CacheMiddleware` | Short-circuits on cache hit; no policy verdict |

The decision rule:

> If the answer to *"should this call proceed?"* or *"is this content
> acceptable?"* is involved, write a **guardrail**.
> If only the surrounding plumbing (logging, metrics, retries, arg
> injection, caching) changes, write a **middleware**.

The contract is enforced at three layers: the `ToolMiddleware`
Protocol docstring, this section, and a normative project rule
loaded into every code-review session.

## Scope: three middleware layers

The framework ships three sibling middleware Protocols, one per
execution layer. All three share the same value-return shape and
the same plumbing-only contract.

| Protocol | Wraps | Slot | Doc |
|---|---|---|---|
| `ToolMiddleware` | `tool.on_invoke` (one tool call) | `Agent.middleware.tools` | this page |
| `AgentMiddleware` | one per-agent block (one agent's tenure inside a run) | `Agent.middleware.agents` | `docs/agents/agent_middleware.md` |
| `LLMMiddleware` | one `LLM.acomplete()` call | `Agent.middleware.llms` | `docs/llms/llm_middleware.md` |

Composition order during one turn is

```
Agent.middleware.agents (outermost)
  Agent.middleware.llms
    LLM.acomplete()
  return
  Agent.middleware.tools (per tool call)
    tool.on_invoke()
  return
return
```

The plumbing-only / no-verdicts contract applies unchanged at every
layer — the forbidden-vs-allowed table in this page is normative
for `AgentMiddleware` and `LLMMiddleware` too. Only the typed
verdict surfaces (`AgentInputGuardrail`, `AgentOutputGuardrail`,
`ToolInputGuardrail`, `ToolOutputGuardrail`, `RunConfig.usage_limits`,
`FunctionTool.requires_approval`, `FunctionTool.rate_limit`,
`FunctionTool.schema_enforcement`) may carry policy decisions.

### Streaming support

All three middleware Protocols apply on both streaming and
non-streaming runs:

- `ToolMiddleware` (this page) — drain of streaming tools'
  `AsyncIterator[ToolStreamEvent]` happens at the innermost
  middleware terminal so middleware sees the final accumulated
  value, not chunks.
- `AgentMiddleware` — composes inside both `run_agent_loop` and
  `run_agent_loop_streamed`.
- `LLMMiddleware` (non-streaming) and `LLMStreamMiddleware`
  (streaming) — sibling Protocols on `Middleware.llms` and
  `Middleware.stream_llms` slots respectively. Use
  `make_logging_middlewares()` to register a paired logger across
  both LLM paths.

## See also

- `docs/agents/agent_middleware.md` — `AgentMiddleware` contract.
- `docs/llms/llm_middleware.md` — `LLMMiddleware` contract.
- `examples/tools/middleware/` — runnable examples (including a
  three-layer demo).
- `src/troopai/adk/tools/tool_middleware.py` — Protocol definition,
  shipped middleware, and the chain composition helpers.
- `docs/tools/toolsets.md` — toolset composition (where
  `WrapperToolset.middleware` lives).
