---
paths:
  - "src/troopai/adk/tools/**/*.py"
  - "src/troopai/adk/validators/**/*.py"
  - "src/troopai/adk/agents/**/*.py"
---

# Tools, Guardrails & Hosted Tools — CRITICAL

## Middleware Is Plumbing — Verdicts Belong in Guardrails

`ToolMiddleware` / `AgentMiddleware` / `LLMMiddleware` MUST be plumbing
only: logging, metrics, tracing, retries, cross-cutting arg injection,
caching. Middleware MUST NEVER encode safety policy, content filtering,
schema validation, rate limiting, approval gates, or any
"should-this-proceed?" decision — those belong in guardrails or dedicated
typed surfaces. This preserves `Guardrail`'s typed verdict
(`allow`/`reject_content`/`raise_exception`) as the single canonical answer
and keeps `RunResult.guardrail_results` audit-complete.

**Authoring checklist** — if ANY of 1–4 is "yes", it's a guardrail not
middleware: (1) returns a typed verdict struct? (2) owns a
`strategy="block"|"redact"|"mask"` param? (3) raises a custom exception
treated as tripwire? (4) mutates `args` to remove disallowed content?
(5) short-circuits only for circuit-breaker/cache reasons? → allowed.

| Forbidden middleware | Use instead |
|---|---|
| `PII*BlockMiddleware` (halt) | `ToolInputGuardrail` → `raise_exception()` |
| `PII*RejectMiddleware` | `ToolInputGuardrail` → `reject_content(...)` |
| `JailbreakMiddleware` | `InputGuardrail` `tripwire_triggered=True` |
| `RBACMiddleware` | `FunctionTool.enabled` / `RunConfig.can_use_tool` |
| `RateLimitMiddleware` | `FunctionTool.rate_limit = ToolRateLimit(...)` |
| `ApprovalMiddleware` | `FunctionTool.requires_approval = True` |
| `SchemaValidationMiddleware` | `FunctionTool.schema_enforcement = STRICT` |
| `ContentFilter`/`JsonRepair` | `OutputGuardrail` with `remediation` |

Allowed (true plumbing): logging/metrics, tracing spans, retry-with-backoff,
request-id injection, cache short-circuit, latency histograms. Forbidden at
every scope; blast radius increases at wider scopes.

## Provider-Hosted Tools

Provider-hosted capabilities (web search, code exec, file search, image
gen, URL context) MUST be typed dataclasses inheriting `HostedTool`.
EVERY provider's converter MUST translate every supported subclass OR raise
`UnsupportedHostedToolError(tool, provider, supported_providers=...)`.
**Silent drops forbidden.** `LLMConfig.extra_body` is the escape hatch ONLY
for genuinely beta/esoteric shapes lacking a typed class.

Every concrete subclass: `@dataclass(kw_only=True)`; `SUPPORTED_PROVIDERS:
ClassVar[tuple[str, ...]]`; class docstring "Provider matrix" section;
attributes for a provider subset tagged `**<Provider> only.**`; concept
name (`WebSearchTool`, never `AnthropicWebSearchTool`).

Add a subclass when ≥2 providers ship it natively, OR 1 provider ships it
with typed knobs worth surfacing. NEVER for beta-only or zero-knob
single-provider markers.

| Tool | Anthropic | OpenAI Resp | OpenAI Chat | Gemini |
|---|---|---|---|---|
| `WebSearchTool` | ✓ | ✓ | ✗ (raises) | ✓ |
| `CodeExecutionTool` | ✗ | ✓ | ✗ | ✓ |
| `FileSearchTool` | ✗ | ✓ | ✗ | ✗ |
| `ImageGenerationTool` | ✗ | ✓ | ✗ | ✗ |
| `URLContextTool` | ✗ | ✗ | ✗ | ✓ |

## Self-Check

1. New middleware class whose name/behaviour matches a forbidden pattern
   (`block`, `redact`, `reject`, `policy`, `pii`, `jailbreak`, `filter`,
   `validate`)? — make it a guardrail.
2. New `HostedTool` subclass missing `SUPPORTED_PROVIDERS`, or a converter
   silently dropping an unsupported subclass?
