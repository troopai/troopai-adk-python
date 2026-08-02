# Routing Module

Ordered-candidate model routing with automatic escalation on failure.

## Files

| File | Purpose |
|---|---|
| `router.py` | `LLMRouter` ABC, `RoutedModel`, `RoutingContext` |
| `cheapest_first.py` | `CheapestFirstRouter` — order by estimated input cost ascending |
| `latency_first.py` | `LatencyFirstRouter` — order by developer-supplied latency map ascending |

## Architecture Decisions

| Decision | What | Why |
|----------|------|-----|
| **Ordered candidates + escalation** | `candidates()` returns a `Sequence[RoutedModel]`; the loop iterates on failure | Simple, explicit fallback chain with no hidden retry logic |
| **Escalation triggers** | Provider exception, output-schema validation failure, `should_escalate()` returning `True` | Covers the two observable failure modes (infrastructure + content) plus a user hook; keeps the contract narrow |
| **Framework errors propagate** | `TroopAIError` subclasses (budget kill, guardrail reject) bypass routing | These are policy decisions, not candidate failures; escalation would mask them |
| **Streaming: pre-token-only** | Escalation is only possible before the first token is yielded | Once streaming begins the response is partially consumed; mid-stream retry is not safe |
| **Routers operate on framework `LLM` instances** | `RoutedModel.llm` is an `LLM` ABC instance, not a litellm model string | Stays provider-agnostic; any `LLM` subclass (LiteLLM, Anthropic native, OpenAI Responses) is a valid candidate |
| **`QualityFirstRouter` deferred** | Not in this module | Requires eval scores from the eval framework, which is not yet complete |
| **`CheapestFirstRouter` calls `estimate_cost` per invocation** | Freshest estimate each time; no cached costs | Avoids stale estimates when message length changes; note: O(candidates × input_tokens), so keep candidate lists small |
| **Unknown-cost candidates sort last** | `math.inf` sentinel in `CheapestFirstRouter` | Priced models are tried before unpriced ones; never silently skip a cheaper priced model |

## Flow

```
RunConfig.router.candidates(RoutingContext) → ordered [RoutedModel, ...]
  ↓ (per candidate)
  llm.acomplete(messages) → success → done
             ↓ exception / schema failure / should_escalate
  next candidate → ... → NoRoutingCandidateError if all exhausted
```

Budget gating runs per-candidate before each attempt (in `run/loop.py`).
Budget kills are not routing failures and do not trigger escalation.

## Pointers

- Usage guide and snippets: `docs/cost/cost.md` (Smart routing section)
- Runnable examples: `examples/cost/smart_routing.py`
- Runner integration field: `run/config.py` (`router`)
- Profile helper: `run/profile.py` (`router`)
- `NoRoutingCandidateError`: `exceptions/__init__.py`
