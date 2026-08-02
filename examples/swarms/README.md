# Swarm Examples

Runnable examples for the `Swarm` primitive — iterative multi-agent
collaboration with cycles and explicit termination.

See `docs/swarms/swarms.md` for the user guide and architecture overview.
All three examples define their swarm with the fluent `Swarm.new(...)`
builder — roster, entry, routing, and termination each on one line.

## Files

| Example | Policy | Pattern |
|---------|--------|---------|
| `llm_handoff_swarm.py` | `LLMHandoffPolicy` | Author ↔ reviewer ↔ security auditor. The LLM picks who speaks next via injected `transfer_to_<name>` tools. AutoGen/Strands parity with explicit `swarm_done` termination. |
| `round_robin_debate.py` | `RoundRobinPolicy` | Two-agent debate with deterministic alternation. Zero routing tokens. Any speaker can call `swarm_done` to end it early. |
| `structured_routing_swarm.py` | `StructuredRoutingPolicy` | Triage agent produces a typed `Intent`; the policy dispatches to the matching specialist via an existing `HandoffRoute`. Zero routing tokens, full type safety. |

## Running

```bash
python examples/swarms/llm_handoff_swarm.py
python examples/swarms/round_robin_debate.py
python examples/swarms/structured_routing_swarm.py
```

Set an LLM API key (e.g. `ANTHROPIC_API_KEY` or any litellm-supported
provider) before running.

## Choosing a Starting Point

- New to swarms? Start with `round_robin_debate.py` — no routing
  tokens, easiest to reason about.
- Want classic AutoGen/Strands semantics? `llm_handoff_swarm.py`.
- Building a typed triage ↔ specialist flow? `structured_routing_swarm.py`
  — the TroopAI differentiator.
