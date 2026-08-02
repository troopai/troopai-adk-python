# Swarm Cost Optimization

Swarms amplify token cost because the same conversation is replayed
across multiple turns and (potentially) multiple agents. TroopAI gives
you **five composable levers** and one absolute safety net.

## The Five Levers

| Lever | Mechanism | Layer | Applies to |
|-------|-----------|-------|------------|
| Per-tool result cap | `FunctionTool.max_result_tokens` | Tool | A single tool's JSON output |
| Per-handoff history cap | `HandoffConfig.budget` | Handoff | History carried across a handoff edge |
| Swarm-wide switch cap | `SwarmConfig.max_handoffs` | Swarm | Total agent switches in the run |
| Swarm-wide token cap | `SwarmConfig.max_total_tokens` | Swarm | Cumulative LLM tokens across the swarm |
| Per-turn context size | `SharedContextStrategy` | Swarm | Messages sent to each agent per turn |

Absolute safety net: **`RunConfig.max_total_turns`** (existing, not
swarm-specific) stops runaway loops. It defaults to `500` per the
cost-conservative-defaults rule — production deployments can override
(raise or lower) for their workload. Set to `None` explicitly only
when you genuinely want unbounded multi-agent turns.

## Minimal-Cost Starter Template

```python
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.swarms import (
    Swarm, SwarmConfig, SharedContextConfig, SharedContextStrategy,
    LLMHandoffPolicy,
    ExplicitDoneTermination, MaxTurnsTermination, TokenBudgetTermination,
)

swarm = Swarm(
    members=(author, reviewer, auditor),
    entry=author,
    policy=LLMHandoffPolicy(),
    termination=(
        ExplicitDoneTermination()        # primary stop
        | MaxTurnsTermination(20)        # soft cap
        | TokenBudgetTermination(80_000) # cost cap
    ),
    config=SwarmConfig(
        max_handoffs=10,
        max_total_tokens=80_000,
        shared_context=SharedContextConfig(
            strategy=SharedContextStrategy.SCOPED,   # default, but stated for clarity
        ),
    ),
)

result = await (
    Runner.configure()
    .with_config(RunConfig(max_total_turns=50))   # absolute safety net
    .swarm(swarm)
    .arun("Refactor this module.")
)
```

## `SharedContextStrategy` — the biggest lever

| Strategy | What each agent sees | Typical use |
|----------|----------------------|-------------|
| `SCOPED` (default) | Its own scratch + the explicit handoff message | Production default — no hidden broadcast |
| `LAST_N` | Last N items of the shared history | When agents need a rolling view but not full history |
| `SUMMARIZED` | Compacted summary + preserved recent items | Long runs where the full trail matters |
| `FULL_BROADCAST` | Every item every agent ever produced | AutoGen parity — debugging only, not production |

Guideline: if you find yourself paying for "the other agents' tool
results" every turn, you're probably on `FULL_BROADCAST` when you
meant `SCOPED`.

## Where Each Lever Wins

### `FunctionTool.max_result_tokens`
A noisy tool (e.g. `get_full_order_history`) will bloat every
downstream replay. Cap the tool output, not the conversation.

### `HandoffConfig.budget`
When agent A hands off to B, you can trim what B inherits. Reuses the
existing handoff budgeting pipeline — the swarm driver calls the same
`prepare_handoff_input` path.

### `SwarmConfig.max_handoffs`
Policy-independent switch cap. Distinct from `max_total_turns` because
a single agent can take many turns between switches.

### `SwarmConfig.max_total_tokens`
Cumulative across the whole swarm run. Checked at the top of each
turn; stops cleanly with `StopReason(kind="max_total_tokens")`.

### `SharedContextStrategy`
The cheapest lever of all because it's per-turn. Moving from
`FULL_BROADCAST` to `SCOPED` can cut replay cost by an order of
magnitude on long runs.

## How the Levers Interact

Think of the run as a nested envelope:

```
RunConfig.max_total_turns      (absolute safety net — never skip)
└── SwarmConfig.max_total_tokens  (cost envelope)
    └── SwarmConfig.max_handoffs  (switch envelope)
        └── TerminationCondition  (explicit stop)
            └── Per-turn:
                ├── SharedContextStrategy (input size)
                ├── HandoffConfig.budget   (handoff carry-over)
                └── FunctionTool.max_result_tokens (tool results)
```

A healthy production swarm trips `ExplicitDoneTermination` first —
every other layer is a safety rail, not the expected exit.

> **Default safety net:** when you don't pass `termination`, a swarm
> gets `DEFAULT_TERMINATION` — `ExplicitDoneTermination() |
> MaxTurnsTermination(25)`. The explicit-done contract is unchanged;
> the 25-turn cap only bounds runs whose members never call
> `swarm_done`. Pass your own `termination=` to tune it.

## Debugging High-Cost Runs

1. Inspect `result.state.cumulative_usage` (or `result.per_member_usage`
   for the per-agent breakdown) — which agent burned tokens?
2. Inspect `result.handoff_count` — too many switches usually means
   the policy isn't converging.
3. If `FULL_BROADCAST` or `LAST_N`, try `SCOPED` and see if quality
   actually drops.
4. If a single agent's turn is huge, check which tool's output is
   dominating — cap it.
5. If tokens explode late in the run, add `TokenBudgetTermination`
   with a soft cap before `MaxTurnsTermination`.

## See Also

- `docs/swarms/swarms.md` — overview and when to use a swarm
- `docs/swarms/policies.md` — how each policy affects cost
