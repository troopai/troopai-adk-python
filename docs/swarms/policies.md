# Swarm Policies

A `SwarmPolicy` answers two questions each turn:

1. **Who speaks next?** — `select_next(state, context) -> Agent`
2. **What tools does the speaker see this turn?** — `build_step_tools(state) -> list[FunctionTool]`

Tools returned by `build_step_tools` are **merged** with the active
agent's own tools at turn-dispatch time. Agent config is never mutated
— a swarm member stays unit-testable in isolation.

## `LLMHandoffPolicy`

The LLM picks the next agent by calling an injected
`transfer_to_<name>(message=...)` tool. All members except the current
speaker get a `transfer_to_<name>` tool, plus every speaker gets
`swarm_done(reason=...)`.

```python
from troopai.adk.swarms import LLMHandoffPolicy, Swarm, ExplicitDoneTermination, MaxTurnsTermination

swarm = Swarm(
    members=(author, reviewer, auditor),
    entry="author",
    policy=LLMHandoffPolicy(),
    termination=ExplicitDoneTermination() | MaxTurnsTermination(20),
)
```

### Routing hints via `handoff_descriptions`

By default each `transfer_to_<name>` tool carries a generic
description. Give the routing LLM *when* to pick each member by
attaching per-member descriptions (builder form shown; the same mapping
is accepted as `Swarm(handoff_descriptions={...})`):

```python
swarm = (
    Swarm.new("code-review")
    .member(author)
    .member(reviewer, handoff_description="Code review: correctness, style, maintainability.")
    .member(auditor, handoff_description="Security audit: injection, secrets, auth flaws.")
    .entry("author")
    .llm_handoff()
    .compile()
)
```

Keys are validated against the roster at construction time — a typo'd
member name is a `ValueError`, not a silently-ignored hint.

When to use:
- Open-ended collaboration where the LLM's judgment should pick the
  next role.
- You want AutoGen/Strands Swarm-style semantics with explicit
  termination and no broadcast-all context explosion.

Cost profile:
- Pays LLM tokens for routing (tool call + arguments).
- Cheapest when paired with `SharedContextStrategy.SCOPED` so each
  turn only replays the agent's own scratch.

## `RoundRobinPolicy`

Deterministic rotation. Ships zero LLM routing tokens. `swarm_done`
is still available, so any agent can stop the swarm explicitly.

```python
from troopai.adk.swarms import RoundRobinPolicy

# Rotate in the declaration order of members:
policy = RoundRobinPolicy()

# Or override:
policy = RoundRobinPolicy(order=("critic", "author", "editor"))
```

When to use:
- Fixed-order pipelines.
- Debates where you want speakers to alternate strictly.
- Deterministic test fixtures.

Gotcha: `order` must reference member names. A bad name raises
`ValueError` at `select_next` time with a clear message.

## `StructuredRoutingPolicy` — the differentiator

Routes based on the active agent's **structured output**. The agent
must have an `output_schema` that yields an `Intent` (or `Respond`, meaning
"I'm answering directly, no routing"). The policy dispatches via an
existing `HandoffRoute.when(IntentType).to(agent)` DSL.

```python
from troopai.adk.handoffs import HandoffRoute
from troopai.adk.swarms import StructuredRoutingPolicy
from troopai.adk.types.intents import Intent
from typing import Literal

class RefundIntent(Intent):
    kind: Literal["refund"] = "refund"
    order_id: str

class OrderIntent(Intent):
    kind: Literal["order"] = "order"
    sku: str

route = (
    HandoffRoute("support")
    .when(RefundIntent).to(refunds_agent)
    .when(OrderIntent).to(orders_agent)
    .otherwise(triage)          # Respond -> stay with triage
)

policy = StructuredRoutingPolicy(route=route)
```

Cost profile:
- **Zero routing tokens.** The agent already had to pick a structured
  output; we reuse it.
- Strongly typed end-to-end.

When to use:
- Triage ↔ specialist flows.
- Any routing that's a classification problem, not a reasoning problem.

## `CustomPolicy`

Plain escape hatch. You supply a selector; you optionally supply a
function that contributes tools per turn.

```python
from troopai.adk.swarms import CustomPolicy, SwarmState

def pick(state: SwarmState) -> str:
    # e.g. pick based on state.total_turns, last yield, or your own context
    return "author" if state.total_turns % 2 == 0 else "reviewer"

policy = CustomPolicy(selector=pick)
```

Signature of `extra_tools_fn`:

```python
def build_tools(state: SwarmState) -> list[FunctionTool]: ...
policy = CustomPolicy(selector=pick, extra_tools_fn=build_tools)
```

Rule of thumb: if you find yourself writing an LLM call inside the
selector, you probably want `LLMHandoffPolicy` or
`StructuredRoutingPolicy` instead.

## Choosing a Policy

| Question | Policy |
|----------|--------|
| Do I want an LLM to decide routing based on natural language? | `LLMHandoffPolicy` |
| Do I have a fixed order or a classic round-robin debate? | `RoundRobinPolicy` |
| Does my routing decision have a finite, typed set of outcomes? | `StructuredRoutingPolicy` |
| None of the above fits? | `CustomPolicy` |

## Writing Your Own Policy

Subclass `SwarmPolicy`:

```python
from troopai.adk.swarms import SwarmPolicy, SwarmState, SwarmYieldSignal
from troopai.adk.tools import FunctionTool
from troopai.adk.agents.agent import Agent

class MyPolicy(SwarmPolicy[None]):
    async def select_next(self, state: SwarmState, context: object) -> Agent:
        ...

    def build_step_tools(self, state: SwarmState) -> list[FunctionTool]:
        return []

    def record_yield(self, state: SwarmState, signal: SwarmYieldSignal) -> None:
        # default no-op; override if you keep internal routing state
        return None
```

Contracts:

- `select_next` MUST return a member agent. Returning `state.current_agent`
  is fine — that's how policies handle "no handoff requested".
- `build_step_tools` MUST NOT return tools that share names with the
  active agent's own tools (name collision). It is the policy's job to
  namespace whatever it injects.
- `record_yield` is called with the most recent `SwarmHandoff` or
  `SwarmDone` — you can mutate internal state here, but not the swarm
  config.

See `tests/unit/swarms/test_policies.py` for behaviour under mocked states.
