# Swarms

Iterative multi-agent collaboration with cycles, explicit termination,
and explicit framework boundaries.

## What a Swarm Is

A `Swarm` is a **configuration object** binding four concerns:

| Concern | Field | Type |
|---------|-------|------|
| Roster | `members`, `entry` | `tuple[Agent, ...]`, `Agent` or member name |
| Routing | `policy` | `SwarmPolicy` (default: `LLMHandoffPolicy`) |
| Stopping | `termination` | `TerminationCondition` (default: `DEFAULT_TERMINATION`) |
| Budgets | `config`, `hooks` | `SwarmConfig`, `Optional[SwarmHooks]` |
| Metadata | `name`, `description`, `handoff_descriptions` | optional |

Like `Agent`, a `Swarm` is immutable (`@dataclass(frozen=True)`). It has
**no `run()` method** — execution lives in `Runner.arun_swarm()`.

## Defining a Swarm

The fluent builder (mirrors `Graph.new(...)`) is the preferred,
readability-first surface — roster, entry, routing, and stopping each
get one line:

```python
from troopai.adk.agents.agent import Agent
from troopai.adk.run.runner import Runner
from troopai.adk.swarms import (
    Swarm, ExplicitDoneTermination, MaxTurnsTermination, TokenBudgetTermination,
)

author = Agent(name="author", system_prompt="You write code.")
reviewer = Agent(name="reviewer", system_prompt="You review code.")
auditor = Agent(name="security_auditor", system_prompt="You audit code for security.")

swarm = (
    Swarm.new("code-review", description="author → reviewer → security auditor")
    .members(author, reviewer, auditor)
    .entry("author")                      # member name or Agent object
    .llm_handoff()                        # .round_robin(), .routed(route), .custom_policy(fn), .policy(p)
    .terminate_on(
        ExplicitDoneTermination()
        | MaxTurnsTermination(20)
        | TokenBudgetTermination(100_000)
    )
    .compile()                            # validates everything up front
)

result = await Runner.arun_swarm(swarm, "Refactor this module.")
print(result)                  # SwarmRunResult(swarm='code-review', stop='explicit_done', turns=8, handoffs=3, tokens=12400, final_output='...')
print(result.final_output)
print(result.stop_reason)      # e.g. StopReason(kind="explicit_done", detail="approved")
print(result.handoff_count)    # number of agent switches
```

`.compile()` fails fast: unknown entry names, duplicate members, and
roster-aware termination errors all surface as `ValueError` at build
time, never mid-run.

### Defaults for the common case

Direct construction works too, and the two most common choices have
defaults, so the minimal useful swarm is one line:

```python
swarm = Swarm(members=(author, reviewer), entry="author")
```

- `policy` defaults to `LLMHandoffPolicy()` — the LLM picks the next
  speaker via injected `transfer_to_<member>` tools.
- `termination` defaults to `DEFAULT_TERMINATION`
  (`ExplicitDoneTermination() | MaxTurnsTermination(25)`): a member must
  still call `swarm_done` for a clean stop; the 25-turn cap is only a
  cost-conservative safety net.
- A single-member builder swarm can omit `.entry(...)` entirely.

### Per-member routing hints

`handoff_descriptions` (builder: `.member(agent, handoff_description=...)`)
tells the routing LLM *when* to pick each member — it becomes the
`transfer_to_<member>` tool description (mirrors the OpenAI Agents SDK
`handoff_description`):

```python
swarm = (
    Swarm.new("support")
    .member(triage, handoff_description="Start here; classifies the request.")
    .member(refunds, handoff_description="Refunds, returns, and order cancellations.")
    .member(billing, handoff_description="Invoices, charges, and payment methods.")
    .entry("triage")
    .llm_handoff()
    .compile()
)
```

### Phrase-based stopping

`TextMentionTermination` (borrowed from AutoGen) stops the swarm when an
agent's message contains a phrase — useful for flows where `swarm_done`
is overkill. Only agent messages are scanned, never user input, and an
optional `member=` restricts who can trigger it (validated against the
roster at construction time). Always compose it with a safety net:

```python
from troopai.adk.swarms import TextMentionTermination, MaxTurnsTermination

termination = TextMentionTermination("VERDICT:", member="judge") | MaxTurnsTermination(20)
```

Explicit `swarm_done` remains the recommended primary stop signal.

## When to Use a Swarm

| You need… | Use |
|-----------|-----|
| Agent A hands off to B, run ends when B finishes | `Handoff` (existing) |
| Agent A delegates a sub-task to B and resumes with the answer | `Agent.as_tool()` (existing) |
| **Agent A ↔ B ↔ C cycling until an explicit stop** | **`Swarm` (this module)** |
| Fan out to N agents in parallel and join | `asyncio.gather` over `Runner.arun(...)` |

If you catch yourself writing "run the reviewer, then decide whether to
go back to the author, then maybe loop to the auditor" — that's a swarm.

## Streaming

`Runner.arun_swarm_streamed(...)` returns a `SwarmRunResultStreaming`.
Its `stream_events()` iterator emits swarm lifecycle events around the
same per-agent stream events used by regular streamed runs.

```python
from troopai.adk.swarms import SwarmTurnStartEvent, SwarmHandoffEvent, SwarmDoneEvent

streamed = await Runner.arun_swarm_streamed(swarm, "Refactor this module.")
async for event in streamed.stream_events():
    match event:
        case SwarmTurnStartEvent(agent=name, turn=t):
            logger.info("[turn %d] %s speaking", t, name)
        case SwarmHandoffEvent(from_agent=a, to_agent=b, message=m):
            logger.info("[handoff] %s -> %s: %s", a, b, m)
        case SwarmDoneEvent(reason=r):
            logger.info("[done] %s", r)
```

Per-agent stream events (`raw_response_event`,
`run_item_stream_event`, `agent_updated_stream_event`) continue to
flow unchanged. Swarm events bracket them so consumers can
pattern-match by type.

## Profile Runner API

```python
result = await (
    Runner.configure(context=my_context)
    .swarm(swarm)
    .hooks(my_run_hooks)
    .session(session)
    .arun("Refactor this module.")
)
```

## Decision Tree

```
Does control need to come back to the same agent later in the run?
├── No, one-shot: use Handoff
├── Yes, but only as a subroutine (A calls B, resumes with B's answer): use Agent.as_tool()
└── Yes, cyclically until a decision point: use Swarm
     |
     What picks the next agent?
     ├── The LLM, via natural tool calls: LLMHandoffPolicy
     ├── A fixed rotation (debates, pipelines, tests): RoundRobinPolicy
     ├── A structured Intent classifier (type-safe triage): StructuredRoutingPolicy
     └── Your own callable: CustomPolicy
```

## No Hidden Behavior

The framework does **not** inject system prompts, preambles, or
cross-agent broadcasts without your opt-in. Specifically:

- System prompts on member agents are used verbatim. If you want
  members to know they're in a swarm, call
  `prompt_with_swarm_instructions(my_prompt)` explicitly.
- Shared context defaults to `SCOPED` — each agent sees only its own
  scratch + the explicit handoff message payload.
- Termination is always explicit: the LLM must call `swarm_done(reason=...)`,
  a termination condition must fire (`MaxTurnsTermination`,
  `TokenBudgetTermination`, …), or a hard guard must trip
  (`SwarmConfig.max_handoffs`, `RunConfig.max_total_turns`, …).

## Related

- `docs/swarms/policies.md` — deep dive on each policy with worked examples
- `docs/swarms/cost_optimization.md` — the five cost levers and how they interact
- `examples/swarms/llm_handoff_swarm.py` — end-to-end code-review swarm
- `examples/swarms/round_robin_debate.py` — deterministic two-agent debate
- `examples/swarms/structured_routing_swarm.py` — triage via `HandoffRoute`
