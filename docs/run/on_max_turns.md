# `RunConfig.on_max_turns` — Salvage Handler for Exhausted Budgets

A recovery hook that fires when an agent runs out of its per-agent turn
budget. Lets you salvage a best-effort final output instead of raising
`MaxTurnsExceeded` — useful for user-facing chat flows where "I couldn't
finish, here's what I have" is a better UX than a traceback.

## Signature

```python
OnMaxTurnsHandler = Callable[[Agent[Any], int], Awaitable[Optional[str]]]
```

The handler receives:

- The agent whose budget was exhausted
- The turn count at exhaustion

It returns either:

- `str` — used as `RunResult.final_output`; the run completes normally
- `None` — falls through to the default behavior, raising `MaxTurnsExceeded`

## Scope

**Per-agent `max_turns` only.** The cross-agent `RunConfig.max_total_turns`
swarm safety limit always raises without routing through the handler — it
represents a runaway workflow, not a recoverable per-agent budget issue.

## Usage

```python
from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner


async def salvage(agent, turns: int) -> str:
    return (
        f"[partial response from {agent.name} after {turns} turns — "
        "budget exhausted]"
    )


config = RunConfig(on_max_turns=salvage)
result = await Runner.arun(
    agent,
    "Long research task...",
    max_turns=5,
    run_config=config,
)
assert result.final_output.startswith("[partial response")
```

## Streaming parity

Works identically in streamed and non-streamed runs. The handler is invoked
from inside the agent loop, so `RunResultStreaming.final_output` is populated
after all buffered events have been drained.

## Interaction with deferred tools

If the run is interrupted for HITL approval before the turn budget is
exhausted, the handler does **not** fire — the run returns normally with
`requires_action=True`, and the handler only runs when resumption hits the
per-agent cap.

## See also

- `examples/agent_patterns/on_max_turns.py` — runnable example
- `src/troopai/adk/run/config.py` — `OnMaxTurnsHandler` type alias + field
- `tests/unit/run/test_on_max_turns.py` — non-streaming + streaming coverage
