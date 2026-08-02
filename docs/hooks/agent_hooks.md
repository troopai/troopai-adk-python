# Agent Hooks

`AgentHooks` attach per-agent lifecycle callbacks to a specific `Agent` via the `hooks=` field. They fire alongside run-level `RunHooks` but are scoped to a single agent instance — the load-bearing use case is multi-agent swarms where each agent wants its own observability or side effects without the run-level hooks accumulating per-agent conditionals.

## When to use which

| Use `RunHooks` when... | Use `AgentHooks` when... |
|------------------------|--------------------------|
| Observing a whole run (any agent) | Observing one specific agent |
| Logging guardrails, skills, sessions | Per-agent metrics / tracing |
| Implementing global policies | Per-agent side effects on handoff |
| You pass hooks through `Runner.arun(hooks=...)` | You attach hooks on `Agent(hooks=...)` |

Both can be active at the same time: `RunHooks` fires first, then `AgentHooks` fires immediately after for the matching lifecycle event.

## Method surface

```python
from troopai.adk.hooks.hooks import AgentHooks

class MyAgentHooks(AgentHooks):
    async def on_start(self, context, agent) -> None: ...
    async def on_end(self, context, agent, output) -> None: ...
    async def on_handoff(self, context, agent, source) -> None: ...
    async def on_llm_start(self, context, agent, messages) -> None: ...
    async def on_llm_end(self, context, agent, response) -> None: ...
    async def on_tool_start(self, context, agent, tool_name, tool_input) -> None: ...
    async def on_tool_end(self, context, agent, tool_name, tool_output) -> None: ...
```

Every method is async and optional — override only what you need. `AgentHooks` intentionally does **not** include guardrail, skill, or session hooks: those are run-level concerns. If you need them, use `RunHooks` instead.

### Handoff semantics

`AgentHooks.on_handoff` fires on the **incoming** agent (the one being handed off TO), with `source` as the outgoing agent. This is the moment the new agent becomes active. The outgoing agent's `on_handoff` is **not** called — use `RunHooks.on_handoff` if you need to observe both sides.

```python
specialist = Agent(name="Specialist", system_prompt="...", hooks=MyAgentHooks())
dispatcher = Agent(name="Dispatcher", system_prompt="...", handoffs=[specialist])

# When dispatcher hands off to specialist:
#   MyAgentHooks.on_handoff(ctx, agent=specialist, source=dispatcher)
```

## Firing order

Per-agent hooks fire **immediately after** the matching run-level hook, so observers see consistent ordering:

```
RunHooks.on_agent_start  ─►  AgentHooks.on_start
RunHooks.on_llm_start    ─►  AgentHooks.on_llm_start
  (LLM call)
RunHooks.on_llm_end      ─►  AgentHooks.on_llm_end
RunHooks.on_tool_start   ─►  AgentHooks.on_tool_start
  (tool execution)
RunHooks.on_tool_end     ─►  AgentHooks.on_tool_end
RunHooks.on_handoff      ─►  AgentHooks.on_handoff   (on the incoming agent)
RunHooks.on_agent_end    ─►  AgentHooks.on_end
```

## Example: per-agent metrics

```python
import logging
from troopai.adk.agents import Agent
from troopai.adk.hooks.hooks import AgentHooks
from troopai.adk.run import Runner

logger = logging.getLogger(__name__)

class MetricsHooks(AgentHooks):
    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_calls = 0

    async def on_llm_start(self, context, agent, messages) -> None:
        del context, messages
        self.llm_calls += 1
        logger.info("agent=%s llm_call_count=%d", agent.name, self.llm_calls)

    async def on_tool_end(self, context, agent, tool_name, tool_output) -> None:
        del context, tool_output
        self.tool_calls += 1
        logger.info("agent=%s tool=%s tool_call_count=%d", agent.name, tool_name, self.tool_calls)

metrics = MetricsHooks()
agent = Agent(name="Assistant", system_prompt="You are helpful.", hooks=metrics)

result = await Runner.arun(agent, "Hello")
logger.info("Final metrics: llm_calls=%d tool_calls=%d", metrics.llm_calls, metrics.tool_calls)
```

## Multi-agent observability

Attach different `AgentHooks` instances to each agent in a swarm to isolate metrics per agent:

```python
router_metrics = MetricsHooks()
specialist_metrics = MetricsHooks()

router = Agent(name="Router", system_prompt="Route to the right specialist.",
               hooks=router_metrics)
specialist = Agent(name="Specialist", system_prompt="Handle the request.",
                   hooks=specialist_metrics)
router.handoffs = [specialist]

await Runner.arun(router, "Fix my billing issue.")

# Each instance holds only its own agent's counters.
logger.info("router llm=%d  specialist llm=%d",
            router_metrics.llm_calls, specialist_metrics.llm_calls)
```

See `examples/agent_patterns/agent_hooks.py` for a runnable version.
