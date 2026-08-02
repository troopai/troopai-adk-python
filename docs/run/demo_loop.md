# `run_demo_loop` — Interactive REPL for Agents

A tiny terminal REPL for agents. Useful for smoke-testing, demos, and local
exploration without writing a wrapper script.

## Signature

```python
async def run_demo_loop(
    agent: Agent[Any],
    *,
    stream: bool = True,
    context: Optional[Any] = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    run_config: Optional[RunConfig] = None,
    hooks: Optional[RunHooks[Any]] = None,
) -> None
```

All parameters are forwarded to `Runner.arun()`. The loop preserves
conversation history across turns by calling `RunResult.to_input_list()`
and feeding the result back in as the next user prompt.

## Behavior

- Reads input from stdin with the prompt `> `
- Streams tokens live when `stream=True` (default)
- Exits cleanly on `exit`, `quit` (case-insensitive), `EOF` (Ctrl-D), or `SIGINT` (Ctrl-C)
- Skips empty input without making an LLM call
- Tracks handoffs — the currently active agent after the last turn becomes the prompt for the next turn

## Usage

```python
import asyncio
from troopai.adk.agents.agent import Agent
from troopai.adk.run.demo import run_demo_loop

agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant. Keep answers concise.",
)

asyncio.run(run_demo_loop(agent))
```

Then, in the terminal:

```
> What's the capital of France?
Paris.
> And of Portugal?
Lisbon.
> exit
```

## Non-streaming mode

Passing `stream=False` buffers each turn's output and prints it once the
response is complete — handy for structured-output agents whose intermediate
tokens aren't meaningful text.

```python
asyncio.run(run_demo_loop(agent, stream=False))
```

## See also

- `examples/agent_patterns/demo_loop.py` — runnable example
- `src/troopai/adk/run/demo.py` — implementation
- `tests/unit/run/test_demo_loop.py` — behavior tests
