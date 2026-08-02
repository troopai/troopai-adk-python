"""Verbose output — Live token-by-token streaming.

Demonstrates the CrewAI-faithful streaming Live widget. When using
``Runner.run(..., stream=True)`` (or ``Runner.arun``), the panel
backend opens a ``rich.live.Live`` panel that grows in place as the
LLM streams tokens:

* Text completions render in the green ``✅ Agent Final Answer`` panel.
* Tool-call argument deltas (JSON streaming during a function call)
  render in the yellow ``🔧 Tool Arguments`` panel.

After the stream closes, the renderer suppresses the duplicate
``agent.finish`` block panel (the Live already painted the answer).

Try it in an interactive TTY for the full effect:

    python examples/verbose/streaming.py

In CI / non-TTY environments, ``mode="auto"`` downgrades to the line
backend and emits ``stream.start`` / ``stream.end`` marker lines
instead of a Live widget.

See also:

- ``examples/agent_patterns/human_in_the_loop_stream.py`` for
  streaming with HITL.
- ``examples/tools/streaming_tools.py`` for streaming with tool
  guardrails.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk import Agent, RunConfig, Runner, VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    agent = Agent(
        name="StoryTeller",
        llm="gpt-4o-mini",
        system_prompt="Write a brief, three-sentence science fiction story.",
    )

    verbose_cfg = VerboseConfig(mode="auto")
    result = Runner.run(
        agent,
        "Tell me a story about a sentient coffee machine.",
        stream=True,
        run_config=RunConfig(verbose=verbose_cfg),
    )

    # Drain the stream. The verbose layer handles the stream.start /
    # stream.end panels automatically; we just consume events to let
    # the underlying coroutine progress.
    async for _event in result.stream_events():
        pass

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
