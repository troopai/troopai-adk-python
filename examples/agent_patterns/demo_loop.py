"""
Example: Interactive REPL with ``run_demo_loop``

Demonstrates the one-liner REPL helper for agents. ``run_demo_loop``
reads user input from stdin, forwards each turn to the Runner with
full streaming, and preserves conversation history across turns via
``RunResult.to_input_list()``.

Exit the loop by typing ``exit`` / ``quit``, pressing Ctrl-D (EOF), or
Ctrl-C (KeyboardInterrupt).

Run:
    python examples/agent_patterns/demo_loop.py

Under the batch runner (or with ``TROOPAI_EXAMPLES_INTERACTIVE_MODE=auto``)
there is no human at the terminal, so the example runs a single scripted
turn instead of the stdin REPL — still exercising a real agent turn.

The example uses the default model, so set ``ANTHROPIC_API_KEY`` in the
environment or in a ``.env`` file alongside the script.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from troopai.adk.agents.agent import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.run.demo import run_demo_loop
from troopai.adk.verbose import VerboseConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_mode import is_auto_mode

logger = logging.getLogger(__name__)


async def main() -> None:
    agent = Agent(
        name="demo_assistant",
        system_prompt=(
            "You are a concise and helpful assistant. "
            "Keep answers to one or two sentences unless the user asks for more."
        ),
    )

    if is_auto_mode():
        # No human at the terminal: run one scripted turn so the example
        # still demonstrates a real agent turn end-to-end.
        question = "In one sentence, what is an agent loop?"
        logger.info("[auto] scripted prompt: %s", question)
        result = await Runner.arun(agent, question, run_config=RunConfig(verbose=VerboseConfig()))
        logger.info("%s", result.final_output)
        return

    logger.info("Interactive demo loop. Type 'exit' or press Ctrl-D to quit.")
    await run_demo_loop(agent, stream=True)


if __name__ == "__main__":
    asyncio.run(main())
