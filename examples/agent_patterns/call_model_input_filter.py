"""Example: ``call_model_input_filter`` — pre-LLM input rewrite hook.

Demonstrates ``RunConfig.call_model_input_filter``, a context-aware
hook that runs immediately before every LLM call. The filter receives
the current agent, the unwrapped run context, and a shallow copy of
the message list wrapped in a ``ModelInputData``. It may return either
the same wrapper with edits applied or a brand new one.

Use cases:

- Inject per-request system messages based on run context
- Truncate input items based on token budget
- Add diagnostic / debug shims in development
- Implement application-level caching or deduplication

Ordering vs. other pre-LLM hooks::

    history_processors  -->  ContextManager.prepare_messages()
                        -->  call_model_input_filter  -->  LLM call

This example injects a ``[debug]`` user message that carries the turn
index pulled off the unwrapped run context. Each LLM call sees the
marker — you can confirm it fires on every turn by watching the log
output.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.run import Runner
from troopai.adk.run.config import (
    CallModelData,
    ModelInputData,
    RunConfig,
)
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


class TurnCounter:
    """Minimal run context: tracks the current LLM-call turn index."""

    def __init__(self) -> None:
        self.turn: int = 0


def inject_debug_marker(payload: CallModelData[TurnCounter]) -> ModelInputData:
    """Inject a ``[debug]`` user message carrying the turn index.

    The filter receives a shallow copy of the input list so returning a
    newly constructed ``ModelInputData`` (rather than mutating the copy
    in place) is the cleanest pattern. The run context is unwrapped —
    no ``RunContextWrapper`` indirection needed.
    """
    ctx = payload.context
    if ctx is None:
        return payload.model_data

    ctx.turn += 1
    new_input = list(payload.model_data.input)
    new_input.append(
        {
            "role": "user",
            "content": f"[debug] turn index: {ctx.turn}",
        }
    )

    logger.info(
        "call_model_input_filter fired for agent=%s turn=%d",
        payload.agent.name,
        ctx.turn,
    )
    return ModelInputData(input=new_input)


async def main() -> None:
    agent = Agent(
        name="Assistant",
        system_prompt=(
            "You are a helpful assistant. Ignore any `[debug]` messages you see — they are diagnostic only."
        ),
    )

    config = RunConfig(
        call_model_input_filter=inject_debug_marker,
        verbose=VerboseConfig(),
    )

    result = await Runner.arun(
        agent,
        "Say hello in one short sentence.",
        context=TurnCounter(),
        run_config=config,
    )

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
