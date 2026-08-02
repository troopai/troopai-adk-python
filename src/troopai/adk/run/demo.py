"""Interactive REPL loop for quick agent testing and debugging.

``run_demo_loop()`` is a lightweight read-eval-print loop that runs an
agent against user input from stdin, preserves conversation state
across turns, and ends when the user types ``exit`` / ``quit`` or sends
``EOF``/``SIGINT``.

It is intended for manual smoke testing and example demos, not for
production chat interfaces — production chat UIs should drive
``Runner.arun()`` directly and own their own rendering.

All output goes through the project logger; configure the logger's
console handler at ``INFO`` or lower to see the conversation.

Example::

    import asyncio
    import logging
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.demo import run_demo_loop

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = Agent(name="demo", system_prompt="You are a helpful assistant.")
    asyncio.run(run_demo_loop(agent))

Reference: OpenAI's ``run_demo_loop`` in
``openai-agents-python/src/agents/repl.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from troopai.adk.run.config import DEFAULT_MAX_TURNS, RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.run.stream import (
    AgentUpdatedStreamEvent,
    RawResponseStreamEvent,
    RunItemStreamEvent,
    RunItemType,
)
from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)


async def run_demo_loop(
    agent: Agent[Any],
    *,
    stream: bool = True,
    context: Any | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    run_config: RunConfig | None = None,
    hooks: RunHooks[Any] | None = None,
) -> None:
    """Run a simple REPL loop against the given agent.

    Preserves conversation state across turns by feeding each turn's
    ``to_input_list()`` back as the next turn's input. Exits cleanly on
    ``exit``/``quit``, ``EOF`` (``Ctrl-D``), or ``KeyboardInterrupt``
    (``Ctrl-C``).

    Args:
        agent: The starting agent. After a handoff, the loop tracks the
            new agent and drives it for subsequent turns.
        stream: If ``True``, runs in streaming mode and logs deltas,
            tool calls, and handoffs as they happen. If ``False``, logs
            only the final output per turn.
        context: Optional user context threaded into the run.
        max_turns: Per-turn maximum iterations inside the agent loop.
        run_config: Optional ``RunConfig`` passed to ``Runner.arun``.
        hooks: Optional lifecycle hooks.

    Output goes through ``logger.info()``; configure a console handler
    on the ``troopai.adk.run.demo`` logger (or the root logger) at
    ``INFO`` level to see the conversation.
    """
    current_agent = agent
    input_items: list[LLMInputContentItem] = []

    while True:
        try:
            user_input = input(" > ")
        except (EOFError, KeyboardInterrupt):
            logger.info("")
            break

        trimmed = user_input.strip()
        if trimmed.lower() in {"exit", "quit"}:
            break
        if len(trimmed) == 0:
            continue

        user_msg: LLMInputEasyMessage = {"role": "user", "content": user_input}
        input_items.append(user_msg)

        if stream:
            streaming = await Runner.arun(
                current_agent,
                input_items,
                stream=True,
                context=context,
                max_turns=max_turns,
                run_config=run_config,
                hooks=hooks,
            )
            buffer: list[str] = []
            async for event in streaming.stream_events():
                if isinstance(event, RawResponseStreamEvent):
                    if isinstance(event.data, str):
                        buffer.append(event.data)
                elif isinstance(event, RunItemStreamEvent):
                    if event.name == RunItemType.TOOL_CALLED:
                        if len(buffer) > 0:
                            logger.info("".join(buffer))
                            buffer = []
                        logger.info("[tool called]")
                    elif event.name == RunItemType.TOOL_OUTPUT:
                        logger.info("[tool output: %s]", event.item)
                elif isinstance(event, AgentUpdatedStreamEvent):
                    if len(buffer) > 0:
                        logger.info("".join(buffer))
                        buffer = []
                    logger.info("[Agent updated: %s]", event.new_agent.name)
            if len(buffer) > 0:
                logger.info("".join(buffer))
            # RunResultStreaming tracks the live agent as ``current_agent``
            # (updated on every handoff); RunResult uses ``last_agent``.
            next_agent: Optional[Agent[Any]] = streaming.current_agent
            input_items = streaming.to_input_list()
        else:
            result = await Runner.arun(
                current_agent,
                input_items,
                stream=False,
                context=context,
                max_turns=max_turns,
                run_config=run_config,
                hooks=hooks,
            )
            if result.final_output is not None:
                logger.info("%s", result.final_output)
            next_agent = result.last_agent
            input_items = result.to_input_list()

        # ``last_agent`` is ``None`` only for pathological runs that
        # never started — keep the current agent in that case.
        if next_agent is not None:
            current_agent = next_agent
