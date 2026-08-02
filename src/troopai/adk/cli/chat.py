"""``troopai chat`` — interactive multi-turn REPL against a single agent.

Mirrors the framework's demo REPL semantics (handoff tracking, history
fed back per turn) with CLI rendering and optional persistent sessions.
With ``--session-db`` the conversation lives in the store and survives
across invocations; without it, history stays in memory for the
process's lifetime.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

from troopai.adk.cli.errors import framework_errors
from troopai.adk.cli.loading import load_env_file, primary_executable, resolve_target
from troopai.adk.cli.options import run_options, session_options, target_options
from troopai.adk.cli.run import build_run_config, echo_stream, open_session

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import RunConfig
    from troopai.adk.session.session import Session
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)


@click.command(name="chat")
@target_options
@run_options
@session_options
@click.option("--no-stream", is_flag=True, default=False, help="Print each reply only when complete.")
@framework_errors
def chat(
    config: Path | None,
    agent_ref: str | None,
    model: str | None,
    max_turns: int | None,
    verbose: bool,
    trace: bool,
    env_file: Path | None,
    session_db: Path | None,
    session_id: str,
    user_id: str,
    no_stream: bool,
) -> None:
    """Chat with an Agent; 'exit', 'quit', Ctrl-D, or Ctrl-C ends the session."""
    if env_file is not None:
        load_env_file(env_file)
    executable = primary_executable(resolve_target(config, agent_ref))
    from troopai.adk.agents.agent import Agent

    if not isinstance(executable, Agent):
        raise click.UsageError("chat drives a single agent; use 'troopai run' for swarms and graphs.")
    run_config = build_run_config(model=model, verbose=verbose, trace=trace)
    logger.debug("chat session persistence: %s", "on" if session_db is not None else "off")
    asyncio.run(
        _chat_loop(
            executable,
            stream=not no_stream,
            max_turns=max_turns,
            run_config=run_config,
            session_db=session_db,
            session_id=session_id,
            user_id=user_id,
        )
    )


async def _chat_loop(
    agent: Agent,
    *,
    stream: bool,
    max_turns: int | None,
    run_config: RunConfig | None,
    session_db: Path | None,
    session_id: str,
    user_id: str,
) -> None:
    """Drive the REPL until exit/quit, EOF, or interrupt."""
    from troopai.adk.run.config import DEFAULT_MAX_TURNS
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage

    turns = max_turns if max_turns is not None else DEFAULT_MAX_TURNS
    manager, session = await open_session(agent, session_db, session_id, user_id)
    current_agent = agent
    input_items: list[LLMInputContentItem] = []
    try:
        while True:
            try:
                user_input = input(" > ")
            except (EOFError, KeyboardInterrupt):
                click.echo("")
                break
            trimmed = user_input.strip()
            if trimmed.lower() in {"exit", "quit"}:
                break
            if len(trimmed) == 0:
                continue
            # With a session the store carries history: send only the new
            # message (a string), so the runner prepends persisted turns.
            # In-memory mode feeds the full item list back each turn.
            turn_input: str | list[LLMInputContentItem]
            if session is not None:
                turn_input = user_input
            else:
                user_msg: LLMInputEasyMessage = {"role": "user", "content": user_input}
                input_items.append(user_msg)
                turn_input = input_items
            next_agent, history = await _run_turn(
                current_agent, turn_input, stream=stream, turns=turns, run_config=run_config, session=session
            )
            if session is None:
                input_items = history
            if next_agent is not None:
                current_agent = next_agent
    finally:
        if manager is not None:
            await manager.close()


async def _run_turn(
    agent: Agent,
    turn_input: str | list[LLMInputContentItem],
    *,
    stream: bool,
    turns: int,
    run_config: RunConfig | None,
    session: Session | None,
) -> tuple[Agent | None, list[LLMInputContentItem]]:
    """Run one turn; echo the reply; return the next agent and history."""
    from troopai.adk.run.runner import Runner

    if stream:
        streaming = await Runner.arun(
            agent, turn_input, stream=True, max_turns=turns, run_config=run_config, session=session
        )
        await echo_stream(streaming)
        return streaming.current_agent, streaming.to_input_list()
    result = await Runner.arun(agent, turn_input, max_turns=turns, run_config=run_config, session=session)
    if result.final_output is not None:
        click.echo(str(result.final_output))
    else:
        # A turn can legitimately end with no text (pure tool runs) — mark it
        # on stderr so silence is distinguishable from a swallowed reply.
        click.echo("[no output]", err=True)
    return result.last_agent, result.to_input_list()
