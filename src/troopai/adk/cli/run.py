"""``troopai run`` — execute an agent, swarm, graph, or topology once.

The prompt comes from the trailing argument or stdin; the final output
lands on stdout (text by default, ``--output json`` for scripting). The
target kind picks the runner entry point automatically; flags that do
not apply to the resolved kind fail loudly instead of being ignored.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from troopai.adk.cli.errors import framework_errors
from troopai.adk.cli.loading import load_env_file, primary_executable, reconcile_positionals, resolve_target
from troopai.adk.cli.options import output_option, run_options, session_options, target_options

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.graphs import Graph
    from troopai.adk.run.config import RunConfig
    from troopai.adk.session.session import Session
    from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions
    from troopai.adk.swarms import Swarm

logger = logging.getLogger(__name__)


@click.command(name="run")
@target_options
@run_options
@session_options
@output_option
@click.option(
    "--stream",
    is_flag=True,
    default=False,
    help="Stream text deltas as they arrive (single-agent targets only).",
)
@click.argument("prompt", required=False)
@framework_errors
def run(
    config: Path | None,
    agent_ref: str | None,
    prompt: str | None,
    model: str | None,
    max_turns: int | None,
    verbose: bool,
    trace: bool,
    env_file: Path | None,
    session_db: Path | None,
    session_id: str,
    user_id: str,
    output: str,
    stream: bool,
) -> None:
    """Run a target once with PROMPT (or stdin) and print the final output."""
    if stream and output == "json":
        raise click.UsageError("--stream and --output json are mutually exclusive.")
    if env_file is not None:
        load_env_file(env_file)
    config, prompt = reconcile_positionals(config, agent_ref, prompt)
    executable = primary_executable(resolve_target(config, agent_ref))
    logger.debug("run target resolved: %s", type(executable).__name__)
    prompt_text = _read_prompt(prompt)
    run_config = build_run_config(model=model, verbose=verbose, trace=trace)
    result = asyncio.run(
        _execute(
            executable,
            prompt_text,
            stream=stream,
            max_turns=max_turns,
            run_config=run_config,
            session_db=session_db,
            session_id=session_id,
            user_id=user_id,
        )
    )
    # Streaming already echoed its output delta-by-delta.
    if result is not None:
        click.echo(_render(result, output))


def build_run_config(*, model: str | None, verbose: bool, trace: bool) -> RunConfig | None:
    """Build a ``RunConfig`` from the passed flags; ``None`` when none was set.

    Args:
        model: Optional model override.
        verbose: Whether to render the run verbosely.
        trace: Whether to enable tracing with a console span exporter.

    Returns:
        A ``RunConfig`` carrying exactly the requested settings, or ``None``
        so the runner keeps its own defaults.
    """
    if model is None and not verbose and not trace:
        return None
    from troopai.adk.run.config import RunConfig

    verbose_config = None
    if verbose:
        from troopai.adk.verbose.config import VerboseConfig

        verbose_config = VerboseConfig()
    if trace:
        _enable_tracing()
    return RunConfig(model=model, verbose=verbose_config, tracing_enabled=trace)


def target_app_name(executable: Agent | Swarm | Graph) -> str:
    """Derive the session ``app_name`` scope for an executable.

    Args:
        executable: The object the command is about to run.

    Returns:
        The agent's name, a swarm's entry-agent name, or a graph's id.
    """
    from troopai.adk.agents.agent import Agent
    from troopai.adk.swarms import Swarm

    if isinstance(executable, Agent):
        return executable.name
    if isinstance(executable, Swarm):
        return executable.entry.name
    return executable.id


def _enable_tracing() -> None:
    """Install an OTel tracer with a console exporter, or guide the install."""
    from troopai.adk.tracing import set_tracer, setup_otel

    if setup_otel is None:
        raise click.UsageError(
            "--trace requires the OpenTelemetry extra. Install with: pip install 'troopai-adk-python[otel]'"
        )
    set_tracer(setup_otel(console=True))


def _read_prompt(prompt: str | None) -> str:
    """Return the prompt argument or read it from piped stdin."""
    if prompt is None:
        stdin = click.get_text_stream("stdin")
        if stdin.isatty():
            raise click.UsageError("Provide PROMPT as an argument or pipe it on stdin.")
        prompt = stdin.read()
    trimmed = prompt.strip()
    if len(trimmed) == 0:
        raise click.UsageError("Prompt is empty.")
    return trimmed


async def _execute(
    executable: Agent | Swarm | Graph,
    prompt: str,
    *,
    stream: bool,
    max_turns: int | None,
    run_config: RunConfig | None,
    session_db: Path | None,
    session_id: str,
    user_id: str,
) -> Any:
    """Dispatch to the runner entry point matching the executable kind."""
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import DEFAULT_MAX_TURNS
    from troopai.adk.run.runner import Runner
    from troopai.adk.swarms import Swarm

    # Kind checks fire BEFORE the session store is touched, so a rejected
    # invocation never leaves a phantom session row behind.
    if not isinstance(executable, Agent):
        if stream:
            raise click.UsageError("--stream applies to single-agent targets.")
        if max_turns is not None:
            raise click.UsageError(
                "--max-turns applies to single-agent targets; set budgets in the swarm/graph config."
            )
    if not isinstance(executable, (Agent, Swarm)) and session_db is not None:
        raise click.UsageError("--session-db applies to agent and swarm targets; graphs persist via checkpointers.")

    manager, session = await open_session(executable, session_db, session_id, user_id)
    try:
        if isinstance(executable, Agent):
            turns = max_turns if max_turns is not None else DEFAULT_MAX_TURNS
            if stream:
                streaming = await Runner.arun(
                    executable, prompt, stream=True, max_turns=turns, run_config=run_config, session=session
                )
                await echo_stream(streaming)
                return None
            return await Runner.arun(executable, prompt, max_turns=turns, run_config=run_config, session=session)
        if isinstance(executable, Swarm):
            return await Runner.arun_swarm(executable, prompt, run_config=run_config, session=session)
        return await Runner.arun_graph(executable, prompt, run_config=run_config)
    finally:
        if manager is not None:
            await manager.close()


async def open_session(
    executable: Agent | Swarm | Graph,
    session_db: Path | None,
    session_id: str,
    user_id: str,
) -> tuple[SQLiteMultiSessions | None, Session | None]:
    """Open (or create) the requested session; ``(None, None)`` without a DB."""
    if session_db is None:
        return None, None
    from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

    manager = SQLiteMultiSessions(path=session_db, app_name=target_app_name(executable))
    try:
        session = await manager.get_or_create(session_id, user_id=user_id)
    except Exception:
        # The manager holds a live DB connection and the caller's finally
        # never sees it when this raises — release it before propagating.
        await manager.close()
        raise
    return manager, session


async def echo_stream(streaming: Any) -> None:
    """Echo text deltas to stdout; tool/handoff markers go to stderr."""
    from troopai.adk.run.stream import (
        AgentUpdatedStreamEvent,
        RawResponseStreamEvent,
        RunItemStreamEvent,
        RunItemType,
    )

    echoed_delta = False
    async for event in streaming.stream_events():
        if isinstance(event, RawResponseStreamEvent):
            if isinstance(event.data, str):
                click.echo(event.data, nl=False)
                echoed_delta = True
        elif isinstance(event, RunItemStreamEvent):
            if event.name == RunItemType.TOOL_CALLED:
                click.echo("[tool called]", err=True)
            elif event.name == RunItemType.TOOL_OUTPUT:
                click.echo(f"[tool output: {event.item}]", err=True)
        elif isinstance(event, AgentUpdatedStreamEvent):
            click.echo(f"[agent updated: {event.new_agent.name}]", err=True)
    if echoed_delta:
        click.echo("")
    elif streaming.final_output is not None:
        # Some providers emit no raw deltas (single terminal event) — fall
        # back to the assembled final output so stream mode never goes mute.
        click.echo(str(streaming.final_output))
    else:
        # A turn can legitimately end with no text (pure tool runs) — mark it
        # on stderr so silence is distinguishable from a broken stream.
        click.echo("[no output]", err=True)


def _render(result: Any, output: str) -> str:
    """Render a run result for stdout in the requested format."""
    if output == "text":
        return "" if result.final_output is None else str(result.final_output)
    from pydantic import BaseModel

    from troopai.adk.types.run.run_result import RunResult

    final_output = result.final_output
    if isinstance(final_output, BaseModel):
        final_output = final_output.model_dump()
    payload: dict[str, Any] = {"final_output": final_output}
    if isinstance(result, RunResult) and result.last_agent is not None:
        payload["agent"] = result.last_agent.name
    return json.dumps(payload, default=str)
