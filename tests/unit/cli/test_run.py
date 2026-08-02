"""Tests for ``troopai run`` (network-free via the scripted stub agent)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from click.testing import CliRunner

from troopai.adk.cli import main

STUB = "cli_stub_agents"


def test_prompt_argument(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--agent", f"{STUB}:support", "hello"])
    assert result.exit_code == 0, result.output
    assert "scripted reply" in result.output


def test_prompt_from_stdin(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--agent", f"{STUB}:support"], input="hello from stdin\n")
    assert result.exit_code == 0, result.output
    assert "scripted reply" in result.output


def test_empty_prompt_rejected(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--agent", f"{STUB}:support"], input="   \n")
    assert result.exit_code == 2
    assert "empty" in result.output.lower()


def test_json_output(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--output", "json", "--agent", f"{STUB}:support", "hello"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["final_output"] == "scripted reply"
    assert payload["agent"] == "support"


def test_stream_and_json_conflict(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--stream", "--output", "json", "--agent", f"{STUB}:support", "hello"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_stream_falls_back_to_final_output(stub_agent_dir: Path) -> None:
    # The plain stub emits a single terminal event with no raw deltas, so
    # this pins the fallback path, not delta-by-delta streaming.
    result = CliRunner().invoke(main, ["run", "--stream", "--agent", f"{STUB}:support", "hello"])
    assert result.exit_code == 0, result.output
    assert "scripted reply" in result.output


def test_stream_echoes_real_deltas(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--stream", "--agent", f"{STUB}:delta_support", "hello"])
    assert result.exit_code == 0, result.output
    assert "streamed reply" in result.output
    assert result.output.endswith("\n")


def test_graph_dispatch(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--agent", f"{STUB}:flow", "hello"])
    assert result.exit_code == 0, result.output
    assert "scripted reply" in result.output


def test_graph_rejects_session_db_without_touching_store(stub_agent_dir: Path) -> None:
    db = stub_agent_dir / "graph_sessions.sqlite"
    result = CliRunner().invoke(main, ["run", "--session-db", str(db), "--agent", f"{STUB}:flow", "hello"])
    assert result.exit_code == 2
    assert "checkpointers" in result.output
    # The rejection must fire before the store is opened: no phantom DB file.
    assert not db.exists()


def test_swarm_dispatch(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--agent", f"{STUB}:team", "hello"])
    assert result.exit_code == 0, result.output


def test_swarm_rejects_stream(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--stream", "--agent", f"{STUB}:team", "hello"])
    assert result.exit_code == 2
    assert "single-agent" in result.output


def test_swarm_rejects_max_turns(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["run", "--max-turns", "3", "--agent", f"{STUB}:team", "hello"])
    assert result.exit_code == 2
    assert "single-agent" in result.output


def test_session_persists_across_invocations(stub_agent_dir: Path) -> None:
    db = stub_agent_dir / "sessions.sqlite"
    runner = CliRunner()
    for prompt in ("first question", "second question"):
        result = runner.invoke(main, ["run", "--session-db", str(db), "--agent", f"{STUB}:support", prompt])
        assert result.exit_code == 0, result.output

    async def count_events() -> int:
        from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

        manager = SQLiteMultiSessions(path=db, app_name="support")
        session = await manager.get("default", user_id="default")
        assert session is not None
        events = await session.get()
        await manager.close()
        return len(events)

    # Two turns persisted: at least two user messages + two replies.
    assert asyncio.run(count_events()) >= 4
