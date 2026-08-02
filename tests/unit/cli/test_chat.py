"""Tests for ``troopai chat`` (network-free via the scripted stub agent)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from click.testing import CliRunner

from troopai.adk.cli import main

STUB = "cli_stub_agents"


def test_chat_roundtrip_and_exit(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["chat", "--no-stream", "--agent", f"{STUB}:support"], input="hello\nexit\n")
    assert result.exit_code == 0, result.output
    assert "scripted reply" in result.output


def test_chat_streaming_default(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["chat", "--agent", f"{STUB}:support"], input="hello\nquit\n")
    assert result.exit_code == 0, result.output
    assert "scripted reply" in result.output


def test_chat_eof_exits_cleanly(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["chat", "--agent", f"{STUB}:support"], input="")
    assert result.exit_code == 0, result.output


def test_chat_blank_lines_reprompt(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(
        main, ["chat", "--no-stream", "--agent", f"{STUB}:support"], input="\n  \nhello\nexit\n"
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("scripted reply") == 1


def test_chat_rejects_swarm(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["chat", "--agent", f"{STUB}:team"], input="hello\nexit\n")
    assert result.exit_code == 2
    assert "single agent" in result.output


def test_chat_session_persists_across_invocations(stub_agent_dir: Path) -> None:
    db = stub_agent_dir / "chat_sessions.sqlite"
    runner = CliRunner()
    for prompt in ("first\nexit\n", "second\nexit\n"):
        result = runner.invoke(
            main,
            ["chat", "--no-stream", "--session-db", str(db), "--agent", f"{STUB}:support"],
            input=prompt,
        )
        assert result.exit_code == 0, result.output

    async def count_events() -> int:
        from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

        manager = SQLiteMultiSessions(path=db, app_name="support")
        session = await manager.get("default", user_id="default")
        assert session is not None
        events = await session.get()
        await manager.close()
        return len(events)

    # Two chat invocations, one turn each: two user messages + two replies.
    assert asyncio.run(count_events()) >= 4
