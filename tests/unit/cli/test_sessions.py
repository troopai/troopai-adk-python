"""Tests for the ``troopai sessions`` group."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from troopai.adk.cli import main

APP = "support"


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """A session store with two sessions, one of which has two events."""
    db = tmp_path / "sessions.sqlite"

    async def seed() -> None:
        from troopai.adk.session.session_event import SessionEvent
        from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

        manager = SQLiteMultiSessions(path=db, app_name=APP)
        first = await manager.create("conv-1", user_id="default")
        await first.add(
            [
                SessionEvent(id="e1", author="user", content={"role": "user", "content": "hi"}, timestamp=1.0),
                SessionEvent(
                    id="e2",
                    author="assistant",
                    content={"role": "assistant", "content": "hello"},
                    timestamp=2.0,
                ),
            ]
        )
        # The second session stays empty: list must still report it.
        await manager.create("conv-2", user_id="alice")
        await manager.close()

    asyncio.run(seed())
    return db


def test_list_shows_both_sessions(seeded_db: Path) -> None:
    result = CliRunner().invoke(main, ["sessions", "list", "--db", str(seeded_db), "--app-name", APP])
    assert result.exit_code == 0, result.output
    assert "conv-1" in result.output
    assert "conv-2" in result.output


def test_list_filters_by_user(seeded_db: Path) -> None:
    result = CliRunner().invoke(
        main, ["sessions", "list", "--db", str(seeded_db), "--app-name", APP, "--user-id", "alice"]
    )
    assert result.exit_code == 0, result.output
    assert "conv-2" in result.output
    assert "conv-1" not in result.output


def test_list_empty_store(tmp_path: Path) -> None:
    empty = tmp_path / "empty.sqlite"
    empty.touch()
    result = CliRunner().invoke(main, ["sessions", "list", "--db", str(empty), "--app-name", APP])
    assert result.exit_code == 0, result.output
    assert "no sessions" in result.output


def test_show_renders_events(seeded_db: Path) -> None:
    result = CliRunner().invoke(main, ["sessions", "show", "--db", str(seeded_db), "--app-name", APP, "--id", "conv-1"])
    assert result.exit_code == 0, result.output
    assert "[user]" in result.output
    assert "hello" in result.output


def test_show_respects_limit(seeded_db: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["sessions", "show", "--db", str(seeded_db), "--app-name", APP, "--id", "conv-1", "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("[") == 1


def test_show_json_output(seeded_db: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["sessions", "show", "--db", str(seeded_db), "--app-name", APP, "--id", "conv-1", "--output", "json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [e["author"] for e in payload] == ["user", "assistant"]


def test_show_missing_session(seeded_db: Path) -> None:
    result = CliRunner().invoke(main, ["sessions", "show", "--db", str(seeded_db), "--app-name", APP, "--id", "nope"])
    assert result.exit_code == 2
    assert "nope" in result.output


def test_delete_with_yes(seeded_db: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["sessions", "delete", "--db", str(seeded_db), "--app-name", APP, "--id", "conv-1", "--yes"],
    )
    assert result.exit_code == 0, result.output
    listing = CliRunner().invoke(main, ["sessions", "list", "--db", str(seeded_db), "--app-name", APP])
    assert "conv-1" not in listing.output


def test_delete_declined_keeps_session(seeded_db: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["sessions", "delete", "--db", str(seeded_db), "--app-name", APP, "--id", "conv-1"],
        input="n\n",
    )
    assert result.exit_code == 1
    listing = CliRunner().invoke(main, ["sessions", "list", "--db", str(seeded_db), "--app-name", APP])
    assert "conv-1" in listing.output


def test_show_empty_session(seeded_db: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["sessions", "show", "--db", str(seeded_db), "--app-name", APP, "--id", "conv-2", "--user-id", "alice"],
    )
    assert result.exit_code == 0, result.output
    assert "(empty session)" in result.output


def test_delete_missing_session_fails(seeded_db: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["sessions", "delete", "--db", str(seeded_db), "--app-name", APP, "--id", "ghost", "--yes"],
    )
    assert result.exit_code == 2
    assert "ghost" in result.output
