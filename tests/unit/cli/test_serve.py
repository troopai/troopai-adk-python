"""Tests for ``troopai serve`` (never binds a port)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

import pytest
from click.testing import CliRunner

from troopai.adk.cli import main

STUB = "cli_stub_agents"

CARD = {
    "name": "support",
    "description": "Scripted support agent for tests.",
    "version": "0.1.0",
    "supportedInterfaces": [{"url": "http://127.0.0.1:8000", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
}


@pytest.fixture
def card_file(tmp_path: Path) -> Path:
    path = tmp_path / "card.json"
    path.write_text(json.dumps(CARD), encoding="utf-8")
    return path


def test_missing_a2a_extra_guides_install(
    stub_agent_dir: Path, card_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import troopai.adk.a2a as a2a_module

    monkeypatch.setattr(a2a_module, "A2AServer", None)
    result = CliRunner().invoke(main, ["serve", "--card", str(card_file), "--agent", f"{STUB}:support"])
    assert result.exit_code == 2
    assert "troopai-adk-python[a2a]" in result.output


def test_invalid_card_json(stub_agent_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "card.json"
    bad.write_text("{not json", encoding="utf-8")
    result = CliRunner().invoke(main, ["serve", "--card", str(bad), "--agent", f"{STUB}:support"])
    assert result.exit_code == 2
    assert "Invalid JSON" in result.output


def test_unknown_card_field_rejected(stub_agent_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "card.json"
    bad.write_text(json.dumps({**CARD, "unknownField": 1}), encoding="utf-8")
    result = CliRunner().invoke(main, ["serve", "--card", str(bad), "--agent", f"{STUB}:support"])
    assert result.exit_code == 2
    assert "Invalid agent card" in result.output


def test_swarm_rejected(stub_agent_dir: Path, card_file: Path) -> None:
    result = CliRunner().invoke(main, ["serve", "--card", str(card_file), "--agent", f"{STUB}:team"])
    assert result.exit_code == 2
    assert "single agent" in result.output


def test_happy_path_builds_app_without_binding(
    stub_agent_dir: Path, card_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: Any, host: str, port: int) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = CliRunner().invoke(
        main,
        ["serve", "--card", str(card_file), "--agent", f"{STUB}:support", "--port", "9123"],
    )
    assert result.exit_code == 0, result.output

    from starlette.applications import Starlette

    assert isinstance(captured["app"], Starlette)
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9123
    assert "agent-card.json" in result.output


def test_max_turns_forwarded_to_server(stub_agent_dir: Path, card_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    import troopai.adk.a2a as a2a_module

    real_server = a2a_module.A2AServer
    captured: dict[str, Any] = {}

    def recording(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_server(**kwargs)

    monkeypatch.setattr(a2a_module, "A2AServer", recording)
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    result = CliRunner().invoke(
        main,
        ["serve", "--card", str(card_file), "--agent", f"{STUB}:support", "--max-turns", "5"],
    )
    assert result.exit_code == 0, result.output
    assert captured["max_turns"] == 5


def test_rest_default_without_card(stub_agent_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn
    from starlette.applications import Starlette

    captured: dict[str, Any] = {}

    def fake_run(app: Any, host: str, port: int) -> None:
        captured["app"] = app

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = CliRunner().invoke(main, ["serve", "--agent", f"{STUB}:support", "--port", "9100"])
    assert result.exit_code == 0, result.output
    assert isinstance(captured["app"], Starlette)
    assert "/run" in result.output
    assert "agent-card.json" not in result.output


def test_nothing_to_serve_errors(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["serve", "--agent", f"{STUB}:support", "--no-rest", "--no-health"])
    assert result.exit_code == 2
    assert "Nothing to serve" in result.output


def test_a2a_without_card_errors(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(main, ["serve", "--agent", f"{STUB}:support", "--a2a"])
    assert result.exit_code == 2
    assert "--card" in result.output


def test_missing_server_extra_guides_install(stub_agent_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import troopai.adk.serving as serving_module

    monkeypatch.setattr(serving_module, "build_app", None)
    result = CliRunner().invoke(main, ["serve", "--agent", f"{STUB}:support"])
    assert result.exit_code == 2
    assert "troopai-adk-python[serve]" in result.output


def test_task_db_creates_durable_store(
    stub_agent_dir: Path, card_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn
    from starlette.testclient import TestClient

    def fake_run(app: Any, host: str, port: int) -> None:
        # uvicorn drives the ASGI lifespan; the durable store is created and
        # recovered on the serving loop, not a throwaway bootstrap loop.
        with TestClient(app):
            pass

    monkeypatch.setattr(uvicorn, "run", fake_run)
    task_db = tmp_path / "tasks.db"
    result = CliRunner().invoke(
        main,
        ["serve", "--card", str(card_file), "--agent", f"{STUB}:support", "--task-db", str(task_db)],
    )
    assert result.exit_code == 0, result.output
    assert task_db.exists()


def test_task_store_recovers_in_lifespan_not_at_build(
    stub_agent_dir: Path, card_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn
    from starlette.testclient import TestClient

    import troopai.adk.a2a.task_store as task_store_module
    from troopai.adk.a2a.task_store import SQLiteTaskStore

    calls: list[str] = []

    class RecordingStore(SQLiteTaskStore):
        @override
        async def recover_on_startup(self) -> int:
            calls.append("recover")
            return await super().recover_on_startup()

    monkeypatch.setattr(task_store_module, "SQLiteTaskStore", RecordingStore)

    def fake_run(app: Any, host: str, port: int) -> None:
        # Building the app must not have run recovery on a bootstrap loop.
        assert calls == []
        with TestClient(app):
            pass

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = CliRunner().invoke(
        main,
        ["serve", "--card", str(card_file), "--agent", f"{STUB}:support", "--task-db", str(tmp_path / "tasks.db")],
    )
    assert result.exit_code == 0, result.output
    # Recovery ran exactly once, driven by the ASGI lifespan on the serving loop.
    assert calls == ["recover"]


def test_session_manager_not_closed_without_lifespan(
    stub_agent_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    import troopai.adk.session.sqlite_multi_sessions as sess_module

    closed: list[bool] = []

    class RecordingManager:
        def __init__(self, *, path: Path, app_name: str) -> None: ...

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(sess_module, "SQLiteMultiSessions", RecordingManager)
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    result = CliRunner().invoke(main, ["serve", "--agent", f"{STUB}:support", "--session-db", str(tmp_path / "s.db")])
    assert result.exit_code == 0, result.output
    # Close is owned by the ASGI lifespan; a runtime that never drives shutdown
    # must not close the manager on a throwaway loop.
    assert closed == []


def test_session_manager_closed_on_lifespan_shutdown(
    stub_agent_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn
    from starlette.testclient import TestClient

    import troopai.adk.session.sqlite_multi_sessions as sess_module

    closed: list[bool] = []

    class RecordingManager:
        def __init__(self, *, path: Path, app_name: str) -> None: ...

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(sess_module, "SQLiteMultiSessions", RecordingManager)

    def fake_run(app: Any, host: str, port: int) -> None:
        with TestClient(app):
            pass

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = CliRunner().invoke(main, ["serve", "--agent", f"{STUB}:support", "--session-db", str(tmp_path / "s.db")])
    assert result.exit_code == 0, result.output
    assert closed == [True]


def test_task_db_and_dsn_mutually_exclusive(stub_agent_dir: Path, card_file: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--card",
            str(card_file),
            "--agent",
            f"{STUB}:support",
            "--task-db",
            "t.db",
            "--task-dsn",
            "postgresql://x/y",
        ],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_session_db_and_dsn_mutually_exclusive(stub_agent_dir: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["serve", "--agent", f"{STUB}:support", "--session-db", "s.db", "--session-dsn", "postgresql://x/y"],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_session_db_wires_session_manager(
    stub_agent_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn
    from starlette.applications import Starlette

    captured: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: captured.setdefault("app", app))
    result = CliRunner().invoke(
        main, ["serve", "--agent", f"{STUB}:support", "--session-db", str(tmp_path / "sess.db")]
    )
    assert result.exit_code == 0, result.output
    assert isinstance(captured["app"], Starlette)


def test_session_dsn_constructs_postgres_manager(stub_agent_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    # The Postgres pool is lazy (no connection until first use), so wiring the
    # manager and closing it on shutdown never touches a database here.
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    result = CliRunner().invoke(
        main, ["serve", "--agent", f"{STUB}:support", "--session-dsn", "postgresql://u@localhost/db"]
    )
    assert result.exit_code == 0, result.output
