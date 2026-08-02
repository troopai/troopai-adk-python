from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from troopai.adk.audit.event import AuditEvent
from troopai.adk.audit.sink import AuditSink, JsonlFileAuditSink


def _event(tool: str) -> AuditEvent:
    return AuditEvent(
        tenant_id="t1",
        agent_name="a",
        tool_name=tool,
        tool_call_id="c1",
        args_hash="h",
        result_hash=None,
        outcome="ok",
        timestamp=datetime.now(UTC),
    )


def test_jsonl_is_an_audit_sink(tmp_path: Path) -> None:
    assert isinstance(JsonlFileAuditSink(tmp_path / "a.jsonl"), AuditSink)


async def test_appends_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlFileAuditSink(path)
    await sink.record(_event("first"))
    await sink.record(_event("second"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["tool_name"] == "first"
    assert json.loads(lines[1])["tool_name"] == "second"


async def test_write_failure_logs_path_before_reraising(tmp_path: Path, caplog) -> None:
    """A failed append logs the file path at ERROR before propagating.

    Governance's best-effort handler logs only a generic warning; the sink
    must name its target path so a compliance team can locate the failed
    write.
    """
    # Parent directory does not exist → open("a") raises FileNotFoundError.
    bad_path = tmp_path / "missing_dir" / "audit.jsonl"
    sink = JsonlFileAuditSink(bad_path)
    with (
        caplog.at_level(logging.ERROR, logger="troopai.adk.audit.sink"),
        pytest.raises(FileNotFoundError),
    ):
        await sink.record(_event("x"))
    assert "FAILED" in caplog.text
    assert "missing_dir" in caplog.text
