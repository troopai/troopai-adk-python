"""Tests for tools/kimi_hooks.py — the Kimi Code hook entrypoint.

The module is a thin adapter: it reuses the docstring-drift detection from
.claude/hooks/docs_sync_reminder.py and reports in Kimi's accepted form
(plain text on stdout, exit 0, never blocking). These tests pin OUR logic —
argv dispatch, output contract, fail-open behavior — with a fake reminder
module, so they stay hermetic (no git state needed).
"""

import io
import sys
from types import SimpleNamespace

import pytest

from tools import kimi_hooks


def _fake_reminder(changed: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        SRC="src/troopai/adk",
        _repo_root=lambda: "/fake/root",
        _changed_py=lambda root: changed,
        _docstring_changed=lambda root, path: True,
    )


@pytest.fixture
def _stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"hook_event_name": "Stop"}'))


@pytest.mark.usefixtures("_stdin")
class TestStop:
    def test_no_changes_is_silent(self, monkeypatch, capsys):
        monkeypatch.setattr(kimi_hooks, "_load_docs_sync_reminder", lambda: _fake_reminder([]))
        kimi_hooks.main(["kimi_hooks", "stop"])
        assert capsys.readouterr().out == ""

    def test_changes_print_plain_text_reminder(self, monkeypatch, capsys):
        changed = ["src/troopai/adk/agents/agent.py"]
        monkeypatch.setattr(kimi_hooks, "_load_docs_sync_reminder", lambda: _fake_reminder(changed))
        kimi_hooks.main(["kimi_hooks", "stop"])
        out = capsys.readouterr().out
        assert "Docstrings changed in 1 file(s)" in out
        assert changed[0] in out
        assert "docs-author" in out
        # Kimi contract: plain text, not the Claude hook's JSON schema.
        assert "systemMessage" not in out

    def test_failopen_when_claude_hook_unavailable(self, monkeypatch, capsys):
        monkeypatch.setattr(kimi_hooks, "_load_docs_sync_reminder", lambda: None)
        kimi_hooks.main(["kimi_hooks", "stop"])  # must not raise
        assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("_stdin")
def test_subagent_stop_is_a_noop(monkeypatch, capsys):
    changed = ["src/troopai/adk/agents/agent.py"]
    monkeypatch.setattr(kimi_hooks, "_load_docs_sync_reminder", lambda: _fake_reminder(changed))
    kimi_hooks.main(["kimi_hooks", "subagent-stop"])
    assert capsys.readouterr().out == ""
