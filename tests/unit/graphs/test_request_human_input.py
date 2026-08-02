"""request_human_input helper + ExecutableInput.resume_reply accessor."""

from __future__ import annotations

import pytest

from troopai.adk.graphs import Interrupt, InterruptException, request_human_input
from troopai.adk.orchestration.executable import ExecutableInput


def _input(metadata: dict[str, object] | None = None) -> ExecutableInput:
    """Build a minimal ExecutableInput for tests — use REAL ctor."""
    return ExecutableInput(content=[], from_nodes=(), edge_label=None, metadata=metadata or {})


def test_resume_reply_returns_none_when_absent() -> None:
    inp = _input()
    assert inp.resume_reply() is None


def test_resume_reply_returns_injected_value() -> None:
    inp = _input({"__resume_reply__": "the-answer"})
    assert inp.resume_reply() == "the-answer"


def test_request_human_input_raises_when_no_reply() -> None:
    inp = _input({"__interrupt_node_id__": "n1"})
    with pytest.raises(InterruptException) as exc_info:
        request_human_input(inp, "Approve?", kind="tool_approval", tool="x", risk="high")
    iv: Interrupt = exc_info.value.interrupt
    assert iv.node_id == "n1"
    assert iv.question == "Approve?"
    assert iv.kind == "tool_approval"
    assert iv.metadata == {"tool": "x", "risk": "high"}


def test_request_human_input_returns_reply_when_injected() -> None:
    inp = _input({"__resume_reply__": "yes", "__interrupt_node_id__": "n1"})
    out = request_human_input(inp, "Approve?", kind="tool_approval")
    assert out == "yes"  # injected reply wins — no exception raised


def test_request_human_input_default_kind_is_generic() -> None:
    inp = _input({"__interrupt_node_id__": "n1"})
    with pytest.raises(InterruptException) as exc_info:
        request_human_input(inp, "Continue?")
    assert exc_info.value.interrupt.kind == "generic"


def test_request_human_input_empty_node_id_when_context_missing() -> None:
    """If the loop hasn't injected __interrupt_node_id__ (e.g. callable test usage
    without the loop), node_id defaults to empty string — non-fatal: the loop
    sets this when it invokes; tests that bypass the loop see "" — documented."""
    inp = _input()
    with pytest.raises(InterruptException) as exc_info:
        request_human_input(inp, "Q?")
    assert exc_info.value.interrupt.node_id == ""


def test_request_human_input_returns_injected_none_reply() -> None:
    """A reply of None is a valid human-supplied value (e.g. 'abstain') —
    presence of the reserved key, not the value, signals that a reply
    was supplied. Prevents a silent infinite-resume on None replies."""
    inp = _input({"__resume_reply__": None, "__interrupt_node_id__": "n1"})
    out = request_human_input(inp, "Continue?")
    assert out is None  # no InterruptException raised; None is the actual reply
