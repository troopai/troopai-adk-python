"""Regression tests for FU-T22b: forwarded handoff history must not carry
an unpaired ``transfer_to_<name>`` tool-call.

When an agent hands off via a tool call, the temporal slice forwarded to
the target agent contains the ``transfer_to_<name>`` ``function_call``
param but NOT its synthetic ``function_call_output`` (that result is
appended to the *source* agent's history after the slice is taken).
Strict providers (Anthropic) reject any ``tool_use`` not immediately
followed by its ``tool_result``. ``_drop_orphan_tool_calls`` removes
unpaired tool-call params; ``_rewrite_trailing_assistant`` composes it
ahead of the trailing-assistant role rewrite.

Fixtures are typed ``list[Any]`` deliberately: the helpers only do
``.get()`` access, and the tests build minimal literal wire-shaped
dicts that intentionally do not conform to a single ``LLMInputContentItem``
TypedDict variant.
"""

from typing import Any

from troopai.adk.run.handoffs_executor import (  # test-only import
    _drop_orphan_tool_calls,
    _rewrite_trailing_assistant,
)


def test_drop_orphan_tool_calls_removes_unpaired_handoff_call() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "I need a refund."},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Transferring you."}]},
        {"type": "function_call", "call_id": "toolu_handoff", "name": "transfer_to_refunds", "arguments": "{}"},
    ]
    result = _drop_orphan_tool_calls(messages)

    assert len(result) == 2
    assert all(m.get("type") != "function_call" for m in result)
    assert result[0].get("role") == "user"


def test_drop_orphan_tool_calls_preserves_paired_tool_call() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "Look up ORD-1."},
        {"type": "function_call", "call_id": "toolu_lookup", "name": "lookup_order", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "toolu_lookup", "output": '{"status":"delivered"}'},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "It is delivered."}]},
    ]
    result = _drop_orphan_tool_calls(messages)

    # The paired call + its result must both survive untouched.
    assert len(result) == 4
    assert result[1].get("call_id") == "toolu_lookup"
    assert result[2].get("type") == "function_call_output"


def test_drop_orphan_tool_calls_drops_orphan_keeps_paired_when_mixed() -> None:
    messages: list[Any] = [
        {"type": "function_call", "call_id": "paired", "name": "lookup_order", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "paired", "output": "ok"},
        {"type": "function_call", "call_id": "orphan_handoff", "name": "transfer_to_x", "arguments": "{}"},
    ]
    result = _drop_orphan_tool_calls(messages)

    call_ids = [m.get("call_id") for m in result if m.get("type") == "function_call"]
    assert call_ids == ["paired"]
    assert len(result) == 2


def test_drop_orphan_tool_calls_noop_without_tool_calls() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "hi"},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    result = _drop_orphan_tool_calls(messages)
    assert len(result) == 2


def test_rewrite_trailing_assistant_drops_orphan_then_flips_role() -> None:
    """The exact reproduced scenario: forwarded slice = [user, assistant
    text, orphan transfer_to call]. The orphan must be dropped AND the
    now-trailing assistant text rewritten to user role."""
    messages: list[Any] = [
        {"role": "user", "content": "I need a refund."},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "I'll transfer you."}]},
        {"type": "function_call", "call_id": "toolu_handoff", "name": "transfer_to_refunds", "arguments": "{}"},
    ]
    result = _rewrite_trailing_assistant(messages)

    # No unpaired tool_use survives — the Anthropic-invalid shape is gone.
    assert all(m.get("type") != "function_call" for m in result)
    # The trailing source-agent assistant turn is rewritten to user role.
    assert result[-1].get("role") == "user"
    assert len(result) == 2


def test_drop_orphan_tool_calls_removes_unpaired_result() -> None:
    """Budget-truncation mirror case: a ``function_call_output`` whose
    ``function_call`` was evicted by FIFO budget truncation is itself an
    invalid orphan and must be dropped."""
    messages: list[Any] = [
        {"role": "user", "content": "hi"},
        {"type": "function_call_output", "call_id": "evicted_call", "output": "stale result"},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    result = _drop_orphan_tool_calls(messages)

    assert all(m.get("type") != "function_call_output" for m in result)
    assert len(result) == 2


def test_drop_orphan_tool_calls_is_idempotent() -> None:
    """Applied at two sites (forwarded-history finalize + post-budget
    truncation); a second pass over a cleaned list must be a no-op."""
    messages: list[Any] = [
        {"role": "user", "content": "Look up ORD-1."},
        {"type": "function_call", "call_id": "paired", "name": "lookup_order", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "paired", "output": "ok"},
        {"type": "function_call", "call_id": "orphan", "name": "transfer_to_x", "arguments": "{}"},
    ]
    once = _drop_orphan_tool_calls(messages)
    twice = _drop_orphan_tool_calls(once)

    assert once == twice
    call_ids = [m.get("call_id") for m in twice if m.get("type") == "function_call"]
    assert call_ids == ["paired"]


def test_rewrite_trailing_assistant_keeps_paired_history() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "Look up ORD-9."},
        {"type": "function_call", "call_id": "tc1", "name": "lookup_order", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "tc1", "output": "delivered"},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Delivered."}]},
    ]
    result = _rewrite_trailing_assistant(messages)

    # Paired tool call preserved; only the trailing assistant flips role.
    assert any(m.get("call_id") == "tc1" and m.get("type") == "function_call" for m in result)
    assert result[-1].get("role") == "user"
    assert len(result) == 4


def test_drop_orphan_tool_calls_keeps_paired_exchange_with_empty_call_id() -> None:
    """A paired ``function_call`` + ``function_call_output`` whose
    ``call_id`` is the empty string must NOT be deleted.

    The pairing-set builder skips items without a valid non-empty
    ``call_id`` (they are never registered). The drop loop must mirror that
    guard — otherwise both halves fail the ``not in`` check against the
    sets they were skipped from and the whole exchange is wiped, even
    though it is legitimately paired.
    """
    messages: list[Any] = [
        {"role": "user", "content": "run it"},
        {"type": "function_call", "call_id": "", "name": "do_thing", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "", "output": "done"},
    ]
    result = _drop_orphan_tool_calls(messages)

    # Both halves of the empty-id pair survive; nothing is dropped.
    assert len(result) == 3
    assert any(m.get("type") == "function_call" for m in result)
    assert any(m.get("type") == "function_call_output" for m in result)


def test_drop_orphan_tool_calls_keeps_tool_call_with_missing_call_id() -> None:
    """A ``function_call`` with NO ``call_id`` key is likewise not subject
    to id-based orphan classification and must be kept — the second loop's
    guard mirrors the pairing-set builder's ``continue``."""
    messages: list[Any] = [
        {"type": "function_call", "name": "do_thing", "arguments": "{}"},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    result = _drop_orphan_tool_calls(messages)
    assert len(result) == 2


def test_drop_orphan_tool_calls_still_drops_valid_id_orphan() -> None:
    """The empty-id guard must not weaken the real orphan-drop: a
    ``function_call`` with a valid, unmatched ``call_id`` is still removed."""
    messages: list[Any] = [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "toolu_handoff", "name": "transfer_to_x", "arguments": "{}"},
    ]
    result = _drop_orphan_tool_calls(messages)

    assert all(m.get("type") != "function_call" for m in result)
    assert len(result) == 1
