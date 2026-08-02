"""Regression tests for ``handoffs_executor`` history preparation.

Covers ``_collapse_history`` (transcript rendering + surviving the
post-handoff system-prompt injection) and ``apply_handoff_budget``
(the drop loop must never empty the forwarded history).

Layer 1 message ``content`` is either a plain string
(``LLMInputEasyMessage``) or a list of typed content-part dicts
(assistant ``LLMResponseMessageParam`` text/refusal parts, multimodal
``input_text``/``image``/``audio`` parts). The collapse transcript must
render the readable text of list-typed content, not Python's list repr —
otherwise the collapsed message sent to the target agent is token-bloated
``[{'type': 'output_text', ...}]`` garbage.

Fixtures are typed ``list[Any]`` deliberately: the helper only does
``.get()`` access, and the tests build minimal literal wire-shaped dicts
that intentionally do not conform to a single ``LLMInputContentItem``
TypedDict variant.
"""

from typing import Any

from troopai.adk.agents.agent import Agent
from troopai.adk.handoffs.handoff_collapse_mode import HandoffCollapseMode
from troopai.adk.handoffs.handoff_config import HandoffConfig
from troopai.adk.handoffs.handoff_target import HandoffTarget
from troopai.adk.run.context import RunContext
from troopai.adk.run.handoffs_executor import (  # test-only import
    _collapse_history,
    _content_to_str,
    apply_handoff_budget,
)
from troopai.adk.run.loop import inject_system_prompt
from troopai.adk.tools.token_budget import TokenBudget


def test_collapse_history_renders_assistant_list_content_as_text() -> None:
    """Assistant turns always carry list content (text-part dicts). The
    collapsed transcript must contain the actual text, never the list
    repr (``[{'type': 'output_text', ...}]``)."""
    messages: list[Any] = [
        {"role": "user", "content": "I need a refund."},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Sure, transferring you."}],
        },
    ]

    result: list[Any] = _collapse_history(messages, mode=HandoffCollapseMode.SYSTEM_MESSAGE)

    assert len(result) == 1
    body = result[0]["content"]
    assert "Sure, transferring you." in body
    # The bug: list repr leaks into the transcript.
    assert "output_text" not in body
    assert "{'type'" not in body
    assert "[{" not in body


def test_collapse_history_joins_multimodal_user_text_parts() -> None:
    """A multimodal user message (list content) must contribute its text
    parts to the transcript, not its dict repr."""
    messages: list[Any] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look at this"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        },
    ]

    result: list[Any] = _collapse_history(messages, mode=HandoffCollapseMode.USER_MESSAGE)

    assert len(result) == 1
    body = result[0]["content"]
    assert "Look at this" in body
    assert "input_image" not in body
    assert "image_url" not in body
    assert "base64" not in body


def test_collapse_history_preserves_plain_string_content() -> None:
    """String content (the common ``LLMInputEasyMessage`` case) must pass
    through unchanged."""
    messages: list[Any] = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "be terse"},
    ]

    result: list[Any] = _collapse_history(messages, mode=HandoffCollapseMode.SYSTEM_MESSAGE)

    assert len(result) == 1
    body = result[0]["content"]
    assert "[user]: hello" in body
    assert "[system]: be terse" in body


def test_collapse_history_uses_role_in_label() -> None:
    """The collapsed transcript labels each line with its role."""
    messages: list[Any] = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        },
    ]

    result: list[Any] = _collapse_history(messages, mode=HandoffCollapseMode.SYSTEM_MESSAGE)

    assert "[assistant]: done" in result[0]["content"]


def test_content_to_str_string_passthrough() -> None:
    assert _content_to_str("plain text") == "plain text"


def test_content_to_str_joins_text_parts() -> None:
    content: list[Any] = [
        {"type": "output_text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert _content_to_str(content) == "first second"


def test_content_to_str_placeholder_for_non_text_list() -> None:
    """A non-empty list with no textual part renders as a placeholder, not
    the list repr."""
    content: list[Any] = [{"type": "input_image", "image_url": "x"}]
    assert _content_to_str(content) == "[non-text content]"


def test_content_to_str_empty_list() -> None:
    assert _content_to_str([]) == ""


async def test_collapse_system_message_survives_system_prompt_injection() -> None:
    """A ``SYSTEM_MESSAGE`` collapse (what ``collapse=True`` maps to) must
    NOT be clobbered by the target's system-prompt injection.

    After a handoff, ``inject_system_prompt`` replaces a leading system
    message with the target agent's own prompt. If the collapsed transcript
    were emitted as a *system* message it would occupy that slot and be
    overwritten — the target would receive ZERO transferred history. The
    transcript is emitted as a user message so the injected system prompt is
    prepended before it and the history survives.
    """
    history: list[Any] = [
        {"role": "user", "content": "I need a refund for ORD-42."},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me transfer you."}]},
    ]
    collapsed: list[Any] = _collapse_history(history, mode=HandoffCollapseMode.SYSTEM_MESSAGE)

    # The collapsed transcript must NOT occupy the leading system slot that
    # inject_system_prompt overwrites.
    assert len(collapsed) == 1
    assert collapsed[0].get("role") != "system"

    target = Agent(name="refunds", system_prompt="You are the refunds specialist.")
    ctx: RunContext[dict[str, Any]] = RunContext(context={})
    final: list[Any] = await inject_system_prompt(target, list(collapsed), ctx)

    joined = "\n".join(str(m.get("content")) for m in final)
    # The target's own system prompt is injected...
    assert any(m.get("role") == "system" and "refunds specialist" in str(m.get("content")) for m in final)
    # ...AND the transferred history survives the injection step.
    assert "ORD-42" in joined
    assert "Previous conversation" in joined


async def test_apply_handoff_budget_never_empties_history() -> None:
    """The budget drop loop must never drop below the newest message.

    With a tiny budget and no system message to preserve, FIFO eviction
    would otherwise drain the whole forwarded history to ``[]`` — a
    zero-message request that strict providers (Anthropic) reject with a
    400. The newest message is retained even when it alone exceeds budget.
    """
    messages: list[Any] = [
        {"role": "user", "content": "first message with plenty of words to exceed the tiny budget"},
        {"role": "user", "content": "second message with plenty of words to exceed the tiny budget"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "the newest message here"}],
        },
    ]
    target: Any = HandoffTarget(
        target=Agent(name="specialist", system_prompt="spec"),
        config=HandoffConfig(budget=TokenBudget(max_tokens=1, drop_policy="oldest_first")),
    )

    result: list[Any] = await apply_handoff_budget(list(messages), target, "gpt-4o")

    # Floor: at least the newest message survives; the history is never empty.
    assert len(result) >= 1
    assert result[-1] is messages[-1]
