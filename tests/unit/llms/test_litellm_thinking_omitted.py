"""Regression: Claude thinking blocks with empty text but a signature survive.

Claude extended thinking can return a thinking block whose text content is
omitted while its ``signature`` is still populated (the ``display: "omitted"``
case), plus ``redacted_thinking`` blocks that carry only opaque ``data``. The
signature is load-bearing: Anthropic rejects a follow-up request in a
multi-turn tool-use exchange unless every prior thinking block is replayed
with its signature intact.

A naive guard that drops a reasoning block when its text is empty (e.g.
``if not thinking_text:``) silently discards these signed-but-empty blocks and
breaks the next API call. The LiteLLM path here must never do that: an
empty-text-with-signature block has to round-trip through both parsing
(wire -> response) and replay (item -> wire). These tests pin that behaviour
in both directions so a future refactor cannot reintroduce the drop.
"""

from __future__ import annotations

from types import SimpleNamespace

from troopai.adk.llms.litellm.litellm_converter import ChatCompletionConverter
from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import LLMResponseReasoning

_MODEL = "claude-3-5-sonnet-20241022"


class TestParsePreservesSignedEmptyThinking:
    """Parsing a response must keep signed thinking blocks that have no text."""

    def _response_with_thinking_blocks(self, blocks: list[dict[str, str]]) -> SimpleNamespace:
        """Wrap thinking blocks in a fake non-streaming litellm response."""
        message = SimpleNamespace(thinking_blocks=blocks, reasoning_content=None, content=None, tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], model=_MODEL, usage=None, created=None, id="resp-1")

    def test_empty_thinking_with_signature_preserved(self) -> None:
        llm = LiteLLM(model=_MODEL)
        resp = self._response_with_thinking_blocks([{"type": "thinking", "thinking": "", "signature": "sig-abc"}])

        result = llm._parse_response(resp)

        reasoning = [p for p in result.response if isinstance(p, LLMResponseReasoning)]
        assert len(reasoning) == 1
        assert reasoning[0].thinking == ""
        assert reasoning[0].signature == "sig-abc", "signature of an omitted thinking block must not be dropped"

    def test_redacted_thinking_preserved(self) -> None:
        llm = LiteLLM(model=_MODEL)
        resp = self._response_with_thinking_blocks([{"type": "redacted_thinking", "data": "redacted-xyz"}])

        result = llm._parse_response(resp)

        reasoning = [p for p in result.response if isinstance(p, LLMResponseReasoning)]
        assert len(reasoning) == 1
        assert reasoning[0].thinking == ""
        assert reasoning[0].signature == "redacted-xyz"


class TestReplayPreservesSignedEmptyThinking:
    """Replaying a reasoning item must re-emit the signed thinking block."""

    def test_signed_empty_reasoning_roundtrips_into_assistant_message(self) -> None:
        items: list[LLMInputContentItem] = [
            {"type": "reasoning", "summary": [], "encrypted_content": "sig-abc"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        ]

        messages = ChatCompletionConverter.items_to_messages(items, model=_MODEL, preserve_thinking_blocks=True)

        thinking_blocks: list[object] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                thinking_blocks.extend(c for c in content if isinstance(c, dict) and c.get("type") == "thinking")
        assert {"type": "thinking", "thinking": "", "signature": "sig-abc"} in thinking_blocks, (
            "a signed reasoning item with no text must replay as a signed thinking block"
        )
