"""Tests for LLMExtractor with mocked LLM."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.exceptions import MemoryExtractionError
from troopai.adk.memory.extractor import LLMExtractor


def _make_extractor(content: str, **kwargs) -> tuple[LLMExtractor, MagicMock]:
    """Create an LLMExtractor with a mock LLM returning the given content."""
    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini"
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm.acomplete = AsyncMock(return_value=mock_response)
    extractor = LLMExtractor(llm=mock_llm, **kwargs)
    return extractor, mock_llm


class TestLLMExtractor:
    @pytest.mark.asyncio
    async def test_extract_parses_json_response(self):
        content = json.dumps(
            [
                {"content": "User prefers dark mode", "importance": 4, "categories": ["preference"]},
                {"content": "User is a developer", "importance": 3, "categories": ["fact"]},
            ]
        )
        extractor, mock_llm = _make_extractor(content)

        results = await extractor.extract(
            [{"role": "user", "content": "I'm a developer who likes dark mode"}],
            namespace="user:1",
        )

        assert len(results) == 2
        assert results[0].content == "User prefers dark mode"
        assert results[0].importance == 4
        assert results[0].categories == ("preference",)
        assert results[1].content == "User is a developer"
        mock_llm.acomplete.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_handles_empty_response(self):
        extractor, _ = _make_extractor("[]")
        results = await extractor.extract([], namespace="ns")
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_raises_on_invalid_json(self):
        extractor, _ = _make_extractor("not valid json")
        with pytest.raises(MemoryExtractionError, match="failed to parse response as JSON"):
            await extractor.extract([], namespace="ns")

    @pytest.mark.asyncio
    async def test_extract_raises_on_non_array_json(self):
        extractor, _ = _make_extractor('{"key": "value"}')
        with pytest.raises(MemoryExtractionError, match="expected JSON array"):
            await extractor.extract([], namespace="ns")

    @pytest.mark.asyncio
    async def test_extract_strips_markdown_fences(self):
        json_content = json.dumps([{"content": "User likes Python", "importance": 3}])
        extractor, _ = _make_extractor(f"```json\n{json_content}\n```")
        results = await extractor.extract([], namespace="ns")
        assert len(results) == 1
        assert results[0].content == "User likes Python"

    @pytest.mark.asyncio
    async def test_extract_respects_max_entries(self):
        items = [{"content": f"Fact {i}", "importance": 3} for i in range(10)]
        extractor, _ = _make_extractor(json.dumps(items), max_entries=2)
        results = await extractor.extract([], namespace="ns")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_extract_clamps_importance(self):
        content = json.dumps(
            [
                {"content": "Too high", "importance": 100},
                {"content": "Too low", "importance": -5},
            ]
        )
        extractor, _ = _make_extractor(content)
        results = await extractor.extract([], namespace="ns")
        assert results[0].importance == 5
        assert results[1].importance == 1

    @pytest.mark.asyncio
    async def test_extract_uses_custom_system_prompt(self):
        extractor, mock_llm = _make_extractor("[]", system_prompt="Custom: {messages}")

        await extractor.extract(
            [{"role": "user", "content": "hello"}],
            namespace="ns",
        )

        call_kwargs = mock_llm.acomplete.call_args[1]
        assert "Custom:" in call_kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_literal_braces_in_custom_prompt_do_not_crash(self):
        """A custom prompt with literal JSON braces must not raise (no str.format)."""
        prompt = 'Extract facts. Example: {"content": "x", "importance": 3}\n{messages}'
        extractor, mock_llm = _make_extractor("[]", system_prompt=prompt)

        results = await extractor.extract([{"role": "user", "content": "hello"}], namespace="ns")

        assert results == []
        sent = mock_llm.acomplete.call_args[1]["messages"][0]["content"]
        # The literal example braces survive untouched...
        assert '{"content": "x", "importance": 3}' in sent
        # ...and the conversation is substituted at the placeholder.
        assert '"hello"' in sent

    @pytest.mark.asyncio
    async def test_prompt_without_placeholder_still_includes_conversation(self):
        """A prompt lacking {messages} must still receive the conversation."""
        extractor, mock_llm = _make_extractor("[]", system_prompt="Just extract facts, no placeholder.")

        await extractor.extract([{"role": "user", "content": "remember me"}], namespace="ns")

        sent = mock_llm.acomplete.call_args[1]["messages"][0]["content"]
        assert "Just extract facts, no placeholder." in sent
        assert "remember me" in sent

    @pytest.mark.asyncio
    async def test_handles_none_content(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o-mini"
        mock_llm.acomplete = AsyncMock(return_value=MagicMock(content=None))
        extractor = LLMExtractor(llm=mock_llm)

        results = await extractor.extract([], namespace="ns")
        assert results == []

    @pytest.mark.asyncio
    async def test_passes_llm_config(self):
        from troopai.adk.llms.llm_config import LLMConfig

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o-mini"
        mock_llm.acomplete = AsyncMock(return_value=MagicMock(content="[]"))
        config = LLMConfig(temperature=0.5)
        extractor = LLMExtractor(llm=mock_llm, llm_config=config)

        await extractor.extract([], namespace="ns")

        call_kwargs = mock_llm.acomplete.call_args[1]
        assert call_kwargs["llm_config"] is config

    @pytest.mark.asyncio
    async def test_malformed_item_emits_warning(self, caplog):
        """Malformed extraction items (missing 'content') must emit a WARNING."""
        import logging

        # Mix one valid and one malformed item
        content = json.dumps(
            [
                {"content": "Good fact"},
                {"importance": 3},  # missing 'content' key — malformed
                "not a dict",  # not a dict at all — also malformed
            ]
        )
        extractor, _ = _make_extractor(content)

        with caplog.at_level(logging.WARNING):
            results = await extractor.extract([], namespace="ns")

        # Only the valid item is returned
        assert len(results) == 1
        assert results[0].content == "Good fact"
        # A WARNING must have been emitted for each malformed item
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 2

    @pytest.mark.asyncio
    async def test_non_numeric_importance_falls_back_to_default(self):
        """A non-numeric importance (e.g. "high") must not raise; defaults to 3.

        Off-spec importance values from a misbehaving model must stay inside
        the documented MemoryExtractionError failure mode, never escape as a
        raw ValueError/TypeError.
        """
        content = json.dumps(
            [
                {"content": "Word importance", "importance": "high"},
                {"content": "Float-string importance", "importance": "4"},
            ]
        )
        extractor, _ = _make_extractor(content)

        results = await extractor.extract([], namespace="ns")

        assert len(results) == 2
        assert results[0].importance == 3
        # A numeric string still coerces and clamps correctly.
        assert results[1].importance == 4

    @pytest.mark.asyncio
    async def test_null_importance_falls_back_to_default(self):
        """JSON null importance (Python None) must not raise; defaults to 3."""
        content = json.dumps([{"content": "Null importance", "importance": None}])
        extractor, _ = _make_extractor(content)

        results = await extractor.extract([], namespace="ns")

        assert len(results) == 1
        assert results[0].importance == 3

    @pytest.mark.asyncio
    async def test_non_iterable_categories_yields_empty_tuple(self):
        """A non-list categories value (e.g. an int) must not raise; yields ()."""
        content = json.dumps([{"content": "Bad categories", "categories": 5}])
        extractor, _ = _make_extractor(content)

        results = await extractor.extract([], namespace="ns")

        assert len(results) == 1
        assert results[0].categories == ()

    @pytest.mark.asyncio
    async def test_string_categories_not_split_into_characters(self):
        """A comma-joined string of categories must not silently corrupt.

        A model emitting "knowledge,facts" instead of a JSON array must not
        produce a per-character tuple; it yields () rather than corrupted tags.
        """
        content = json.dumps([{"content": "String categories", "categories": "knowledge,facts"}])
        extractor, _ = _make_extractor(content)

        results = await extractor.extract([], namespace="ns")

        assert len(results) == 1
        # NOT ('k', 'n', 'o', 'w', ...) — the string is rejected, not iterated.
        assert results[0].categories == ()
