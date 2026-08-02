"""Tests for ``trim_tool_output`` (G3.5).

Covers: argument validation, char cap, token cap (mocked counter),
``content_and_artifact`` response format, non-string results, and
original-tool immutability.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.tools.tool_output_trimmer import (
    DEFAULT_TRUNCATION_MARKER,
    trim_tool_output,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_ctx() -> ToolContext:
    return ToolContext(
        tool_name="test",
        tool_call_id="call-1",
        tool_arguments={},
        raw_arguments="{}",
    )


def _make_tool(
    *,
    returns: Any = "hello",
    response_format: str = "text",
) -> FunctionTool:
    async def on_invoke(ctx: ToolContext, raw: str) -> Any:
        return returns

    return FunctionTool(
        name="test_tool",
        description="test",
        schema={"type": "object", "properties": {}},
        on_invoke=on_invoke,
        response_format=response_format,
    )


# ── Argument validation ─────────────────────────────────────────────


class TestValidation:
    def test_raises_when_no_limits_given(self) -> None:
        tool = _make_tool()
        with pytest.raises(ValueError, match="at least one of"):
            trim_tool_output(tool)

    def test_raises_when_max_tokens_without_model(self) -> None:
        tool = _make_tool()
        with pytest.raises(ValueError, match="model"):
            trim_tool_output(tool, max_tokens=100)

    def test_raises_on_non_positive_max_chars(self) -> None:
        tool = _make_tool()
        with pytest.raises(ValueError, match="max_chars"):
            trim_tool_output(tool, max_chars=0)
        with pytest.raises(ValueError, match="max_chars"):
            trim_tool_output(tool, max_chars=-5)

    def test_raises_on_non_positive_max_tokens(self) -> None:
        tool = _make_tool()
        with pytest.raises(ValueError, match="max_tokens"):
            trim_tool_output(tool, max_tokens=0, model="gpt-4o")

    def test_raises_when_on_invoke_is_none(self) -> None:
        tool = FunctionTool(
            name="empty",
            description="no invoke",
            schema={"type": "object", "properties": {}},
            on_invoke=None,
        )
        with pytest.raises(ValueError, match="on_invoke is None"):
            trim_tool_output(tool, max_chars=100)


# ── Character cap ────────────────────────────────────────────────────


class TestCharCap:
    @pytest.mark.asyncio
    async def test_short_text_passes_through(self) -> None:
        tool = _make_tool(returns="short")
        trimmed = trim_tool_output(tool, max_chars=100)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert result == "short"

    @pytest.mark.asyncio
    async def test_long_text_truncated_with_marker(self) -> None:
        tool = _make_tool(returns="a" * 200)
        trimmed = trim_tool_output(tool, max_chars=50)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert len(result) == 50
        assert result.endswith(DEFAULT_TRUNCATION_MARKER)
        # Everything before the marker is the original leading chars
        body = result[: -len(DEFAULT_TRUNCATION_MARKER)]
        assert body == "a" * (50 - len(DEFAULT_TRUNCATION_MARKER))

    @pytest.mark.asyncio
    async def test_custom_marker(self) -> None:
        tool = _make_tool(returns="z" * 200)
        trimmed = trim_tool_output(tool, max_chars=30, marker="<cut>")
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert result.endswith("<cut>")
        assert len(result) == 30

    @pytest.mark.asyncio
    async def test_max_chars_smaller_than_marker_honors_hard_cap(self) -> None:
        # max_chars below the default marker length (15) must still be a
        # hard cap: the result may not exceed max_chars even though the
        # marker would not fit.
        assert len(DEFAULT_TRUNCATION_MARKER) > 10
        tool = _make_tool(returns="a" * 200)
        trimmed = trim_tool_output(tool, max_chars=10)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert len(result) == 10
        # Marker did not fit, so it is dropped rather than overshooting.
        assert result == "a" * 10

    @pytest.mark.asyncio
    async def test_max_chars_equal_to_marker_honors_hard_cap(self) -> None:
        marker = "<cut>"
        tool = _make_tool(returns="b" * 200)
        trimmed = trim_tool_output(tool, max_chars=len(marker), marker=marker)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert len(result) == len(marker)
        assert result == "b" * len(marker)


# ── Non-string results ──────────────────────────────────────────────


class TestNonStringResults:
    @pytest.mark.asyncio
    async def test_int_result_stringified(self) -> None:
        tool = _make_tool(returns=42)
        trimmed = trim_tool_output(tool, max_chars=100)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_dict_result_stringified_and_trimmed(self) -> None:
        big_dict = {f"k{i}": "x" * 20 for i in range(20)}
        tool = _make_tool(returns=big_dict)
        trimmed = trim_tool_output(tool, max_chars=80)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert isinstance(result, str)
        assert len(result) == 80
        assert result.endswith(DEFAULT_TRUNCATION_MARKER)


# ── Token cap (mocked counter) ──────────────────────────────────────


class TestTokenCap:
    @pytest.mark.asyncio
    async def test_under_budget_passes_through(self) -> None:
        tool = _make_tool(returns="small payload")
        with patch(
            "troopai.adk.tools.tool_output_trimmer.TokenCounter.count_text",
            return_value=5,
        ) as counter:
            trimmed = trim_tool_output(tool, max_tokens=100, model="gpt-4o")
            assert trimmed.on_invoke is not None
            result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert result == "small payload"
        counter.assert_called_once()

    @pytest.mark.asyncio
    async def test_over_budget_triggers_shrink_loop(self) -> None:
        tool = _make_tool(returns="x" * 1000)
        counts = iter([500, 90, 50])  # first over, then under
        with patch(
            "troopai.adk.tools.tool_output_trimmer.TokenCounter.count_text",
            side_effect=lambda *a, **kw: next(counts),
        ):
            trimmed = trim_tool_output(tool, max_tokens=100, model="gpt-4o")
            assert trimmed.on_invoke is not None
            result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert isinstance(result, str)
        assert len(result) < 1000
        assert result.endswith(DEFAULT_TRUNCATION_MARKER)


# ── content_and_artifact ─────────────────────────────────────────────


class TestContentAndArtifact:
    @pytest.mark.asyncio
    async def test_trims_content_preserves_artifact(self) -> None:
        artifact = [{"doc_id": "d1", "score": 0.9}]
        tool = _make_tool(
            returns=("a" * 500, artifact),
            response_format="content_and_artifact",
        )
        trimmed = trim_tool_output(tool, max_chars=60)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        assert isinstance(result, tuple)
        assert len(result) == 2
        content, art = result
        assert isinstance(content, str)
        assert len(content) == 60
        assert content.endswith(DEFAULT_TRUNCATION_MARKER)
        # Artifact is unchanged — same object reference
        assert art is artifact

    @pytest.mark.asyncio
    async def test_malformed_result_passes_through(self) -> None:
        tool = _make_tool(
            returns="not_a_tuple",  # wrong shape
            response_format="content_and_artifact",
        )
        trimmed = trim_tool_output(tool, max_chars=50)
        assert trimmed.on_invoke is not None
        result = await trimmed.on_invoke(_make_ctx(), "{}")
        # Passed through unchanged
        assert result == "not_a_tuple"


# ── Immutability of the original tool ──────────────────────────────


class TestOriginalUnchanged:
    @pytest.mark.asyncio
    async def test_returns_new_instance(self) -> None:
        tool = _make_tool(returns="a" * 200)
        trimmed = trim_tool_output(tool, max_chars=50)
        assert trimmed is not tool
        assert trimmed.on_invoke is not tool.on_invoke

    @pytest.mark.asyncio
    async def test_original_on_invoke_still_returns_full_text(self) -> None:
        tool = _make_tool(returns="a" * 200)
        trim_tool_output(tool, max_chars=50)
        # Original tool is not mutated — still returns the full 200 chars
        assert tool.on_invoke is not None
        result = await tool.on_invoke(_make_ctx(), "{}")
        assert result == "a" * 200

    def test_preserves_schema_and_metadata(self) -> None:
        tool = _make_tool()
        trimmed = trim_tool_output(tool, max_chars=50, marker="<end>")
        assert trimmed.name == tool.name
        assert trimmed.description == tool.description
        assert trimmed.schema == tool.schema
        assert trimmed.response_format == tool.response_format


# ── Finding 4: clone() + streaming guard ─────────────────────────────


class TestClonePreservesInternalState:
    """Finding 4: trim_tool_output must use clone() not dataclasses.replace()."""

    def test_trimmed_tool_shares_cache_with_original(self) -> None:
        """clone() shares _cache; dataclasses.replace() would create a fresh dict."""
        tool = _make_tool(returns="hello world")
        trimmed = trim_tool_output(tool, max_chars=100)
        # They share the same _cache object
        assert trimmed._cache is tool._cache

    def test_trimmed_tool_shares_rate_state_with_original(self) -> None:
        """clone() shares _rate_state; dataclasses.replace() would not."""
        tool = _make_tool(returns="hello world")
        trimmed = trim_tool_output(tool, max_chars=100)
        assert trimmed._rate_state is tool._rate_state

    def test_streaming_tool_raises(self) -> None:
        """Streaming tools must be rejected before str() corrupts the async generator."""
        from collections.abc import AsyncIterator

        from troopai.adk.tools.function_tool import function_tool

        @function_tool(streaming=True)
        async def streamer() -> AsyncIterator[str]:
            yield "chunk"

        with pytest.raises(ValueError, match="streaming"):
            trim_tool_output(streamer, max_chars=100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
