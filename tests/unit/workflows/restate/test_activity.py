"""Tests for :func:`~troopai.adk.workflows.restate.activity.invoke_model_handler`.

Covers:
- Handler forwards ``tools`` and ``output_schema`` to ``llm.acomplete`` when
  provided as JSON strings.
- Handler calls ``llm.acomplete`` with ``tools=None`` and
  ``output_schema=None`` when those parameters are omitted (backward-compat
  default: empty string → no tools, no schema).
- Handler raises ``TypeError`` when ``acomplete`` returns a non-LLMResponse.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.llms.llm import LLM
from troopai.adk.types.responses.llm_response import LLMResponse
from troopai.adk.workflows.restate.activity import invoke_model_handler
from troopai.adk.workflows.temporal.activity import register_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response() -> LLMResponse:
    return LLMResponse(
        response_id="resp-test",
        model="test-model",
        response=[],
        finish_reason="stop",
        usage=None,
        timestamp=None,
    )


def _make_mock_llm(response: LLMResponse) -> MagicMock:
    mock = MagicMock(spec=LLM)
    mock.acomplete = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInvokeModelHandlerToolsAndSchema:
    """tools_json / output_schema_json are forwarded to llm.acomplete."""

    @pytest.mark.asyncio
    async def test_no_tools_no_schema_calls_acomplete_with_none(self) -> None:
        """Omitting tools_json/output_schema_json passes None to acomplete."""
        resp = _make_llm_response()
        mock_llm = _make_mock_llm(resp)
        register_model("test-handler-no-tools", mock_llm)

        ctx = MagicMock()
        await invoke_model_handler(
            ctx,
            "test-handler-no-tools",
            messages=[],
            config=None,
            tools_json="",
            output_schema_json="",
        )

        mock_llm.acomplete.assert_awaited_once()
        _, kwargs = mock_llm.acomplete.call_args
        assert kwargs.get("tools") is None
        assert kwargs.get("output_schema") is None

    @pytest.mark.asyncio
    async def test_empty_tools_json_list_passes_none(self) -> None:
        """An empty tools list in JSON passes tools=None to acomplete."""
        resp = _make_llm_response()
        mock_llm = _make_mock_llm(resp)
        register_model("test-handler-empty-tools", mock_llm)

        ctx = MagicMock()
        await invoke_model_handler(
            ctx,
            "test-handler-empty-tools",
            messages=[],
            config=None,
            tools_json=json.dumps([]),
            output_schema_json="",
        )

        mock_llm.acomplete.assert_awaited_once()
        _, kwargs = mock_llm.acomplete.call_args
        assert kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_tools_json_deserialized_and_forwarded(self) -> None:
        """Non-empty tools_json is deserialized and forwarded to acomplete."""
        resp = _make_llm_response()
        mock_llm = _make_mock_llm(resp)
        register_model("test-handler-with-tools", mock_llm)

        tool_dict = {"name": "my_tool", "description": "does something", "parameters": {}}
        tools_json = json.dumps([tool_dict])

        fake_tool = MagicMock()
        ctx = MagicMock()

        with patch(
            "troopai.adk.workflows.temporal.serialization.tool_from_json_dict",
            return_value=fake_tool,
        ) as mock_deserialize:
            await invoke_model_handler(
                ctx,
                "test-handler-with-tools",
                messages=[],
                config=None,
                tools_json=tools_json,
                output_schema_json="",
            )

        mock_deserialize.assert_called_once_with(tool_dict)
        _, kwargs = mock_llm.acomplete.call_args
        assert kwargs.get("tools") == [fake_tool]
        assert kwargs.get("output_schema") is None

    @pytest.mark.asyncio
    async def test_output_schema_json_deserialized_and_forwarded(self) -> None:
        """Non-empty output_schema_json is deserialized and forwarded."""
        resp = _make_llm_response()
        mock_llm = _make_mock_llm(resp)
        register_model("test-handler-with-schema", mock_llm)

        schema_dict = {"type": "object", "properties": {"answer": {"type": "string"}}}
        output_schema_json = json.dumps(schema_dict)
        fake_schema = MagicMock()
        ctx = MagicMock()

        with patch(
            "troopai.adk.workflows.temporal.serialization.output_schema_from_json_dict",
            return_value=fake_schema,
        ) as mock_deserialize:
            await invoke_model_handler(
                ctx,
                "test-handler-with-schema",
                messages=[],
                config=None,
                tools_json="",
                output_schema_json=output_schema_json,
            )

        mock_deserialize.assert_called_once_with(schema_dict)
        _, kwargs = mock_llm.acomplete.call_args
        assert kwargs.get("tools") is None
        assert kwargs.get("output_schema") is fake_schema

    @pytest.mark.asyncio
    async def test_type_error_on_non_llm_response(self) -> None:
        """TypeError raised when acomplete returns something other than LLMResponse."""
        mock_llm = MagicMock(spec=LLM)
        mock_llm.acomplete = AsyncMock(return_value="not a response")
        register_model("test-handler-bad-return", mock_llm)

        ctx = MagicMock()
        with pytest.raises(TypeError, match="LLMResponse"):
            await invoke_model_handler(
                ctx,
                "test-handler-bad-return",
                messages=[],
                config=None,
            )

    @pytest.mark.asyncio
    async def test_returns_asdict_of_response(self) -> None:
        """Return value is dataclasses.asdict of the LLMResponse."""
        resp = _make_llm_response()
        mock_llm = _make_mock_llm(resp)
        register_model("test-handler-return-shape", mock_llm)

        ctx = MagicMock()
        result = await invoke_model_handler(
            ctx,
            "test-handler-return-shape",
            messages=[],
            config=None,
        )

        assert result == dataclasses.asdict(resp)

    @pytest.mark.asyncio
    async def test_returned_dict_is_json_serializable_with_datetime_timestamp(self) -> None:
        """A real datetime timestamp is nullified so the dict is JSON-serializable.

        Real LLM responses (OpenAI/Anthropic via litellm) carry a populated
        ``datetime`` timestamp. ``dataclasses.asdict`` preserves the datetime
        verbatim, which Restate's default JSON serde cannot encode. The handler
        must nullify the timestamp before serialization.
        """
        resp = LLMResponse(
            response_id="resp-test",
            model="test-model",
            response=[],
            finish_reason="stop",
            usage=None,
            timestamp=datetime.fromtimestamp(1_700_000_000, tz=UTC),
        )
        mock_llm = _make_mock_llm(resp)
        register_model("test-handler-datetime-timestamp", mock_llm)

        ctx = MagicMock()
        result = await invoke_model_handler(
            ctx,
            "test-handler-datetime-timestamp",
            messages=[],
            config=None,
        )

        assert result["timestamp"] is None
        # Must round-trip through JSON without raising (mirrors Restate serde).
        json.dumps(result)
