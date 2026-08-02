"""Tests for :mod:`troopai.adk.workflows.temporal.mcp`.

Covers:
- ``TemporalMCPToolSet`` stores ``name`` and ``connection_params`` correctly.
- ``TemporalMCPToolSet`` derives the expected list-tools and call-tool activity names.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

temporalio = pytest.importorskip("temporalio")

from troopai.adk.workflows.temporal.mcp import TemporalMCPToolSet


class TestTemporalMCPToolSetStoresConfig:
    def test_temporal_mcp_toolset_stores_name(self) -> None:
        """The ``name`` attribute is stored as provided."""
        toolset = TemporalMCPToolSet(name="my-mcp-server")
        assert toolset.name == "my-mcp-server"

    def test_temporal_mcp_toolset_stores_connection_params(self) -> None:
        """The ``connection_params`` attribute is stored as provided."""
        params = {"url": "http://localhost:8080", "token": "secret"}
        toolset = TemporalMCPToolSet(name="my-mcp-server", connection_params=params)
        assert toolset.connection_params == params

    def test_temporal_mcp_toolset_default_connection_params_is_empty(self) -> None:
        """``connection_params`` defaults to an empty dict."""
        toolset = TemporalMCPToolSet(name="my-mcp-server")
        assert toolset.connection_params == {}

    def test_temporal_mcp_toolset_default_timeout(self) -> None:
        """``start_to_close_timeout`` defaults to 30 seconds."""
        toolset = TemporalMCPToolSet(name="my-mcp-server")
        assert toolset.start_to_close_timeout == 30

    def test_temporal_mcp_toolset_custom_timeout(self) -> None:
        """A custom ``start_to_close_timeout`` is stored correctly."""
        toolset = TemporalMCPToolSet(name="my-mcp-server", start_to_close_timeout=60)
        assert toolset.start_to_close_timeout == 60


class TestTemporalMCPToolSetActivityNames:
    def test_list_tools_activity_name(self) -> None:
        """``list_tools_activity_name`` follows the ``{name}-mcp-list-tools`` pattern."""
        toolset = TemporalMCPToolSet(name="weather")
        assert toolset.list_tools_activity_name == "weather-mcp-list-tools"

    def test_call_tool_activity_name(self) -> None:
        """``call_tool_activity_name`` follows the ``{name}-mcp-call-tool`` pattern."""
        toolset = TemporalMCPToolSet(name="weather")
        assert toolset.call_tool_activity_name == "weather-mcp-call-tool"

    def test_activity_names_reflect_custom_name(self) -> None:
        """Activity names incorporate whatever ``name`` was supplied."""
        toolset = TemporalMCPToolSet(name="search-engine")
        assert toolset.list_tools_activity_name == "search-engine-mcp-list-tools"
        assert toolset.call_tool_activity_name == "search-engine-mcp-call-tool"


class TestTemporalMCPToolSetRetriesOffByDefault:
    """Cost-conservative invariant: MCP activities are never re-run unless the
    developer opts in via ``maximum_attempts``.
    """

    def test_default_maximum_attempts_is_one(self) -> None:
        """``maximum_attempts`` defaults to 1 (retries off).

        Regression: both dispatch methods hardcoded ``maximum_attempts=2``
        (retries ON) with no field to opt out.
        """
        toolset = TemporalMCPToolSet(name="srv")
        assert toolset.maximum_attempts == 1

    def test_custom_maximum_attempts_stored(self) -> None:
        """An explicit ``maximum_attempts`` is retained."""
        toolset = TemporalMCPToolSet(name="srv", maximum_attempts=3)
        assert toolset.maximum_attempts == 3

    def test_zero_maximum_attempts_rejected(self) -> None:
        """``maximum_attempts`` below 1 is rejected at construction."""
        with pytest.raises(ValueError, match="maximum_attempts"):
            TemporalMCPToolSet(name="srv", maximum_attempts=0)

    def test_non_positive_timeout_rejected(self) -> None:
        """A non-positive ``start_to_close_timeout`` is rejected at construction."""
        with pytest.raises(ValueError, match="start_to_close_timeout"):
            TemporalMCPToolSet(name="srv", start_to_close_timeout=0)

    async def test_list_tools_uses_configured_attempts(self) -> None:
        """``list_tools_in_workflow`` dispatches with the configured attempts."""
        import temporalio.workflow as twf

        toolset = TemporalMCPToolSet(name="srv")
        with patch.object(twf, "execute_activity", new=AsyncMock(return_value=[])) as exec_mock:
            await toolset.list_tools_in_workflow()
        assert exec_mock.call_args.kwargs["retry_policy"].maximum_attempts == 1

    async def test_call_tool_uses_configured_attempts(self) -> None:
        """``call_tool_in_workflow`` dispatches with the opted-in attempts."""
        import temporalio.workflow as twf

        toolset = TemporalMCPToolSet(name="srv", maximum_attempts=5)
        with patch.object(twf, "execute_activity", new=AsyncMock(return_value="ok")) as exec_mock:
            await toolset.call_tool_in_workflow("some_tool", {"a": 1})
        assert exec_mock.call_args.kwargs["retry_policy"].maximum_attempts == 5
