"""Tests for the MCP-typed dispatch from ``LLMResponseProviderItem``.

Three concerns:
1. ``ItemHelpers.response_to_run_items`` lifts ``mcp_approval_request``
   and ``mcp_list_tools`` items into the dedicated typed RunItem
   classes (instead of dropping them into the generic ``ProviderItem``).
2. ``MCPApprovalRequestItem`` / ``MCPApprovalResponseItem`` /
   ``MCPListToolsItem`` round-trip through ``to_param`` as
   ``LLMResponseProviderItemParam`` carrying the OpenAI wire shape
   (``server_label``, ``approve``).
3. Unknown ``item_type`` values still fall through to the generic
   ``ProviderItem`` — the typed dispatch is additive, not exclusive.
"""

from __future__ import annotations

from typing import Any, cast

from troopai.adk.types.items import (
    ItemHelpers,
    MCPApprovalRequestItem,
    MCPApprovalResponseItem,
    MCPListToolsItem,
)
from troopai.adk.types.items.items import ProviderItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseProviderItem,
)
from troopai.adk.types.tools.builtin_tool_types import (
    MCPApprovalRequest,
    MCPApprovalResponse,
    MCPListTools,
    MCPListToolsTool,
)


def _provider_param(item: Any) -> dict[str, Any]:
    """Cast the union-typed ``to_param()`` return to a dict for these
    tests — every item under test produces an
    ``LLMResponseProviderItemParam`` (a TypedDict) at runtime, but
    pyright can't narrow that union from the assertion side."""
    return cast(dict[str, Any], item.to_param())


def _response_with(parts: list[LLMResponseProviderItem]) -> LLMResponse:
    return LLMResponse(response=list(parts), response_id="r1", model="gpt-4o")


class TestDispatchToTypedItems:
    def test_mcp_approval_request_lifts_to_typed_item(self) -> None:
        resp = _response_with(
            [
                LLMResponseProviderItem(
                    item_type="mcp_approval_request",
                    raw={
                        "type": "mcp_approval_request",
                        "id": "mar_1",
                        "server_label": "calendar",
                        "name": "create_event",
                        "arguments": '{"title":"meeting"}',
                    },
                )
            ]
        )
        items = ItemHelpers.response_to_run_items(resp)
        assert len(items) == 1
        item = items[0]
        assert isinstance(item, MCPApprovalRequestItem)
        # Field-name remap: wire `server_label` → framework `server`.
        assert item.raw.server == "calendar"
        assert item.raw.name == "create_event"
        assert item.raw.id == "mar_1"
        assert item.raw.arguments == '{"title":"meeting"}'

    def test_mcp_list_tools_lifts_to_typed_item(self) -> None:
        resp = _response_with(
            [
                LLMResponseProviderItem(
                    item_type="mcp_list_tools",
                    raw={
                        "type": "mcp_list_tools",
                        "server_label": "calendar",
                        "tools": [
                            {"name": "create_event", "description": "Create a calendar event."},
                            {"name": "list_events"},
                        ],
                    },
                )
            ]
        )
        items = ItemHelpers.response_to_run_items(resp)
        assert len(items) == 1
        item = items[0]
        assert isinstance(item, MCPListToolsItem)
        assert item.raw.server == "calendar"
        assert len(item.raw.tools) == 2
        assert item.raw.tools[0].name == "create_event"
        assert item.raw.tools[0].description == "Create a calendar event."

    def test_unknown_item_type_falls_through_to_provider_item(self) -> None:
        resp = _response_with(
            [
                LLMResponseProviderItem(
                    item_type="image_generation_call",
                    raw={"id": "ig_1", "type": "image_generation_call"},
                )
            ]
        )
        items = ItemHelpers.response_to_run_items(resp)
        assert len(items) == 1
        assert isinstance(items[0], ProviderItem)
        assert items[0].raw.item_type == "image_generation_call"

    def test_mixed_response_dispatches_per_item(self) -> None:
        resp = _response_with(
            [
                LLMResponseProviderItem(
                    item_type="mcp_approval_request",
                    raw={
                        "type": "mcp_approval_request",
                        "id": "mar_1",
                        "server_label": "calendar",
                        "name": "create_event",
                        "arguments": "{}",
                    },
                ),
                LLMResponseProviderItem(item_type="file_search_call", raw={"id": "fs_1"}),
            ]
        )
        items = ItemHelpers.response_to_run_items(resp)
        assert len(items) == 2
        assert isinstance(items[0], MCPApprovalRequestItem)
        assert isinstance(items[1], ProviderItem)


class TestRoundTripViaToParam:
    def test_request_item_to_param_round_trip(self) -> None:
        item = MCPApprovalRequestItem(
            raw=MCPApprovalRequest(
                id="mar_1",
                server="calendar",
                name="create_event",
                arguments="{}",
            )
        )
        param = _provider_param(item)
        # Round-trips through the provider_item channel so the existing
        # converter._convert_input_item plumbing replays it verbatim.
        assert param["type"] == "provider_item"
        assert param["item_type"] == "mcp_approval_request"
        # Wire field name is `server_label`, not `server`.
        raw = param["raw"]
        assert raw["server_label"] == "calendar"
        assert raw["name"] == "create_event"
        assert raw["id"] == "mar_1"
        assert raw["arguments"] == "{}"

    def test_response_item_to_param_uses_approve_wire_field(self) -> None:
        item = MCPApprovalResponseItem(
            raw=MCPApprovalResponse(
                approval_request_id="mar_1",
                approved=True,
                reason="user clicked accept",
            )
        )
        param = _provider_param(item)
        assert param["type"] == "provider_item"
        assert param["item_type"] == "mcp_approval_response"
        raw = param["raw"]
        # Wire field is `approve`, not `approved`.
        assert raw["approve"] is True
        assert "approved" not in raw
        assert raw["approval_request_id"] == "mar_1"
        assert raw["reason"] == "user clicked accept"

    def test_response_item_omits_none_optional_fields(self) -> None:
        item = MCPApprovalResponseItem(
            raw=MCPApprovalResponse(
                approval_request_id="mar_1",
                approved=False,
            )
        )
        raw = _provider_param(item)["raw"]
        assert "reason" not in raw
        assert "id" not in raw

    def test_list_tools_item_to_param(self) -> None:
        item = MCPListToolsItem(
            raw=MCPListTools(
                id="lt_1",
                server="calendar",
                tools=[MCPListToolsTool(name="create_event", description="d")],
            )
        )
        param = _provider_param(item)
        assert param["type"] == "provider_item"
        assert param["item_type"] == "mcp_list_tools"
        raw = param["raw"]
        assert raw["server_label"] == "calendar"
        assert raw["tools"][0]["name"] == "create_event"
        # Preserve `id` when set so the round-trip is lossless.
        assert raw["id"] == "lt_1"

    def test_list_tools_item_to_param_omits_empty_id_and_none_error(self) -> None:
        item = MCPListToolsItem(raw=MCPListTools(server="calendar", tools=[]))
        raw = _provider_param(item)["raw"]
        assert "id" not in raw
        assert "error" not in raw

    def test_list_tools_dispatch_preserves_id(self) -> None:
        # The dispatcher reads `id` from the wire payload — pin that.
        resp = _response_with(
            [
                LLMResponseProviderItem(
                    item_type="mcp_list_tools",
                    raw={
                        "type": "mcp_list_tools",
                        "id": "lt_42",
                        "server_label": "srv",
                        "tools": [],
                    },
                )
            ]
        )
        items = ItemHelpers.response_to_run_items(resp)
        assert isinstance(items[0], MCPListToolsItem)
        assert items[0].raw.id == "lt_42"

    def test_list_tools_dispatch_warns_on_malformed_tools(self, caplog: Any) -> None:
        # Non-list `tools` payloads silently produced an empty
        # MCPListToolsItem before the warning was added; pin the new
        # contract that surfaces the corruption.
        import logging

        resp = _response_with(
            [
                LLMResponseProviderItem(
                    item_type="mcp_list_tools",
                    raw={"type": "mcp_list_tools", "server_label": "srv", "tools": "not-a-list"},
                )
            ]
        )
        with caplog.at_level(logging.WARNING):
            items = ItemHelpers.response_to_run_items(resp)
        assert isinstance(items[0], MCPListToolsItem)
        assert len(items[0].raw.tools) == 0
        assert any("expected list" in record.message for record in caplog.records)


class TestEndToEndDispatchAndReplay:
    """A response with an mcp_approval_request lands as
    ``MCPApprovalRequestItem`` in ``new_items``; the developer
    appends an ``MCPApprovalResponseItem`` whose ``to_param`` produces
    a valid wire-format ``mcp_approval_response`` for the next turn."""

    def test_full_round_trip(self) -> None:
        resp = _response_with(
            [
                LLMResponseProviderItem(
                    item_type="mcp_approval_request",
                    raw={
                        "type": "mcp_approval_request",
                        "id": "mar_42",
                        "server_label": "github",
                        "name": "create_pr",
                        "arguments": '{"title":"fix"}',
                    },
                )
            ]
        )
        items = ItemHelpers.response_to_run_items(resp)
        request_item = items[0]
        assert isinstance(request_item, MCPApprovalRequestItem)

        # Developer constructs the response with their decision.
        response_item = MCPApprovalResponseItem(
            raw=MCPApprovalResponse(
                approval_request_id=request_item.raw.id,
                approved=True,
            )
        )
        wire = _provider_param(response_item)
        # The next turn would pass this back to the converter as part
        # of `messages`; converter._convert_input_item forwards
        # provider_item.raw verbatim to the API.
        assert wire["raw"]["approve"] is True
        assert wire["raw"]["approval_request_id"] == "mar_42"
