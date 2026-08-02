"""Tests for the LLMResponseProviderItem response-part variant.

Verifies the new catch-all part type constructs with the expected
defaults, round-trips through ``to_param()`` without losing data,
fits the ``LLMResponsePart`` union's discriminator pattern, and is
accumulated into a ``ProviderItem`` run item by
``ItemHelpers.response_to_run_items``.
"""

from __future__ import annotations

from typing import Any

from troopai.adk.types.items.items import ItemHelpers, ProviderItem
from troopai.adk.types.output import LLMResponseProviderItemParam
from troopai.adk.types.responses import (
    LLMResponse,
    LLMResponseProviderItem,
    LLMResponseText,
)


class TestLLMResponseProviderItem:
    def test_defaults(self) -> None:
        """Construction with no kwargs yields empty payload + the discriminator."""
        item = LLMResponseProviderItem()
        assert item.type == "provider_item"
        assert item.item_type == ""
        assert item.raw == {}

    def test_construct_with_payload(self) -> None:
        payload: dict[str, Any] = {
            "id": "fsc_01",
            "status": "completed",
            "queries": ["What is attention?"],
        }
        item = LLMResponseProviderItem(item_type="file_search_call", raw=payload)
        assert item.item_type == "file_search_call"
        assert item.raw == payload
        # Same object — dataclass is intentionally a reference, not a copy.
        # Consumers MUST treat ``raw`` as read-only.
        assert item.raw is payload

    def test_to_param_round_trip(self) -> None:
        payload: dict[str, Any] = {"id": "ig_01", "status": "completed"}
        item = LLMResponseProviderItem(item_type="image_generation_call", raw=payload)
        p: LLMResponseProviderItemParam = item.to_param()
        assert p["type"] == "provider_item"
        assert p["item_type"] == "image_generation_call"
        assert p["raw"] == payload

    def test_discriminator_isolates_from_other_parts(self) -> None:
        """``isinstance`` dispatch MUST distinguish provider items from other parts."""
        text = LLMResponseText(text="hi")
        provider = LLMResponseProviderItem(item_type="web_search_call", raw={"id": "ws_1"})
        assert isinstance(text, LLMResponseText)
        assert not isinstance(text, LLMResponseProviderItem)
        assert isinstance(provider, LLMResponseProviderItem)
        assert not isinstance(provider, LLMResponseText)


class TestResponseToRunItems:
    def test_provider_item_becomes_provider_run_item(self) -> None:
        """``response_to_run_items`` MUST emit a ``ProviderItem`` for each
        ``LLMResponseProviderItem`` part, preserving ``raw`` by reference."""
        provider = LLMResponseProviderItem(
            item_type="file_search_call",
            raw={"id": "fsc_abc", "status": "completed"},
        )
        response = LLMResponse(
            response_id="resp_01",
            model="gpt-5.1",
            response=[provider],
        )

        items = ItemHelpers.response_to_run_items(response, agent_name="orchestrator")

        provider_items = [i for i in items if isinstance(i, ProviderItem)]
        assert len(provider_items) == 1
        assert provider_items[0].raw is provider
        assert provider_items[0].agent_name == "orchestrator"
        assert provider_items[0].type == "provider_item"

    def test_provider_items_do_not_affect_text_message_output(self) -> None:
        """Mixing a provider item with text parts MUST still yield one
        ``MessageOutputItem`` (for the text) and one ``ProviderItem``."""
        response = LLMResponse(
            response_id="resp_02",
            model="gpt-5.1",
            response=[
                LLMResponseText(text="Here is what I found:"),
                LLMResponseProviderItem(
                    item_type="web_search_call",
                    raw={"id": "ws_1", "status": "completed"},
                ),
            ],
        )

        items = ItemHelpers.response_to_run_items(response)

        from troopai.adk.types.items.items import MessageOutputItem

        assert sum(isinstance(i, MessageOutputItem) for i in items) == 1
        assert sum(isinstance(i, ProviderItem) for i in items) == 1

    def test_provider_item_to_param_returns_replay_dict(self) -> None:
        """The Layer 3 ``ProviderItem.to_param()`` MUST echo the Layer 1 payload.

        ``RunItemBase.to_param()`` is typed ``LLMInputContentItem``; the
        concrete narrowing to :class:`LLMResponseProviderItemParam` happens
        at runtime via the ``type`` discriminator, so we assert it here.
        """
        provider = LLMResponseProviderItem(
            item_type="code_interpreter_call",
            raw={"id": "ci_1", "status": "completed", "code": "print('hi')"},
        )
        run_item = ProviderItem(raw=provider)
        param = run_item.to_param()
        # ``to_param()`` is typed against the broad ``LLMInputContentItem``
        # union — all members are TypedDicts, which are plain dicts at
        # runtime. Access payload fields via ``dict.get()`` directly and
        # assert the discriminator + payload values. No narrowing cast is
        # needed because every assertion goes through ``dict.get()``, which
        # is valid across the union.
        assert param.get("type") == "provider_item"
        assert param.get("item_type") == "code_interpreter_call"
        raw_payload = param.get("raw")
        assert isinstance(raw_payload, dict)
        assert raw_payload["code"] == "print('hi')"


class TestLegacyResponsePartsUnchanged:
    """Regression: adding the new variant MUST NOT change behaviour when the
    response contains only pre-existing part types."""

    def test_text_only_response(self) -> None:
        from troopai.adk.types.items.items import MessageOutputItem

        response = LLMResponse(
            response_id="resp_03",
            model="claude-3",
            response=[LLMResponseText(text="pong")],
        )
        items = ItemHelpers.response_to_run_items(response)
        assert len(items) == 1
        assert isinstance(items[0], MessageOutputItem)
