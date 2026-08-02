"""LLMResponseProviderItemParam: TypedDict version of LLMResponseProviderItem for replay.

Represents provider-native hosted-tool output items (file search, web
search, code interpreter, image generation, computer use, MCP calls,
etc.) that do not fit the text / reasoning / function-call / refusal
taxonomy. The OpenAI Responses API accepts the same item dict as
input on subsequent turns, so replay is a verbatim echo of the
provider payload carried in ``raw``.

``raw`` is intentionally ``dict[str, Any]`` because it carries
genuinely dynamic data (provider-specific extensions). Each provider
module that emits a provider item MUST include whatever identity /
state fields the provider requires for replay (typically ``id`` and
``status``).
"""

from __future__ import annotations

from typing import Any, Literal, Required

from typing_extensions import TypedDict

__all__ = ["LLMResponseProviderItemParam"]


class LLMResponseProviderItemParam(TypedDict, total=False):
    """TypedDict version of ``LLMResponseProviderItem`` for replay.

    Attributes:
        type: Discriminator. Always ``"provider_item"``.
        item_type: Provider-owned item-type string (e.g.
            ``"file_search_call"``, ``"image_generation_call"``,
            ``"web_search_call"``).
        raw: Provider-owned payload dict. Replayed verbatim.
    """

    type: Required[Literal["provider_item"]]
    """Discriminator. Always ``"provider_item"``."""

    item_type: Required[str]
    """Provider-owned item-type string (e.g. ``"file_search_call"``)."""

    raw: Required[dict[str, Any]]
    """Provider-owned payload dict. Replayed verbatim."""
