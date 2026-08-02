"""Provider-agnostic LLM response types.

Provides ``LLMResponse`` — the unified response type returned by all
``LLM`` implementations — with a parts-based ``response`` field that
preserves the order and structure of interleaved text, thinking, tool
calls, and refusals.

Each ``LLMResponsePart`` is a typed dataclass that normalizes provider-
specific output into a single format.  Both litellm and Anthropic
converters produce the same part types — the Runner handles ONE format
regardless of provider.

Following Pydantic-AI's pattern: custom normalized types (not provider
SDK types), every provider converts SDK types → framework types.

Example::

    response: LLMResponse = await llm.acomplete(messages="Hello!")
    logger.info(response.content)  # "Hi there!" (convenience property)
    logger.info(response.response)  # [LLMResponseText(text="Hi there!")]
    logger.info(response.usage.total_tokens)  # 15
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from troopai.adk.types.output.llm_response_function_tool_call_param import (
        LLMResponseFunctionToolCallParam,
    )
    from troopai.adk.types.output.llm_response_provider_item_param import (
        LLMResponseProviderItemParam,
    )
    from troopai.adk.types.output.llm_response_reasoning_param import (
        LLMResponseReasoningParam,
        ReasoningRedactedThinkingParam,
    )
    from troopai.adk.types.output.llm_response_refusal_param import LLMResponseRefusalParam
    from troopai.adk.types.output.llm_response_text_param import LLMResponseTextParam
    from troopai.adk.types.output.reasoning_content_text_param import ReasoningContentTextParam
    from troopai.adk.types.output.reasoning_summary_text_param import ReasoningSummaryTextParam
    from troopai.adk.types.tokens.llm_usage import LLMUsage


# ==================================================================
# Response part types (the normalization layer)
# ==================================================================


@dataclasses.dataclass
class LLMResponseText:
    """Text content from the model.

    Supported by: All providers.
    """

    type: Literal["text"] = "text"
    """Part type discriminator."""

    text: str = ""
    """The text content."""

    annotations: list[LLMResponseAnnotation] | None = None
    """URL citations from web search tools.

    Supported by: OpenAI (web search tool).
    """

    def to_param(self) -> LLMResponseTextParam:
        """Convert to ``LLMResponseTextParam`` for conversation replay.

        Produces ``{"type": "output_text", "text": ..., "annotations": ...}``.
        """
        result: LLMResponseTextParam = {"type": "output_text", "text": self.text}
        if self.annotations is not None:
            result["annotations"] = [dataclasses.asdict(a) for a in self.annotations]
        return result


@dataclasses.dataclass
class LLMResponseReasoning:
    """Thinking / reasoning content from the model.

    Each thinking block from the model becomes one
    ``LLMResponseReasoning`` part in the response list.

    The ``signature`` field is critical for multi-turn context
    preservation — providers require it to be passed back in
    subsequent requests during tool-use flows.

    Supported by: Anthropic, Gemini, DeepSeek, OpenAI o-series.

    Attributes:
        type: Part type discriminator.
        thinking: The reasoning / chain-of-thought text content.
        signature: Cryptographic signature for context preservation.
        id: Unique item ID for session persistence.
        summary: Summary of reasoning (for providers that only expose
            summaries, e.g. OpenAI o-series with ``reasoning_effort``).
        encrypted_content: Opaque thinking block data (Anthropic).
            Concatenated redacted thinking block data for multi-turn
            tool use with extended thinking.
        status: Item processing status.
    """

    type: Literal["thinking"] = "thinking"
    """Part type discriminator."""

    thinking: str = ""
    """The reasoning / chain-of-thought text content."""

    signature: str | None = None
    """Cryptographic signature for context preservation.

    Must be preserved verbatim in conversation history.
    Required for Anthropic and Gemini multi-turn tool use.
    ``None`` for providers that don't use signatures (DeepSeek, OpenAI).
    """

    id: str | None = None
    """Unique item ID for session persistence."""

    summary: str | None = None
    """Summary of reasoning.

    For providers that only expose summaries (e.g. OpenAI o-series
    with ``reasoning_effort``). ``None`` when the full chain-of-thought
    is available in ``thinking``.
    """

    encrypted_content: str | None = None
    """Opaque thinking block data (Anthropic).

    Concatenated redacted thinking block data required for multi-turn
    tool use with extended thinking. Must be preserved and replayed
    in subsequent turns.
    """

    is_redacted: bool = False
    """Whether this is an Anthropic redacted-thinking block.

    When ``True``, ``encrypted_content`` carries the opaque redacted
    payload and ``to_param`` emits a ``redacted_thinking`` replay entry,
    so the block round-trips as a redacted block rather than a plain
    thinking block (which Anthropic rejects as a signature mismatch on
    multi-turn extended-thinking tool use).
    """

    status: Literal["in_progress", "completed", "incomplete"] | None = None
    """Item processing status."""

    def to_param(self) -> LLMResponseReasoningParam:
        """Convert to ``LLMResponseReasoningParam`` for conversation replay.

        Maps the flat ``thinking``/``summary``/``signature`` fields into the
        structured ``summary``/``content``/``encrypted_content`` format that
        the replay TypedDict and converters expect. ``summary`` is populated
        only from an explicit ``self.summary`` — the chain-of-thought text
        flows solely into ``content`` so the round-trip never fabricates a
        summary the model did not emit.
        """
        summary_parts: list[ReasoningSummaryTextParam] = []
        if self.summary is not None:
            summary_parts.append({"type": "summary_text", "text": self.summary})
        result: LLMResponseReasoningParam = {"type": "reasoning", "summary": summary_parts}
        if self.id is not None:
            result["id"] = self.id
        if self.is_redacted:
            redacted_payload = self.encrypted_content
            if redacted_payload is None:
                redacted_payload = self.signature
            redacted_part: ReasoningRedactedThinkingParam = {
                "type": "redacted_thinking",
                "data": redacted_payload if redacted_payload is not None else "",
            }
            result["content"] = [redacted_part]
        elif len(self.thinking) > 0:
            content_part: ReasoningContentTextParam = {"type": "reasoning_text", "text": self.thinking}
            result["content"] = [content_part]
        if self.encrypted_content is not None:
            result["encrypted_content"] = self.encrypted_content
        elif self.signature is not None:
            result["encrypted_content"] = self.signature
        if self.status is not None:
            result["status"] = self.status
        return result


@dataclasses.dataclass
class LLMResponseFunctionToolCall:
    """A function tool invocation from the model.

    Supported by: All providers with function calling.
    """

    type: Literal["function_call"] = "function_call"
    """Part type discriminator."""

    call_id: str = ""
    """Correlation ID linking this call to its result."""

    name: str = ""
    """Function name to invoke."""

    arguments: str = ""
    """JSON-encoded arguments string."""

    id: str | None = None
    """Unique item ID (framework-assigned)."""

    status: Literal["in_progress", "completed", "incomplete"] | None = None
    """Processing status."""

    signature: str | None = None
    """Base64-encoded opaque provider signature for context preservation.

    Carries a thinking model's per-tool-call signature bytes — Gemini
    attaches a ``thought_signature`` to a ``function_call`` part — encoded
    as base64 (lossless for arbitrary, non-utf-8 bytes). Must be replayed
    verbatim in conversation history to preserve the model's thinking
    context across multi-turn tool use. ``None`` for providers that do not
    attach a signature to tool calls (Anthropic, OpenAI).
    """

    def to_param(self) -> LLMResponseFunctionToolCallParam:
        """Convert to ``LLMResponseFunctionToolCallParam`` for conversation replay."""
        result: LLMResponseFunctionToolCallParam = {
            "type": "function_call",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.id is not None:
            result["id"] = self.id
        if self.status is not None:
            result["status"] = self.status
        if self.signature is not None:
            result["signature"] = self.signature
        return result


@dataclasses.dataclass
class LLMResponseRefusal:
    """Model refusal to respond.

    Supported by: OpenAI (gpt-4o+).
    """

    type: Literal["refusal"] = "refusal"
    """Part type discriminator."""

    refusal: str = ""
    """The refusal message."""

    def to_param(self) -> LLMResponseRefusalParam:
        """Convert to ``LLMResponseRefusalParam`` for conversation replay."""
        result: LLMResponseRefusalParam = {"type": "refusal", "refusal": self.refusal}
        return result


@dataclasses.dataclass
class LLMResponseAnnotation:
    """Content annotation (URL citation from web search).

    Supported by: OpenAI (web search tool).
    """

    type: Literal["url_citation"] = "url_citation"
    """Annotation type."""

    url: str = ""
    """URL of the cited web resource."""

    title: str = ""
    """Title of the cited web resource."""

    start_index: int = 0
    """Start character index in the text."""

    end_index: int = 0
    """End character index in the text."""


@dataclasses.dataclass
class LLMResponseProviderItem:
    """Provider-native hosted-tool output item.

    Catch-all variant for output items that do not fit the text /
    reasoning / function-call / refusal taxonomy — e.g. OpenAI
    Responses API's ``file_search_call``, ``web_search_call``,
    ``image_generation_call``, ``code_interpreter_call``,
    ``computer_call``, ``mcp_call``. The framework does NOT wrap
    each provider-hosted tool in a dedicated class; instead the raw
    provider payload is carried verbatim in ``raw`` and replayed
    verbatim via :meth:`to_param`.

    ``raw`` is intentionally ``dict[str, Any]`` because it carries
    genuinely dynamic data (provider-specific extensions).

    Supported by: OpenAI Responses API. Other providers currently
    never emit this variant.

    Attributes:
        type: Part type discriminator. Always ``"provider_item"``.
        item_type: Provider-owned item-type string (e.g.
            ``"file_search_call"``, ``"image_generation_call"``).
            Framework code uses this for routing / tracing; no
            discriminated-union branching is expected.
        raw: Provider-owned payload dict, replayed verbatim on
            subsequent turns. MUST include any identity fields
            (``id``, ``status``) the provider requires.
    """

    type: Literal["provider_item"] = "provider_item"
    """Part type discriminator. Always ``"provider_item"``."""

    item_type: str = ""
    """Provider-owned item-type string (e.g. ``"file_search_call"``)."""

    raw: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Provider-owned payload dict. Replayed verbatim."""

    def to_param(self) -> LLMResponseProviderItemParam:
        """Convert to ``LLMResponseProviderItemParam`` for conversation replay.

        Echoes the verbatim provider payload back under
        ``{"type": "provider_item", "item_type": ..., "raw": ...}``.
        """
        result: LLMResponseProviderItemParam = {
            "type": "provider_item",
            "item_type": self.item_type,
            "raw": self.raw,
        }
        return result


# ==================================================================
# Union of all response parts
# ==================================================================

LLMResponsePart = (
    LLMResponseText | LLMResponseReasoning | LLMResponseFunctionToolCall | LLMResponseRefusal | LLMResponseProviderItem
)
"""Discriminated union of response part types.

Each part has a ``type`` field for discrimination. Consumers that
exhaustively ``match`` on the union MUST include a branch for
``LLMResponseProviderItem`` — it carries provider-hosted-tool output
that does not fit the function-call/text/reasoning/refusal taxonomy.
"""


# ==================================================================
# LLMResponse container
# ==================================================================


@dataclasses.dataclass
class LLMResponse:
    """Provider-agnostic response from an LLM call.

    Contains the model's output as an ordered list of typed parts,
    plus metadata (usage, finish reason, timestamp).

    The ``response`` field preserves the structure of the model's
    output — interleaved text, thinking, tool calls, and refusals
    appear in the order the model produced them.

    Convenience properties (``.content``, ``.thinking``, ``.tool_calls``,
    ``.refusal``) provide flat access to the most common fields.

    Validated structured output does NOT live here — it belongs on
    ``RunResult.final_output`` (same as Pydantic-AI and OpenAI SDK).

    Attributes:
        response_id: Response ID from the provider.
        model: The model that generated this response.
        response: Ordered list of typed response parts.
        usage: Token usage statistics.
        finish_reason: Why the model stopped generating.
        timestamp: When the response was received.
    """

    response_id: str
    """Response ID from the provider.

    Format varies (e.g., ``"chatcmpl-..."`` for OpenAI, ``"msg_..."``
    for Anthropic).
    """

    model: str
    """The model that generated this response.

    May differ from the requested model if the provider applied
    routing or aliasing.
    """

    response: list[LLMResponsePart] = dataclasses.field(default_factory=list)
    """Ordered list of typed response parts.

    Contains ``LLMResponseText``, ``LLMResponseReasoning``,
    ``LLMResponseFunctionToolCall``, and ``LLMResponseRefusal``
    parts in the order the model produced them.
    """

    usage: LLMUsage | None = None
    """Token usage statistics for this call."""

    finish_reason: str | None = None
    """Why the model stopped generating.

    Common values:

    - ``"stop"`` / ``"end_turn"`` — natural completion
    - ``"tool_calls"`` / ``"tool_use"`` — model wants to call tools
    - ``"length"`` / ``"max_tokens"`` — hit token limit (truncated)

    Provider-specific — litellm normalizes some values.
    """

    timestamp: datetime | None = None
    """When the response was received.

    Populated from the provider's response metadata when available.
    """

    # ------------------------------------------------------------------
    # Convenience properties — flat access to common fields
    # ------------------------------------------------------------------

    @property
    def content(self) -> str | None:
        """Concatenate text from all ``LLMResponseText`` parts, or ``None``.

        Parts are joined with ``""`` (no separator) — consistent with
        ``ItemHelpers.text_message_output`` — so the reconstructed assistant
        text matches the persisted message-item text and JSON split across
        parts is not corrupted by a spurious newline before validation.
        """
        texts = [p.text for p in self.response if isinstance(p, LLMResponseText)]
        return "".join(texts) if len(texts) > 0 else None

    @property
    def thinking(self) -> str | None:
        """Join thinking from all ``LLMResponseReasoning`` parts, or ``None``."""
        parts = [p.thinking for p in self.response if isinstance(p, LLMResponseReasoning)]
        return "\n".join(parts) if len(parts) > 0 else None

    @property
    def tool_calls(self) -> list[LLMResponseFunctionToolCall]:
        """All ``LLMResponseFunctionToolCall`` parts."""
        return [p for p in self.response if isinstance(p, LLMResponseFunctionToolCall)]

    @property
    def refusal(self) -> str | None:
        """Text from the first ``LLMResponseRefusal`` part, or ``None``."""
        for p in self.response:
            if isinstance(p, LLMResponseRefusal):
                return p.refusal
        return None

    @property
    def thinking_parts(self) -> list[LLMResponseReasoning]:
        """All ``LLMResponseReasoning`` parts (with signatures for replay)."""
        return [p for p in self.response if isinstance(p, LLMResponseReasoning)]


# ==================================================================
# Streaming events (part-based)
# ==================================================================


@dataclasses.dataclass
class LLMStreamEvent:
    """A single event from a streaming LLM response.

    Part-based events that map 1:1 with response parts:

    - ``"part_start"``: A new part begins. ``index`` identifies which
      part, ``part`` contains the initial (possibly empty) part.
    - ``"part_delta"``: Incremental update to a part. ``index``
      identifies which part, ``delta`` contains the text fragment.
    - ``"part_end"``: A part is finalized. ``index`` identifies which.
    - ``"done"``: Streaming complete. ``response`` contains the full
      ``LLMResponse`` with all parts assembled.

    Attributes:
        type: Event type.
        index: Part index (for part_start, part_delta, part_end).
        part: Initial part object (for part_start).
        delta: Text fragment (for part_delta).
        response: Complete response (for done).
    """

    type: Literal["part_start", "part_delta", "part_end", "done"]
    """Event type discriminator."""

    index: int | None = None
    """Part index (which part this event applies to)."""

    part: LLMResponsePart | None = None
    """Initial part (``part_start`` events only)."""

    delta: str | None = None
    """Text fragment (``part_delta`` events only)."""

    response: LLMResponse | None = None
    """Complete response (``done`` events only)."""
