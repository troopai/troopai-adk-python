"""Google Gemini native LLM implementation using the ``google-genai`` SDK.

Calls ``client.aio.models.generate_content`` /
``generate_content_stream`` directly — no litellm indirection. Single
class supports both the public Gemini Developer API (api_key auth)
and Vertex AI (project + location + credentials) — the SDK's own
constructor dispatches on the ``vertexai`` flag.

Provider-hosted capabilities (Google Search, code execution, URL
context) are wired via the framework's typed hosted-tool classes
(``WebSearchTool``, ``CodeExecutionTool``, ``URLContextTool``).

Usage::

    from troopai.adk.llms.gemini import GeminiLLM
    from troopai.adk.tools import function_tool


    @function_tool(name="lookup", description="Look up a record")
    def lookup(record_id: str) -> str:
        return f"Record {record_id}"


    agent = Agent(
        llm=GeminiLLM(model="gemini-2.5-flash"),
        tools=[lookup],
    )

Refs:
    - Gemini API: https://ai.google.dev/api
    - Python SDK: https://googleapis.github.io/python-genai/
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, overload, override

from google.genai.types import (
    GenerateContentConfig,
    GenerateContentResponse,
)
from httpx import Timeout

from troopai.adk.llms.gemini.gemini_boundary import (
    headers_as_sdk,
    sanitize_for_log,
)
from troopai.adk.llms.gemini.gemini_config import GeminiConfig
from troopai.adk.llms.gemini.gemini_converter import GeminiConverter
from troopai.adk.llms.gemini.gemini_reasoning_resolver import resolve_thinking
from troopai.adk.llms.gemini.gemini_retry import call_with_retry
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.stream_error import stream_with_error_contract
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseReasoning,
    LLMResponseText,
    LLMStreamEvent,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai import Client
    from google.genai.types import (
        Content,
        HttpOptions,
        Tool as GeminiTool,
        ToolConfig,
    )

    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool

logger = logging.getLogger(__name__)


def _clean_schema_for_gemini(
    schema: Any,
    *,
    root: dict[str, Any] | None = None,
    active_refs: frozenset[str] | None = None,
) -> Any:
    """Strip JSON Schema fields that Gemini's ``response_schema`` rejects.

    The Gemini API supports a constrained subset of JSON Schema —
    ``additionalProperties``, ``$ref``, ``$defs``, ``oneOf``, ``allOf``
    are not accepted. The framework's ``AgentOutputSchema`` defaults
    to OpenAI's strict mode which adds ``additionalProperties: false``;
    that field has to come out before the schema reaches Gemini.

    Nested models are emitted by Pydantic as ``{"$ref": "#/$defs/Name"}``
    properties plus a top-level ``$defs`` block. Gemini rejects both, so
    each ``$ref`` is RESOLVED in place — the referenced definition is
    inlined recursively — before the ``$defs`` block is dropped. Deleting
    a ``$ref`` without inlining would leave the property unconstrained
    (an empty ``{}``), silently discarding the nested model's shape.

    ``active_refs`` tracks the ``$ref`` pointers currently being inlined so
    a self-referential model (which Gemini cannot express anyway) drops the
    back-reference instead of recursing without bound.

    Recursively walks the schema and drops the unsupported keys. The
    visit is best-effort — Gemini surfaces a clear 400 error when a
    schema feature is genuinely incompatible, so we strip only the
    common offenders.
    """
    unsupported_keys = {
        "additionalProperties",
        "$schema",
        "$id",
        "$defs",
        "definitions",
        "oneOf",
        "allOf",
    }
    refs = active_refs if active_refs is not None else frozenset()
    if isinstance(schema, list):
        return [_clean_schema_for_gemini(v, root=root, active_refs=refs) for v in schema]
    if not isinstance(schema, dict):
        return schema

    # The first dict seen is the root that holds the ``$defs`` block used
    # to resolve every ``$ref`` encountered deeper in the tree.
    resolution_root = root if root is not None else schema

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref not in refs:
        resolved = _resolve_gemini_ref(ref, resolution_root)
        if resolved is not None:
            # Inline the definition, then layer any sibling keys on top
            # (sibling keys win, mirroring JSON Schema $ref-merge order).
            merged = dict(resolved)
            for key, value in schema.items():
                if key != "$ref":
                    merged[key] = value
            return _clean_schema_for_gemini(merged, root=resolution_root, active_refs=refs | {ref})

    return {
        k: _clean_schema_for_gemini(v, root=resolution_root, active_refs=refs)
        for k, v in schema.items()
        if k not in unsupported_keys and k != "$ref"
    }


def _resolve_gemini_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a local ``#/$defs/<name>`` (or ``#/definitions/<name>``) ref.

    Returns the referenced definition dict, or ``None`` when the ref is
    external or the target is missing — callers then fall back to dropping
    the unresolved ``$ref`` rather than raising.
    """
    for container in ("$defs", "definitions"):
        prefix = f"#/{container}/"
        if ref.startswith(prefix):
            def_name = ref[len(prefix) :]
            defs = root.get(container)
            if isinstance(defs, dict):
                target = defs.get(def_name)
                if isinstance(target, dict):
                    return target
    return None


@dataclasses.dataclass
class _PartAccumulator:
    """Per-streaming-part accumulator.

    Buffers text / tool-input / thought fragments across SSE chunks
    keyed on ``(part_index, part_kind)`` — the chunk-local ``parts``
    position paired with the kind (call / thought / text), since Gemini
    reuses position 0 across kinds in separate chunks. Each Gemini
    streaming chunk carries only the delta text for the current part;
    the framework concatenates them into the final ``LLMResponsePart``.
    """

    is_thought: bool = False
    text_buf: list[str] = dataclasses.field(default_factory=list)
    thought_signature: bytes | None = None
    function_call_id: str | None = None
    function_call_name: str | None = None
    function_call_args: dict[str, Any] | None = None


def _timeout_to_millis(timeout: float | Timeout) -> int | None:
    """Convert an ``LLMConfig.timeout`` (seconds or ``httpx.Timeout``) to milliseconds.

    Gemini's ``HttpOptions.timeout`` is an integer millisecond value. A plain
    number is treated as seconds; an ``httpx.Timeout`` contributes its read
    timeout (the bound that dominates long generations), falling back to its
    connect timeout. Returns ``None`` when no bounded value is available so the
    caller omits the timeout entirely.
    """
    if isinstance(timeout, (int, float)):
        return int(timeout * 1000)
    seconds = timeout.read if timeout.read is not None else timeout.connect
    if seconds is None:
        return None
    return int(seconds * 1000)


def _merge_extra_body(config: LLMConfig) -> dict[str, Any] | None:
    """Merge ``extra_body`` + ``extra_args`` into a single request-body dict.

    Mirrors the OpenAI path: both escape hatches feed the request body, which
    for Gemini rides on ``HttpOptions.extra_body`` because
    ``GenerateContentConfig`` forbids unknown keyword arguments. ``extra_args``
    wins on key collision. ``extra_body`` is only merged when it is a mapping
    (its declared type is the framework-wide ``object``). Returns ``None`` when
    neither field contributes anything.
    """
    merged: dict[str, Any] = {}
    if isinstance(config.extra_body, Mapping):
        merged.update(config.extra_body)
    if config.extra_args is not None:
        merged.update(config.extra_args)
    return merged if len(merged) > 0 else None


class GeminiLLM(LLM):
    """Native Gemini LLM via the ``google-genai`` SDK.

    Single class supporting both backend modes:

    - **Gemini Developer API** (default): pass ``api_key=`` or set
      ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` in the environment.
    - **Vertex AI**: pass ``vertexai=True`` plus ``project=`` and
      ``location=``. Credentials come from ``credentials=`` or
      Application Default Credentials.

    Args:
        model: Gemini model ID (e.g., ``"gemini-2.5-flash"``,
            ``"gemini-2.5-pro"``).
        api_key: API key for the Gemini Developer API. Falls back to
            ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` env vars. Mutually
            exclusive with ``vertexai=True``.
        vertexai: When ``True``, use Vertex AI instead of the Gemini
            Developer API. Requires ``project`` and ``location``.
        project: GCP project ID for Vertex AI mode. Falls back to
            ``GOOGLE_CLOUD_PROJECT``.
        location: GCP region for Vertex AI mode (e.g., ``"us-central1"``).
            Falls back to ``GOOGLE_CLOUD_LOCATION``.
        credentials: Optional ``google.auth.credentials.Credentials``
            for Vertex AI. ``None`` triggers ADC discovery.
        base_url: Optional custom base URL (for testing / proxies).
            Forwarded via ``HttpOptions(base_url=...)``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        credentials: Credentials | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._vertexai = vertexai
        self._project = project
        self._location = location
        self._credentials = credentials
        self._base_url = base_url
        self._client: Client | None = None

    @property
    def model(self) -> str:
        """The model identifier."""
        return self._model

    def _get_client(self) -> Client:
        """Lazy-initialize the google-genai async client."""
        if self._client is None:
            try:
                from google.genai import Client
                from google.genai.types import HttpOptions
            except ImportError as e:
                raise ImportError(
                    "The 'google-genai' package is required for GeminiLLM. Install it with: pip install google-genai"
                ) from e

            http_options: HttpOptions | None = None
            if self._base_url is not None:
                http_options = HttpOptions(base_url=self._base_url)

            if self._vertexai:
                self._client = Client(
                    vertexai=True,
                    project=self._project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    location=self._location or os.environ.get("GOOGLE_CLOUD_LOCATION"),
                    credentials=self._credentials,
                    http_options=http_options,
                )
            else:
                self._client = Client(
                    api_key=self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
                    http_options=http_options,
                )
        return self._client

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @overload
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: Literal[False] = False,
    ) -> LLMResponse: ...

    # PyCharm pairs each subclass overload against the base class
    # implementation rather than the corresponding base overload,
    # flagging the narrower Literal[True] / AsyncIterator return as
    # an LSP mismatch. mypy + pyright clear the site.
    # noinspection PyMethodOverriding
    @overload
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        *,
        stream: Literal[True],
    ) -> AsyncIterator[LLMStreamEvent]: ...

    @override
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        """Call the Gemini API.

        Orchestrates: convert → resolve params → build
        ``GenerateContentConfig`` → call ``generate_content`` /
        ``generate_content_stream`` → parse / stream.
        """
        from troopai.adk.llms.llm_config import LLMConfig

        config = llm_config or LLMConfig()

        # Plain-text output schemas behave like no schema at all.
        wants_structured = output_schema is not None and not output_schema.is_plain_text()

        # 1. Convert messages → (system_instruction, contents).
        system_instruction, contents = GeminiConverter.items_to_contents(messages)

        # 2. Convert tools.
        wire_tools = GeminiConverter.convert_tools(tools) if tools is not None and len(tools) > 0 else None

        # 3. Resolve tool_choice → ToolConfig.
        tool_config = GeminiConverter.convert_tool_choice(
            config.tool_choice,
            tools_present=wire_tools is not None,
        )

        # 4. Resolve thinking.
        thinking_config = resolve_thinking(config)

        # 5. Build GenerateContentConfig.
        gen_config = self._build_generate_content_config(
            config=config,
            system_instruction=system_instruction,
            wire_tools=wire_tools,
            tool_config=tool_config,
            thinking_config=thinking_config,
            wants_structured=wants_structured,
            output_schema=output_schema,
        )

        logger.info(
            "Gemini API call: model=%s, contents=%d, tools=%d, stream=%s, structured=%s",
            sanitize_for_log(self._model),
            len(contents),
            sum(
                len(t.function_declarations or [])
                + (1 if t.google_search is not None else 0)
                + (1 if t.code_execution is not None else 0)
                + (1 if t.url_context is not None else 0)
                for t in (wire_tools or [])
            ),
            stream,
            wants_structured,
        )

        if stream is True:
            # Streaming defers structured-output validation to the
            # Runner's pass over the accumulated text — the
            # ``output_schema`` argument is intentionally not threaded
            # into ``_stream`` (the Gemini synthetic-tool pattern is
            # not used; ``response_schema`` is set on ``gen_config``
            # above and the model produces JSON text directly).
            # Cross-provider streaming-error contract: a mid-stream failure
            # emits a terminal done(finish_reason="error") then re-raises.
            return stream_with_error_contract(self._stream(contents, gen_config), model=self._model, logger=logger)

        client = self._get_client()

        # ``contents`` is ``list[Content]`` after our converter; the
        # SDK's ``ContentListUnion`` accepts ``list[Content | str | ...]``
        # but Python's list is invariant so mypy refuses the structural
        # subtype. The runtime accepts the value unchanged. ``cast``
        # would also work; widening the local annotation avoids the
        # ``cast`` import.
        contents_arg: Any = contents

        async def _do_call() -> GenerateContentResponse:
            return await client.aio.models.generate_content(
                model=self._model,
                contents=contents_arg,
                config=gen_config,
            )

        if config.retry_policy is not None:
            response = await call_with_retry(
                _do_call,
                config.retry_policy,
                model=self._model,
            )
        else:
            response = await _do_call()

        usage_md = response.usage_metadata
        logger.info(
            "Gemini response: model=%s, input_tokens=%d, output_tokens=%d, total=%d",
            sanitize_for_log(response.model_version or self._model),
            usage_md.prompt_token_count if usage_md is not None else 0,
            usage_md.candidates_token_count if usage_md is not None else 0,
            usage_md.total_token_count if usage_md is not None else 0,
        )

        return GeminiConverter.response_to_llm_response(response)

    # ------------------------------------------------------------------
    # GenerateContentConfig assembly
    # ------------------------------------------------------------------

    def _build_generate_content_config(
        self,
        *,
        config: LLMConfig,
        system_instruction: str | None,
        wire_tools: list[GeminiTool] | None,
        tool_config: ToolConfig | None,
        thinking_config: Any,
        wants_structured: bool,
        output_schema: AgentOutputSchemaBase | None,
    ) -> GenerateContentConfig:
        """Assemble ``GenerateContentConfig`` from the resolved params.

        ``response_schema`` is set ONLY when structured output is
        requested AND the schema is not plain text. The framework's
        ``output_schema.json_schema()`` returns a JSON Schema dict —
        the SDK accepts it directly for ``response_schema`` (along
        with Pydantic types and dict).
        """
        kwargs: dict[str, Any] = {}
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if wire_tools is not None:
            kwargs["tools"] = wire_tools
        if tool_config is not None:
            kwargs["tool_config"] = tool_config
        if thinking_config is not None:
            kwargs["thinking_config"] = thinking_config
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        if config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = config.max_output_tokens
        if config.stop_sequences is not None:
            kwargs["stop_sequences"] = list(config.stop_sequences)
        if config.frequency_penalty is not None:
            kwargs["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            kwargs["presence_penalty"] = config.presence_penalty
        if config.seed is not None:
            kwargs["seed"] = config.seed
        if config.response_logprobs is not None:
            kwargs["response_logprobs"] = config.response_logprobs
        if config.top_logprobs is not None:
            kwargs["logprobs"] = config.top_logprobs

        # HTTP boundary: headers + request timeout + extra_body/extra_args,
        # folded into one per-request HttpOptions (omitted when all unset).
        http_options = self._build_http_options(config)
        if http_options is not None:
            kwargs["http_options"] = http_options

        # GeminiConfig-specific typed fields.
        if isinstance(config, GeminiConfig):
            if config.safety_settings is not None:
                kwargs["safety_settings"] = config.safety_settings
            if config.cached_content_name is not None:
                kwargs["cached_content"] = config.cached_content_name
            if config.response_modalities is not None:
                kwargs["response_modalities"] = config.response_modalities

        # Native structured output. Gemini's ``response_schema`` accepts
        # a constrained JSON Schema dialect — it rejects fields like
        # ``additionalProperties`` that OpenAI's strict mode adds. We
        # clean the schema before passing.
        if wants_structured and output_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = _clean_schema_for_gemini(output_schema.json_schema())

        return GenerateContentConfig(**kwargs)

    def _build_http_options(self, config: LLMConfig) -> HttpOptions | None:
        """Assemble a per-request ``HttpOptions`` from generic ``LLMConfig`` fields.

        Folds the header boundary together with the request timeout and the
        ``extra_body`` / ``extra_args`` escape hatches so none of them are
        silently dropped. Without a forwarded timeout a Gemini call can hang
        indefinitely; ``config.timeout`` (seconds, or an ``httpx.Timeout``) is
        converted to Gemini's millisecond ``HttpOptions.timeout``. Extra
        request-body fields ride on ``HttpOptions.extra_body`` because
        ``GenerateContentConfig`` forbids unknown keyword arguments. Returns
        ``None`` when no field is set so the caller omits ``http_options``.
        """
        from google.genai.types import HttpOptions

        opts: dict[str, Any] = {}
        if config.extra_headers is not None:
            sdk_headers = headers_as_sdk(config.extra_headers)
            if sdk_headers is not None:
                opts["headers"] = sdk_headers
        if config.timeout is not None:
            timeout_ms = _timeout_to_millis(config.timeout)
            if timeout_ms is not None:
                opts["timeout"] = timeout_ms
        extra_body = _merge_extra_body(config)
        if extra_body is not None:
            opts["extra_body"] = extra_body
        if len(opts) == 0:
            return None
        return HttpOptions(**opts)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream(
        self,
        contents: list[Content],
        gen_config: GenerateContentConfig,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Process Gemini streaming chunks → ``LLMStreamEvent`` yields.

        Each Gemini streaming chunk is a complete ``GenerateContentResponse``
        whose ``candidates[0].content.parts[i].text`` carries only the
        delta for that chunk. The framework concatenates across chunks
        keyed on ``(part_index, part_kind)`` to assemble the final
        response. ``usage_metadata`` arrives only on the last chunk.

        Yields ``part_start`` / ``part_delta`` / ``part_end`` events
        per part, ending with a ``"done"`` event carrying the assembled
        :class:`LLMResponse`.
        """
        client = self._get_client()
        # Same widening rationale as the non-streaming path: the SDK
        # accepts ``list[Content]`` at runtime; Python's invariant
        # list narrows the static type away from the SDK's structural
        # union.
        contents_arg: Any = contents
        stream = await client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents_arg,
            config=gen_config,
        )

        # Gemini streaming carries one part per chunk at ``parts[0]``; the
        # ``enumerate`` position resets to 0 on every chunk and is reused
        # across part kinds (a thought delta and the answer-text delta both
        # arrive at index 0 in separate chunks). Keying purely on that
        # position would fold the answer text into the reasoning slot and
        # drop it from the response. The accumulator key therefore pairs the
        # position with the part kind (``"call"`` / ``"thought"`` / ``"text"``)
        # so distinct kinds at the same position stay separate. Each distinct
        # key is assigned a monotonic integer for ``LLMStreamEvent.index``,
        # which downstream consumers use as an ``int`` set membership key.
        #
        # Function calls need a stronger key than (position, kind): Gemini
        # streams each parallel call complete in its own chunk at parts[0],
        # so every one would collide on ``(0, "call")`` and overwrite the
        # previous call. They therefore key on a monotonic ``call_seq`` so
        # each distinct call gets its own accumulator and is appended.
        accumulators: dict[tuple[int, str], _PartAccumulator] = {}
        part_order: list[tuple[int, str]] = []
        emit_index: dict[tuple[int, str], int] = {}
        emitted_starts: set[tuple[int, str]] = set()
        message_id = ""
        model_name = self._model
        final_usage = None
        final_finish_reason: str | None = None
        call_seq = 0

        async for chunk in stream:
            if chunk.response_id is not None and len(message_id) == 0:
                message_id = chunk.response_id
            if chunk.model_version is not None:
                # Sanitise the API-sourced model string before it lands
                # on ``LLMResponse.model``; downstream callers may log
                # the field directly and a model string with embedded
                # line terminators would split log records.
                model_name = sanitize_for_log(chunk.model_version)
            if chunk.usage_metadata is not None:
                final_usage = chunk.usage_metadata

            candidates = chunk.candidates or []
            if len(candidates) == 0:
                continue
            cand = candidates[0]
            if cand.finish_reason is not None:
                final_finish_reason = (
                    cand.finish_reason.value if hasattr(cand.finish_reason, "value") else str(cand.finish_reason)
                )

            content = cand.content
            if content is None or content.parts is None:
                continue

            for idx, part in enumerate(content.parts):
                if part.function_call is not None:
                    # Distinct monotonic key so parallel calls never collide.
                    kind = "call"
                    key_index = call_seq
                    call_seq += 1
                elif part.thought is True:
                    kind = "thought"
                    key_index = idx
                else:
                    kind = "text"
                    key_index = idx
                acc_key = (key_index, kind)

                acc = accumulators.get(acc_key)
                if acc is None:
                    acc = _PartAccumulator(is_thought=kind == "thought")
                    accumulators[acc_key] = acc
                    part_order.append(acc_key)
                    emit_index[acc_key] = len(emit_index)
                out_index = emit_index[acc_key]

                if part.function_call is not None:
                    fc = part.function_call
                    acc.function_call_id = fc.id or fc.name
                    acc.function_call_name = fc.name
                    acc.function_call_args = dict(fc.args) if fc.args is not None else {}
                    if acc_key not in emitted_starts:
                        yield LLMStreamEvent(
                            type="part_start",
                            index=out_index,
                            part=LLMResponseFunctionToolCall(
                                call_id=acc.function_call_id or "",
                                name=acc.function_call_name or "",
                            ),
                        )
                        emitted_starts.add(acc_key)
                    continue

                if part.thought is True:
                    acc.is_thought = True
                    if part.thought_signature is not None:
                        acc.thought_signature = part.thought_signature
                    text = part.text or ""
                    if acc_key not in emitted_starts:
                        yield LLMStreamEvent(
                            type="part_start",
                            index=out_index,
                            part=LLMResponseReasoning(),
                        )
                        emitted_starts.add(acc_key)
                    if len(text) > 0:
                        acc.text_buf.append(text)
                        yield LLMStreamEvent(type="part_delta", index=out_index, delta=text)
                    continue

                text = part.text or ""
                if acc_key not in emitted_starts:
                    yield LLMStreamEvent(
                        type="part_start",
                        index=out_index,
                        part=LLMResponseText(),
                    )
                    emitted_starts.add(acc_key)
                if len(text) > 0:
                    acc.text_buf.append(text)
                    yield LLMStreamEvent(type="part_delta", index=out_index, delta=text)

        # Build final response parts.
        parts: list[LLMResponsePart] = []
        for acc_key in part_order:
            acc = accumulators[acc_key]
            out_index = emit_index[acc_key]
            if acc.function_call_args is not None:
                import json

                parts.append(
                    LLMResponseFunctionToolCall(
                        call_id=acc.function_call_id or "",
                        name=acc.function_call_name or "",
                        arguments=json.dumps(acc.function_call_args),
                    )
                )
                yield LLMStreamEvent(type="part_end", index=out_index)
            elif acc.is_thought:
                # Base64-encode the opaque signature bytes (lossless) so the
                # replay round-trip through ``encrypted_content`` preserves
                # them exactly — matches the non-streaming converter path.
                sig = (
                    base64.b64encode(acc.thought_signature).decode("ascii")
                    if acc.thought_signature is not None
                    else None
                )
                parts.append(
                    LLMResponseReasoning(
                        thinking="".join(acc.text_buf),
                        encrypted_content=sig,
                    )
                )
                yield LLMStreamEvent(type="part_end", index=out_index)
            else:
                parts.append(LLMResponseText(text="".join(acc.text_buf)))
                yield LLMStreamEvent(type="part_end", index=out_index)

        usage = GeminiConverter._parse_usage(final_usage) if final_usage is not None else None

        response = LLMResponse(
            response_id=message_id,
            model=model_name,
            response=parts,
            usage=usage,
            finish_reason=final_finish_reason,
        )

        yield LLMStreamEvent(type="done", response=response)
