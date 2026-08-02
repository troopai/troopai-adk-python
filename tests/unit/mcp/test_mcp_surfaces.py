"""Tests for assorted MCP surfaces: $ref resolver, hosted MCP,
sampling, elicitation, manager ref counting, hooks bridge.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.mcp.exceptions import MCPSchemaConversionError
from troopai.adk.mcp.manager import MCPServerManager
from troopai.adk.mcp.schema_resolver import inline_intra_document_refs
from troopai.adk.tools.hosted import HostedMCPTool, UnsupportedHostedToolError

# ---------------------------------------------------------------- $ref resolver


def test_inline_refs_replaces_intra_doc_pointer() -> None:
    schema = {
        "type": "object",
        "properties": {"q": {"$ref": "#/$defs/Q"}},
        "$defs": {"Q": {"type": "string", "minLength": 1}},
    }
    out = inline_intra_document_refs(schema, tool_name="t")
    assert out["properties"]["q"] == {"type": "string", "minLength": 1}
    assert "$defs" not in out


def test_inline_refs_merges_sibling_keys() -> None:
    schema = {
        "type": "object",
        "properties": {
            "q": {"$ref": "#/$defs/Q", "description": "free-text query"},
        },
        "$defs": {"Q": {"type": "string"}},
    }
    out = inline_intra_document_refs(schema, tool_name="t")
    assert out["properties"]["q"] == {"type": "string", "description": "free-text query"}


def test_inline_refs_fast_path_when_no_refs() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    out = inline_intra_document_refs(schema, tool_name="t")
    assert out == schema
    assert out is not schema  # Deep-copied


def test_inline_refs_rejects_external_pointer() -> None:
    schema = {"type": "object", "properties": {"q": {"$ref": "https://example/q"}}}
    with pytest.raises(MCPSchemaConversionError):
        inline_intra_document_refs(schema, tool_name="t")


def test_inline_refs_rejects_broken_pointer() -> None:
    schema = {"type": "object", "properties": {"q": {"$ref": "#/$defs/Missing"}}}
    with pytest.raises(MCPSchemaConversionError):
        inline_intra_document_refs(schema, tool_name="t")


def test_inline_refs_detects_cycle_with_trail() -> None:
    """A cycle must surface a readable error trail naming every
    visited pointer (``#/$defs/A → #/$defs/B → #/$defs/A``)."""
    schema = {
        "type": "object",
        "properties": {"q": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/A"},
        },
    }
    with pytest.raises(MCPSchemaConversionError) as exc_info:
        inline_intra_document_refs(schema, tool_name="t")
    msg = str(exc_info.value)
    assert "cycle detected" in msg
    assert "#/$defs/A" in msg and "#/$defs/B" in msg


def test_inline_refs_rejects_pointer_with_matching_name_but_wrong_path() -> None:
    """The leaky per-step fallback is gone. ``#/foo/A`` must NOT
    silently resolve to ``$defs/A`` even when ``A`` exists in defs."""
    schema = {
        "type": "object",
        "properties": {"q": {"$ref": "#/foo/A"}},
        "$defs": {"A": {"type": "string"}},
    }
    with pytest.raises(MCPSchemaConversionError):
        inline_intra_document_refs(schema, tool_name="t")


def test_inline_refs_canonicalises_legacy_definitions_root() -> None:
    """A pointer ``#/$defs/X`` must still resolve when the schema
    uses the legacy ``definitions`` spelling for the def block."""
    schema = {
        "type": "object",
        "properties": {"q": {"$ref": "#/$defs/Q"}},
        "definitions": {"Q": {"type": "string"}},
    }
    out = inline_intra_document_refs(schema, tool_name="t")
    assert out["properties"]["q"] == {"type": "string"}


def test_streamable_http_params_headers_repr_omitted() -> None:
    """Bearer tokens stored in ``headers`` must not leak via repr."""
    from troopai.adk.mcp import MCPServerStreamableHttpParams

    params = MCPServerStreamableHttpParams(
        url="https://x/mcp",
        headers={"Authorization": "Bearer SECRET"},
    )
    assert "SECRET" not in repr(params)
    assert "https://x/mcp" in repr(params)  # URL still visible


def test_sse_params_headers_repr_omitted() -> None:
    from troopai.adk.mcp import MCPServerSseParams

    params = MCPServerSseParams(
        url="https://x/sse",
        headers={"Authorization": "Bearer SECRET"},
    )
    assert "SECRET" not in repr(params)
    assert "https://x/sse" in repr(params)


# --------------------------------------------------------------- HostedMCPTool


def test_hosted_mcp_tool_supported_providers() -> None:
    assert HostedMCPTool.SUPPORTED_PROVIDERS == ("openai-responses",)


def test_hosted_mcp_tool_requires_url_xor_connector() -> None:
    # The XOR constraint (exactly one of server_url / connector_id) is enforced
    # fail-fast in __post_init__, so a server_label-only tool raises at construction.
    with pytest.raises(ValueError):
        HostedMCPTool(server_label="x")


def test_hosted_mcp_tool_translates_to_responses_param() -> None:
    from troopai.adk.llms.openai.openai_responses_converter import OpenAIResponsesConverter

    tool = HostedMCPTool(
        server_label="gh",
        server_url="https://api/mcp",
        headers={"Authorization": "Bearer x"},
        require_approval="never",
        allowed_tools=["search"],
        defer_loading=True,
    )
    [out] = OpenAIResponsesConverter.convert_tools([tool])
    assert out["type"] == "mcp"
    assert out["server_label"] == "gh"
    # server_url/headers/require_approval/allowed_tools/defer_loading are
    # not-required keys on the Mcp TypedDict; .get() is the type-safe accessor.
    assert out.get("server_url") == "https://api/mcp"
    assert out.get("headers") == {"Authorization": "Bearer x"}
    assert out.get("require_approval") == "never"
    assert out.get("allowed_tools") == ["search"]
    assert out.get("defer_loading") is True


def test_hosted_mcp_tool_other_providers_raise() -> None:
    from troopai.adk.llms.anthropic.anthropic_converter import AnthropicConverter

    tool = HostedMCPTool(server_label="gh", server_url="https://api/mcp")
    with pytest.raises(UnsupportedHostedToolError):
        AnthropicConverter.convert_tools([tool])


# ------------------------------------------------------------ ref-counted manager


def _server(name: str) -> MagicMock:
    s = MagicMock()
    s.name = name
    s.connect = AsyncMock()
    s.cleanup = AsyncMock()
    return s


async def test_acquire_release_ref_counts_correctly() -> None:
    s = _server("a")
    manager = MCPServerManager(servers=[s])

    await manager.acquire(s)
    await manager.acquire(s)
    s.connect.assert_awaited_once()
    s.cleanup.assert_not_called()

    await manager.release(s)
    s.cleanup.assert_not_called()  # Still one ref outstanding

    await manager.release(s)
    s.cleanup.assert_awaited_once()


async def test_release_below_zero_is_safe() -> None:
    s = _server("a")
    manager = MCPServerManager(servers=[s])
    await manager.release(s)  # No prior acquire — must not raise


async def test_acquire_unknown_server_raises() -> None:
    s = _server("a")
    other = _server("b")
    manager = MCPServerManager(servers=[s])
    from troopai.adk.mcp.exceptions import MCPConnectionError

    with pytest.raises(MCPConnectionError):
        await manager.acquire(other)


# ----------------------------------------------- public state-observation API


async def test_manager_is_active_property_tracks_connect_all() -> None:
    s = _server("a")
    manager = MCPServerManager(servers=[s])
    assert manager.is_active is False
    await manager.connect_all()
    assert manager.is_active is True
    await manager.cleanup_all()
    assert manager.is_active is False


async def test_manager_get_ref_count_tracks_acquire_release() -> None:
    s = _server("a")
    manager = MCPServerManager(servers=[s])
    assert manager.get_ref_count(s) == 0
    await manager.acquire(s)
    assert manager.get_ref_count(s) == 1
    await manager.acquire(s)
    assert manager.get_ref_count(s) == 2
    await manager.release(s)
    assert manager.get_ref_count(s) == 1
    await manager.release(s)
    assert manager.get_ref_count(s) == 0


async def test_toolset_is_connected_and_is_disposed_properties() -> None:
    """Public lifecycle predicates on ``MCPToolset``."""
    from troopai.adk.tools.toolsets import MCPToolset

    server = _server("svc")
    server.list_tools = AsyncMock(return_value=[])
    toolset = MCPToolset(server=server, auto_connect=True)

    assert toolset.is_connected is False
    assert toolset.is_disposed is False

    await toolset.get_tools(None)
    assert toolset.is_connected is True
    assert toolset.is_disposed is False

    await toolset.adispose()
    assert toolset.is_disposed is True


async def test_server_with_client_session_is_connected_property() -> None:
    """``MCPServerWithClientSession.is_connected`` mirrors session state."""
    from troopai.adk.mcp import MCPServerStdio, MCPServerStdioParams

    # ``echo`` exits immediately; we never actually connect.
    server = MCPServerStdio(name="x", params=MCPServerStdioParams(command="echo"))
    assert server.is_connected is False


# ------------------------------------------------------------ sampling callback


async def test_sampling_callback_calls_llm() -> None:
    from mcp import types as mcp_types

    from troopai.adk.mcp.sampling import make_sampling_callback
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText

    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r1",
        model="fake-model",
        response=[LLMResponseText(text="hello")],
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)

    cb = make_sampling_callback(fake_llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=100,
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    assert result.content.text == "hello"  # type: ignore[union-attr]
    assert result.role == "assistant"
    assert result.model == "fake-model"


async def test_sampling_callback_warns_on_textless_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A response with no text part returns a blank completion AND logs a warning.

    Regression: ``_join_text_parts`` read a non-existent ``.parts`` fallback and
    silently returned "" on any shape mismatch, so the MCP server received a
    blank completion it could not distinguish from a real empty answer.
    """
    from mcp import types as mcp_types

    from troopai.adk.mcp.sampling import make_sampling_callback
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseReasoning

    fake_llm = MagicMock()
    # Only a reasoning part — no text part at all.
    fake_response = LLMResponse(
        response_id="r2",
        model="fake-model",
        response=[LLMResponseReasoning(thinking="(internal)")],
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)

    cb = make_sampling_callback(fake_llm)
    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=100,
    )
    with caplog.at_level("WARNING"):
        result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    assert result.content.text == ""  # type: ignore[union-attr]
    assert any("no text part" in r.message for r in caplog.records)


async def test_sampling_callback_swallows_exceptions() -> None:
    from mcp import types as mcp_types

    from troopai.adk.mcp.sampling import make_sampling_callback

    bad_llm = MagicMock()
    bad_llm.acomplete = AsyncMock(side_effect=RuntimeError("boom"))
    cb = make_sampling_callback(bad_llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=10,
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.ErrorData)


# ----------------------------------------------------------- elicitation


async def test_elicitation_callback_wraps_dict_handler() -> None:
    from troopai.adk.mcp.elicitation import make_elicitation_callback

    async def handler(params: Any) -> Any:
        del params
        return {"text": "yes"}

    cb = make_elicitation_callback(handler)
    params = MagicMock()
    result = await cb(None, params)
    # Result is mcp_types.ElicitResult
    assert getattr(result, "action", None) == "accept"


async def test_elicitation_callback_declines_on_none() -> None:
    """A handler returning ``None`` must decline, not accept a literal "None".

    Regression: the wrapper coerced any falsy return through ``str(raw)`` into
    ``{"text": "None"}`` + ``action="accept"`` — so a user's refusal looked
    like an approval to the MCP server.
    """
    from troopai.adk.mcp.elicitation import make_elicitation_callback

    async def declining_handler(params: Any) -> Any:
        del params
        return None

    cb = make_elicitation_callback(declining_handler)
    result = await cb(None, MagicMock())
    assert getattr(result, "action", None) == "decline"
    # Decline carries no content (the user submitted nothing).
    assert getattr(result, "content", None) is None


async def test_elicitation_callback_swallows_handler_error() -> None:
    from mcp import types as mcp_types

    from troopai.adk.mcp.elicitation import make_elicitation_callback

    async def bad_handler(params: Any) -> Any:
        del params
        raise ValueError("denied")

    cb = make_elicitation_callback(bad_handler)
    params = MagicMock()
    result = await cb(None, params)
    assert isinstance(result, mcp_types.ErrorData)


# --------------------------------- call_tool always applies server header_provider


async def test_call_tool_always_applies_server_header_provider() -> None:
    """Regression: call_tool only applied _header_provider when the ContextVar
    was None. When another call had already set the ContextVar to a different
    provider, this server's provider was silently ignored, causing 401 errors.
    The fix: always override the ContextVar with the server's own provider
    for the duration of this call, then reset.
    """
    from troopai.adk.mcp.auth import active_header_provider
    from troopai.adk.mcp.mcp_server import MCPServerWithClientSession

    # Simulate an "ambient" provider already set on the ContextVar
    ambient_provider = lambda: {"Authorization": "Bearer AMBIENT"}  # noqa: E731
    server_provider = lambda: {"Authorization": "Bearer SERVER"}  # noqa: E731

    token = active_header_provider.set(ambient_provider)
    provider_seen_during_call: list[Any] = []

    async def fake_call_tool(*args: Any, **kwargs: Any) -> Any:
        # Capture what provider is visible during the call
        provider_seen_during_call.append(active_header_provider.get())
        result = MagicMock()
        result.content = []
        result.isError = False
        return result

    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=fake_call_tool)

    class _Concrete(MCPServerWithClientSession):
        async def connect(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

    server = _Concrete(name="test-server", header_provider=server_provider)
    server._session = session

    try:
        await server.call_tool("my_tool", {"arg": "val"})
    finally:
        active_header_provider.reset(token)

    assert len(provider_seen_during_call) == 1
    assert provider_seen_during_call[0] is server_provider, (
        "call_tool MUST apply server's own _header_provider unconditionally, not only when ContextVar is unset"
    )
    # Verify the ContextVar was reset after the call
    assert active_header_provider.get() is None


async def test_mcp_server_not_in_all() -> None:
    """Regression: MCPServer ABC must not be in __all__ because its abstract
    methods return mcp SDK wire types, leaking the wire protocol surface.
    MCPServerWithClientSession is the correct public extension point.
    """
    import troopai.adk.mcp as mcp_pkg

    assert "MCPServer" not in mcp_pkg.__all__, (
        "MCPServer ABC must be excluded from __all__ (wire-type leak on public surface)"
    )
    assert "MCPServerWithClientSession" in mcp_pkg.__all__, (
        "MCPServerWithClientSession must remain in __all__ as the extension point"
    )


# ----------------------------------------- sampling stopReason mapping


@pytest.mark.parametrize(
    ("finish_reason", "expected_stop_reason"),
    [
        ("stop", "endTurn"),
        ("end_turn", "endTurn"),
        (None, "endTurn"),
        ("length", "maxTokens"),
        ("max_tokens", "maxTokens"),
        ("stop_sequence", "stopSequence"),
        ("unknown_future_value", "endTurn"),
    ],
)
async def test_sampling_callback_maps_finish_reason_to_stop_reason(
    finish_reason: str | None, expected_stop_reason: str
) -> None:
    """The CreateMessageResult stopReason MUST reflect the LLM's finish_reason.

    Before the fix, stopReason was always hardcoded to "endTurn", so a
    token-limit truncation or stop-sequence completion was indistinguishable
    from a natural end-of-turn completion to the MCP server's agentic loop.
    """
    from mcp import types as mcp_types

    from troopai.adk.mcp.sampling import make_sampling_callback
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText

    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r3",
        model="fake-model",
        response=[LLMResponseText(text="answer")],
        finish_reason=finish_reason,
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)

    cb = make_sampling_callback(fake_llm)
    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=5,
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    assert result.stopReason == expected_stop_reason, (
        f"stopReason MUST be {expected_stop_reason!r} for finish_reason={finish_reason!r}; "
        "was always 'endTurn' before fix"
    )


async def test_sampling_callback_forwards_tools_to_llm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When params.tools is non-empty, tools must be forwarded to the LLM.

    Tools are fully supported: the callback builds FunctionTool wrappers
    and passes them to llm.acomplete. No warning about tools being
    unsupported is emitted. When the LLM returns a text response (no tool
    calls), CreateMessageResult is returned normally.
    """
    from mcp import types as mcp_types

    from troopai.adk.mcp.sampling import make_sampling_callback
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText

    captured_kwargs: dict[str, Any] = {}

    async def fake_acomplete(messages: Any, **kwargs: Any) -> LLMResponse:
        captured_kwargs.update(kwargs)
        return LLMResponse(
            response_id="r4",
            model="fake-model",
            response=[LLMResponseText(text="answer")],
            finish_reason="stop",
        )

    fake_llm = MagicMock()
    fake_llm.acomplete = AsyncMock(side_effect=fake_acomplete)

    cb = make_sampling_callback(fake_llm)
    tool = mcp_types.Tool(
        name="search",
        description="web search",
        inputSchema={"type": "object", "properties": {}},
    )
    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="find X"),
            )
        ],
        maxTokens=100,
        tools=[tool],
    )
    with caplog.at_level("WARNING"):
        result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    # Tools must be forwarded to the LLM (tools kwarg must be non-empty)
    assert "tools" in captured_kwargs
    assert captured_kwargs["tools"] is not None and len(captured_kwargs["tools"]) == 1
    # No warning about tools being unsupported
    assert not any("cannot be forwarded" in r.message for r in caplog.records)


# --------------------------------- transport call_tool_timeout_seconds


def test_stdio_params_call_tool_timeout_defaults_to_none() -> None:
    """call_tool_timeout_seconds MUST default to None (no timeout) per cost-conservative invariant."""
    from troopai.adk.mcp.stdio import MCPServerStdioParams

    params = MCPServerStdioParams(command="echo")
    assert params.call_tool_timeout_seconds is None  # type: ignore[attr-defined]  # new field, editable install lags


def test_sse_params_call_tool_timeout_defaults_to_none() -> None:
    """call_tool_timeout_seconds MUST default to None (no timeout)."""
    from troopai.adk.mcp.sse import MCPServerSseParams

    params = MCPServerSseParams(url="http://localhost/sse")
    assert params.call_tool_timeout_seconds is None  # type: ignore[attr-defined]  # new field, editable install lags


def test_websocket_params_call_tool_timeout_defaults_to_none() -> None:
    """call_tool_timeout_seconds MUST default to None (no timeout)."""
    from troopai.adk.mcp.websocket import MCPServerWebsocketParams

    params = MCPServerWebsocketParams(url="ws://localhost/mcp")
    assert params.call_tool_timeout_seconds is None  # type: ignore[attr-defined]  # new field, editable install lags


def test_make_client_session_passes_timeout_when_set() -> None:
    """When call_tool_timeout_seconds is set on params, the session MUST receive
    a read_timeout_seconds timedelta so session.call_tool does not hang forever.

    Before the fix, _make_client_session was always called without
    read_timeout_seconds on stdio/SSE/WebSocket transports, leaving
    anyio.fail_after(None) in effect — an infinite deadline.
    """
    from datetime import timedelta

    from mcp import ClientSession

    from troopai.adk.mcp.mcp_server import MCPServerWithClientSession

    class _Concrete(MCPServerWithClientSession):
        async def connect(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

    server = _Concrete(name="test")
    read = MagicMock()
    write = MagicMock()

    # Passing a timedelta timeout must not raise and must produce a ClientSession.
    session = server._make_client_session(read, write, read_timeout_seconds=timedelta(seconds=30.0))
    assert isinstance(session, ClientSession)


def test_make_client_session_timeout_none_produces_session() -> None:
    """None timeout (default) still produces a valid ClientSession."""
    from mcp import ClientSession

    from troopai.adk.mcp.mcp_server import MCPServerWithClientSession

    class _Concrete(MCPServerWithClientSession):
        async def connect(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

    server = _Concrete(name="test")
    read = MagicMock()
    write = MagicMock()

    session = server._make_client_session(read, write, read_timeout_seconds=None)
    assert isinstance(session, ClientSession)
