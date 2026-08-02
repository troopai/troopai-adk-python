"""Regression tests for ``tool_middleware.wrap_tool_with_middleware``.

Pins the contract that toolset-scoped middleware must respect a tool's
declared ``response_format``: only a ``content_and_artifact`` tool's
2-tuple return is unpacked into ``(content, artifact)``. A text-format
tool that legitimately returns a 2-tuple value (e.g. coordinates) must
reach the LLM as a single stringified value — identical to the
no-middleware executor path — instead of having its second element
silently diverted into the hidden artifact channel.
"""

from __future__ import annotations

import json
from typing import Any

from troopai.adk.tools import (
    ToolLoggingMiddleware,
    function_tool,
    wrap_tool_with_middleware,
)
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.tools.tool_stream_event import ToolStreamEvent


def _ctx(args: dict[str, Any]) -> ToolContext:
    return ToolContext(
        tool_name="pair",
        tool_call_id="call_1",
        tool_arguments=args,
        raw_arguments=json.dumps(args),
    )


class TestTextTupleNotTreatedAsArtifact:
    """A text-format tool returning a 2-tuple keeps both elements visible."""

    async def test_text_two_tuple_stringified_not_split_into_artifact(self) -> None:
        # A text-format tool (the default) may legitimately return a
        # 2-tuple value such as coordinates. With middleware attached
        # the result must match what the LLM sees with NO middleware:
        # a single stringified tuple, with nothing diverted to the
        # hidden artifact channel.
        @function_tool(name="pair")
        def pair() -> tuple[int, str]:
            return (42, "hello")

        wrapped = wrap_tool_with_middleware(pair, [ToolLoggingMiddleware()])
        assert wrapped.on_invoke is not None
        result = await wrapped.on_invoke(_ctx({}), "")
        # No artifact -> wrapped_on_invoke returns a plain value, not a
        # ``(output, artifact)`` tuple. The single stringify of the
        # original tuple is what the no-middleware path produces too.
        assert isinstance(result, str)
        assert result == str((42, "hello"))

    async def test_text_two_tuple_with_none_second_element_preserved(self) -> None:
        # The degenerate ``(x, None)`` case must NOT collapse to just
        # ``str(x)``: the second element belongs in the model-visible
        # output for a text-format tool.
        @function_tool(name="pair")
        def pair() -> tuple[int, None]:
            return (7, None)

        wrapped = wrap_tool_with_middleware(pair, [ToolLoggingMiddleware()])
        assert wrapped.on_invoke is not None
        result = await wrapped.on_invoke(_ctx({}), "")
        assert isinstance(result, str)
        assert result == str((7, None))


class TestContentAndArtifactStillUnpacks:
    """The genuine artifact path must keep working after the guard."""

    async def test_content_and_artifact_tuple_split_preserved(self) -> None:
        artifact_payload = [{"doc_id": "d1", "score": 0.99}]

        @function_tool(name="rag", response_format="content_and_artifact")
        def rag(query: str) -> tuple[str, Any]:
            return "Found 1 result", artifact_payload

        wrapped = wrap_tool_with_middleware(rag, [ToolLoggingMiddleware()])
        assert wrapped.on_invoke is not None
        result = await wrapped.on_invoke(_ctx({"query": "test"}), '{"query": "test"}')
        # content_and_artifact tools still surface (output, artifact)
        # so the executor rebuilds both fields.
        assert isinstance(result, tuple)
        assert len(result) == 2
        content, artifact = result
        assert content == "Found 1 result"
        assert artifact is artifact_payload


class TestEmptyArgsSerialization:
    """An empty args dict must reach the tool as ``"{}"`` (valid JSON).

    Serialising it to ``""`` is not JSON and diverges from the
    no-middleware executor path, which always hands the tool a JSON
    string it can parse.
    """

    async def test_empty_args_serialize_to_json_object_not_empty_string(self) -> None:
        seen: list[str] = []

        @function_tool(name="noarg")
        def noarg() -> str:
            return "ok"

        async def recording_invoke(ctx: ToolContext, raw_args: str) -> Any:
            seen.append(raw_args)
            return "ok"

        tool = noarg.clone(on_invoke=recording_invoke)
        wrapped = wrap_tool_with_middleware(tool, [ToolLoggingMiddleware()])
        assert wrapped.on_invoke is not None
        await wrapped.on_invoke(_ctx({}), "{}")
        # The terminal re-serialises the parsed (empty) args dict; it must
        # be the JSON object "{}" so the tool's on_invoke sees valid JSON.
        assert seen == ["{}"]


class TestJsonErrorPathDrainsStreaming:
    """The malformed-JSON deferral path must drain a streaming result.

    When ``wrapped_on_invoke`` cannot parse the raw args it defers to the
    original invoker. A streaming tool's ``on_invoke`` returns an async
    iterator even then, so it must be drained to the final value —
    mirroring the terminal closure — rather than handed back undrained to
    the executor, which treats a middleware-wrapped tool as already drained.
    """

    async def test_invalid_json_defers_and_drains_streaming_result(self) -> None:
        @function_tool(name="streamer")
        def streamer() -> str:
            return "unused"

        async def streaming_invoke(ctx: ToolContext, raw_args: str) -> Any:
            async def gen() -> Any:
                yield ToolStreamEvent(type="part_delta", delta="chunk")
                yield ToolStreamEvent(type="done", response="FINAL")

            return gen()

        tool = streamer.clone(on_invoke=streaming_invoke)
        wrapped = wrap_tool_with_middleware(tool, [ToolLoggingMiddleware()])
        assert wrapped.on_invoke is not None
        # "oops" is not valid JSON, so wrapped_on_invoke's own json.loads
        # raises and it defers to the original streaming invoker. The result
        # must be the drained final value, not the async iterator.
        result = await wrapped.on_invoke(_ctx({}), "oops")
        assert result == "FINAL"
        assert not hasattr(result, "__aiter__")
