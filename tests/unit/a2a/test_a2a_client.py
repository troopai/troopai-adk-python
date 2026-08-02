"""Tests for ``A2AClient`` — framework-typed wrapper over ``a2a.client``.

The tests stub out the underlying a2a-sdk ``Client`` so we don't need
a live server. Verified behaviours:

* Construction validates URL and accepts injected ``httpx.AsyncClient``.
* ``send_message`` aggregates a streaming-from-SDK response into an
  ``A2ARunResult`` and raises typed ``A2A*Error`` on failure states.
* ``stream_message`` yields ``A2AStreamEvent`` TypedDicts.
* ``submit_background`` returns an ``A2AContinuationToken`` from the
  first task identifier on the stream.
* ``poll_task`` returns an ``A2ATaskStatus`` snapshot.
* ``cancel_task`` forwards to the SDK.
* Lifecycle: ``close()`` releases owned ``httpx.AsyncClient``.
"""

from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.a2a.exceptions import A2ATaskInterruptedError

# Skip module if extra missing.
pytest.importorskip("a2a.client")

import httpx
from a2a.client import ClientConfig
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from troopai.adk.a2a import (
    A2AClient,
    A2AContinuationToken,
    A2AProtocolError,
    A2ARunResult,
    A2ATaskCancelledError,
    A2ATaskError,
    A2ATransportError,
    converters,
)

# ----------------------------------------------------------------------
# Helpers — async iterator wrappers
# ----------------------------------------------------------------------


async def _async_iter(items: list[StreamResponse]) -> AsyncIterator[StreamResponse]:
    for item in items:
        yield item


def _completed_task_response(text: str = "done") -> StreamResponse:
    return StreamResponse(
        task=Task(
            id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[Artifact(artifact_id="a1", parts=[Part(text=text)])],
        )
    )


def _failed_status_response(message_text: str = "boom") -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id="t1",
            context_id="c1",
            status=TaskStatus(
                state=TaskState.TASK_STATE_FAILED,
                message=Message(role=Role.ROLE_AGENT, parts=[Part(text=message_text)]),
            ),
        )
    )


def _cancelled_status_response() -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
        )
    )


# ----------------------------------------------------------------------
# Construction & lifecycle
# ----------------------------------------------------------------------


class TestConstruction:
    def test_empty_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty URL"):
            A2AClient(url="")

    def test_default_owns_internal_httpx_client(self) -> None:
        client = A2AClient(url="http://example.com")
        # Internal flag indicating the client was constructed (so close()
        # tears it down).
        assert client._owns_http_client is True
        assert isinstance(client._http_client, httpx.AsyncClient)

    def test_injected_httpx_client_not_owned(self) -> None:
        # Shared httpx clients flow through ``client_config.httpx_client``
        # — the only injection surface.
        external = httpx.AsyncClient()
        config = ClientConfig(httpx_client=external)
        client = A2AClient(url="http://example.com", client_config=config)
        assert client._owns_http_client is False
        assert client._http_client is external


class TestLifecycle:
    async def test_close_idempotent(self) -> None:
        client = A2AClient(url="http://example.com")
        await client.close()
        # Second call: no error.
        await client.close()

    async def test_close_releases_owned_httpx(self) -> None:
        client = A2AClient(url="http://example.com")
        owned = client._http_client
        assert isinstance(owned, httpx.AsyncClient)
        await client.close()
        assert owned.is_closed

    async def test_close_does_not_release_external_httpx(self) -> None:
        external = httpx.AsyncClient()
        config = ClientConfig(httpx_client=external)
        client = A2AClient(url="http://example.com", client_config=config)
        await client.close()
        assert external.is_closed is False
        await external.aclose()

    async def test_aenter_aexit_closes(self) -> None:
        async with A2AClient(url="http://example.com") as client:
            assert client._owns_http_client is True
        assert client._http_client.is_closed if isinstance(client._http_client, httpx.AsyncClient) else True


# ----------------------------------------------------------------------
# send_message — happy path + every typed-error path
# ----------------------------------------------------------------------


class TestSendMessage:
    @pytest.fixture
    def patched_client(self) -> Iterator[tuple[A2AClient, MagicMock]]:
        """A2AClient with its underlying SDK client patched."""
        client = A2AClient(url="http://example.com")
        sdk_client = MagicMock()
        # _ensure_client returns this stub.
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk_client)):
            yield client, sdk_client

    async def test_happy_path_returns_run_result(self, patched_client: tuple[A2AClient, MagicMock]) -> None:
        client, sdk = patched_client
        sdk.send_message = MagicMock(return_value=_async_iter([_completed_task_response("hi!")]))
        result = await client.send_message("anything")
        assert isinstance(result, A2ARunResult)
        assert result.text == "hi!"
        assert result.task_id == "t1"
        assert result.context_id == "c1"

    async def test_failed_state_raises_task_error(self, patched_client: tuple[A2AClient, MagicMock]) -> None:
        client, sdk = patched_client
        # Synthesise: status_update transitioning to FAILED.
        sdk.send_message = MagicMock(return_value=_async_iter([_failed_status_response("LLM error")]))
        with pytest.raises(A2ATaskError) as exc_info:
            await client.send_message("anything")
        assert exc_info.value.state == "failed"
        assert exc_info.value.remote_message == "LLM error"

    async def test_cancelled_state_raises_cancelled_error(self, patched_client: tuple[A2AClient, MagicMock]) -> None:
        client, sdk = patched_client
        sdk.send_message = MagicMock(return_value=_async_iter([_cancelled_status_response()]))
        with pytest.raises(A2ATaskCancelledError) as exc_info:
            await client.send_message("anything")
        assert exc_info.value.state == "cancelled"

    async def test_transport_error_raises_typed_exception(self, patched_client: tuple[A2AClient, MagicMock]) -> None:
        client, sdk = patched_client

        # Explicit async-iterator class instead of an async-generator
        # function — avoids the "unreachable yield" idiom that
        # Pyright flags as structurally unreachable code. The
        # forward reference is quoted because this test file does
        # not use ``from __future__ import annotations``.
        class _RaisingStream:
            def __aiter__(self) -> "_RaisingStream":
                return self

            async def __anext__(self) -> StreamResponse:
                raise httpx.ConnectError("connection refused")

        sdk.send_message = MagicMock(return_value=_RaisingStream())
        with pytest.raises(A2ATransportError, match="transport failure"):
            await client.send_message("anything")

    async def test_empty_stream_raises_protocol_error(self, patched_client: tuple[A2AClient, MagicMock]) -> None:
        client, sdk = patched_client
        sdk.send_message = MagicMock(return_value=_async_iter([]))
        with pytest.raises(A2AProtocolError, match="closed the stream"):
            await client.send_message("anything")


# ----------------------------------------------------------------------
# stream_message — events drained correctly
# ----------------------------------------------------------------------


class TestStreamMessage:
    async def test_yields_text_deltas_then_completion(self) -> None:
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()

        # Two chunks of the SAME artifact then a completed status. The
        # continuation chunk MUST carry append=True: that is how a real A2A
        # server signals "concatenate onto artifact a1". The first chunk is
        # append=False (the initial/create). Two append=False chunks would
        # instead REPLACE (last wins) per the A2A append-flag contract, so a
        # streaming continuation is modeled with append=True here.
        chunk1 = StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a1", parts=[Part(text="hel")]),
            )
        )
        chunk2 = StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a1", parts=[Part(text="lo")]),
                append=True,
            )
        )
        completion = StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([chunk1, chunk2, completion]))

        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)):
            events = [ev async for ev in client.stream_message("hi")]

        types = [ev["type"] for ev in events]
        assert types == ["text_delta", "text_delta", "completed"]
        # The completion event carries the accumulated text. Use
        # ``.get`` because ``result_text`` is not a required key on
        # the A2AStreamEvent TypedDict — Pyright otherwise flags
        # the subscript access as unsafe.
        assert events[-1].get("result_text") == "hello"

    async def test_streaming_interrupt_carries_task_id(self) -> None:
        # Regression: _iter_stream_events was raising
        # A2ATaskInterruptedError(task_id="") because no task_id accumulator
        # was present in the streaming path. The task_id must be populated
        # from the status_update chunk.
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()

        input_required = StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="real-task-id",
                context_id="ctx-1",
                status=TaskStatus(
                    state=TaskState.TASK_STATE_INPUT_REQUIRED,
                    message=Message(role=Role.ROLE_AGENT, parts=[Part(text="What date?")]),
                ),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([input_required]))

        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2ATaskInterruptedError) as exc_info,
        ):
            async for _ in client.stream_message("book flight"):
                pass

        assert exc_info.value.task_id == "real-task-id", (
            f"Expected task_id='real-task-id', got {exc_info.value.task_id!r}"
        )
        assert exc_info.value.state == "input_required"

    async def test_streaming_interrupt_carries_prompt(self) -> None:
        # Regression: the interrupt status event dropped status.message, so
        # A2ATaskInterruptedError.prompt arrived empty. The human-readable
        # prompt the server asks for ("What date?") must survive to the
        # raised exception.
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()

        input_required = StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="real-task-id",
                context_id="ctx-1",
                status=TaskStatus(
                    state=TaskState.TASK_STATE_INPUT_REQUIRED,
                    message=Message(role=Role.ROLE_AGENT, parts=[Part(text="What date?")]),
                ),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([input_required]))

        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2ATaskInterruptedError) as exc_info,
        ):
            async for _ in client.stream_message("book flight"):
                pass

        assert exc_info.value.prompt == "What date?"

    async def test_stream_closed_without_terminal_yields_failed_not_completed(self) -> None:
        # Regression: stream_message used to synthesise a 'completed' event
        # when the server closed the connection without a terminal event,
        # silently hiding crashes and premature closes.
        # Now it must yield a 'failed' event instead.
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()

        # Only emit one text delta — no completed/failed terminal event.
        chunk = StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a1", parts=[Part(text="partial")]),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([chunk]))

        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)):
            events = [ev async for ev in client.stream_message("hi")]

        final = events[-1]
        assert final["type"] == "failed", (
            f"Expected final event type='failed' when stream closes without terminal, got {final['type']!r}"
        )

    async def test_a2a_client_error_in_stream_maps_to_transport_error(self) -> None:
        # Regression: stream_message only caught httpx.HTTPError; A2AClientError
        # and other SDK errors escaped raw, breaking the typed-exception contract.
        from a2a.client import A2AClientError as SDKClientError

        client = A2AClient(url="http://example.com")
        sdk = MagicMock()

        class _RaisingStream:
            def __aiter__(self) -> "_RaisingStream":
                return self

            async def __anext__(self) -> StreamResponse:
                raise SDKClientError("SDK transport blew up")

        sdk.send_message = MagicMock(return_value=_RaisingStream())

        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises((A2ATransportError, A2AProtocolError)),
        ):
            async for _ in client.stream_message("hi"):
                pass

    async def test_a2a_protocol_error_in_stream_propagates_typed(self) -> None:
        # A2AProtocolError raised inside _iter_stream_events (e.g. volume cap)
        # must surface typed, not as a raw SDK type.
        client = A2AClient(url="http://example.com", max_stream_chunks=1)
        sdk = MagicMock()

        # Two chunks — second one should trigger the cap.
        chunk1 = StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a1", parts=[Part(text="x")]),
            )
        )
        chunk2 = StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a1", parts=[Part(text="y")]),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([chunk1, chunk2]))

        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2AProtocolError, match="max_stream_chunks"),
        ):
            async for _ in client.stream_message("hi"):
                pass


# ----------------------------------------------------------------------
# submit_background + poll_task + cancel_task
# ----------------------------------------------------------------------


class TestBackgroundSubmit:
    async def test_returns_continuation_token_from_first_task_id(self) -> None:
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()
        # First chunk carries identifiers; we should return immediately.
        first = StreamResponse(
            task=Task(
                id="task-abc",
                context_id="ctx-xyz",
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([first]))
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)):
            tok = await client.submit_background("long task")
        assert isinstance(tok, A2AContinuationToken)
        assert tok.task_id == "task-abc"
        assert tok.context_id == "ctx-xyz"
        assert tok.remote_url == "http://example.com"


class TestPollTask:
    async def test_returns_typed_status(self) -> None:
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()
        completed_task = Task(
            id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[Artifact(artifact_id="a1", parts=[Part(text="result")])],
        )
        sdk.get_task = AsyncMock(return_value=completed_task)
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)):
            tok = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://example.com")
            status = await client.poll_task(tok)
        assert status.state == "completed"
        assert status.result == "result"


class TestCancelTask:
    async def test_forwards_request(self) -> None:
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()
        sdk.cancel_task = AsyncMock()
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)):
            tok = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://example.com")
            await client.cancel_task(tok)
        sdk.cancel_task.assert_awaited_once()


# ----------------------------------------------------------------------
# _consume_until_terminal — fresh-Task-on-return semantics
# ----------------------------------------------------------------------


class TestConsumeUntilTerminalFreshness:
    """Guards that the terminal-consume path never aliases SDK-vended objects.

    ``_consume_until_terminal`` must not aggregate streaming chunks by
    mutating the SDK-vended ``Task`` (via ``CopyFrom`` / ``artifacts.append``)
    — that mutates objects whose lifetime the SDK iterator owns. Instead it
    tracks id / context_id / status / artifacts as plain locals on a typed
    ``_StreamAccumulator`` and constructs a **fresh** :class:`Task` at return
    time; protobuf-python copies sub-messages at constructor time, so the
    returned Task does not alias any SDK-vended sub-message.
    """

    @pytest.fixture
    def patched_client(self) -> Iterator[tuple[A2AClient, MagicMock]]:
        """A2AClient with its underlying SDK client patched.

        Class-scoped duplicate of ``TestSendMessage.patched_client``
        — pytest fixtures defined on a class are scoped to that
        class only and don't leak to siblings.
        """
        client = A2AClient(url="http://example.com")
        sdk_client = MagicMock()
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk_client)):
            yield client, sdk_client

    async def test_returned_task_is_fresh_object_not_sdk_chunk(
        self,
        patched_client: tuple[A2AClient, MagicMock],
    ) -> None:
        # Drive ``_consume_until_terminal`` directly — going through
        # the public ``send_message`` surface returns an
        # ``A2ARunResult`` whose ``.text`` is an immutable Python
        # ``str`` extracted before we could probe; mutations to the
        # SDK chunk after that can't possibly affect the string,
        # making the assertion a tautology even if the code aliased.
        # This test instead probes object identity on the Task
        # itself, which IS the falsifiable invariant.
        client, _ = patched_client
        sdk_task_chunk = StreamResponse(
            task=Task(
                id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[Artifact(artifact_id="a1", parts=[Part(text="answer")])],
            )
        )
        sdk_status = sdk_task_chunk.task.status
        sdk_artifact = sdk_task_chunk.task.artifacts[0]

        returned_task = await client._consume_until_terminal(_async_iter([sdk_task_chunk]))

        # Identity probe: protobuf copies sub-messages at
        # constructor time, so the returned Task's status and
        # artifacts are NOT the same Python objects as the SDK's.
        assert returned_task.status is not sdk_status
        assert len(returned_task.artifacts) == 1
        assert returned_task.artifacts[0] is not sdk_artifact
        # Mutation probe: extracting text from the returned Task
        # AFTER mutating the SDK chunk's artifact must reflect the
        # original snapshot, not the mutation. On the prior
        # CopyFrom-based implementation, mutating the SDK part
        # would bleed through; on the fresh-Task implementation,
        # it cannot.
        sdk_artifact.parts[0].text = "MUTATED"
        assert converters.extract_text_from_task(returned_task) == "answer"

    async def test_status_update_only_synthesises_task_with_warning(
        self,
        patched_client: tuple[A2AClient, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Server emits a terminal status_update with no prior Task
        # chunk — this used to require a synthesised placeholder
        # in the old implementation; the new implementation handles
        # it natively but still warns for operator visibility.
        client, sdk = patched_client
        terminal_only = _failed_status_response(message_text="instant fail")
        sdk.send_message = MagicMock(return_value=_async_iter([terminal_only]))

        with (
            caplog.at_level("WARNING", logger="troopai.adk.a2a.a2a_client"),
            pytest.raises(A2ATaskError) as excinfo,
        ):
            await client.send_message("hi")

        # The warning fires exactly once for the no-prior-Task case.
        assert any("synthesising a Task from the status_update" in r.getMessage() for r in caplog.records)
        # The error carries the task identifiers from the status_update.
        assert excinfo.value.task_id == "t1"
        assert excinfo.value.context_id == "c1"

    async def test_artifact_chunks_accumulate_into_fresh_task(
        self,
        patched_client: tuple[A2AClient, MagicMock],
    ) -> None:
        # Streamed ``artifact_update`` chunks land on the
        # accumulator's own list, never on an SDK-held Task.
        # ``extract_text_from_task`` returns only the last
        # artifact's text — that pre-existing convention is the
        # invariant we're checking here.
        client, sdk = patched_client
        chunk1 = StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a1", parts=[Part(text="first ")]),
            )
        )
        chunk2 = StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a2", parts=[Part(text="last")]),
            )
        )
        completion = StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([chunk1, chunk2, completion]))

        result = await client.send_message("hi")
        # The accumulator captured both artifact chunks; the
        # extractor returns the last artifact's text.
        assert result.text == "last"


# ----------------------------------------------------------------------
# Regression: premature stream close on a non-terminal state
# (_consume_until_terminal must surface a protocol error, not success)
# ----------------------------------------------------------------------


def _working_status_response() -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
        )
    )


def _message_response(text: str) -> StreamResponse:
    return StreamResponse(message=Message(role=Role.ROLE_AGENT, parts=[Part(text=text)]))


class TestPrematureStreamClose:
    @pytest.fixture
    def patched_client(self) -> Iterator[tuple[A2AClient, MagicMock]]:
        client = A2AClient(url="http://example.com")
        sdk_client = MagicMock()
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk_client)):
            yield client, sdk_client

    async def test_send_message_raises_when_stream_closes_on_working_state(
        self,
        patched_client: tuple[A2AClient, MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Regression: the server emits a non-terminal ``working``
        # status_update then the SSE stream closes cleanly (no
        # exception). _consume_until_terminal used to return that
        # non-terminal Task, so send_message reported text='' as a
        # success. It must now surface a protocol error instead,
        # matching stream_message's premature-close handling.
        client, sdk = patched_client
        sdk.send_message = MagicMock(return_value=_async_iter([_working_status_response()]))
        with (
            caplog.at_level("WARNING", logger="troopai.adk.a2a.a2a_client"),
            pytest.raises(A2AProtocolError, match="before reaching a terminal state"),
        ):
            await client.send_message("anything")
        assert any("closed without a terminal event" in r.getMessage() for r in caplog.records)

    async def test_consume_until_terminal_returns_on_interrupt_state(
        self,
        patched_client: tuple[A2AClient, MagicMock],
    ) -> None:
        # An interrupt state (input_required) is non-terminal but DOES
        # end the consume loop early via _apply_chunk -> True; the
        # premature-close guard must not fire for it.
        client, _ = patched_client
        interrupt = StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
            )
        )
        task = await client._consume_until_terminal(_async_iter([interrupt]))
        assert converters.task_state_to_literal(task.status.state) == "input_required"


# ----------------------------------------------------------------------
# Regression: text delivered via free-standing 'message' chunks
# ----------------------------------------------------------------------


class TestMessageChunkText:
    @pytest.fixture
    def patched_client(self) -> Iterator[tuple[A2AClient, MagicMock]]:
        client = A2AClient(url="http://example.com")
        sdk_client = MagicMock()
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk_client)):
            yield client, sdk_client

    async def test_send_message_surfaces_message_chunk_text(
        self,
        patched_client: tuple[A2AClient, MagicMock],
    ) -> None:
        # Regression: a server that delivers its answer as a
        # free-standing ``message`` chunk (no artifacts) produced
        # A2ARunResult(text="") from send_message because _apply_chunk
        # dropped the message and to_task() never set history. The text
        # must now round-trip via the Task history fallback.
        client, sdk = patched_client
        completion = StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        )
        sdk.send_message = MagicMock(return_value=_async_iter([_message_response("the answer"), completion]))
        result = await client.send_message("anything")
        assert result.text == "the answer"

    async def test_message_chunk_text_counts_toward_byte_cap(
        self,
        patched_client: tuple[A2AClient, MagicMock],
    ) -> None:
        # A message chunk's text must not bypass the max_stream_bytes
        # volume bound — a runaway server cannot exhaust memory via the
        # message route any more than via artifact deltas.
        client = A2AClient(url="http://example.com", max_stream_bytes=4)
        sdk = MagicMock()
        sdk.send_message = MagicMock(return_value=_async_iter([_message_response("way too long")]))
        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2AProtocolError, match="max_stream_bytes"),
        ):
            await client.send_message("anything")


# ----------------------------------------------------------------------
# Regression: artifact append/last_chunk semantics (blocking path)
# ----------------------------------------------------------------------


def _artifact_chunk(text: str, *, artifact_id: str = "a1", append: bool = False) -> StreamResponse:
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id="t1",
            context_id="c1",
            artifact=Artifact(artifact_id=artifact_id, parts=[Part(text=text)]),
            append=append,
        )
    )


def _completed_status() -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        )
    )


class TestArtifactAppendSemantics:
    @pytest.fixture
    def patched_client(self) -> Iterator[tuple[A2AClient, MagicMock]]:
        client = A2AClient(url="http://example.com")
        sdk_client = MagicMock()
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk_client)):
            yield client, sdk_client

    async def test_append_true_concatenates_multi_chunk_artifact(
        self, patched_client: tuple[A2AClient, MagicMock]
    ) -> None:
        # Regression: _apply_chunk appended every artifact_update as a
        # separate list entry, so extract_text_from_task (last artifact only)
        # returned just the final chunk — truncating multi-chunk answers.
        client, sdk = patched_client
        sdk.send_message = MagicMock(
            return_value=_async_iter(
                [
                    _artifact_chunk("hel", append=False),
                    _artifact_chunk("lo", append=True),
                    _completed_status(),
                ]
            )
        )
        result = await client.send_message("hi")
        assert result.text == "hello"

    async def test_append_false_replaces_same_id_artifact(self, patched_client: tuple[A2AClient, MagicMock]) -> None:
        # append=False for an already-seen artifact_id replaces it (the A2A
        # append-flag contract), so a cumulative-snapshot server does not
        # duplicate content.
        client, sdk = patched_client
        sdk.send_message = MagicMock(
            return_value=_async_iter(
                [
                    _artifact_chunk("Hello", append=False),
                    _artifact_chunk("Hello world", append=False),
                    _completed_status(),
                ]
            )
        )
        result = await client.send_message("hi")
        assert result.text == "Hello world"


# ----------------------------------------------------------------------
# Regression: bare message-only reply is a success, not a protocol error
# ----------------------------------------------------------------------


class TestBareMessageReply:
    @pytest.fixture
    def patched_client(self) -> Iterator[tuple[A2AClient, MagicMock]]:
        client = A2AClient(url="http://example.com")
        sdk_client = MagicMock()
        with patch.object(client, "_ensure_client", AsyncMock(return_value=sdk_client)):
            yield client, sdk_client

    async def test_message_only_stream_end_is_success(self, patched_client: tuple[A2AClient, MagicMock]) -> None:
        # Regression: a server that answers with only free-standing Message
        # chunks (no Task, no status_update) left acc.status None, so
        # send_message raised A2AProtocolError even though the peer replied.
        # It must now assemble the messages into a completed result.
        client, sdk = patched_client
        sdk.send_message = MagicMock(return_value=_async_iter([_message_response("bare reply")]))
        result = await client.send_message("hi")
        assert result.text == "bare reply"


# ----------------------------------------------------------------------
# Regression: unknown TaskState wraps to A2AProtocolError (send_message)
# ----------------------------------------------------------------------


class TestUnknownStateWrapping:
    async def test_send_message_unknown_state_maps_to_protocol_error(self) -> None:
        # Regression: an unmapped TaskState (protobuf-default UNSPECIFIED=0)
        # made task_state_to_literal raise a raw ValueError from inside
        # _apply_chunk, escaping the typed A2AError hierarchy.
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()
        unmapped = StreamResponse(task=Task(id="t1", context_id="c1", status=TaskStatus()))
        sdk.send_message = MagicMock(return_value=_async_iter([unmapped]))
        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2AProtocolError, match="unrecognised task state"),
        ):
            await client.send_message("hi")


# ----------------------------------------------------------------------
# Regression: close() ownership — never tear down a caller's httpx client
# ----------------------------------------------------------------------


class TestCloseOwnership:
    async def test_close_does_not_close_sdk_client_when_httpx_injected(self) -> None:
        # The a2a Client.close() calls httpx_client.aclose(); when the caller
        # injected their own httpx client, driving that close would tear down
        # a transport we do not own. close() must skip the SDK close then.
        external = httpx.AsyncClient()
        config = ClientConfig(httpx_client=external)
        client = A2AClient(url="http://example.com", client_config=config)
        sdk = MagicMock()
        sdk.close = AsyncMock()
        client._client = sdk  # simulate a constructed SDK client
        await client.close()
        sdk.close.assert_not_awaited()
        assert external.is_closed is False
        await external.aclose()

    async def test_close_closes_sdk_client_when_owned(self) -> None:
        # When we own the httpx client, close() must still tear down the SDK
        # client (guards against over-correcting into a resource leak).
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()
        sdk.close = AsyncMock()
        client._client = sdk
        await client.close()
        sdk.close.assert_awaited_once()


# ----------------------------------------------------------------------
# Regression: _read_first_identifiers / poll_task typed-error mapping
# ----------------------------------------------------------------------


class TestTypedErrorMapping:
    async def test_submit_background_maps_sdk_timeout_to_transport_error(self) -> None:
        # Regression: _read_first_identifiers only caught httpx.HTTPError,
        # leaking raw a2a-sdk A2AClientTimeoutError to callers that branch
        # on the typed exception hierarchy.
        from a2a.client import A2AClientTimeoutError as SDKTimeoutError

        client = A2AClient(url="http://example.com")
        sdk = MagicMock()

        class _RaisingStream:
            def __aiter__(self) -> "_RaisingStream":
                return self

            async def __anext__(self) -> StreamResponse:
                raise SDKTimeoutError("read timed out")

        sdk.send_message = MagicMock(return_value=_RaisingStream())
        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2ATransportError, match="timed out"),
        ):
            await client.submit_background("long task")

    async def test_submit_background_maps_sdk_client_error_to_protocol_error(self) -> None:
        from a2a.client import A2AClientError as SDKClientError

        client = A2AClient(url="http://example.com")
        sdk = MagicMock()

        class _RaisingStream:
            def __aiter__(self) -> "_RaisingStream":
                return self

            async def __anext__(self) -> StreamResponse:
                raise SDKClientError("protocol violation")

        sdk.send_message = MagicMock(return_value=_RaisingStream())
        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2AProtocolError, match="protocol failure"),
        ):
            await client.submit_background("long task")

    async def test_poll_task_maps_unknown_state_to_protocol_error(self) -> None:
        # Regression: task_to_status ran outside poll_task's try block,
        # so a Task with an unmapped TaskState (e.g. the protobuf-default
        # UNSPECIFIED=0) threw a raw ValueError instead of the typed
        # A2AProtocolError callers expect.
        client = A2AClient(url="http://example.com")
        sdk = MagicMock()
        # Default TaskStatus -> state 0 (UNSPECIFIED), which is not mapped.
        unmapped_task = Task(id="t1", context_id="c1", status=TaskStatus())
        sdk.get_task = AsyncMock(return_value=unmapped_task)
        with (
            patch.object(client, "_ensure_client", AsyncMock(return_value=sdk)),
            pytest.raises(A2AProtocolError, match="unrecognised task state"),
        ):
            tok = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://example.com")
            await client.poll_task(tok)
