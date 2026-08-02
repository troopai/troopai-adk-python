"""Tests for the A2A bidirectional converter functions.

Verifies:
* TaskState protobuf-int <-> framework string-literal mapping.
* Terminal-state and failure-state classification.
* Inbound Message -> text extraction (text + skipped non-text parts).
* Inbound Task -> text extraction (artifacts preferred, history fallback).
* Outbound text -> Message / SendMessageRequest construction.
* Task -> A2ATaskStatus snapshot projection.
* StreamResponse oneof discriminator.
"""

import logging

import pytest

# Skip the entire module if a2a-sdk is not installed. Mirrors how
# the rest of the framework gates optional-extra modules.
a2a_types = pytest.importorskip("a2a.types")

from troopai.adk.a2a import converters
from troopai.adk.a2a.a2a_continuation_token import A2ATaskStatus


class TestTaskStateMapping:
    def test_every_known_state_round_trips(self) -> None:
        # Map every known integer to a string and confirm the inverse
        # is_terminal/is_failure classification matches the protocol.
        cases = [
            (a2a_types.TaskState.TASK_STATE_SUBMITTED, "submitted", False, False),
            (a2a_types.TaskState.TASK_STATE_WORKING, "working", False, False),
            (a2a_types.TaskState.TASK_STATE_INPUT_REQUIRED, "input_required", False, False),
            (a2a_types.TaskState.TASK_STATE_AUTH_REQUIRED, "auth_required", False, False),
            (a2a_types.TaskState.TASK_STATE_COMPLETED, "completed", True, False),
            (a2a_types.TaskState.TASK_STATE_CANCELED, "cancelled", True, True),
            (a2a_types.TaskState.TASK_STATE_REJECTED, "rejected", True, True),
            (a2a_types.TaskState.TASK_STATE_FAILED, "failed", True, True),
        ]
        for proto_value, literal, terminal, failure in cases:
            got = converters.task_state_to_literal(proto_value)
            assert got == literal, f"{proto_value} -> {got!r} != {literal!r}"
            assert converters.is_terminal_state(got) is terminal
            assert converters.is_failure_state(got) is failure

    def test_unknown_state_raises(self) -> None:
        # Defends against future protocol extensions adding new state
        # values without a typed mapping — silent fall-through would
        # mask a real protocol mismatch.
        with pytest.raises(ValueError, match="Unknown A2A TaskState"):
            converters.task_state_to_literal(99)


class TestMessageExtraction:
    def test_extract_text_concatenates_text_parts(self) -> None:
        msg = a2a_types.Message(
            role=a2a_types.Role.ROLE_USER,
            parts=[a2a_types.Part(text="hello "), a2a_types.Part(text="world")],
        )
        assert converters.extract_text_from_message(msg) == "hello world"

    def test_extract_text_skips_non_text_parts_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        msg = a2a_types.Message(
            role=a2a_types.Role.ROLE_USER,
            parts=[
                a2a_types.Part(text="prefix "),
                a2a_types.Part(url="http://example.com/file.png", media_type="image/png"),
                a2a_types.Part(text="suffix"),
            ],
        )
        with caplog.at_level(logging.WARNING, logger="troopai.adk.a2a.converters"):
            result = converters.extract_text_from_message(msg)
        assert result == "prefix suffix"
        # Exactly one warning should have fired for the URL part.
        assert any("non-text Part" in r.getMessage() for r in caplog.records)

    def test_extract_text_empty_message(self) -> None:
        msg = a2a_types.Message(role=a2a_types.Role.ROLE_USER, parts=[])
        assert converters.extract_text_from_message(msg) == ""


class TestTaskExtraction:
    def test_artifacts_preferred_over_history(self) -> None:
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_COMPLETED),
            artifacts=[
                a2a_types.Artifact(
                    artifact_id="a1",
                    parts=[a2a_types.Part(text="final answer")],
                ),
            ],
            history=[
                a2a_types.Message(
                    role=a2a_types.Role.ROLE_AGENT,
                    parts=[a2a_types.Part(text="ignored historical text")],
                ),
            ],
        )
        assert converters.extract_text_from_task(task) == "final answer"

    def test_falls_back_to_history(self) -> None:
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_COMPLETED),
            history=[
                a2a_types.Message(
                    role=a2a_types.Role.ROLE_AGENT,
                    parts=[a2a_types.Part(text="from history")],
                ),
            ],
        )
        assert converters.extract_text_from_task(task) == "from history"

    def test_returns_empty_when_neither_present(self) -> None:
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_COMPLETED),
        )
        assert converters.extract_text_from_task(task) == ""


class TestOutboundConstruction:
    # Outbound text Parts are built via the SDK's
    # ``a2a.helpers.proto_helpers.new_text_part``. Coverage for the
    # text-Part shape lives upstream and is exercised here through
    # ``build_user_message`` below (which uses the SDK helper).

    def test_build_user_message_defaults_message_id(self) -> None:
        msg = converters.build_user_message("hi")
        # Auto-generated UUID4 — non-empty, deterministic shape.
        assert len(msg.message_id) >= 32
        assert msg.role == a2a_types.Role.ROLE_USER
        assert len(msg.parts) == 1
        assert msg.parts[0].text == "hi"

    def test_build_user_message_explicit_ids(self) -> None:
        msg = converters.build_user_message(
            "continue please",
            context_id="c1",
            task_id="t1",
            message_id="m1",
        )
        assert msg.message_id == "m1"
        assert msg.context_id == "c1"
        assert msg.task_id == "t1"

    def test_build_send_request_default_blocking(self) -> None:
        msg = converters.build_user_message("hi")
        req = converters.build_send_request(msg)
        # Default: server blocks until terminal — return_immediately is False.
        assert req.configuration.return_immediately is False

    def test_build_send_request_background(self) -> None:
        msg = converters.build_user_message("hi")
        req = converters.build_send_request(msg, return_immediately=True)
        assert req.configuration.return_immediately is True

    def test_build_send_request_accepted_output_modes(self) -> None:
        msg = converters.build_user_message("hi")
        req = converters.build_send_request(msg, accepted_output_modes=["text/plain", "application/json"])
        assert list(req.configuration.accepted_output_modes) == ["text/plain", "application/json"]


class TestTaskToStatus:
    def test_completed_task_carries_result(self) -> None:
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_COMPLETED),
            artifacts=[
                a2a_types.Artifact(
                    artifact_id="a1",
                    parts=[a2a_types.Part(text="done")],
                ),
            ],
        )
        status = converters.task_to_status(task)
        assert isinstance(status, A2ATaskStatus)
        assert status.state == "completed"
        assert status.result == "done"

    def test_failed_task_carries_message(self) -> None:
        failure_msg = a2a_types.Message(
            role=a2a_types.Role.ROLE_AGENT,
            parts=[a2a_types.Part(text="LLM rate limit hit")],
        )
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(
                state=a2a_types.TaskState.TASK_STATE_FAILED,
                message=failure_msg,
            ),
        )
        status = converters.task_to_status(task)
        assert status.state == "failed"
        assert status.result is None
        assert status.message == "LLM rate limit hit"

    def test_working_task_no_result_no_message(self) -> None:
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_WORKING),
        )
        status = converters.task_to_status(task)
        assert status.state == "working"
        assert status.result is None
        assert status.message is None


class TestStreamResponseDiscriminator:
    def test_task_payload(self) -> None:
        resp = a2a_types.StreamResponse(
            task=a2a_types.Task(
                id="t1",
                context_id="c1",
                status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_WORKING),
            )
        )
        assert converters.stream_response_kind(resp) == "task"

    def test_status_update_payload(self) -> None:
        resp = a2a_types.StreamResponse(
            status_update=a2a_types.TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_COMPLETED),
            )
        )
        assert converters.stream_response_kind(resp) == "status_update"


class TestStreamResponseToEvent:
    """Tests for stream_response_to_event accumulator correctness."""

    def test_task_kind_completed_populates_accumulator(self) -> None:
        """task-kind completed event appends final_text to the accumulator."""
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_COMPLETED),
            artifacts=[
                a2a_types.Artifact(
                    artifact_id="a1",
                    parts=[a2a_types.Part(text="the answer")],
                ),
            ],
        )
        chunk = a2a_types.StreamResponse(task=task)
        accumulator: list[str] = []

        event = converters.stream_response_to_event(chunk, accumulator)

        assert event is not None
        assert event["type"] == "completed"
        # ``.get`` because result_text is not a required key on the
        # A2AStreamEvent TypedDict (pyright flags a bare subscript).
        assert event.get("result_text") == "the answer"
        # The accumulator must contain the final text so the caller's
        # span output ("".join(accumulator)) records it.
        assert accumulator == ["the answer"]

    def test_task_kind_completed_empty_text_does_not_append(self) -> None:
        """Empty final_text is not appended to avoid a spurious empty entry."""
        task = a2a_types.Task(
            id="t1",
            context_id="c1",
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_COMPLETED),
        )
        chunk = a2a_types.StreamResponse(task=task)
        accumulator: list[str] = []

        event = converters.stream_response_to_event(chunk, accumulator)

        assert event is not None
        assert event.get("result_text") == ""
        assert accumulator == []

    def test_artifact_append_true_concatenates(self) -> None:
        """append=True concatenates its delta onto the running accumulator."""
        accumulator: list[str] = ["hel"]
        chunk = a2a_types.StreamResponse(
            artifact_update=a2a_types.TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=a2a_types.Artifact(artifact_id="a1", parts=[a2a_types.Part(text="lo")]),
                append=True,
            )
        )
        event = converters.stream_response_to_event(chunk, accumulator)
        assert event is not None
        assert event["type"] == "text_delta"
        assert accumulator == ["hel", "lo"]

    def test_artifact_append_false_resets_accumulator(self) -> None:
        """append=False replaces the snapshot: the running text is reset so a
        cumulative-snapshot chunk is not double-counted as a delta."""
        accumulator: list[str] = ["Hello"]
        chunk = a2a_types.StreamResponse(
            artifact_update=a2a_types.TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=a2a_types.Artifact(artifact_id="a1", parts=[a2a_types.Part(text="Hello world")]),
                append=False,
            )
        )
        event = converters.stream_response_to_event(chunk, accumulator)
        assert event is not None
        assert event["type"] == "text_delta"
        # Reset-then-append: the accumulator reflects only the replacement,
        # not "Hello" + "Hello world".
        assert accumulator == ["Hello world"]

    def test_interrupt_status_carries_message(self) -> None:
        """An input_required status event surfaces status.message as prompt."""
        accumulator: list[str] = []
        chunk = a2a_types.StreamResponse(
            status_update=a2a_types.TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=a2a_types.TaskStatus(
                    state=a2a_types.TaskState.TASK_STATE_INPUT_REQUIRED,
                    message=a2a_types.Message(
                        role=a2a_types.Role.ROLE_AGENT,
                        parts=[a2a_types.Part(text="What date?")],
                    ),
                ),
            )
        )
        event = converters.stream_response_to_event(chunk, accumulator)
        assert event is not None
        assert event["type"] == "status"
        assert event.get("state") == "input_required"
        assert event.get("message") == "What date?"


class TestInterruptStateClassification:
    def test_input_required_is_interrupt(self) -> None:
        assert converters.is_interrupt_state("input_required") is True

    def test_auth_required_is_interrupt(self) -> None:
        assert converters.is_interrupt_state("auth_required") is True

    def test_terminal_states_are_not_interrupts(self) -> None:
        for state in ("completed", "failed", "rejected", "cancelled"):
            assert converters.is_interrupt_state(state) is False  # type: ignore[arg-type]

    def test_working_state_is_not_interrupt(self) -> None:
        assert converters.is_interrupt_state("working") is False
