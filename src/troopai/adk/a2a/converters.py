"""Bidirectional conversion between A2A wire types and framework types.

This module is the single boundary where ``a2a.types`` (protobuf-generated
wire types shipped by the ``a2a-sdk`` package) crosses into framework-owned
types. Every function here has either an ``a2a.types`` argument or an
``a2a.types`` return — and nothing else in the framework imports from
``a2a.types`` directly. This mirrors the discipline that ``llms/litellm/``
applies to ``litellm.types``: provider wire types stay confined to one
module so the rest of the framework remains provider-agnostic.

Two directions, kept symmetric:

* **Inbound** — a remote A2A response (Task, Message, StreamResponse) is
  decomposed into framework-typed pieces (text strings, state literals,
  ``A2ATaskStatus`` snapshots).
* **Outbound** — a framework-typed prompt is wrapped into an A2A
  ``Message`` + ``SendMessageRequest`` for transmission.

The surface covers both the client path (decoding peer responses)
and the server path (encoding local agent output for the executor).
Multi-modal parts beyond plain text — file references, structured
data, etc. — are not yet handled; add them at the call site that
needs them.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

try:
    # Order matters: TaskState is an enum used by every helper. If the
    # extra is missing, every public function raises at import time
    # with a clear message — same soft-import guard the mcp package uses.
    from a2a.helpers.proto_helpers import new_text_part
    from a2a.types import (
        CancelTaskRequest,
        GetTaskRequest,
        Message,
        Role,
        SendMessageConfiguration,
        SendMessageRequest,
        StreamResponse,
        Task,
        TaskState,
    )
except ImportError as ie:
    if ie.name is not None and ie.name != "a2a" and not ie.name.startswith("a2a."):
        # A transitive dependency of an installed a2a-sdk failed — surface
        # the real error instead of mislabeling it "extra not installed".
        raise
    raise ImportError(
        "Please install the 'a2a' extra to use A2A protocol support. Run: pip install 'troopai-adk-python[a2a]'",
        # name="a2a" lets optional-extra guards (e.g. the graphs adapter's
        # A2A fallthrough) recognize the missing extra and degrade gracefully.
        name="a2a",
    ) from ie

from troopai.adk.a2a.a2a_continuation_token import A2ATaskStateLiteral, A2ATaskStatus

if TYPE_CHECKING:
    # Forward reference — A2AStreamEvent is defined in a2a_agent.py;
    # importing it at runtime would create a converters <-> a2a_agent
    # cycle (a2a_agent imports A2AClient which imports converters).
    from troopai.adk.a2a.a2a_agent import A2AStreamEvent

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# TaskState <-> framework string literal
# ----------------------------------------------------------------------

# Single source of truth for the protobuf-enum-int <-> string-literal
# mapping. Kept as plain dicts (not Enum) so callers can branch on the
# string literal in pattern-matched code without importing protobuf.
_PROTO_TO_LITERAL: dict[int, A2ATaskStateLiteral] = {
    TaskState.TASK_STATE_SUBMITTED: "submitted",
    TaskState.TASK_STATE_WORKING: "working",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input_required",
    TaskState.TASK_STATE_AUTH_REQUIRED: "auth_required",
    TaskState.TASK_STATE_COMPLETED: "completed",
    TaskState.TASK_STATE_CANCELED: "cancelled",
    TaskState.TASK_STATE_REJECTED: "rejected",
    TaskState.TASK_STATE_FAILED: "failed",
}

_TERMINAL_STATES: frozenset[A2ATaskStateLiteral] = frozenset({"completed", "cancelled", "rejected", "failed"})
"""States from which the task cannot transition further."""

_FAILURE_TERMINAL_STATES: frozenset[A2ATaskStateLiteral] = frozenset({"cancelled", "rejected", "failed"})
"""Terminal states that are NOT a successful completion — caller should raise."""

_INTERRUPT_STATES: frozenset[A2ATaskStateLiteral] = frozenset({"input_required", "auth_required"})
"""Non-terminal states that pause the task awaiting external input.

A server may emit either of these as a transient state on the way
to ``completed``/``failed``, or as a final state requiring caller
action before the conversation can continue. Both
``_consume_until_terminal`` and the executor must STOP iterating
on these states — looping past them with the chunk cap as the
only bound produces a misleading "exceeded max_stream_chunks"
error instead of a clear "task requires input" error.
"""


def task_state_to_literal(state: int) -> A2ATaskStateLiteral:
    """Map a protobuf ``TaskState`` value to a framework string literal.

    Args:
        state: The protobuf enum integer value (e.g.
            ``TaskState.TASK_STATE_WORKING``).

    Returns:
        The matching framework state literal.

    Raises:
        ValueError: If the integer does not correspond to any known
            ``TaskState`` value (defends against future protocol
            extensions surfacing unknown states without a typed
            mapping).
    """
    literal = _PROTO_TO_LITERAL.get(state)
    if literal is None:
        raise ValueError(f"Unknown A2A TaskState integer value: {state}")
    return literal


def is_terminal_state(state: A2ATaskStateLiteral) -> bool:
    """Return True if the state is terminal (no further transitions).

    Args:
        state: A framework task-state literal.

    Returns:
        ``True`` when ``state`` is one of ``"completed"``,
        ``"cancelled"``, ``"rejected"``, or ``"failed"``.
    """
    return state in _TERMINAL_STATES


def is_failure_state(state: A2ATaskStateLiteral) -> bool:
    """Return True if the terminal state represents a failure.

    Args:
        state: A framework task-state literal.

    Returns:
        ``True`` when ``state`` is one of ``"cancelled"``,
        ``"rejected"``, or ``"failed"`` (i.e., terminal but not
        ``"completed"``).
    """
    return state in _FAILURE_TERMINAL_STATES


def is_interrupt_state(state: A2ATaskStateLiteral) -> bool:
    """Return True if the state pauses the task awaiting external input.

    ``input_required`` and ``auth_required`` are NOT terminal but
    callers MUST stop iterating when they arrive — no further
    progress will be made until the caller either provides input
    via a follow-up turn or supplies auth credentials.

    Args:
        state: A framework task-state literal.

    Returns:
        ``True`` when ``state`` is ``"input_required"`` or
        ``"auth_required"``.
    """
    return state in _INTERRUPT_STATES


# ----------------------------------------------------------------------
# Message <-> text
# ----------------------------------------------------------------------


def extract_text_from_message(msg: Message) -> str:
    """Concatenate every text Part in a Message into a single string.

    Non-text parts (file/url/data) are logged at WARNING and skipped —
    the framework currently doesn't have a typed surface for streaming
    binary attachments back through the runner. A future enhancement
    can extend the executor to surface multi-modal Parts via
    ``LLMInputContentItem`` when a real call site needs it.

    Args:
        msg: The inbound A2A ``Message`` (from a Task's ``history``,
            from a ``Task.status.message``, or from a streaming
            response payload).

    Returns:
        The concatenated text content. Empty string when the message
        had no text parts.
    """
    chunks: list[str] = []
    for part in msg.parts:
        which = part.WhichOneof("content")
        if which == "text":
            chunks.append(part.text)
        elif which is None:
            continue
        else:
            logger.warning(
                "Skipping non-text Part of kind %r in A2A message; multi-modal parts are not yet surfaced.",
                which,
            )
    return "".join(chunks)


def extract_text_from_task(task: Task) -> str:
    """Pull the user-visible result text out of a completed Task.

    Looks first at the most recently emitted artifact (the typical
    location of the agent's final answer), then falls back to the
    last message in the task ``history``.

    Args:
        task: The completed :class:`a2a.types.Task` whose text content
            should be extracted.

    Returns:
        The concatenated text from the final artifact or last history
        message, or an empty string if neither has any text content.
    """
    if len(task.artifacts) > 0:
        last_artifact = task.artifacts[-1]
        chunks = [p.text for p in last_artifact.parts if p.WhichOneof("content") == "text"]
        if len(chunks) > 0:
            return "".join(chunks)
    if len(task.history) > 0:
        return extract_text_from_message(task.history[-1])
    return ""


def build_user_message(
    text: str,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
    message_id: str | None = None,
) -> Message:
    """Construct an A2A ``Message`` from a plain prompt string.

    Args:
        text: The user's prompt.
        context_id: Optional conversation context identifier — when
            continuing a multi-turn conversation, callers supply the
            same ``context_id`` returned by the prior task.
        task_id: Optional remote task identifier — set when continuing
            a task that previously paused for ``input_required``.
        message_id: Optional explicit message identifier. When omitted,
            a UUID4 is generated. Always populating ``message_id`` makes
            replay across retries idempotent on the server side.

    Returns:
        A protobuf ``Message`` ready to wrap in a ``SendMessageRequest``.
    """
    return Message(
        message_id=message_id if message_id is not None else str(uuid.uuid4()),
        role=Role.ROLE_USER,
        parts=[new_text_part(text)],
        context_id=context_id if context_id is not None else "",
        task_id=task_id if task_id is not None else "",
    )


def build_send_request(
    message: Message,
    *,
    return_immediately: bool = False,
    accepted_output_modes: list[str] | None = None,
) -> SendMessageRequest:
    """Wrap a ``Message`` in a ``SendMessageRequest`` for transmission.

    Args:
        message: The user/agent message to send.
        return_immediately: Inverse of "blocking". When ``False``
            (default) the server holds the response open until the
            task reaches a terminal state — used by both blocking
            (``A2AClient.send_message``) and streaming
            (``A2AClient.stream_message``) callers, since each wants
            the server to keep producing events through to terminal.
            When ``True`` the server returns as soon as a
            ``task_id`` / ``context_id`` pair has been minted —
            used by :meth:`A2AClient.submit_background` only.
            Wire-level field name comes from the
            ``a2a.types.SendMessageConfiguration`` protobuf.
        accepted_output_modes: Optional content-type filter passed to
            the server (``["text/plain"]``, ``["application/json"]``,
            etc.). Defaults to ``None`` — the server picks.

    Returns:
        A :class:`a2a.types.SendMessageRequest` ready to pass to the
        protocol client.
    """
    config = SendMessageConfiguration(return_immediately=return_immediately)
    if accepted_output_modes is not None and len(accepted_output_modes) > 0:
        config.accepted_output_modes.extend(accepted_output_modes)
    return SendMessageRequest(message=message, configuration=config)


# ----------------------------------------------------------------------
# Task <-> A2ATaskStatus snapshot (for poll_task)
# ----------------------------------------------------------------------


def task_to_status(task: Task) -> A2ATaskStatus:
    """Project an A2A ``Task`` into a framework ``A2ATaskStatus``.

    Used by ``A2AClient.poll_task()`` to surface the remote task's
    current state without leaking protobuf types. ``result`` is
    populated only when the task reached ``"completed"``.

    Args:
        task: The :class:`a2a.types.Task` snapshot returned by the
            remote server's ``get_task`` endpoint.

    Returns:
        An :class:`A2ATaskStatus` reflecting the current state,
        with ``result`` populated if completed and ``message``
        populated if the status carries human-readable detail.
    """
    state_literal = task_state_to_literal(task.status.state)
    result_text: str | None = None
    if state_literal == "completed":
        result_text = extract_text_from_task(task)
    failure_message: str | None = None
    if task.status.message is not None and len(task.status.message.parts) > 0:
        msg_text = extract_text_from_message(task.status.message)
        if len(msg_text) > 0:
            failure_message = msg_text
    return A2ATaskStatus(
        task_id=task.id,
        context_id=task.context_id,
        state=state_literal,
        result=result_text,
        message=failure_message,
    )


# ----------------------------------------------------------------------
# StreamResponse discriminator
# ----------------------------------------------------------------------


def stream_response_kind(response: StreamResponse) -> str | None:
    """Identify which payload variant a ``StreamResponse`` carries.

    The protobuf ``StreamResponse`` is a ``oneof payload`` with four
    options: ``task``, ``message``, ``status_update``, ``artifact_update``.

    Args:
        response: A single :class:`a2a.types.StreamResponse` chunk
            from the protocol iterator.

    Returns:
        The active payload field name (``"task"``, ``"message"``,
        ``"status_update"``, or ``"artifact_update"``), or ``None``
        when no payload is set.
    """
    # Protobuf's WhichOneof is typed as Any in the stubs; the runtime
    # contract is "string field name or None". Typed local pins the
    # framework-facing return shape.
    kind: str | None = response.WhichOneof("payload")
    return kind


# ----------------------------------------------------------------------
# Request builders for poll/cancel
# ----------------------------------------------------------------------


def build_get_task_request(task_id: str, *, history_length: int = 0) -> GetTaskRequest:
    """Build a ``GetTaskRequest`` for polling a remote task by id.

    Args:
        task_id: The remote task identifier to fetch.
        history_length: Number of history messages to include in the
            response. Default 0 (status only, no history).

    Returns:
        A :class:`a2a.types.GetTaskRequest` ready to pass to the
        protocol client's ``get_task`` method.
    """
    return GetTaskRequest(id=task_id, history_length=history_length)


def build_cancel_task_request(task_id: str) -> CancelTaskRequest:
    """Build a ``CancelTaskRequest`` for cancelling a remote task by id.

    Args:
        task_id: The remote task identifier to cancel.

    Returns:
        A :class:`a2a.types.CancelTaskRequest` ready to pass to the
        protocol client's ``cancel_task`` method.
    """
    return CancelTaskRequest(id=task_id)


def stream_response_to_event(
    chunk: StreamResponse,
    accumulator: list[str],
) -> A2AStreamEvent | None:
    """Convert a raw protobuf ``StreamResponse`` into a framework event.

    Returns ``None`` when the chunk carries no actionable update (e.g.
    a status_update for a non-terminal state with no text). The
    ``accumulator`` argument is mutated in-place to collect text
    deltas so a downstream final ``"completed"`` event can return
    the full accumulated text.

    Lives in ``converters.py`` (rather than ``a2a_client.py``) because
    its job is the single boundary crossing from ``a2a.types`` to
    framework-typed events — exactly what this module owns. Reusable
    by the server-side executor without an inter-module dependency
    on the client.

    Args:
        chunk: A single :class:`a2a.types.StreamResponse` from the
            protocol iterator.
        accumulator: Mutable list collecting text-delta strings so a
            downstream ``"completed"`` event can return the full
            accumulated text.

    Returns:
        An :class:`A2AStreamEvent` TypedDict when the chunk carries
        actionable content, or ``None`` when the chunk has no
        text-bearing update.
    """
    kind = stream_response_kind(chunk)
    if kind == "artifact_update":
        update = chunk.artifact_update
        text_parts = [p.text for p in update.artifact.parts if p.WhichOneof("content") == "text"]
        text = "".join(text_parts)
        if len(text) == 0:
            return None
        if not update.append:
            # append=False is a fresh / replacement artifact snapshot, not a
            # delta onto the prior one. Reset the running text so a
            # cumulative-snapshot server (or the start of a new artifact) is
            # not double-counted as a delta — matching the last-artifact
            # semantics of extract_text_from_task. append=True concatenates.
            accumulator.clear()
        accumulator.append(text)
        return {"type": "text_delta", "text_delta": text}
    if kind == "status_update":
        status = chunk.status_update.status
        state_literal = task_state_to_literal(status.state)
        if state_literal == "completed":
            return {"type": "completed", "result_text": "".join(accumulator)}
        msg = ""
        if status.message is not None and len(status.message.parts) > 0:
            msg = extract_text_from_message(status.message)
        if is_failure_state(state_literal):
            return {"type": "failed", "state": state_literal, "message": msg}
        # Non-terminal status (including the interrupt states
        # input_required / auth_required, whose status.message carries the
        # prompt the caller must answer). Surface that prompt so the
        # downstream interrupt raise does not report an empty message.
        if len(msg) > 0:
            return {"type": "status", "state": state_literal, "message": msg}
        return {"type": "status", "state": state_literal}
    if kind == "task":
        state_literal = task_state_to_literal(chunk.task.status.state)
        if state_literal == "completed":
            final_text = extract_text_from_task(chunk.task)
            # Append to the accumulator so the caller's span output reflects
            # the real response text for non-incremental servers that emit a
            # single task chunk instead of artifact_update deltas.
            if final_text:
                accumulator.append(final_text)
            return {"type": "completed", "result_text": final_text}
        if is_failure_state(state_literal):
            msg = ""
            if chunk.task.status.message is not None and len(chunk.task.status.message.parts) > 0:
                msg = extract_text_from_message(chunk.task.status.message)
            return {"type": "failed", "state": state_literal, "message": msg}
        return {"type": "status", "state": state_literal}
    if kind == "message":
        # A free-standing agent message outside a task lifecycle —
        # surface as a single text delta.
        text = extract_text_from_message(chunk.message)
        if len(text) == 0:
            return None
        accumulator.append(text)
        return {"type": "text_delta", "text_delta": text}
    return None
