"""Typed continuation handles for long-running remote A2A tasks.

The ``A2AContinuationToken`` is the framework's typed equivalent of an
A2A protocol task identifier pair (``task_id`` + ``context_id``) plus
the remote URL needed to resume the task from a different process or
context. It is surfaced by ``A2ARunner.arun(agent, prompt, background=True)``
and accepted by ``A2ARunner.arun(agent, prompt, continuation_token=token)``
for resumption; ``A2ARunner.poll_task(agent, token)`` returns a status snapshot.

The token is JSON-serialisable via ``dataclasses.asdict()``, so callers
can persist it to a database or queue and resume the task from a fresh
process — durable as long as the remote ``TaskStore`` retains the task.

The matching ``A2ATaskStatus`` snapshot describes the remote task's
state at poll time. ``result`` is populated only when the task reached
``TASK_STATE_COMPLETED``; ``message`` carries human-readable detail for
non-terminal interruptions (``input_required`` / ``auth_required``) and
terminal failures (``failed`` / ``rejected``).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

A2ATaskStateLiteral = Literal[
    "submitted",
    "working",
    "input_required",
    "auth_required",
    "completed",
    "cancelled",
    "rejected",
    "failed",
]
"""Framework-typed task-state literals.

These map 1:1 to the protobuf ``TaskState`` enum values shipped by
``a2a-sdk``, but live here as a string ``Literal`` so framework code
never has to import a wire-format enum to discriminate state.
"""


@dataclasses.dataclass(frozen=True, kw_only=True)
class A2AContinuationToken:
    """Typed handle for resuming a long-running remote A2A task.

    Attributes:
        task_id: The remote task identifier issued by the A2A server
            when the task was submitted.
        context_id: The conversation context identifier the task
            belongs to. Multi-turn conversations re-use the same
            ``context_id`` across distinct task ids.
        remote_url: The base URL of the remote A2A endpoint that owns
            the task. Required for resume so the framework can route
            the poll/cancel request to the correct server even from a
            fresh process that never held the original ``A2AAgent``.
    """

    task_id: str
    """Remote task identifier."""

    context_id: str
    """Conversation context identifier."""

    remote_url: str
    """Base URL of the remote A2A endpoint that owns the task.

    Not used for routing — ``A2AClient.poll_task`` and
    ``A2AClient.cancel_task`` always route via the URL the client
    was constructed with. Because the field is carried in a
    serialisable token, callers deserialising tokens from untrusted
    sources (e.g. user-uploaded JSON) MUST validate this URL before
    use — it is a trust boundary. ``__post_init__`` enforces an
    ``http(s)://`` scheme as a baseline SSRF guard, but callers SHOULD
    still allowlist the host against the known remote agent.
    """

    def __post_init__(self) -> None:
        """Validate the token at construction (the single deserialization gate).

        A continuation token is frequently rebuilt from an untrusted source —
        persisted JSON, a queue message — via ``A2AContinuationToken(**data)``,
        which runs this hook. Validating here means every construction path is
        checked once, rather than trusting each call site.

        Raises:
            ValueError: If any identifier is empty, or ``remote_url`` does not
                use an ``http://`` / ``https://`` scheme.
        """
        if len(self.task_id) == 0:
            raise ValueError("A2AContinuationToken.task_id must be non-empty")
        if len(self.context_id) == 0:
            raise ValueError("A2AContinuationToken.context_id must be non-empty")
        if len(self.remote_url) == 0:
            raise ValueError("A2AContinuationToken.remote_url must be non-empty")
        # SSRF baseline: the URL may arrive from an untrusted token, so reject
        # any non-HTTP(S) scheme (file://, gopher://, …) before it can reach an
        # HTTP client on the resume path.
        if not (self.remote_url.startswith("https://") or self.remote_url.startswith("http://")):
            raise ValueError(
                f"A2AContinuationToken.remote_url must use an http:// or https:// scheme (got {self.remote_url!r})"
            )


@dataclasses.dataclass(frozen=True, kw_only=True)
class A2ATaskStatus:
    """Snapshot of a remote task's state at poll time.

    Attributes:
        task_id: The remote task identifier.
        context_id: The conversation context identifier.
        state: Current task-state literal.
        result: The completed task's output text, populated only when
            ``state == "completed"``. ``None`` otherwise.
        message: Optional human-readable detail. Populated for
            interrupted states (``input_required`` / ``auth_required``)
            and terminal failure states (``failed`` / ``rejected``).
    """

    task_id: str
    """Remote task identifier."""

    context_id: str
    """Conversation context identifier."""

    state: A2ATaskStateLiteral
    """Current task-state literal."""

    result: str | None = None
    """Completed task output, or ``None`` if not yet completed."""

    message: str | None = None
    """Human-readable detail for interrupted or failed states."""
