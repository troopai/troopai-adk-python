"""Exception hierarchy for the A2A (Agent-to-Agent) protocol layer.

All A2A errors derive from :class:`A2AError`, which itself derives from
:class:`troopai.adk.exceptions.TroopAIError`. Callers can catch any
framework error including A2A failures with a single ``except TroopAIError``.

Mirrors the convention established by ``llms/litellm/exceptions.py`` —
each subclass models a distinct failure mode at the protocol boundary so
calling code can branch on a typed cause rather than a string match.
"""

from __future__ import annotations

from typing import override

from troopai.adk.a2a.a2a_continuation_token import A2ATaskStateLiteral
from troopai.adk.exceptions import TroopAIError


class A2AError(TroopAIError):
    """Base class for all A2A protocol errors."""


class A2ATransportError(A2AError):
    """Transport-layer failure when talking to a remote A2A endpoint.

    Raised for connection refusal, DNS failures, TLS errors, and HTTP
    timeouts. The remote agent never received the request, or the
    response could not be retrieved.
    """


class A2AProtocolError(A2AError):
    """A2A-level protocol violation surfaced by the ADK.

    Raised for malformed responses, unsupported protocol versions,
    auth-scheme rejection (HTTP 401/403), or unexpected payload shapes
    that the a2a-sdk's wire types cannot parse. The remote endpoint
    answered, but did not honour the protocol contract.
    """


class A2ATaskError(A2AError):
    """The remote task reached a terminal failure state.

    Raised for terminal ``TASK_STATE_FAILED`` and ``TASK_STATE_REJECTED``.
    For ``TASK_STATE_CANCELED`` the more specific subclass
    ``A2ATaskCancelledError`` is raised (it IS-A ``A2ATaskError``, so
    callers catching ``A2ATaskError`` still see all three). Exposes the
    task identifiers and the remote agent's failure message so callers
    can route on the typed state literal rather than parse the message.

    Attributes:
        task_id: The remote task identifier.
        context_id: The conversation context identifier the task
            belongs to.
        state: The terminal-state literal — one of ``"failed"``,
            ``"rejected"``, ``"cancelled"``.
        remote_message: Human-readable reason from the remote agent,
            empty string if the remote provided none.

    .. warning::
       ``remote_message`` is **untrusted input** sourced from the peer
       agent. It MAY contain prompt-injection bait, escape sequences,
       or misleading text designed to influence downstream rendering.
       Applications surfacing this field to end-users (in error pages,
       chat UIs, etc.) MUST escape / sanitise it appropriately.
    """

    task_id: str
    context_id: str
    state: A2ATaskStateLiteral
    remote_message: str

    def __init__(
        self,
        *,
        task_id: str,
        context_id: str,
        state: A2ATaskStateLiteral,
        remote_message: str = "",
    ) -> None:
        self.task_id = task_id
        self.context_id = context_id
        self.state = state
        self.remote_message = remote_message
        suffix = f": {remote_message}" if len(remote_message) > 0 else ""
        super().__init__(f"A2A task {task_id} ended in state {state}{suffix}")

    @override
    def __str__(self) -> str:
        return self.message


class A2ATaskCancelledError(A2ATaskError):
    """The remote task was cancelled before completion.

    Distinct subclass so callers that want to treat cancellation as a
    non-error path can pattern-match on the type rather than inspect
    the ``state`` field.
    """


class A2ATaskInterruptedError(A2AError):
    """The remote task is paused awaiting external input.

    Raised when a remote task transitions to a non-terminal state
    that requires caller action before progress can continue:

    * ``input_required`` — the agent is asking the user / caller
      for additional information; the caller should send a follow-up
      message in the same ``context_id`` to provide it.
    * ``auth_required`` — the agent is asking the caller to supply
      authentication; the caller should configure an
      :class:`a2a.client.AuthInterceptor` or otherwise satisfy the
      auth scheme advertised in the AgentCard, then resume.

    Distinct from :class:`A2ATaskError`: interruption is not a
    failure — the task can still complete once the input is supplied.
    Callers that don't care about the distinction may catch
    :class:`A2AError` for both cases.

    Attributes:
        task_id: The remote task identifier (use to resume).
        context_id: The conversation context identifier.
        state: Either ``"input_required"`` or ``"auth_required"``.
        prompt: Human-readable detail from the remote agent
            describing what input or auth is needed. May be empty.

    .. warning::
       ``prompt`` is **untrusted input** sourced from the peer
       agent. It MAY contain prompt-injection bait, ANSI escape
       sequences, Unicode control characters, or misleading text
       designed to influence downstream rendering or
       prompt-conditioning. The string flows verbatim into
       ``__str__`` and from there into framework spans / logs.
       Applications surfacing this field to end-users (in
       error pages, chat UIs, follow-up LLM prompts) MUST escape /
       sanitise it appropriately. Same caution as
       :class:`A2ATaskError.remote_message`.
    """

    task_id: str
    context_id: str
    state: A2ATaskStateLiteral
    prompt: str

    def __init__(
        self,
        *,
        task_id: str,
        context_id: str,
        state: A2ATaskStateLiteral,
        prompt: str = "",
    ) -> None:
        self.task_id = task_id
        self.context_id = context_id
        self.state = state
        self.prompt = prompt
        suffix = f": {prompt}" if len(prompt) > 0 else ""
        super().__init__(f"A2A task {task_id} paused at {state}{suffix}")

    @override
    def __str__(self) -> str:
        return self.message
