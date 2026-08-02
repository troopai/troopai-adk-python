"""``A2ARunner`` — execution entry point for :class:`A2AAgent`.

``A2ARunner`` is the execution side of the A2A peer surface, paired
with :class:`A2AAgent` as its config side — the same config/execution
split used everywhere else:

* :class:`troopai.adk.run.runner.Runner` is execution for local
  :class:`Agent`, :class:`Swarm`, and :class:`Graph`.
* :class:`A2ARunner` (this class) is execution for remote
  :class:`A2AAgent` peers — and ONLY for ``A2AAgent``.

.. important::

   Every public method on :class:`A2ARunner` accepts only an
   :class:`A2AAgent` as the agent argument — NOT :class:`Agent`,
   :class:`BaseAgent`, :class:`Swarm`, :class:`Graph`, or any other
   primitive. Local-agent execution belongs on :class:`Runner`; this
   runner is exclusively for the remote-A2A peer surface.

   Static type checkers reject the wrong type at the call site. A
   defensive ``isinstance(agent, A2AAgent)`` guard at runtime raises
   :class:`TypeError` with a message pointing the caller at
   :class:`Runner` for local agents. Both layers exist deliberately:
   the static check catches most misuses at edit time; the runtime
   guard catches ``Any``-typed boundary cases (dynamic dispatch
   tables, JSON-loaded configs, untyped third-party callers).

``A2ARunner`` is the only execution entry point for :class:`A2AAgent`;
the agent itself is pure config.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Literal, overload

from troopai.adk.a2a.a2a_agent import A2AAgent, A2ARunResult, A2AStreamEvent
from troopai.adk.a2a.a2a_continuation_token import A2AContinuationToken, A2ATaskStatus

logger = logging.getLogger(__name__)


def _check_a2a_agent(agent: object, method_name: str) -> A2AAgent:
    """Validate that ``agent`` is an :class:`A2AAgent`; narrow the type.

    Raises :class:`TypeError` with a message pointing the caller at
    :class:`Runner` for local agents. Used by every public method
    on :class:`A2ARunner` to enforce the A2AAgent-only contract.
    """
    if not isinstance(agent, A2AAgent):
        raise TypeError(
            f"A2ARunner.{method_name} accepts only A2AAgent instances; "
            f"got {type(agent).__name__}. For local Agent / Swarm / Graph "
            f"primitives, use troopai.adk.run.runner.Runner."
        )
    return agent


class A2ARunner:
    """Execution entry point for :class:`A2AAgent` peers (remote A2A endpoints).

    Pure namespace of ``@classmethod`` entry points — no instance
    state. The agent itself carries the lazy
    :class:`A2AClient`; this runner translates the caller's flag
    combinations into client method calls and enforces the type +
    mutex contracts.

    .. important::
       Every method here accepts ONLY :class:`A2AAgent` instances.
       Passing a local :class:`Agent`, :class:`Swarm`, or any other
       primitive raises :class:`TypeError`. Use
       :class:`troopai.adk.run.runner.Runner` for those.

    Why a separate runner from :class:`Runner`? :class:`Runner`
    already exists for local primitives. Multiplexing local + remote
    on a single runner would force every entry point to discriminate
    at the type level — for no benefit, since the wire format,
    lifecycle, and error model are entirely different. A separate
    runner per agent kind is the cheaper and clearer model.
    """

    @overload
    @classmethod
    async def arun(
        cls,
        agent: A2AAgent,
        prompt: str,
        *,
        stream: Literal[False] = False,
        background: Literal[False] = False,
        continuation_token: A2AContinuationToken | None = None,
        context_id: str | None = None,
    ) -> A2ARunResult: ...

    @overload
    @classmethod
    async def arun(
        cls,
        agent: A2AAgent,
        prompt: str,
        *,
        stream: Literal[True],
        background: Literal[False] = False,
        continuation_token: A2AContinuationToken | None = None,
        context_id: str | None = None,
    ) -> AsyncIterator[A2AStreamEvent]: ...

    @overload
    @classmethod
    async def arun(
        cls,
        agent: A2AAgent,
        prompt: str,
        *,
        stream: Literal[False] = False,
        background: Literal[True],
        context_id: str | None = None,
    ) -> A2AContinuationToken: ...

    # mypy[misc]: overload-implementation bool-vs-Literal mismatch.
    # The runtime flag dispatch below covers every literal overload
    # domain; mypy cannot prove this. Pyright accepts the same
    # construct without a marker. The ignore is load-bearing:
    # removing it raises a real mypy error that no source-level fix
    # can clear without abandoning the ``@overload`` contract.
    @classmethod
    async def arun(  # type: ignore[misc]
        cls,
        agent: A2AAgent,
        prompt: str,
        *,
        stream: bool = False,
        background: bool = False,
        continuation_token: A2AContinuationToken | None = None,
        context_id: str | None = None,
    ) -> A2ARunResult | AsyncIterator[A2AStreamEvent] | A2AContinuationToken:
        """Run a remote A2A peer; dispatch by ``stream`` / ``background``.

        Flag combinations and return types (typed via ``@overload``):
        default → :class:`A2ARunResult` (blocks until terminal);
        ``stream=True`` → ``AsyncIterator[A2AStreamEvent]``;
        ``background=True`` → :class:`A2AContinuationToken`.

        Args:
            agent: The remote :class:`A2AAgent` — MUST be an
                ``A2AAgent``; other primitives raise ``TypeError``.
            prompt: User-facing prompt text.
            stream: Mutually exclusive with ``background``.
            background: Mutually exclusive with ``stream``.
            continuation_token: Mutually exclusive with ``context_id``;
                the token's ``context_id`` is authoritative.
            context_id: Continue a multi-turn conversation.

        Raises:
            TypeError: ``agent`` is not an :class:`A2AAgent`.
            ValueError: mutually-exclusive flags combined.
            A2A*Error: see the ``A2A*Error`` class hierarchy in
                :mod:`troopai.adk.a2a.exceptions`.
        """
        checked = _check_a2a_agent(agent, "arun")
        if stream and background:
            raise ValueError("A2ARunner.arun cannot combine stream=True with background=True")
        if continuation_token is not None and context_id is not None:
            raise ValueError(
                "A2ARunner.arun cannot accept both continuation_token and "
                "context_id; the token's context_id is authoritative."
            )
        if stream and continuation_token is not None:
            raise ValueError(
                "A2ARunner.arun cannot combine stream=True with continuation_token; "
                "streaming resume is not supported. Use stream=False to resume a "
                "paused task via continuation_token."
            )
        client = await checked.get_client()
        if background:
            return await client.submit_background(prompt, context_id=context_id)
        if stream:
            return client.stream_message(prompt, context_id=context_id)
        return await client.send_message(
            prompt,
            context_id=context_id,
            continuation_token=continuation_token,
        )

    @classmethod
    async def poll_task(
        cls,
        agent: A2AAgent,
        token: A2AContinuationToken,
    ) -> A2ATaskStatus:
        """One-shot status snapshot for a previously-submitted task.

        Does NOT wait for the task to reach a terminal state — returns
        whatever state the remote currently holds. To wait, call this
        in a polling loop bounded by your own timeout / retry budget.

        Args:
            agent: MUST be an :class:`A2AAgent` instance. Raises
                :class:`TypeError` for any other primitive.
            token: The continuation token returned by a prior
                ``arun(background=True)`` call.

        Raises:
            TypeError: If ``agent`` is not an :class:`A2AAgent`.
            A2ATransportError: Network failure polling the task.
            A2AProtocolError: Server-side protocol violation.
        """
        checked = _check_a2a_agent(agent, "poll_task")
        client = await checked.get_client()
        return await client.poll_task(token)

    @classmethod
    async def cancel_task(
        cls,
        agent: A2AAgent,
        token: A2AContinuationToken,
    ) -> None:
        """Request cancellation of an in-flight remote task.

        Returns once the cancellation request is acknowledged by the
        server. Does not wait for the task to actually transition to
        ``cancelled``; caller can :meth:`poll_task` to confirm.

        Args:
            agent: MUST be an :class:`A2AAgent` instance. Raises
                :class:`TypeError` for any other primitive.
            token: The continuation token returned by a prior
                ``arun(background=True)`` call.

        Raises:
            TypeError: If ``agent`` is not an :class:`A2AAgent`.
            A2ATransportError: Network failure on the cancel request.
            A2AProtocolError: Server-side protocol violation.
        """
        checked = _check_a2a_agent(agent, "cancel_task")
        client = await checked.get_client()
        await client.cancel_task(token)


__all__ = ["A2ARunner"]
