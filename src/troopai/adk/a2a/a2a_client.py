"""Framework-typed client for talking to a remote A2A endpoint.

:class:`A2AClient` is a thin wrapper over the protocol-library
:class:`Client` abstraction (constructed via
:func:`a2a.client.create_client`). It does three jobs the bare
``Client`` does not:

1. Translates between framework-typed inputs/outputs (plain ``str``
   prompts, :class:`A2ARunResult`, :class:`A2AStreamEvent` TypedDicts,
   :class:`A2AContinuationToken`) and the protobuf wire types — every
   ``a2a.types`` import stays inside this file or :mod:`converters`.
2. Maps protocol-level failures (transport timeouts, malformed
   responses, terminal failure ``TaskState`` values) to the typed
   :mod:`exceptions` hierarchy so callers can catch by cause rather
   than parse strings.
3. Owns an internal :class:`httpx.AsyncClient` lifecycle when no client
   was injected, and emits a :func:`function_span` per public method so
   A2A traffic shows up in OpenTelemetry next to local tool calls.

The class is constructed eagerly with a URL and (optionally) an
explicit :class:`AgentCard`. The remote card is fetched lazily on first
use unless one was supplied — this lets construction stay synchronous
while still benefiting from card-based capability negotiation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

try:
    import httpx
    from a2a.client import (
        A2ACardResolver,
        A2AClientError,
        A2AClientTimeoutError,
        AgentCardResolutionError,
        Client,
        ClientCallInterceptor,
        ClientConfig,
        ClientFactory,
    )
    from a2a.types import (
        AgentCard,
        Artifact,
        Message,
        StreamResponse,
        Task,
        TaskState,
        TaskStatus,
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

from troopai.adk.a2a import converters
from troopai.adk.a2a.a2a_continuation_token import A2AContinuationToken, A2ATaskStatus
from troopai.adk.a2a.converters import stream_response_to_event
from troopai.adk.a2a.exceptions import (
    A2AProtocolError,
    A2ATaskCancelledError,
    A2ATaskError,
    A2ATaskInterruptedError,
    A2ATransportError,
)
from troopai.adk.tracing import function_span

if TYPE_CHECKING:
    # Forward references — the runtime imports happen inside method
    # bodies (see send_message / stream_message) to break the
    # client <-> agent module cycle while still letting pyright narrow
    # the return-annotation types.
    from troopai.adk.a2a.a2a_agent import A2ARunResult, A2AStreamEvent

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_POLL_INTERVAL_SECONDS: float = 1.0

# Per-call streaming hot path is bounded so a runaway remote agent
# cannot exhaust memory or stall indefinitely. These are framework-
# side safety bounds — they do NOT belong on
# :class:`a2a.client.ClientConfig`, which governs protocol-library
# behaviour rather than ours.
DEFAULT_MAX_STREAM_CHUNKS: int = 10_000
DEFAULT_MAX_STREAM_BYTES: int = 8 * 1024 * 1024  # 8 MiB total accumulated text


class _Closed:
    """Marker sentinel for client teardown state.

    Plain class (not a dataclass) — the field-less sentinel needs no
    generated ``__init__`` / ``__repr__`` / ``__eq__``; identity
    comparison via the singleton ``_CLOSED`` is the only contract.
    """


_CLOSED = _Closed()


@dataclasses.dataclass
class _StreamAccumulator:
    """Typed accumulator for the streaming-Task assembly hot path.

    Centralises the contract between
    :meth:`A2AClient._consume_until_terminal` and
    :meth:`A2AClient._apply_chunk`. Carries plain locals — a
    ``task_id`` / ``context_id`` pair, the latest ``TaskStatus``,
    the running ``artifacts`` list, and the streamed-bytes counter.

    The final :meth:`to_task` constructs a **fresh** :class:`Task`
    whose sub-messages protobuf has copied — so the returned Task
    does not alias any object the SDK iterator may still hold a
    reference to.

    Attributes:
        task_id: Most-recent task identifier seen on the stream.
        context_id: Most-recent context identifier seen on the stream.
        status: Latest :class:`a2a.types.TaskStatus`; ``None`` until
            the first signal arrives.
        artifacts: Artifacts accumulated across ``artifact_update``
            chunks (or copied from a Task snapshot).
        messages: Free-standing ``message`` chunks accumulated across
            the stream — a server may deliver its final answer as a
            message rather than artifact deltas.
        accumulated_bytes: Running byte count of streamed text-part
            content for the framework-side ``max_stream_bytes`` cap.
    """

    task_id: str = ""
    """Most-recent task identifier seen on the stream."""

    context_id: str = ""
    """Most-recent context identifier seen on the stream."""

    status: TaskStatus | None = None
    """Latest :class:`a2a.types.TaskStatus`; ``None`` until the first signal arrives."""

    artifacts: list[Artifact] = dataclasses.field(default_factory=list)
    """Artifacts accumulated across ``artifact_update`` chunks (or copied from a Task snapshot)."""

    messages: list[Message] = dataclasses.field(default_factory=list)
    """Free-standing ``message`` chunks accumulated across the stream
    (a server may deliver its final answer as a message instead of
    artifact deltas)."""

    accumulated_bytes: int = 0
    """Running byte count of streamed text-part content for the
    framework-side ``max_stream_bytes`` cap."""

    def to_task(self) -> Task:
        """Construct a fresh :class:`Task` owning its sub-messages.

        Protobuf-python copies sub-messages and repeated-field
        elements at constructor time, so the returned Task does
        NOT alias the SDK-vended status or artifact instances the
        accumulator captured during streaming.

        Returns:
            A new :class:`a2a.types.Task` built from the accumulated
            ``task_id``, ``context_id``, ``status``, ``artifacts``, and
            free-standing ``messages`` (surfaced as Task ``history`` so
            :func:`converters.extract_text_from_task` can fall back to a
            message-delivered answer when no artifacts are present).
        """
        return Task(
            id=self.task_id,
            context_id=self.context_id,
            status=self.status,
            artifacts=list(self.artifacts),
            history=list(self.messages),
        )

    def add_artifact(self, artifact: Artifact, *, append: bool) -> None:
        """Merge a streamed ``artifact_update`` into :attr:`artifacts`.

        Honors the A2A ``append`` flag exactly as the protocol library's
        server-side artifact assembler does: a chunk with ``append=False``
        replaces the same-id artifact (or is added when the id is unseen);
        ``append=True`` concatenates its parts onto the same-id artifact.
        Without this, every chunk of a multi-chunk artifact would land as a
        separate list entry and :func:`converters.extract_text_from_task`
        — which reads only the last artifact — would return just the final
        chunk, truncating the answer.

        The concat builds a **fresh** :class:`Artifact` (protobuf copies its
        parts at constructor time), so no SDK-vended sub-message is mutated
        in place — preserving the accumulator's freshness contract.

        Args:
            artifact: The artifact carried by the ``artifact_update`` chunk.
            append: The chunk's ``append`` flag — ``True`` concatenates
                onto the same-id artifact, ``False`` replaces it (or adds a
                new one when the id is unseen).
        """
        index = self._artifact_index(artifact.artifact_id)
        if not append:
            if index is not None:
                self.artifacts[index] = artifact
            else:
                self.artifacts.append(artifact)
            return
        if index is None:
            # append=True with no prior same-id artifact is a server
            # protocol quirk; keep the content (add it) rather than drop it
            # or raise, so a lenient client never loses text.
            self.artifacts.append(artifact)
            return
        existing = self.artifacts[index]
        self.artifacts[index] = Artifact(
            artifact_id=existing.artifact_id,
            name=existing.name,
            description=existing.description,
            parts=list(existing.parts) + list(artifact.parts),
            metadata=existing.metadata,
            extensions=list(existing.extensions),
        )

    def _artifact_index(self, artifact_id: str) -> int | None:
        """Return the index of the same-id artifact in :attr:`artifacts`, or ``None``."""
        for i, existing in enumerate(self.artifacts):
            if existing.artifact_id == artifact_id:
                return i
        return None


class A2AClient:
    """Framework-typed client for a single remote A2A endpoint.

    All public methods are async. The client manages an internal
    :class:`httpx.AsyncClient` unless one is injected via the
    constructor; in either case, calling :meth:`close` (or using the
    instance as an async context manager) releases the underlying
    transport.

    .. note::
       **Streaming wall-clock exposure**. ``max_stream_chunks`` and
       ``max_stream_bytes`` are volume bounds, not time bounds. The
       httpx per-read timeout (default ``timeout=30s``) only governs
       the gap *between* chunks — a server that slow-drips one
       chunk every 29s can keep the connection alive for
       ``max_stream_chunks * 29s`` (~80 hours at the default cap)
       before the chunk cap fires. Operators concerned with
       adversarial servers MUST wrap streaming calls in
       :func:`asyncio.wait_for` with an explicit deadline.
    """

    def __init__(
        self,
        *,
        url: str,
        agent_card: AgentCard | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        interceptors: list[ClientCallInterceptor] | None = None,
        client_config: ClientConfig | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        max_stream_chunks: int = DEFAULT_MAX_STREAM_CHUNKS,
        max_stream_bytes: int = DEFAULT_MAX_STREAM_BYTES,
    ) -> None:
        """Construct an :class:`A2AClient` for the given endpoint URL.

        Args:
            url: Base URL of the remote A2A server (e.g.
                ``"https://research.example.com"``).
            agent_card: Optional pre-fetched ``AgentCard``. When
                ``None``, the card is fetched lazily on first call
                via :class:`A2ACardResolver`.
            timeout: Per-call HTTP timeout in seconds. Default 30s.
                Applied via the auto-constructed
                :class:`httpx.AsyncClient` when ``client_config`` does
                not carry one; otherwise the caller's
                ``client_config.httpx_client`` governs.
            interceptors: A list of :class:`ClientCallInterceptor`
                instances (e.g. :class:`a2a.client.AuthInterceptor`)
                inserted into every request. Pure pass-through to the
                protocol client — no new auth surface in this ADK.
            client_config: Optional :class:`a2a.client.ClientConfig`
                for full pass-through to the underlying protocol
                library. Use this for wire-transport selection
                (``supported_protocol_bindings``), connection-pool
                sharing (``httpx_client``), and every other knob the
                a2a SDK exposes. ``client_config.httpx_client`` is the
                **only** way to inject a shared
                :class:`httpx.AsyncClient` — when present, the
                framework does NOT take ownership of it (the caller
                closes it). When absent, this client auto-constructs
                an httpx client and owns its lifecycle.
            poll_interval: Seconds between polls when waiting on a
                background task to reach a terminal state via
                :meth:`poll_task`. Default 1.0s.
            max_stream_chunks: Maximum streaming chunks before raising
                :class:`A2AProtocolError`. Default 10 000.
            max_stream_bytes: Maximum total accumulated streamed bytes
                before raising :class:`A2AProtocolError`. Default 8 MiB.

        Raises:
            ValueError: If ``url`` is empty.
        """
        if len(url) == 0:
            raise ValueError("A2AClient.url MUST be a non-empty URL.")
        self._url = url
        self._agent_card = agent_card
        self._timeout = timeout
        self._interceptors = list(interceptors) if interceptors is not None else []
        # Caller-supplied protocol-library config (streaming /
        # polling / grpc / output-modes / etc). ``None`` means
        # "build a default at first use" — see :meth:`_ensure_client`.
        self._client_config = client_config
        # Framework-side bounds on the streaming hot path; not part
        # of the protocol library's ``ClientConfig``.
        self._poll_interval = poll_interval
        self._max_stream_chunks = max_stream_chunks
        self._max_stream_bytes = max_stream_bytes
        # Single httpx-client surface: caller-supplied via
        # ``client_config.httpx_client`` OR auto-constructed here.
        # The flag tracks ownership so :meth:`close` only tears down
        # what we built.
        injected = client_config.httpx_client if client_config is not None else None
        self._owns_http_client = injected is None
        self._http_client: httpx.AsyncClient | _Closed = (
            injected if injected is not None else httpx.AsyncClient(timeout=timeout)
        )
        self._client: Client | None = None
        self._client_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> A2AClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        del exc
        await self.close()

    async def close(self) -> None:
        """Release the underlying transport.

        Safe to call multiple times — second and subsequent calls are
        no-ops. Closes the internal :class:`httpx.AsyncClient` only if
        the client was constructed without an externally-supplied
        instance.
        """
        if isinstance(self._http_client, _Closed):
            return
        if self._client is not None:
            # The a2a.client transports close by calling
            # ``httpx_client.aclose()``. When the caller injected their own
            # httpx client via ``client_config.httpx_client``, driving the
            # SDK close here would tear down a transport we do not own. Only
            # close the SDK client when we own the underlying httpx client;
            # otherwise just drop the reference and leave the injected
            # client for its owner to close.
            if self._owns_http_client:
                await self._client.close()
            self._client = None
        if self._owns_http_client:
            await self._http_client.aclose()
        self._http_client = _CLOSED

    # ------------------------------------------------------------------
    # Card discovery
    # ------------------------------------------------------------------

    async def fetch_agent_card(self) -> AgentCard:
        """Fetch (and cache) the remote ``AgentCard`` via the well-known URL.

        Subsequent calls return the cached card without a network round
        trip.

        Returns:
            The fetched (or previously cached) :class:`a2a.types.AgentCard`.

        Raises:
            A2AProtocolError: On any failure to parse or validate the
                card.
            A2ATransportError: On network failure fetching the card.
        """
        if self._agent_card is not None:
            return self._agent_card
        http = self._require_http()
        resolver = A2ACardResolver(httpx_client=http, base_url=self._url)
        try:
            card = await resolver.get_agent_card()
        except AgentCardResolutionError as exc:
            raise A2AProtocolError(f"Failed to resolve AgentCard at {self._url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise A2ATransportError(f"Transport error fetching AgentCard from {self._url}: {exc}") from exc
        self._agent_card = card
        return card

    async def _ensure_client(self) -> Client:
        """Lazily construct the underlying protocol :class:`Client`.

        Double-checked locking: the outer ``self._client is not None``
        avoids the lock on the hot path; the inner re-check inside
        the lock guards against another coroutine winning the race
        between our outer check and the lock acquisition. Both
        checks are real — Pyright's flow analysis cannot prove
        ``self._client`` is unchanged across the ``await`` on the
        lock acquisition (and indeed, the contract is that it
        MAY change in that window).

        Returns:
            The lazily-constructed protocol :class:`a2a.client.Client`.

        Raises:
            A2AProtocolError: If the :class:`a2a.client.ClientFactory`
                fails to construct the client.
        """
        if self._client is not None:
            return self._client
        async with self._client_lock:
            # Race re-check: another coroutine may have completed
            # construction between our outer check and the lock
            # acquisition above. Both static checkers narrow
            # ``self._client`` to ``None`` from the outer check
            # because they cannot model the cross-coroutine update.
            # The ignore is load-bearing: the runtime invariant is
            # that the lock acquisition is the only synchronisation
            # point, so a re-read here MAY observe a non-None value
            # set by a racing coroutine while we awaited the lock.
            inflight: Client | None = self._client  # type: ignore[unreachable]
            if inflight is not None:
                return inflight
            card = await self.fetch_agent_card()
            http = self._require_http()
            # Use the caller-supplied protocol-library ClientConfig
            # if any; fall back to a minimal default with our httpx
            # client wired in. The ``a2a.client.ClientConfig``
            # governs protocol behaviour (streaming flag, transport
            # bindings, output modes, push-notification config, ...);
            # we don't shadow it.
            if self._client_config is not None:
                config = self._client_config
                # If the caller provided a config without an httpx
                # client, splice ours in so card resolution and the
                # client share one connection pool.
                if config.httpx_client is None:
                    config = dataclasses.replace(config, httpx_client=http)
            else:
                config = ClientConfig(httpx_client=http)
            factory = ClientFactory(config=config)
            try:
                client = factory.create(card=card, interceptors=self._interceptors)
            except A2AClientError as exc:
                raise A2AProtocolError(f"Failed to construct A2A client for {self._url}: {exc}") from exc
            self._client = client
            return client

    def _require_http(self) -> httpx.AsyncClient:
        """Return the live :class:`httpx.AsyncClient` or raise if closed.

        Returns:
            The active :class:`httpx.AsyncClient`.

        Raises:
            RuntimeError: The client has already been closed.
        """
        if isinstance(self._http_client, _Closed):
            raise RuntimeError("A2AClient has been closed; create a new instance to make further calls.")
        return self._http_client

    # ------------------------------------------------------------------
    # Public call surface
    # ------------------------------------------------------------------

    async def send_message(
        self,
        prompt: str,
        *,
        context_id: str | None = None,
        continuation_token: A2AContinuationToken | None = None,
        accepted_output_modes: list[str] | None = None,
    ) -> A2ARunResult:
        """Send a prompt and block until the task reaches a terminal state.

        Maps non-completed terminal states to typed exceptions:
        ``failed`` / ``rejected`` -> :class:`A2ATaskError`,
        ``cancelled`` -> :class:`A2ATaskCancelledError`.

        Args:
            prompt: User-facing prompt text. Multi-modal inputs are
                not yet supported on this surface — open a follow-up
                if you need a parts-accepting overload.
            context_id: When continuing a multi-turn conversation, the
                ``context_id`` returned by the prior task. Otherwise
                ``None`` and the server creates a fresh context.
            continuation_token: When resuming a task that previously
                paused (e.g. ``input_required``), pass the token from
                the prior call.
            accepted_output_modes: Optional content-type filter
                (``["text/plain"]``, ``["application/json"]``, ...).

        Returns:
            An :class:`A2ARunResult` carrying the final text, task id,
            and context id.

        Raises:
            A2ATaskInterruptedError: Task paused at
                ``input_required`` or ``auth_required``.
            A2ATaskCancelledError: Task was cancelled before
                completion.
            A2ATaskError: Task reached ``failed`` or ``rejected``
                terminal state.
            A2ATransportError: Network failure before or during the
                call.
            A2AProtocolError: Protocol-level failure from the remote
                endpoint.
        """
        from troopai.adk.a2a.a2a_agent import A2ARunResult  # avoid cycle

        client = await self._ensure_client()
        message = converters.build_user_message(
            prompt,
            context_id=context_id if continuation_token is None else continuation_token.context_id,
            task_id=continuation_token.task_id if continuation_token is not None else None,
        )
        request = converters.build_send_request(
            message,
            return_immediately=False,
            accepted_output_modes=accepted_output_modes,
        )
        with function_span(
            name=f"client.send_message:{self._url}",
            input=prompt,
            a2a_data={"remote_url": self._url, "context_id": message.context_id},
        ) as span:
            terminal_task = await self._consume_until_terminal(client.send_message(request))
            # Order matters: interrupt states (input_required /
            # auth_required) are non-terminal but still exit the
            # consume loop; raise them as a typed exception before
            # checking failure terminal states.
            self._raise_on_interrupt(terminal_task)
            self._raise_on_failure(terminal_task)
            text = converters.extract_text_from_task(terminal_task)
            span.data = dataclasses.replace(span.data, output=text)
            return A2ARunResult(
                text=text,
                task_id=terminal_task.id,
                context_id=terminal_task.context_id,
            )

    async def stream_message(
        self,
        prompt: str,
        *,
        context_id: str | None = None,
    ) -> AsyncIterator[A2AStreamEvent]:
        """Stream a prompt response chunk-by-chunk.

        Yields :class:`A2AStreamEvent` TypedDicts terminating in a
        ``"completed"`` or ``"failed"`` event. Bounded by
        ``max_stream_chunks`` / ``max_stream_bytes``; callers
        concerned with slow-drip adversaries should wrap in
        :func:`asyncio.wait_for` (see class docstring's wall-clock
        note).

        Args:
            prompt: User-facing prompt text.
            context_id: When continuing a multi-turn conversation, the
                ``context_id`` returned by the prior task. Otherwise
                ``None`` and the server creates a fresh context.

        Returns:
            An async iterator of :class:`A2AStreamEvent` TypedDicts,
            terminating with a ``"completed"`` or ``"failed"`` event.

        Raises:
            A2ATaskInterruptedError: Non-terminal interrupt state
                (``input_required`` / ``auth_required``) encountered
                in the stream.
            A2ATransportError: Network failure during streaming.
            A2AProtocolError: Volume cap exceeded.
        """
        client = await self._ensure_client()
        message = converters.build_user_message(prompt, context_id=context_id)
        # ``return_immediately=False`` keeps the SSE connection open
        # so the server streams events through to terminal state.
        request = converters.build_send_request(message, return_immediately=False)

        # Manage the span lifecycle manually (start/finish in try/finally)
        # rather than via `with function_span(...) as span:`.  An async
        # generator's `with` block is cleaned up when `aclose()` is called,
        # which may happen from a different asyncio context than the one that
        # created the ContextVar token.  `ContextVar.reset(token)` raises
        # `ValueError` in that case, leaving `_current_span` pointing at the
        # stale span for every subsequent span created in the caller's task.
        # The explicit `try/finally` with a guarded `finish()` call ensures
        # cleanup always runs, and a `ValueError` from cross-context reset is
        # caught so it does not propagate as an unhandled exception to the
        # caller.
        span = function_span(
            name=f"client.stream_message:{self._url}",
            input=prompt,
            a2a_data={"remote_url": self._url, "context_id": message.context_id},
        )
        span.start()
        accumulated: list[str] = []
        task_id_acc: list[str] = [""]
        saw_terminal = False
        try:
            try:
                async for event in self._iter_stream_events(
                    client.send_message(request),
                    accumulated,
                    message.context_id,
                    task_id_acc,
                ):
                    yield event
                    if event["type"] == "completed" or event["type"] == "failed":
                        saw_terminal = True
                        break
            except A2ATaskInterruptedError:
                # Already a typed framework exception — re-raise as-is.
                # (Raised inside _iter_stream_events on interrupt states.)
                raise
            except A2AProtocolError:
                # Already a typed framework exception — re-raise as-is.
                # (Raised by _iter_stream_events volume-cap guards.)
                raise
            except A2AClientTimeoutError as exc:
                raise A2ATransportError(f"Streaming timed out from {self._url}: {exc}") from exc
            except A2AClientError as exc:
                raise A2AProtocolError(f"Streaming protocol failure from {self._url}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise A2ATransportError(f"Streaming transport failure: {exc}") from exc
            except ValueError as exc:
                # An unmapped TaskState makes stream_response_to_event raise a
                # raw ValueError; wrap it typed so the stream contract holds
                # (mirrors the blocking _consume_until_terminal path).
                raise A2AProtocolError(
                    f"Streaming received an unrecognised task state from {self._url}: {exc}"
                ) from exc
            span.data = dataclasses.replace(span.data, output="".join(accumulated))
            if not saw_terminal:
                # Stream closed without a terminal event — the server
                # crashed, was cancelled, or CANCELED the connection. Emitting
                # a 'completed' here is actively deceptive; emit 'failed'
                # instead so callers and monitoring can distinguish a clean
                # finish from a premature close.
                logger.warning(
                    "A2A stream from %s closed without a terminal event; emitting 'failed' event.",
                    self._url,
                )
                final: A2AStreamEvent = {
                    "type": "failed",
                    "state": "failed",
                    "message": "Stream closed without terminal event",
                }
                yield final
        finally:
            try:
                span.finish()
            except ValueError:
                # ContextVar.reset() raises ValueError when called from a
                # different context than the one that created the token.  This
                # can happen when the caller abandons the generator (e.g.
                # `break` in `async for`) and Python's asyncio calls aclose()
                # during GC or loop shutdown from a different task context.
                # Suppress and log so the stale span is still marked finished.
                logger.debug(
                    "stream_message: span.finish() raised ValueError (cross-context reset); "
                    "span %r marked finished but ContextVar not reset.",
                    span.data.name,
                )

    async def _iter_stream_events(
        self,
        stream: AsyncIterator[StreamResponse],
        accumulated: list[str],
        context_id: str,
        task_id_acc: list[str],
    ) -> AsyncIterator[A2AStreamEvent]:
        """Generator: walk the SDK stream, enforce bounds, raise interrupts.

        Splits :meth:`stream_message` so the public entry stays
        under the 60-line function limit. ``accumulated`` is mutated
        in place to collect text deltas for the final-event
        fallback in the caller. ``task_id_acc`` is mutated in place
        to capture the first non-empty ``task_id`` seen on the stream
        so that :class:`A2ATaskInterruptedError` carries the real
        task identifier rather than an empty string.

        Args:
            stream: The raw :class:`a2a.types.StreamResponse` iterator
                from the protocol client.
            accumulated: Mutable list collecting text-delta strings
                for the final ``"completed"`` event fallback.
            context_id: The conversation context identifier, surfaced
                on :class:`A2ATaskInterruptedError` when an interrupt
                state is detected.
            task_id_acc: Single-element mutable list; populated with
                the first non-empty task identifier seen on the stream
                so the interrupt raise can carry it. Pass
                ``[""]`` — the list is mutated in place.

        Raises:
            A2AProtocolError: Volume cap (``max_stream_chunks`` or
                ``max_stream_bytes``) exceeded.
            A2ATaskInterruptedError: Interrupt state
                (``input_required`` / ``auth_required``) encountered.
        """
        accumulated_bytes = 0
        chunks_seen = 0
        async for chunk in stream:
            chunks_seen += 1
            if chunks_seen > self._max_stream_chunks:
                raise A2AProtocolError(
                    f"Streaming response exceeded max_stream_chunks={self._max_stream_chunks} from {self._url}"
                )
            # Accumulate task_id from status_update and task chunks so
            # A2ATaskInterruptedError carries the real task identifier.
            kind = converters.stream_response_kind(chunk)
            if len(task_id_acc[0]) == 0:
                if kind == "status_update" and len(chunk.status_update.task_id) > 0:
                    task_id_acc[0] = chunk.status_update.task_id
                elif kind == "task" and len(chunk.task.id) > 0:
                    task_id_acc[0] = chunk.task.id
            event = stream_response_to_event(chunk, accumulated)
            if event is None:
                continue
            if "text_delta" in event:
                accumulated_bytes += len(event["text_delta"].encode("utf-8"))
                if accumulated_bytes > self._max_stream_bytes:
                    raise A2AProtocolError(
                        f"Streaming response exceeded max_stream_bytes={self._max_stream_bytes} from {self._url}"
                    )
            # Interrupt detection mirrors the blocking-path
            # behaviour. Without this, an input_required /
            # auth_required would yield a "status" event but the
            # caller loop would keep iterating until the chunk
            # cap fired with a misleading message.
            state = event.get("state")
            if event["type"] == "status" and state is not None and converters.is_interrupt_state(state):
                raise A2ATaskInterruptedError(
                    task_id=task_id_acc[0],
                    context_id=context_id,
                    state=state,
                    prompt=event.get("message", ""),
                )
            yield event

    async def submit_background(
        self,
        prompt: str,
        *,
        context_id: str | None = None,
    ) -> A2AContinuationToken:
        """Submit a task with non-blocking semantics; return the token immediately.

        The server is asked to return as soon as a ``task_id`` /
        ``context_id`` pair has been issued (typically right after
        the first ``status_update``). Use :meth:`poll_task` or
        ``A2ARunner.arun(agent, prompt, continuation_token=token)``
        to resume.

        Args:
            prompt: User-facing prompt text.
            context_id: Optional conversation context identifier to
                continue an existing multi-turn conversation.

        Returns:
            An :class:`A2AContinuationToken` carrying the
            ``task_id``, ``context_id``, and ``remote_url`` needed
            to resume or poll the task from any process.

        Raises:
            A2ATransportError: Network failure submitting the task.
            A2AProtocolError: Server closed the stream before issuing
                a task identifier.
        """
        client = await self._ensure_client()
        message = converters.build_user_message(prompt, context_id=context_id)
        request = converters.build_send_request(message, return_immediately=True)
        with function_span(
            name=f"client.submit_background:{self._url}",
            input=prompt,
            a2a_data={"remote_url": self._url, "context_id": message.context_id},
        ) as span:
            task_id, ctx_id = await self._read_first_identifiers(client.send_message(request))
            span.data = dataclasses.replace(
                span.data,
                output=f"task_id={task_id} context_id={ctx_id}",
            )
            return A2AContinuationToken(
                task_id=task_id,
                context_id=ctx_id,
                remote_url=self._url,
            )

    async def poll_task(self, token: A2AContinuationToken) -> A2ATaskStatus:
        """One-shot status snapshot for a previously-submitted task.

        Does NOT wait for the task to reach a terminal state — returns
        whatever state the remote currently holds. To wait, call this
        in a polling loop bounded by your own timeout / retry budget.

        Args:
            token: The :class:`A2AContinuationToken` returned by a
                prior :meth:`submit_background` call.

        Returns:
            An :class:`A2ATaskStatus` snapshot of the task's current
            state.

        Raises:
            A2ATransportError: Network failure polling the task.
            A2AProtocolError: Server-side protocol violation.
        """
        client = await self._ensure_client()
        request = converters.build_get_task_request(token.task_id)
        with function_span(
            name=f"client.poll_task:{self._url}",
            input=token.task_id,
            a2a_data={"remote_url": self._url, "task_id": token.task_id, "context_id": token.context_id},
        ) as span:
            try:
                task = await client.get_task(request)
                status = converters.task_to_status(task)
            except A2AClientTimeoutError as exc:
                raise A2ATransportError(f"poll_task timed out for task {token.task_id}: {exc}") from exc
            except A2AClientError as exc:
                raise A2AProtocolError(f"poll_task failed for task {token.task_id}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise A2ATransportError(f"Transport failure polling task {token.task_id}: {exc}") from exc
            except ValueError as exc:
                raise A2AProtocolError(
                    f"poll_task received an unrecognised task state for task {token.task_id}: {exc}"
                ) from exc
            span.data = dataclasses.replace(span.data, output=status.state)
            return status

    async def cancel_task(self, token: A2AContinuationToken) -> None:
        """Request the remote agent cancel an in-flight task.

        Does not block on cancellation completion — the server returns
        a (potentially still ``working``) ``Task`` snapshot which we
        discard. Caller should :meth:`poll_task` if confirmation
        matters.

        Args:
            token: The :class:`A2AContinuationToken` returned by a
                prior :meth:`submit_background` call.

        Raises:
            A2AProtocolError: Server-side protocol violation on the
                cancel request.
            A2ATransportError: Network failure on the cancel request.
        """
        client = await self._ensure_client()
        request = converters.build_cancel_task_request(token.task_id)
        with function_span(
            name=f"client.cancel_task:{self._url}",
            input=token.task_id,
            a2a_data={"remote_url": self._url, "task_id": token.task_id, "context_id": token.context_id},
        ):
            try:
                await client.cancel_task(request)
            except A2AClientError as exc:
                raise A2AProtocolError(f"cancel_task failed for task {token.task_id}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise A2ATransportError(f"Transport failure cancelling task {token.task_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _consume_until_terminal(self, stream: AsyncIterator[StreamResponse]) -> Task:
        """Drive a streaming send_message to a terminal/interrupt Task snapshot.

        Aggregates incremental Task / status_update / artifact_update
        events into a :class:`_StreamAccumulator` until a terminal
        OR interrupt state is observed. Returns a **fresh**
        :class:`Task` constructed from the accumulator — the returned
        object does NOT alias any SDK-vended sub-message the
        streaming iterator may still reference.

        Per-chunk dispatch lives in :meth:`_apply_chunk`; this
        method owns only the loop, the bounds, and the wire-error
        mapping.

        Bounded by ``self._max_stream_chunks`` and
        ``self._max_stream_bytes`` so a runaway remote cannot grow
        memory without bound — it raises :class:`A2AProtocolError`.

        Args:
            stream: The raw :class:`a2a.types.StreamResponse` iterator
                from the protocol client.

        Returns:
            A fresh :class:`a2a.types.Task` built from the accumulated
            stream events.

        Raises:
            A2AProtocolError: Volume cap exceeded or server closed
                the stream without any ``Task`` or status update.
            A2ATransportError: Network failure during streaming.
        """
        acc = _StreamAccumulator()
        chunks_seen = 0
        try:
            async for chunk in stream:
                chunks_seen += 1
                if chunks_seen > self._max_stream_chunks:
                    raise A2AProtocolError(
                        f"Streaming response exceeded max_stream_chunks={self._max_stream_chunks} from {self._url}"
                    )
                if self._apply_chunk(chunk, acc):
                    return acc.to_task()
        except A2AClientTimeoutError as exc:
            raise A2ATransportError(f"send_message timed out for {self._url}: {exc}") from exc
        except A2AClientError as exc:
            raise A2AProtocolError(f"send_message protocol failure for {self._url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise A2ATransportError(f"send_message transport failure for {self._url}: {exc}") from exc
        except ValueError as exc:
            # A chunk carrying an unmapped TaskState makes
            # converters.task_state_to_literal raise a raw ValueError from
            # inside _apply_chunk. Wrap it in the typed hierarchy so callers
            # branching on A2AError see it, mirroring poll_task.
            raise A2AProtocolError(f"send_message received an unrecognised task state from {self._url}: {exc}") from exc
        if acc.status is None:
            if len(acc.messages) > 0:
                # A server may answer with only free-standing Message chunks
                # and never emit a Task or status_update (a bare "message
                # reply"). The stream ended cleanly with content, so treat it
                # as a successful completion assembled from the accumulated
                # messages rather than a protocol error.
                acc.status = TaskStatus(state=TaskState.TASK_STATE_COMPLETED)
                return acc.to_task()
            raise A2AProtocolError(
                f"Server at {self._url} closed the stream without emitting any Task or status update."
            )
        # The accumulator loop only returns early on a terminal OR
        # interrupt state. Reaching here means the SDK iterator ended
        # cleanly (no exception) while the last-seen state was still
        # non-terminal (e.g. ``working`` / ``submitted``) — a premature
        # close (server crash/restart, proxy idle-close, partial
        # response after a clean EOF). Returning that snapshot would let
        # send_message report empty text as a success; surface it as a
        # protocol error instead, mirroring the streaming path.
        state_literal = converters.task_state_to_literal(acc.status.state)
        if not converters.is_terminal_state(state_literal) and not converters.is_interrupt_state(state_literal):
            logger.warning(
                "A2A stream from %s closed without a terminal event (last state: %s); treating as failure.",
                self._url,
                state_literal,
            )
            raise A2AProtocolError(
                f"Server at {self._url} closed the stream before reaching a terminal state (last state: {state_literal})."
            )
        return acc.to_task()

    def _apply_chunk(
        self,
        chunk: StreamResponse,
        acc: _StreamAccumulator,
    ) -> bool:
        """Apply one ``StreamResponse`` to the accumulator.

        Mutates ``acc`` in place; does NOT mutate any SDK-vended
        sub-message (no ``CopyFrom``, no ``artifacts.append`` on a
        Task held by the iterator). The accumulator slots
        (``acc.status``, items of ``acc.artifacts``) hold shared
        references to SDK-vended sub-messages until
        :meth:`_StreamAccumulator.to_task` performs the protobuf
        constructor copy — so the SDK iterator MUST NOT mutate
        sub-messages it has already yielded. The current ``a2a-sdk``
        contract honours this.

        Args:
            chunk: A single :class:`a2a.types.StreamResponse` from
                the protocol client iterator.
            acc: The mutable :class:`_StreamAccumulator` to update.

        Returns:
            ``True`` iff this chunk drove the accumulator into a
            terminal or interrupt state — caller's loop stops and
            constructs the fresh :class:`Task` via
            :meth:`_StreamAccumulator.to_task`.

        Raises:
            A2AProtocolError: The ``max_stream_bytes`` cap was
                exceeded by an artifact-update or message chunk.
        """
        kind = converters.stream_response_kind(chunk)
        if kind == "task":
            t = chunk.task
            acc.task_id = t.id
            acc.context_id = t.context_id
            acc.status = t.status
            # A Task chunk is a snapshot — replace, not extend.
            acc.artifacts = list(t.artifacts)
            state_literal = converters.task_state_to_literal(t.status.state)
            return converters.is_terminal_state(state_literal) or converters.is_interrupt_state(state_literal)
        if kind == "status_update":
            update = chunk.status_update
            update_state = converters.task_state_to_literal(update.status.state)
            is_done = converters.is_terminal_state(update_state) or converters.is_interrupt_state(update_state)
            first_signal = acc.status is None
            if len(acc.task_id) == 0:
                acc.task_id = update.task_id
                acc.context_id = update.context_id
            acc.status = update.status
            if is_done and first_signal:
                logger.warning(
                    "A2A server at %s emitted terminal status %s without a prior Task or artifact;"
                    " synthesising a Task from the status_update for the caller.",
                    self._url,
                    update_state,
                )
            return is_done
        if kind == "artifact_update":
            art_update = chunk.artifact_update
            artifact = art_update.artifact
            acc.accumulated_bytes += sum(
                len(p.text.encode("utf-8")) for p in artifact.parts if p.WhichOneof("content") == "text"
            )
            if acc.accumulated_bytes > self._max_stream_bytes:
                raise A2AProtocolError(
                    f"Streaming response exceeded max_stream_bytes={self._max_stream_bytes} from {self._url}"
                )
            # Honor the append flag: concat same-id chunks when append=True,
            # replace (or add) when append=False. Blindly appending every
            # chunk as a separate artifact truncated multi-chunk answers,
            # since extract_text_from_task reads only the last artifact.
            acc.add_artifact(artifact, append=art_update.append)
        elif kind == "message":
            # A server may deliver its answer as a free-standing message
            # rather than artifact deltas. Accumulate it so to_task()
            # surfaces it as Task history and extract_text_from_task can
            # fall back to it — matching the streaming path, which emits
            # the message text as a delta. Count its text toward the byte
            # cap so the message route cannot bypass the volume bound.
            message = chunk.message
            acc.accumulated_bytes += sum(
                len(p.text.encode("utf-8")) for p in message.parts if p.WhichOneof("content") == "text"
            )
            if acc.accumulated_bytes > self._max_stream_bytes:
                raise A2AProtocolError(
                    f"Streaming response exceeded max_stream_bytes={self._max_stream_bytes} from {self._url}"
                )
            acc.messages.append(message)
        return False

    async def _read_first_identifiers(self, stream: AsyncIterator[StreamResponse]) -> tuple[str, str]:
        """Pull the first ``task_id`` / ``context_id`` pair off the stream.

        Used by :meth:`submit_background` to surface the identifiers
        immediately without waiting for terminal state.

        Args:
            stream: The raw :class:`a2a.types.StreamResponse` iterator
                from the protocol client.

        Returns:
            A ``(task_id, context_id)`` tuple from the first chunk
            that carries non-empty identifiers.

        Raises:
            A2ATransportError: Network failure while reading from the
                stream.
            A2AProtocolError: Server closed the stream before issuing
                a task identifier.
        """
        try:
            async for chunk in stream:
                kind = converters.stream_response_kind(chunk)
                if kind == "task" and len(chunk.task.id) > 0:
                    return chunk.task.id, chunk.task.context_id
                if kind == "status_update" and len(chunk.status_update.task_id) > 0:
                    return chunk.status_update.task_id, chunk.status_update.context_id
        except A2AClientTimeoutError as exc:
            raise A2ATransportError(f"submit_background timed out for {self._url}: {exc}") from exc
        except A2AClientError as exc:
            raise A2AProtocolError(f"submit_background protocol failure for {self._url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise A2ATransportError(f"submit_background transport failure for {self._url}: {exc}") from exc
        raise A2AProtocolError(f"Server at {self._url} closed the stream before issuing a task_id.")

    @staticmethod
    def _raise_on_interrupt(task: Task) -> None:
        """Raise :class:`A2ATaskInterruptedError` if the task is paused.

        ``input_required`` and ``auth_required`` are non-terminal
        states the server uses to ask the caller for more input or
        authentication. Surface them as a typed exception so callers
        can branch distinctly from terminal failures.
        """
        state_literal = converters.task_state_to_literal(task.status.state)
        if not converters.is_interrupt_state(state_literal):
            return
        prompt_text = ""
        if task.status.message is not None and len(task.status.message.parts) > 0:
            prompt_text = converters.extract_text_from_message(task.status.message)
        raise A2ATaskInterruptedError(
            task_id=task.id,
            context_id=task.context_id,
            state=state_literal,
            prompt=prompt_text,
        )

    @staticmethod
    def _raise_on_failure(task: Task) -> None:
        """Map terminal failure states to typed exceptions."""
        state_literal = converters.task_state_to_literal(task.status.state)
        if state_literal == "completed":
            return
        if not converters.is_failure_state(state_literal):
            return
        message = ""
        if task.status.message is not None and len(task.status.message.parts) > 0:
            message = converters.extract_text_from_message(task.status.message)
        if state_literal == "cancelled":
            raise A2ATaskCancelledError(
                task_id=task.id,
                context_id=task.context_id,
                state=state_literal,
                remote_message=message,
            )
        raise A2ATaskError(
            task_id=task.id,
            context_id=task.context_id,
            state=state_literal,
            remote_message=message,
        )
