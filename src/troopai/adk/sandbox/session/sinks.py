"""Audit-event sinks for sandbox sessions.

Backends emit ``SandboxSessionEvent`` instances; sinks consume
them. The sink hierarchy is pluggable so an operator can compose
a callback for in-process observability + a JSONL file for
long-term forensics + a custom HTTP exporter — without touching
the backend.

Concrete sinks shipped here:

* ``CallbackSink`` — fan events out to a developer-supplied
  ``Callable[[SandboxSessionEvent], object]``. Sync or async.
* ``JsonlOutboxSink`` — append one JSONL line per event to a host
  filesystem path. POSIX-locked when the platform supports it.
* ``HttpPostSink`` — POST each event to a URL as JSON.
* ``ChainedSink`` — compose multiple sinks behind a single
  ``handle()``; sub-sinks are awaited sequentially with their
  own per-sink ``on_error`` policy controlling whether a failure
  aborts the chain.

Per-sink ``on_error`` controls failure propagation:

* ``"raise"`` — failure aborts the handle call (default for sinks
  feeding business logic).
* ``"log"`` — log the failure at WARNING and continue.
* ``"ignore"`` — drop silently (rare; only when an outbox is
  truly best-effort).
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import errno
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType
from typing import Literal, override
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from troopai.adk.sandbox.session.events import EventPayloadPolicy, SandboxSessionEvent
from troopai.adk.sandbox.session.utils import event_to_json_line

logger = logging.getLogger(__name__)

__all__ = [
    "NEVER_DOWNGRADE_EXC",
    "CallbackSink",
    "ChainedSink",
    "DeliveryMode",
    "EventSink",
    "HttpPostSink",
    "JsonlOutboxSink",
    "OnErrorPolicy",
]

DeliveryMode = Literal["sync", "async", "best_effort"]
"""How the host backend invokes the sink — ``sync`` blocks the calling op,
``async`` schedules and continues, ``best_effort`` is fire-and-forget."""

OnErrorPolicy = Literal["raise", "log", "ignore"]
"""What to do when a sink's handler raises."""

NEVER_DOWNGRADE_EXC: tuple[type[BaseException], ...] = (
    MemoryError,
    RecursionError,
    SystemError,
    NameError,
    TypeError,
    AttributeError,
)
"""Programmer-bug / interpreter-fault exceptions that ``on_error="log"``
or ``"ignore"`` MUST NEVER downgrade. A sink raising one of these has a
real defect the operator must see — not a transient transport hiccup
the delivery layer should hide. Shared by ``ChainedSink`` AND
``Instrumentation`` so both the chain and the wider event-delivery
layer enforce the same contract (a chain decomposed by the
instrumentation layer must not lose this guarantee)."""


class EventSink(abc.ABC):
    """Abstract base — concrete sinks override ``handle``.

    Attributes:
        name: Optional descriptive label surfaced in logs.
        mode: Delivery semantics the host should honor.
        on_error: Failure-propagation policy.
        payload_policy: Per-sink override of the session's
            ``EventPayloadPolicy``; ``None`` defers to the session.
    """

    name: str | None = None
    mode: DeliveryMode = "sync"
    on_error: OnErrorPolicy = "raise"
    payload_policy: EventPayloadPolicy | None = None

    @abc.abstractmethod
    async def handle(self, event: SandboxSessionEvent) -> None:
        """Consume one event; may raise per ``on_error`` policy."""


class CallbackSink(EventSink):
    """Deliver events to a developer-supplied callable.

    The callable may be sync or async — async returns are awaited.
    The session reference is NOT bound on this sink (unlike the
    upstream design) — keeping it stateless lets the same
    ``CallbackSink`` serve every concurrent session in a multi-
    tenant process without leaking one session's reference into
    another's events.
    """

    def __init__(
        self,
        callback: Callable[[SandboxSessionEvent], object | Awaitable[object]],
        *,
        mode: DeliveryMode = "sync",
        on_error: OnErrorPolicy = "raise",
        payload_policy: EventPayloadPolicy | None = None,
        name: str | None = None,
    ) -> None:
        self._callback = callback
        self.mode = mode
        self.on_error = on_error
        self.payload_policy = payload_policy
        self.name = name

    @override
    async def handle(self, event: SandboxSessionEvent) -> None:
        result = self._callback(event)
        # ``inspect.isawaitable`` covers the full awaitable protocol (custom
        # __await__, Task, Future) — not just native ``async def``
        # coroutines, which is what ``asyncio.iscoroutine`` recognizes. The
        # ``CallbackSink`` callback signature explicitly advertises
        # ``Awaitable[object]`` so the wider check is the correct one.
        if inspect.isawaitable(result):
            await result


class JsonlOutboxSink(EventSink):
    """Append events to a host-filesystem JSONL file.

    Uses an ``asyncio.Lock`` to serialize writes from the same
    sink instance + ``fcntl.flock`` (when available) so cross-
    process appends to the same file remain line-atomic.
    """

    def __init__(
        self,
        path: Path,
        *,
        mode: DeliveryMode = "best_effort",
        on_error: OnErrorPolicy = "log",
        payload_policy: EventPayloadPolicy | None = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self.mode = mode
        self.on_error = on_error
        self.payload_policy = payload_policy
        self.name = name
        self._write_lock = asyncio.Lock()

    @override
    async def handle(self, event: SandboxSessionEvent) -> None:
        line = event_to_json_line(event)
        async with self._write_lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(
                "JsonlOutboxSink: mkdir(%s) failed (errno=%s): %s",
                self.path.parent,
                exc.errno,
                exc,
            )
            raise

        fcntl_mod: ModuleType | None
        try:
            import fcntl as fcntl_mod
        except ImportError:
            # fcntl is POSIX-only — Windows simply skips file locking
            # and relies on the asyncio.Lock above for in-process safety.
            fcntl_mod = None

        try:
            with self.path.open("a", encoding="utf-8") as handle:
                if fcntl_mod is not None:
                    self._acquire_flock(fcntl_mod, handle.fileno())
                handle.write(line)
                handle.flush()
                if fcntl_mod is not None:
                    # Best-effort unlock; the OS releases on close anyway.
                    with contextlib.suppress(OSError):
                        fcntl_mod.flock(handle.fileno(), fcntl_mod.LOCK_UN)
        except OSError as exc:
            logger.error(
                "JsonlOutboxSink: write to %s failed (errno=%s): %s — audit event lost",
                self.path,
                exc.errno,
                exc,
            )
            raise

    def _acquire_flock(self, fcntl_mod: ModuleType, fd: int) -> None:
        """Acquire ``LOCK_EX``; tolerate "filesystem doesn't support flock" only.

        ``EINVAL`` is the kernel's "this filesystem doesn't support
        advisory locks" signal (common on NFS without lockd, certain
        FUSE mounts, some Docker tmpfs configurations). Every other
        ``OSError`` from ``flock`` (``ENOLCK``, ``EBADF``, ``EINTR``,
        ...) is an operationally meaningful failure — surface them
        so the audit trail's cross-process atomicity guarantee is
        honest about its current state.
        """
        try:
            fcntl_mod.flock(fd, fcntl_mod.LOCK_EX)
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                logger.warning(
                    "JsonlOutboxSink(%s): flock unsupported on this filesystem "
                    "(errno=EINVAL); cross-process atomicity degraded to in-process only",
                    self.path,
                )
                return
            logger.error(
                "JsonlOutboxSink(%s): flock(LOCK_EX) failed (errno=%s); write proceeding WITHOUT cross-process lock",
                self.path,
                exc.errno,
            )


class HttpPostSink(EventSink):
    """POST each event to a URL as JSON.

    Uses stdlib ``urllib`` to avoid pulling ``httpx`` into the
    framework's required-dep surface. The request is dispatched
    off the event loop via ``asyncio.to_thread``; transport
    errors are routed through the sink's ``on_error`` policy.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
        mode: DeliveryMode = "best_effort",
        on_error: OnErrorPolicy = "log",
        payload_policy: EventPayloadPolicy | None = None,
        name: str | None = None,
    ) -> None:
        if len(url) == 0:
            raise ValueError("HttpPostSink: url must be non-empty")
        self.url = url
        self.headers = dict(headers) if headers is not None else {}
        self.timeout_s = timeout_s
        self.mode = mode
        self.on_error = on_error
        self.payload_policy = payload_policy
        self.name = name

    @override
    async def handle(self, event: SandboxSessionEvent) -> None:
        body = event.model_dump_json().encode("utf-8")
        request_headers = {"Content-Type": "application/json", **self.headers}
        await asyncio.to_thread(self._post, body, request_headers)

    def _post(self, body: bytes, headers: dict[str, str]) -> None:
        request = Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                # Drain the body so the connection can be released.
                response.read()
        except HTTPError as exc:
            # HTTPError carries the response body — read up to 2 KiB for
            # diagnostics. ``HTTPError.read()`` can itself fail on closed
            # streams; the inner suppression keeps the wrapped exception
            # informative even when the body is unavailable.
            response_body = "<body unreadable>"
            with contextlib.suppress(Exception):
                response_body = exc.read(2048).decode("utf-8", errors="replace")
            logger.error(
                "HttpPostSink(%s): POST returned %s %s; body=%r",
                self.url,
                exc.code,
                exc.reason,
                response_body,
            )
            raise RuntimeError(
                f"HttpPostSink({self.url}): POST returned {exc.code} {exc.reason}: {response_body!r}"
            ) from exc
        except URLError as exc:
            logger.error("HttpPostSink(%s): connection failed: %r", self.url, exc.reason)
            raise RuntimeError(f"HttpPostSink({self.url}): connection failed: {exc.reason!r}") from exc
        except TimeoutError as exc:
            logger.error("HttpPostSink(%s): timed out after %ss", self.url, self.timeout_s)
            raise RuntimeError(f"HttpPostSink({self.url}): timed out after {self.timeout_s}s") from exc
        # Note: bare OSError intentionally NOT caught — it indicates a
        # process-wide resource problem (EMFILE, etc.), not a transport
        # failure, and should propagate to the caller's framework handler.


class ChainedSink(EventSink):
    """Compose multiple sinks behind a single ``handle()`` call.

    Sinks are invoked in order. Each sub-sink's individual
    ``on_error`` policy governs whether a failure aborts the chain
    or just logs — the chained sink itself doesn't impose policy.
    """

    def __init__(
        self,
        sinks: list[EventSink],
        *,
        mode: DeliveryMode = "sync",
        on_error: OnErrorPolicy = "raise",
        payload_policy: EventPayloadPolicy | None = None,
        name: str | None = None,
    ) -> None:
        self._sinks = list(sinks)
        self.mode = mode
        self.on_error = on_error
        self.payload_policy = payload_policy
        self.name = name

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        """Inner sinks in invocation order — read-only view for introspection."""
        return tuple(self._sinks)

    @override
    async def handle(self, event: SandboxSessionEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.handle(event)
            except NEVER_DOWNGRADE_EXC:
                # Re-raise unconditionally — these are not transport errors.
                raise
            except Exception as exc:
                if sink.on_error == "raise":
                    raise
                sink_label = sink.name or type(sink).__name__
                if sink.on_error == "log":
                    logger.exception(
                        "ChainedSink: sink %r raised %s while handling "
                        "session=%s seq=%s op=%s — continuing per on_error=log",
                        sink_label,
                        type(exc).__name__,
                        event.session_id,
                        event.seq,
                        event.op,
                    )
                else:
                    # on_error == "ignore" — leave a DEBUG breadcrumb so the
                    # drop is traceable in post-mortem analysis.
                    logger.debug(
                        "ChainedSink: sink %r raised %s while handling session=%s seq=%s — dropped per on_error=ignore",
                        sink_label,
                        type(exc).__name__,
                        event.session_id,
                        event.seq,
                    )
