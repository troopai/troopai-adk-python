"""Adapter wrapping a ``bytes`` iterator as an ``io.IOBase`` readable stream.

Backends that stream chunks (Docker ``get_archive``, gRPC
``recv``, HTTP chunked transfer) yield iterators of ``bytes``;
``IteratorIO`` lets the rest of the framework treat those streams
as ordinary readable files without buffering the whole payload
in memory.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterator
from typing import Any, cast, override

logger = logging.getLogger(__name__)

__all__ = ["IteratorIO"]


class IteratorIO(io.IOBase):
    """Wrap an ``Iterator[bytes]`` as a readable, non-seekable ``IOBase``.

    Reads draw from the underlying iterator on demand; chunks are
    buffered only long enough to satisfy the next ``read(size)``
    call. The optional ``on_close`` callback fires exactly once,
    after the iterator is drained or the stream is explicitly
    closed — useful for releasing a backend connection.

    Attributes:
        on_close: Optional callable invoked once at finalization.
    """

    def __init__(
        self,
        iterator: Iterator[bytes],
        *,
        on_close: Callable[[], object] | None = None,
    ) -> None:
        self._it = iterator
        self._on_close = on_close
        self._buffer = bytearray()
        self._closed = False
        self._finalized = False

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        # Guard iterator close so a failing backend doesn't suppress
        # the on_close callback — otherwise a transient backend error
        # leaks the resource the callback was meant to release.
        # cast(Any, ...) documents that getattr on a generic Iterator
        # widens to object; the runtime callable() check below is the
        # real type narrowing the static checker can't see.
        close = cast(Any, getattr(self._it, "close", None))
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning(
                    "IteratorIO: iterator close() raised %s: %s",
                    type(exc).__name__,
                    exc,
                )

        if self._on_close is not None:
            try:
                self._on_close()
            except Exception as exc:
                logger.error(
                    "IteratorIO: on_close callback raised %s: %s",
                    type(exc).__name__,
                    exc,
                )
                raise

    @override
    def readable(self) -> bool:
        return True

    @override
    def read(self, size: int = -1) -> bytes:
        if self._closed:
            return b""

        if size < 0:
            chunks: list[bytes] = []
            if len(self._buffer) > 0:
                chunks.append(bytes(self._buffer))
                self._buffer.clear()
            for chunk in self._it:
                if len(chunk) > 0:
                    chunks.append(chunk)
            self._closed = True
            self._finalize()
            return b"".join(chunks)

        if size == 0:
            return b""

        while len(self._buffer) < size and not self._closed:
            try:
                chunk = next(self._it)
            except StopIteration:
                self._closed = True
                self._finalize()
                break
            if len(chunk) > 0:
                self._buffer.extend(chunk)

        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        return out

    def readinto(self, target: bytearray) -> int:
        if self._closed:
            return 0

        while len(self._buffer) == 0:
            try:
                chunk = next(self._it)
            except StopIteration:
                self._closed = True
                self._finalize()
                return 0
            if len(chunk) > 0:
                self._buffer.extend(chunk)

        copied = min(len(target), len(self._buffer))
        target[:copied] = self._buffer[:copied]
        del self._buffer[:copied]
        return copied

    @override
    def close(self) -> None:
        self._closed = True
        self._finalize()
        super().close()
