"""Coerce arbitrary ``IOBase`` streams into a bounded binary read interface.

Backends that wrap remote-VM bridges or hosted providers can't
trust the developer-supplied stream: it might be text-mode, it
might be a urllib response, it might lie about ``read()`` return
types. ``coerce_write_payload`` normalizes the stream behind a
defensive adapter that:

* refuses non-bytes / non-bytearray chunks (raises
  ``WorkspaceWriteTypeError`` immediately, never silently corrupts).
* surfaces a best-effort content length when one is available (so
  the backend can choose chunked vs single-shot uploads).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import override

from troopai.adk.exceptions.exceptions import WorkspaceWriteTypeError

logger = logging.getLogger(__name__)

__all__ = ["WritePayload", "coerce_write_payload"]


@dataclass(frozen=True)
class WritePayload:
    """Wrapped, type-checked write input passed to backend write primitives.

    Attributes:
        stream: A readable binary stream guaranteed to yield
            ``bytes`` on ``read()`` (or raise ``WorkspaceWriteTypeError``).
        content_length: Best-effort total byte count when known;
            ``None`` when the source can't report a length without
            consuming the stream.
    """

    stream: io.IOBase
    """Defensive binary-only adapter wrapping the original input."""

    content_length: int | None = None
    """Best-effort total byte count; ``None`` when unavailable."""


class _BinaryReadAdapter(io.IOBase):
    """Wrap ``io.IOBase`` and reject non-bytes chunks at the boundary.

    The class is module-private — callers receive instances via
    ``coerce_write_payload(...).stream``.
    """

    def __init__(self, *, path: Path, stream: io.IOBase) -> None:
        self._path = path
        self._stream = stream

    @override
    def readable(self) -> bool:
        return True

    @override
    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        if chunk is None:
            return b""
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, bytearray):
            return bytes(chunk)
        raise WorkspaceWriteTypeError(path=self._path, actual_type=type(chunk).__name__)

    def readinto(self, target: bytearray) -> int:
        data = self.read(len(target))
        copied = len(data)
        target[:copied] = data
        return copied

    @override
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return int(self._stream.seek(offset, whence))

    @override
    def tell(self) -> int:
        return int(self._stream.tell())


def coerce_write_payload(*, path: Path, data: io.IOBase) -> WritePayload:
    """Wrap ``data`` in a binary-only adapter + extract a best-effort length.

    Args:
        path: Target workspace path — surfaced in
            ``WorkspaceWriteTypeError`` when the stream yields the
            wrong type so the operator can trace the offending call.
        data: Original developer-supplied stream.

    Returns:
        A ``WritePayload`` ready for backend consumption.
    """
    stream = _BinaryReadAdapter(path=path, stream=data)
    return WritePayload(stream=stream, content_length=_best_effort_content_length(data))


def _best_effort_content_length(stream: io.IOBase) -> int | None:
    """Probe ``stream`` for a length without consuming it.

    Priority: explicit ``content_length`` / ``length`` attribute,
    HTTP-style ``headers["Content-Length"]``, ``tell()``+``seek()``
    round-trip on seekable streams. Returns ``None`` when none of
    the above is available.

    Every fallback path logs at DEBUG (or WARNING for the
    EOF-corruption case) so an operator chasing a "why was my
    Content-Length lost?" question can find the probe outcome in
    the log instead of having to bisect the stream class.
    """
    attribute_length = _length_from_attribute(stream)
    if attribute_length is not None:
        return attribute_length

    header_length = _content_length_from_headers(stream)
    if header_length is not None:
        return header_length

    return _length_from_seek_probe(stream)


def _length_from_attribute(stream: io.IOBase) -> int | None:
    """Try ``stream.content_length`` then ``stream.length`` as integer attributes."""
    for attr in ("content_length", "length"):
        try:
            value = getattr(stream, attr, None)
        except Exception as exc:
            logger.debug("coerce_write_payload: getattr(%s) raised: %s", attr, exc)
            continue
        if not isinstance(value, int):
            continue
        if value < 0:
            logger.debug("coerce_write_payload: ignoring negative %s=%d on stream", attr, value)
            continue
        return value
    return None


def _length_from_seek_probe(stream: io.IOBase) -> int | None:
    """Use ``tell()`` + ``seek()`` to learn the remaining-bytes count without consuming.

    Three failure paths — each logged distinctly so partial seeks
    that leave the stream at EOF (a corruption hazard) escalate to
    a WARNING, while the routine "not seekable" path stays at DEBUG.
    """
    if not (hasattr(stream, "tell") and hasattr(stream, "seek")):
        return None
    try:
        pos = stream.tell()
    except (OSError, io.UnsupportedOperation) as exc:
        logger.debug("coerce_write_payload: tell() probe failed: %s", exc)
        return None
    try:
        stream.seek(0, io.SEEK_END)
        end = stream.tell()
    except (OSError, io.UnsupportedOperation) as exc:
        logger.debug("coerce_write_payload: seek(END) probe failed: %s", exc)
        return None
    try:
        stream.seek(pos, io.SEEK_SET)
    except (OSError, io.UnsupportedOperation) as exc:
        logger.warning(
            "coerce_write_payload: seek-to-original-position failed after seek(END); "
            "stream is now at EOF and a subsequent read() will return b''. "
            "Caller should treat the upload payload as compromised: %s",
            exc,
        )
        return None
    length = int(end - pos)
    if length < 0:
        logger.debug("coerce_write_payload: tell/seek probe yielded negative length=%d", length)
        return None
    return length


def _content_length_from_headers(stream: io.IOBase) -> int | None:
    """Pull ``Content-Length`` off a urllib-style ``headers`` attribute when present.

    Logs at DEBUG on every malformed / negative header so server
    bugs (e.g. ``Content-Length: -1`` from a buggy upstream) don't
    silently vanish — the framework keeps working but the operator
    keeps a trail of the upstream regression.
    """
    headers = getattr(stream, "headers", None)
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if not callable(get):
        return None
    raw = get("Content-Length")
    if not isinstance(raw, str):
        return None
    try:
        parsed = int(raw)
    except ValueError:
        logger.debug("coerce_write_payload: malformed Content-Length header value: %r", raw)
        return None
    if parsed < 0:
        logger.debug("coerce_write_payload: negative Content-Length header value: %d", parsed)
        return None
    return parsed
