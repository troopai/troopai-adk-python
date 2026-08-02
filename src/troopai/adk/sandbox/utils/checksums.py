"""SHA-256 streaming digest helpers.

``sha256_file`` reads a path in 1 MiB chunks; ``sha256_io`` hashes
any readable stream and rewinds it if the stream is seekable so
the caller can reuse the contents (typical workspace-snapshot
flow: hash the tar payload then upload it to a remote store).
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["sha256_file", "sha256_io"]

_CHUNK_SIZE = 1024 * 1024

# Safety bound for ``sha256_io``: a stream whose ``read()`` never returns an
# empty chunk would otherwise spin forever. 1 TiB / 1 MiB-per-chunk = 1M
# iterations — vastly higher than any realistic workspace payload.
_SHA256_IO_MAX_CHUNKS: int = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash ``path`` and return the hex digest.

    Reads the file in 1 MiB chunks so very large workspace tars do
    not pin the digest computation in RAM. ``iter(..., b"")`` is the
    idiomatic bounded form: it terminates at the EOF sentinel
    returned by ``read()`` on regular files.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    hex_digest = digest.hexdigest()
    logger.debug("sha256_file(%s) -> %s", path, hex_digest)
    return hex_digest


def _read_chunk(stream: io.IOBase, chunk_size: int) -> bytes | str | None:
    """Read one chunk; return ``None`` at EOF, the chunk otherwise.

    Raises ``TypeError`` on any chunk that isn't ``bytes`` /
    ``bytearray`` / ``str``.
    """
    chunk: object = stream.read(chunk_size)
    if chunk == b"" or chunk == "":
        return None
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes | bytearray):
        return bytes(chunk)
    raise TypeError(f"sha256_io() requires a bytes-or-str readable stream, got chunk of type {type(chunk).__name__}")


def sha256_io(stream: io.IOBase, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Hash a readable stream and rewind it when possible.

    The stream may yield ``bytes`` or ``str`` chunks; ``str`` chunks
    are encoded as UTF-8 before hashing. If the stream is seekable
    the position is restored on success so the caller can resume
    reading from the same offset.

    Raises:
        TypeError: A chunk is neither ``bytes``, ``bytearray``, nor ``str``.
        RuntimeError: Stream produced more than
            ``_SHA256_IO_MAX_CHUNKS`` non-empty chunks without
            reaching EOF — almost certainly a buggy / hostile stream.
    """
    start_position: int | None = None
    if stream.seekable():
        start_position = stream.tell()

    digest = hashlib.sha256()
    eof_reached = False
    for _ in range(_SHA256_IO_MAX_CHUNKS):
        chunk = _read_chunk(stream, chunk_size)
        if chunk is None:
            eof_reached = True
            break
        digest.update(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)

    if not eof_reached:
        raise RuntimeError(f"sha256_io() exceeded {_SHA256_IO_MAX_CHUNKS} reads without EOF; stream may be unbounded")

    if start_position is not None:
        stream.seek(start_position)

    return digest.hexdigest()
