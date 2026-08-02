"""Tests for ``troopai.adk.sandbox.utils.checksums``."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import override

import pytest

from troopai.adk.sandbox.utils.checksums import sha256_file, sha256_io


class TestSha256File:
    def test_hashes_known_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "payload.bin"
        path.write_bytes(b"hello world")
        assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()

    def test_hashes_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert sha256_file(path) == hashlib.sha256(b"").hexdigest()

    def test_hashes_chunked_file(self, tmp_path: Path) -> None:
        payload = b"x" * (3 * 1024 * 1024 + 7)
        path = tmp_path / "big.bin"
        path.write_bytes(payload)
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


class TestSha256IO:
    def test_hashes_bytes_stream_and_rewinds(self) -> None:
        stream = io.BytesIO(b"hello world")
        # Seek mid-stream first: sha256_io reads from THIS position to EOF and
        # MUST restore the position when done. The digest below is computed for
        # "lo world" — the portion the stream actually exposes to the hasher.
        stream.seek(3)
        digest = sha256_io(stream)
        assert digest == hashlib.sha256(b"lo world").hexdigest()
        assert stream.tell() == 3, "sha256_io MUST rewind a seekable stream"

    def test_full_stream_hash_from_zero(self) -> None:
        stream = io.BytesIO(b"hello world")
        digest = sha256_io(stream)
        assert digest == hashlib.sha256(b"hello world").hexdigest()
        assert stream.tell() == 0

    def test_accepts_str_chunks(self) -> None:
        class StrStream(io.IOBase):
            def __init__(self, payload: str) -> None:
                self._payload = payload
                self._pos = 0

            @override
            def read(self, size: int = -1) -> str:
                if size < 0:
                    out = self._payload[self._pos :]
                    self._pos = len(self._payload)
                    return out
                out = self._payload[self._pos : self._pos + size]
                self._pos += len(out)
                return out

            @override
            def seekable(self) -> bool:
                return False

        stream = StrStream("héllo")
        assert sha256_io(stream) == hashlib.sha256("héllo".encode()).hexdigest()

    def test_rejects_non_bytes_str_chunks(self) -> None:
        class IntStream(io.IOBase):
            @override
            def read(self, size: int = -1) -> object:
                _ = size
                return 42

            @override
            def seekable(self) -> bool:
                return False

        with pytest.raises(TypeError, match="bytes-or-str"):
            sha256_io(IntStream())
