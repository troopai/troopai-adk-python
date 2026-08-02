"""Tests for ``troopai.adk.sandbox.session.workspace_payloads``."""

from __future__ import annotations

import io
from pathlib import Path
from typing import override

import pytest

from troopai.adk.exceptions.exceptions import WorkspaceWriteTypeError
from troopai.adk.sandbox.session.workspace_payloads import coerce_write_payload


def test_coerce_normal_bytes_stream() -> None:
    src = io.BytesIO(b"hello world")
    payload = coerce_write_payload(path=Path("x.txt"), data=src)
    assert payload.content_length == 11
    assert payload.stream.read() == b"hello world"


def test_coerce_rejects_str_chunks() -> None:
    class StrStream(io.IOBase):
        def __init__(self) -> None:
            self._pos = 0

        @override
        def read(self, size: int = -1) -> str:
            _ = size
            return "not bytes"

        @override
        def readable(self) -> bool:
            return True

        @override
        def seekable(self) -> bool:
            return False

    payload = coerce_write_payload(path=Path("x.txt"), data=StrStream())
    with pytest.raises(WorkspaceWriteTypeError, match="str"):
        payload.stream.read()


def test_coerce_accepts_bytearray_and_returns_bytes() -> None:
    class ByteArrayStream(io.IOBase):
        def __init__(self) -> None:
            self._data = bytearray(b"foo")
            self._consumed = False

        @override
        def read(self, size: int = -1) -> bytearray | bytes:
            _ = size
            if self._consumed:
                return b""
            self._consumed = True
            return self._data

    payload = coerce_write_payload(path=Path("x"), data=ByteArrayStream())
    out = payload.stream.read()
    assert isinstance(out, bytes)
    assert out == b"foo"


def test_content_length_from_explicit_attribute() -> None:
    src = io.BytesIO(b"abc")
    src.length = 99  # type: ignore[attr-defined]  # mimic urllib response wrappers
    payload = coerce_write_payload(path=Path("x"), data=src)
    assert payload.content_length == 99


def test_content_length_from_headers_dict() -> None:
    class HeaderedStream(io.IOBase):
        def __init__(self) -> None:
            self.headers = {"Content-Length": "42"}

        @override
        def read(self, size: int = -1) -> bytes:
            _ = size
            return b""

        @override
        def seekable(self) -> bool:
            return False

    payload = coerce_write_payload(path=Path("x"), data=HeaderedStream())
    assert payload.content_length == 42


def test_content_length_unavailable_returns_none() -> None:
    class OpaqueStream(io.IOBase):
        @override
        def read(self, size: int = -1) -> bytes:
            _ = size
            return b""

        @override
        def seekable(self) -> bool:
            return False

    payload = coerce_write_payload(path=Path("x"), data=OpaqueStream())
    assert payload.content_length is None


def test_readinto_copies_into_buffer() -> None:
    src = io.BytesIO(b"abcdef")
    payload = coerce_write_payload(path=Path("x"), data=src)
    buf = bytearray(4)
    n = payload.stream.readinto(buf)
    assert n == 4
    assert bytes(buf) == b"abcd"
