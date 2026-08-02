"""Tests for ``troopai.adk.sandbox.utils.iterator_io``."""

from __future__ import annotations

from troopai.adk.sandbox.utils.iterator_io import IteratorIO


def test_read_all() -> None:
    stream = IteratorIO(iter([b"hello ", b"world"]))
    assert stream.read() == b"hello world"
    assert stream.read() == b""


def test_read_chunked() -> None:
    stream = IteratorIO(iter([b"abc", b"def", b"ghi"]))
    assert stream.read(2) == b"ab"
    assert stream.read(2) == b"cd"
    assert stream.read() == b"efghi"


def test_read_zero_returns_empty() -> None:
    stream = IteratorIO(iter([b"abc"]))
    assert stream.read(0) == b""
    assert stream.read() == b"abc"


def test_readinto() -> None:
    stream = IteratorIO(iter([b"abc", b"def"]))
    buf = bytearray(4)
    n = stream.readinto(buf)
    assert n == 3
    assert buf[:n] == b"abc"


def test_skips_empty_chunks() -> None:
    stream = IteratorIO(iter([b"", b"abc", b"", b"def"]))
    assert stream.read() == b"abcdef"


def test_on_close_called_once() -> None:
    calls: list[int] = []
    stream = IteratorIO(iter([b"abc"]), on_close=lambda: calls.append(1))
    assert stream.read() == b"abc"
    stream.close()
    stream.close()
    assert calls == [1], "on_close MUST fire exactly once"


def test_readable_returns_true() -> None:
    stream = IteratorIO(iter([b""]))
    assert stream.readable() is True
