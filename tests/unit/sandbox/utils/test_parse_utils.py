"""Tests for ``troopai.adk.sandbox.utils.parse_utils``."""

from __future__ import annotations

from troopai.adk.sandbox.utils.parse_utils import EntryKind, parse_ls_la

_SAMPLE = """total 20
drwxr-xr-x  2 root root 4096 Jan  1 00:00 sub
-rw-r--r--  1 alice users 12 Jan  1 00:00 hello.txt
lrwxrwxrwx  1 bob   bob   12 Jan  1 00:00 link -> target
drwxr-xr-x  2 root root 4096 Jan  1 00:00 .
drwxr-xr-x  3 root root 4096 Jan  1 00:00 ..
"""


def test_skips_total_dot_and_dotdot() -> None:
    entries = parse_ls_la(_SAMPLE, base="/work")
    names = [e.path for e in entries]
    assert names == ["/work/sub", "/work/hello.txt", "/work/link"]


def test_resolves_paths_against_base() -> None:
    entries = parse_ls_la(_SAMPLE, base="/")
    assert [e.path for e in entries[:2]] == ["/sub", "/hello.txt"]


def test_owner_group_size_parsing() -> None:
    entries = parse_ls_la(_SAMPLE, base="/x")
    sub = entries[0]
    assert sub.owner == "root"
    assert sub.group == "root"
    assert sub.size == 4096
    txt = entries[1]
    assert txt.owner == "alice"
    assert txt.group == "users"
    assert txt.size == 12


def test_kind_classification() -> None:
    entries = parse_ls_la(_SAMPLE, base="/x")
    assert entries[0].kind == EntryKind.DIRECTORY
    assert entries[0].is_directory is True
    assert entries[1].kind == EntryKind.FILE
    assert entries[2].kind == EntryKind.SYMLINK


def test_symlink_target_stripped_from_path() -> None:
    entries = parse_ls_la(_SAMPLE, base="/x")
    link = entries[2]
    assert link.path.endswith("/link")
    assert " -> " not in link.path


def test_unparseable_lines_skipped() -> None:
    junk = "wat???\ntotal 0\n-rw-r--r-- 1 a b NOT_AN_INT Jan 1 00:00 weird\n"
    assert parse_ls_la(junk, base="/x") == []


def test_too_few_columns_skipped() -> None:
    sample = "drwxr-xr-x  2 root root\n"
    assert parse_ls_la(sample, base="/x") == []
