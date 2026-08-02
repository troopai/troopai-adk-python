"""V4A diff parser and applier — pure functions on text.

The V4A patch format is OpenAI's apply_patch dialect: a single
diff envelope that may contain create-file, update-file, and
delete-file operations. This module owns the part that turns one
diff (the ``diff`` field of a single operation) into the new file
contents.

``apply_diff`` is the only public entry point; everything else is
parser plumbing. The algorithm is faithful to the upstream
reference so cross-implementation interoperability holds: a diff
produced for OpenAI's editor applies to ours, and vice versa.

The matcher tolerates three fuzzy-context tiers:

* exact line equality (fuzz 0),
* equal after ``rstrip`` (fuzz 1),
* equal after ``strip`` (fuzz 100).

EOF-anchored hunks additionally fall back to a from-start search
with a +10000 fuzz penalty, so a wildly misplaced hunk still
applies but the metric records the deviation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = ["ApplyDiffMode", "apply_diff"]

ApplyDiffMode = Literal["default", "create"]
"""``"default"`` for update diffs; ``"create"`` for ``+``-only create-file diffs."""


@dataclass(slots=True)
class _Chunk:
    """One contiguous delete/insert hunk inside an update diff."""

    orig_index: int
    del_lines: list[str]
    ins_lines: list[str]


@dataclass(slots=True)
class _ParserState:
    """Mutable cursor over the diff's lines."""

    lines: list[str]
    index: int = 0
    fuzz: int = 0


@dataclass(slots=True)
class _ParsedUpdateDiff:
    """All hunks parsed out of one update diff, plus the fuzz accumulator."""

    chunks: list[_Chunk]
    fuzz: int


@dataclass(slots=True)
class _ReadSectionResult:
    """Output of ``_read_section`` — the parsed body plus continuation state."""

    next_context: list[str]
    section_chunks: list[_Chunk]
    end_index: int
    eof: bool


@dataclass(slots=True)
class _ContextMatch:
    """Where context matched in the source plus the fuzz penalty incurred."""

    new_index: int
    fuzz: int


_END_PATCH: str = "*** End Patch"
_END_FILE: str = "*** End of File"
_SECTION_TERMINATORS: tuple[str, ...] = (
    _END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
)
_END_SECTION_MARKERS: tuple[str, ...] = (*_SECTION_TERMINATORS, _END_FILE)


def apply_diff(input_text: str, diff: str, mode: ApplyDiffMode = "default") -> str:
    """Apply a V4A diff to ``input_text`` and return the rewritten contents.

    Args:
        input_text: The current text (empty string when ``mode == "create"``).
        diff: The V4A diff body (no envelope headers; just hunks).
        mode: ``"default"`` for update diffs, ``"create"`` for create-file
            diffs (``+``-prefixed lines only).

    Returns:
        The text that results from applying ``diff`` to ``input_text``.
        Newline style is preserved: if either ``input_text`` or ``diff``
        contains CRLF, the output uses CRLF; otherwise LF.

    Raises:
        ValueError: The diff is malformed or context can't be located.
    """
    newline = _detect_newline(input_text, diff, mode)
    diff_lines = _normalize_diff_lines(diff)
    if mode == "create":
        return _parse_create_diff(diff_lines, newline=newline)

    normalized_input = _normalize_text_newlines(input_text)
    parsed = _parse_update_diff(diff_lines, normalized_input)
    return _apply_chunks(normalized_input, parsed.chunks, newline=newline)


def _normalize_diff_lines(diff: str) -> list[str]:
    lines = [line.rstrip("\r") for line in re.split(r"\r?\n", diff)]
    if len(lines) > 0 and lines[-1] == "":
        lines.pop()
    return lines


def _detect_newline_from_text(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _detect_newline(input_text: str, diff: str, mode: ApplyDiffMode) -> str:
    if mode != "create" and "\n" in input_text:
        return _detect_newline_from_text(input_text)
    return _detect_newline_from_text(diff)


def _normalize_text_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def _is_done(state: _ParserState, prefixes: Sequence[str]) -> bool:
    if state.index >= len(state.lines):
        return True
    return any(state.lines[state.index].startswith(prefix) for prefix in prefixes)


def _read_str(state: _ParserState, prefix: str) -> str:
    if state.index >= len(state.lines):
        return ""
    current = state.lines[state.index]
    if current.startswith(prefix):
        state.index += 1
        return current[len(prefix) :]
    return ""


def _parse_create_diff(lines: list[str], *, newline: str) -> str:
    parser = _ParserState(lines=[*lines, _END_PATCH])
    output: list[str] = []

    while not _is_done(parser, _SECTION_TERMINATORS):
        if parser.index >= len(parser.lines):
            break
        line = parser.lines[parser.index]
        parser.index += 1
        if not line.startswith("+"):
            raise ValueError(f"Invalid Add File Line: {line}")
        output.append(line[1:])

    return newline.join(output)


def _parse_update_diff(lines: list[str], input_text: str) -> _ParsedUpdateDiff:
    parser = _ParserState(lines=[*lines, _END_PATCH])
    input_lines = input_text.split("\n")
    chunks: list[_Chunk] = []
    cursor = 0

    while not _is_done(parser, _END_SECTION_MARKERS):
        anchor = _read_str(parser, "@@ ")
        has_bare_anchor = anchor == "" and parser.index < len(parser.lines) and parser.lines[parser.index] == "@@"
        if has_bare_anchor:
            parser.index += 1

        if not (len(anchor) > 0 or has_bare_anchor or cursor == 0):
            current_line = parser.lines[parser.index] if parser.index < len(parser.lines) else ""
            raise ValueError(f"Invalid Line:\n{current_line}")

        if len(anchor.strip()) > 0:
            cursor = _advance_cursor_to_anchor(anchor, input_lines, cursor, parser)

        section = _read_section(parser.lines, parser.index)
        find_result = _find_context(input_lines, section.next_context, cursor, section.eof)
        if find_result.new_index == -1:
            ctx_text = "\n".join(section.next_context)
            if section.eof:
                raise ValueError(f"Invalid EOF Context {cursor}:\n{ctx_text}")
            raise ValueError(f"Invalid Context {cursor}:\n{ctx_text}")

        cursor = find_result.new_index + len(section.next_context)
        parser.fuzz += find_result.fuzz
        parser.index = section.end_index

        for ch in section.section_chunks:
            chunks.append(
                _Chunk(
                    orig_index=ch.orig_index + find_result.new_index,
                    del_lines=list(ch.del_lines),
                    ins_lines=list(ch.ins_lines),
                )
            )

    return _ParsedUpdateDiff(chunks=chunks, fuzz=parser.fuzz)


def _advance_cursor_to_anchor(
    anchor: str,
    input_lines: list[str],
    cursor: int,
    parser: _ParserState,
) -> int:
    found = False

    if not any(line == anchor for line in input_lines[:cursor]):
        for i in range(cursor, len(input_lines)):
            if input_lines[i] == anchor:
                cursor = i + 1
                found = True
                break

    if not found and not any(line.strip() == anchor.strip() for line in input_lines[:cursor]):
        for i in range(cursor, len(input_lines)):
            if input_lines[i].strip() == anchor.strip():
                cursor = i + 1
                parser.fuzz += 1
                break

    return cursor


def _is_section_break(raw: str) -> bool:
    return (
        raw.startswith("@@")
        or raw.startswith(_END_PATCH)
        or raw.startswith("*** Update File:")
        or raw.startswith("*** Delete File:")
        or raw.startswith("*** Add File:")
        or raw.startswith(_END_FILE)
    )


def _read_section(lines: list[str], start_index: int) -> _ReadSectionResult:
    context: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    section_chunks: list[_Chunk] = []
    mode: Literal["keep", "add", "delete"] = "keep"
    index = start_index
    orig_index = index

    while index < len(lines):
        raw = lines[index]
        if _is_section_break(raw):
            break
        if raw == "***":
            break
        if raw.startswith("***"):
            raise ValueError(f"Invalid Line: {raw}")

        index += 1
        last_mode = mode
        line = raw if len(raw) > 0 else " "
        prefix = line[0]
        if prefix == "+":
            mode = "add"
        elif prefix == "-":
            mode = "delete"
        elif prefix == " ":
            mode = "keep"
        else:
            raise ValueError(f"Invalid Line: {line}")

        line_content = line[1:]
        switching_to_context = mode == "keep" and last_mode != mode
        if switching_to_context and (len(del_lines) > 0 or len(ins_lines) > 0):
            section_chunks.append(
                _Chunk(
                    orig_index=len(context) - len(del_lines),
                    del_lines=list(del_lines),
                    ins_lines=list(ins_lines),
                )
            )
            del_lines = []
            ins_lines = []

        if mode == "delete":
            del_lines.append(line_content)
            context.append(line_content)
        elif mode == "add":
            ins_lines.append(line_content)
        else:
            context.append(line_content)

    if len(del_lines) > 0 or len(ins_lines) > 0:
        section_chunks.append(
            _Chunk(
                orig_index=len(context) - len(del_lines),
                del_lines=list(del_lines),
                ins_lines=list(ins_lines),
            )
        )

    if index < len(lines) and lines[index] == _END_FILE:
        return _ReadSectionResult(context, section_chunks, index + 1, True)

    if index == orig_index:
        next_line = lines[index] if index < len(lines) else ""
        raise ValueError(f"Nothing in this section - index={index} {next_line}")

    return _ReadSectionResult(context, section_chunks, index, False)


def _find_context(
    lines: list[str],
    context: list[str],
    start: int,
    eof: bool,
) -> _ContextMatch:
    if eof:
        end_start = max(0, len(lines) - len(context))
        end_match = _find_context_core(lines, context, end_start)
        if end_match.new_index != -1:
            return end_match
        fallback = _find_context_core(lines, context, start)
        return _ContextMatch(new_index=fallback.new_index, fuzz=fallback.fuzz + 10000)
    return _find_context_core(lines, context, start)


def _find_context_core(lines: list[str], context: list[str], start: int) -> _ContextMatch:
    if len(context) == 0:
        return _ContextMatch(new_index=start, fuzz=0)

    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value):
            return _ContextMatch(new_index=i, fuzz=0)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.rstrip()):
            return _ContextMatch(new_index=i, fuzz=1)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.strip()):
            return _ContextMatch(new_index=i, fuzz=100)

    return _ContextMatch(new_index=-1, fuzz=0)


def _equals_slice(
    source: list[str],
    target: list[str],
    start: int,
    map_fn: Callable[[str], str],
) -> bool:
    if start + len(target) > len(source):
        return False
    return all(map_fn(source[start + offset]) == map_fn(target_value) for offset, target_value in enumerate(target))


def _apply_chunks(input_text: str, chunks: list[_Chunk], *, newline: str) -> str:
    orig_lines = input_text.split("\n")
    dest_lines: list[str] = []
    cursor = 0

    for chunk in chunks:
        if chunk.orig_index > len(orig_lines):
            raise ValueError(f"apply_diff: chunk.orig_index {chunk.orig_index} > input length {len(orig_lines)}")
        if cursor > chunk.orig_index:
            raise ValueError(f"apply_diff: overlapping chunk at {chunk.orig_index} (cursor {cursor})")

        dest_lines.extend(orig_lines[cursor : chunk.orig_index])
        cursor = chunk.orig_index

        if len(chunk.ins_lines) > 0:
            dest_lines.extend(chunk.ins_lines)

        cursor += len(chunk.del_lines)

    dest_lines.extend(orig_lines[cursor:])
    return newline.join(dest_lines)
