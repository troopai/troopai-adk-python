"""Approximate-token-budget text truncation.

Sandbox tool outputs (shell stdout, file reads, memory summary)
must fit inside the LLM's context. We don't have a tokenizer in
the framework's hot path, so we approximate: roughly four bytes
per token. Good enough for budget enforcement; the exact figure
is then folded into the message that surfaces to the LLM
("…N tokens truncated…").

Truncation is two-sided: keep the head AND the tail of the
output, drop a contiguous middle. The truncation marker lands
in the middle and reports how much was discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "APPROX_BYTES_PER_TOKEN",
    "TruncationMode",
    "TruncationPolicy",
    "approx_bytes_for_tokens",
    "approx_token_count",
    "approx_tokens_from_byte_count",
    "assemble_truncated_output",
    "format_truncation_marker",
    "formatted_truncate_text",
    "formatted_truncate_text_with_token_count",
    "removed_units_for_source",
    "split_budget",
    "split_string",
    "truncate_text",
    "truncate_with_byte_estimate",
    "truncate_with_token_budget",
]

APPROX_BYTES_PER_TOKEN: int = 4
"""Heuristic UTF-8 bytes per LLM token used by every estimator below."""

TruncationMode = Literal["bytes", "tokens"]
"""Whether the truncation policy's limit is a byte or token budget."""


@dataclass(frozen=True)
class TruncationPolicy:
    """Frozen policy describing how to truncate.

    Attributes:
        mode: ``"bytes"`` or ``"tokens"``.
        limit: Upper bound in the mode's unit.
    """

    mode: TruncationMode
    """``"bytes"`` or ``"tokens"``."""

    limit: int
    """Upper bound in the mode's unit."""

    @classmethod
    def bytes(cls, limit: int) -> TruncationPolicy:
        """Build a byte-budget policy; ``limit`` is clamped to ``>= 0``."""
        return cls(mode="bytes", limit=max(0, limit))

    @classmethod
    def tokens(cls, limit: int) -> TruncationPolicy:
        """Build a token-budget policy; ``limit`` is clamped to ``>= 0``."""
        return cls(mode="tokens", limit=max(0, limit))

    def token_budget(self) -> int:
        """Return the policy's limit expressed in tokens."""
        if self.mode == "bytes":
            return approx_tokens_from_byte_count(self.limit)
        return self.limit

    def byte_budget(self) -> int:
        """Return the policy's limit expressed in bytes."""
        if self.mode == "bytes":
            return self.limit
        return approx_bytes_for_tokens(self.limit)


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def formatted_truncate_text(content: str, policy: TruncationPolicy) -> str:
    """Truncate ``content`` per ``policy``; prepend ``"Total output lines: N"`` on truncation."""
    if _byte_len(content) <= policy.byte_budget():
        return content
    total_lines = len(content.splitlines())
    result = truncate_text(content, policy)
    return f"Total output lines: {total_lines}\n\n{result}"


def truncate_text(content: str, policy: TruncationPolicy) -> str:
    """Apply the policy; returns ``content`` unchanged if within budget."""
    if policy.mode == "bytes":
        return truncate_with_byte_estimate(content, policy)
    truncated, _ = truncate_with_token_budget(content, policy)
    return truncated


def formatted_truncate_text_with_token_count(content: str, max_output_tokens: int | None) -> tuple[str, int | None]:
    """Truncate by token budget and return (truncated_text, approx_original_token_count_or_None)."""
    if max_output_tokens is None:
        return content, None

    policy = TruncationPolicy.tokens(max_output_tokens)
    if _byte_len(content) <= policy.byte_budget():
        return content, None

    truncated, original_token_count = truncate_with_token_budget(content, policy)
    total_lines = len(content.splitlines())
    return f"Total output lines: {total_lines}\n\n{truncated}", original_token_count


def truncate_with_token_budget(s: str, policy: TruncationPolicy) -> tuple[str, int | None]:
    """Token-mode truncation; second tuple element is original-token estimate when truncated."""
    if s == "":
        return "", None

    max_tokens = policy.token_budget()
    byte_len = _byte_len(s)
    if max_tokens > 0 and byte_len <= approx_bytes_for_tokens(max_tokens):
        return s, None

    truncated = truncate_with_byte_estimate(s, policy)
    approx_total = approx_token_count(s)
    if truncated == s:
        return truncated, None
    return truncated, approx_total


def truncate_with_byte_estimate(s: str, policy: TruncationPolicy) -> str:
    """Byte-mode truncation: split head/tail by byte budget, inline a marker."""
    if s == "":
        return ""

    total_chars = len(s)
    max_bytes = policy.byte_budget()
    source_bytes = s.encode("utf-8")

    if max_bytes == 0:
        return format_truncation_marker(
            policy,
            removed_units_for_source(policy, len(source_bytes), total_chars),
        )

    if len(source_bytes) <= max_bytes:
        return s

    left_budget, right_budget = split_budget(max_bytes)
    removed_chars, left, right = split_string(s, left_budget, right_budget)
    marker = format_truncation_marker(
        policy,
        removed_units_for_source(policy, len(source_bytes) - max_bytes, removed_chars),
    )
    return assemble_truncated_output(left, right, marker)


def split_string(s: str, beginning_bytes: int, end_bytes: int) -> tuple[int, str, str]:
    """Split ``s`` so the head fits ``beginning_bytes`` and the tail fits ``end_bytes``.

    Returns ``(removed_chars, head, tail)``. UTF-8 safe — split
    points always fall on character boundaries.
    """
    if s == "":
        return 0, "", ""

    source_bytes = s.encode("utf-8")
    length = len(source_bytes)
    tail_start_target = max(0, length - end_bytes)
    prefix_end = 0
    suffix_start = length
    removed_chars = 0
    suffix_started = False

    byte_idx = 0
    for ch in s:
        ch_len = len(ch.encode("utf-8"))
        char_end = byte_idx + ch_len
        if char_end <= beginning_bytes:
            prefix_end = char_end
            byte_idx = char_end
            continue

        if byte_idx >= tail_start_target:
            if not suffix_started:
                suffix_start = byte_idx
                suffix_started = True
            byte_idx = char_end
            continue

        removed_chars += 1
        byte_idx = char_end

    if suffix_start < prefix_end:
        suffix_start = prefix_end

    before = source_bytes[:prefix_end].decode("utf-8", errors="strict")
    after = source_bytes[suffix_start:].decode("utf-8", errors="strict")
    return removed_chars, before, after


def format_truncation_marker(policy: TruncationPolicy, removed_count: int) -> str:
    """Render the inline ``"…N tokens/chars truncated…"`` marker."""
    if policy.mode == "tokens":
        return f"…{removed_count} tokens truncated…"
    return f"…{removed_count} chars truncated…"


def split_budget(budget: int) -> tuple[int, int]:
    """Split a byte budget into (left_half, right_half) — left rounds down."""
    left = budget // 2
    return left, budget - left


def removed_units_for_source(
    policy: TruncationPolicy,
    removed_bytes: int,
    removed_chars: int,
) -> int:
    """Convert removed bytes/chars to the policy's unit for the marker."""
    if policy.mode == "tokens":
        return approx_tokens_from_byte_count(removed_bytes)
    return removed_chars


def assemble_truncated_output(prefix: str, suffix: str, marker: str) -> str:
    """Concatenate head + marker + tail."""
    return f"{prefix}{marker}{suffix}"


def approx_token_count(text: str) -> int:
    """Estimate the LLM-token count of ``text`` from its UTF-8 byte length."""
    byte_len = _byte_len(text)
    return (byte_len + (APPROX_BYTES_PER_TOKEN - 1)) // APPROX_BYTES_PER_TOKEN


def approx_bytes_for_tokens(tokens: int) -> int:
    """Estimate the UTF-8 byte length of ``tokens`` tokens."""
    return max(0, tokens) * APPROX_BYTES_PER_TOKEN


def approx_tokens_from_byte_count(byte_count: int) -> int:
    """Estimate token count from a byte count; ``<= 0`` returns ``0``."""
    if byte_count <= 0:
        return 0
    return (byte_count + (APPROX_BYTES_PER_TOKEN - 1)) // APPROX_BYTES_PER_TOKEN
