"""The one shared regex scanner reused by every pattern-based built-in guardrail.

``PatternScanner`` holds a mapping of label → pre-compiled pattern and answers
two questions over a piece of text: which labels matched (``scan``) and where
each match sits (``find_spans``). Both the PII and the prompt-injection
guardrails construct one of these instead of carrying their own loop, so the
scanning behaviour stays identical across the hub.

``find_spans`` returns the scanner's own observation of the checked text. It is
distinct from a verdict's ``changed_spans``: the latter is what a transforming
guardrail reports for audit, even though both happen to use the same spans here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from troopai.adk.types.guardrails.action import GuardrailSpan

__all__ = ["PatternScanner"]


@dataclass(frozen=True, kw_only=True)
class PatternScanner:
    """A reusable, immutable scanner over a fixed set of labelled regexes.

    Patterns arrive already compiled (the built-in pattern maps are
    module-level ``re.compile`` constants), so scanning never pays a compile
    cost per call. The mapping is validated once at construction.

    Attributes:
        patterns: Label → compiled pattern. ``scan`` reports the labels that
            matched; ``find_spans`` reports a ``GuardrailSpan`` per match,
            tagged with the matching label as its ``reason``.
    """

    patterns: Mapping[str, re.Pattern[str]]
    """Label → compiled pattern, validated non-empty at construction."""

    def __post_init__(self) -> None:
        """Reject an empty pattern map or a blank label at construction time."""
        if len(self.patterns) == 0:
            raise ValueError("PatternScanner requires at least one pattern")
        for label in self.patterns:
            if len(label) == 0:
                raise ValueError("PatternScanner pattern labels must be non-empty")

    def scan(self, text: str) -> list[str]:
        """Return the sorted labels of every pattern that matches ``text``.

        Args:
            text: The text to scan.

        Returns:
            Sorted, de-duplicated labels of all matching patterns (empty when
            ``text`` is empty or nothing matches).
        """
        if len(text) == 0:
            return []
        matched = [label for label, pattern in self.patterns.items() if pattern.search(text) is not None]
        return sorted(matched)

    def find_spans(self, text: str) -> list[GuardrailSpan]:
        """Return a ``GuardrailSpan`` for every individual match in ``text``.

        Args:
            text: The text to scan.

        Returns:
            Spans ordered left to right by ``(start, end)``, each carrying the
            matching pattern's label as its ``reason`` (empty when ``text`` is
            empty or nothing matches).
        """
        if len(text) == 0:
            return []
        spans: list[GuardrailSpan] = []
        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                spans.append(GuardrailSpan(start=match.start(), end=match.end(), reason=label))
        spans.sort(key=lambda span: (span.start, span.end))
        return spans
