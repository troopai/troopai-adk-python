"""Output PII guardrail — scan-and-halt by default, scan-and-mask under TRANSFORM.

The default (``on_fail=RAISE``) is cost-conservative: any match halts the run.
``on_fail=TRANSFORM`` is opt-in and applies to ``str`` outputs only — it computes
the COMPLETE anonymized text and returns it as ``transformed_output`` so the
runner substitutes the output wholesale. The matched ranges ride along as
``changed_spans`` for audit, but they are never spliced: the masked string is the
validator's own whole-string computation.

A transforming verdict also sets ``tripwire_triggered=True`` as a halt fallback,
so the run still stops when the runner cannot apply the substitution (a non-text
output, or no transform sink wired). The factory therefore refuses to pair a
TRANSFORM with a non-halting severity, which would silence that fallback and let
masked PII through. Note that already-streamed tokens cannot be recalled — the
mask lands on the final output and the persisted history, not on a live stream.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from typing import Any

from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrailSeverity,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
)
from troopai.adk.guardrails.scan import PatternScanner
from troopai.adk.types.guardrails.action import GuardrailAction, GuardrailSpan

__all__ = [
    "DEFAULT_PII_MASK",
    "DEFAULT_PII_PATTERNS",
    "mask_pii_spans",
    "pii_guardrail",
]

logger = logging.getLogger(__name__)

DEFAULT_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    # Internationalized-address aware: ``\w`` and the Unicode-letter TLD class
    # (``[^\W\d_]``) match non-ASCII local parts and IDN domains
    # (e.g. ``josé@exämple.com``), not just ASCII.
    "email": re.compile(r"[\w.%+-]+@[\w.-]+\.[^\W\d_]{2,}"),
    "url": re.compile(r"https?://[^\s)]+"),
    "phone": re.compile(r"(?<!\d)\+?\d[\d ().-]{7,}\d(?!\d)"),
}
"""Cheap, deterministic markers for the common injected identifiers: email
addresses (ASCII and internationalized), URLs, and phone numbers (``\\d`` matches
non-Latin numerals). Override via the ``patterns`` argument."""

DEFAULT_PII_MASK = "[REDACTED_PII]"
"""Replacement string the default redactor splices over each matched span."""


def mask_pii_spans(text: str, spans: list[GuardrailSpan], *, mask: str = DEFAULT_PII_MASK) -> str:
    """Return ``text`` with ``mask`` spliced over every span.

    Spans are applied left to right; a span that starts inside an already-masked
    range is skipped, so overlapping matches collapse into a single mask instead
    of producing nested replacements.

    Args:
        text: The original text being anonymized.
        spans: Ranges to mask, as produced by ``PatternScanner.find_spans``.
        mask: The replacement string for each (non-overlapping) span.

    Returns:
        The complete anonymized text.
    """
    if len(spans) == 0:
        return text
    parts: list[str] = []
    cursor = 0
    for span in spans:
        if span.start < cursor:
            continue  # overlapping range already covered by an earlier span
        parts.append(text[cursor : span.start])
        parts.append(mask)
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts)


def _build_default_redactor(scanner: PatternScanner) -> Callable[[str], str]:
    """Return a redactor that masks every span ``scanner`` finds in its input."""

    def redact(text: str) -> str:
        return mask_pii_spans(text, scanner.find_spans(text))

    return redact


def pii_guardrail(
    *,
    on_fail: GuardrailAction = GuardrailAction.RAISE,
    patterns: Mapping[str, re.Pattern[str]] | None = None,
    redactor: Callable[[str], str] | None = None,
    name: str = "pii",
    severity: AgentGuardrailSeverity | None = None,
) -> AgentOutputGuardrail[Any]:
    """Build an output guardrail that detects PII in an agent's response.

    Args:
        on_fail: What to do on a match. ``RAISE`` (default, cost-conservative)
            halts the run. ``TRANSFORM`` (opt-in, ``str`` outputs only)
            substitutes the masked text instead of halting.
        patterns: Override the default label → compiled-pattern map.
        redactor: Override the default masker. Receives the original text and
            returns the complete anonymized text. Ignored unless
            ``on_fail=TRANSFORM``.
        name: Guardrail name surfaced in results and tracing.
        severity: Verdict severity, applied only in ``RAISE`` mode (e.g.
            ``WARNING`` to detect-and-log without halting).

    Returns:
        An ``AgentOutputGuardrail`` ready to register on an agent.

    Raises:
        ValueError: If ``on_fail`` is not ``RAISE``/``TRANSFORM``, or if a
            non-halting severity is paired with ``TRANSFORM`` (which would
            suppress the tripwire fallback).
    """
    if on_fail not in (GuardrailAction.RAISE, GuardrailAction.TRANSFORM):
        raise ValueError(f"pii_guardrail on_fail must be RAISE or TRANSFORM, got {on_fail!r}")
    if on_fail is GuardrailAction.TRANSFORM and severity is not None and severity is not AgentGuardrailSeverity.ERROR:
        raise ValueError(
            f"pii_guardrail cannot pair TRANSFORM with a non-halting severity ({severity!r}): "
            "a WARNING/INFO verdict would suppress the tripwire fallback that halts when the "
            "masked output cannot be substituted."
        )
    scanner = PatternScanner(patterns=patterns if patterns is not None else DEFAULT_PII_PATTERNS)
    active_redactor = redactor if redactor is not None else _build_default_redactor(scanner)

    async def check(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
        output = data.output
        text = output if isinstance(output, str) else str(output)
        spans = scanner.find_spans(text)
        if len(spans) == 0:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)
        labels = sorted({span.reason for span in spans})
        if on_fail is GuardrailAction.TRANSFORM and isinstance(output, str):
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                transformed_output=active_redactor(output),
                changed_spans=spans,
                output_info={"matched": labels},
            )
        return AgentGuardrailFunctionOutput(tripwire_triggered=True, severity=severity, output_info={"matched": labels})

    return AgentOutputGuardrail(guardrail_function=check, name=name)
