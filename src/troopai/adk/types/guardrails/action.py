"""Framework-owned guardrail action vocabulary and observability span.

These types are cross-cutting over the agent, tool, and flow guardrail levels:
each level's verdict type maps its own outcome onto a single shared
``GuardrailAction`` so the runner can dispatch uniformly. The vocabulary is
framework-owned and intentionally minimal — it is not adopted from any external
validation library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GuardrailAction(StrEnum):
    """What the runner does with a guardrail verdict.

    Agent and tool verdicts may ``TRANSFORM`` — substitute the checked artifact
    wholesale. A flow step is ``PASS``/``RAISE`` only: it has no replaceable
    return value, and the developer's typed shared state is off-limits to the
    verdict contract.
    """

    PASS = "pass"
    """Proceed — the checked artifact is accepted unchanged."""

    RAISE = "raise"
    """Halt the run. Agent ``ERROR`` severity / tripwire, tool
    ``raise_exception``, and flow rejection (both the routed and the surfaced
    variants) resolve here."""

    TRANSFORM = "transform"
    """Substitute the checked artifact wholesale with a replacement the guardrail
    supplies. A tool ``reject_content`` substitutes a rejection notice; an agent
    output guardrail substitutes repaired content."""


@dataclass(frozen=True, kw_only=True)
class GuardrailSpan:
    """A ``(start, end, reason)`` text range that a guardrail flagged.

    A conventional character span over the checked text — observability only. It
    reports a range that moved or matched; it is never consumed by the runner to
    construct or apply a fix. A transforming guardrail computes the complete
    replacement itself, and the runner substitutes that whole value.
    """

    start: int
    """Inclusive start index into the checked text."""

    end: int
    """Exclusive end index into the checked text."""

    reason: str
    """Why this range was flagged (e.g. the matched pattern label)."""
