"""Per-action guardrail audit record.

A privacy-preserving trail of what each guardrail did across the agent, tool,
and flow levels: the action taken, payload hashes (never the raw checked text),
and observability spans. Modeled on the tool-call audit event — hashes give
correlation and tamper-evidence without turning the audit log into a sink for
the very PII a guardrail is meant to catch.

This module is import-light by design: the framework types it references appear
only in annotations, so they are imported under ``TYPE_CHECKING`` and the record
carries no runtime dependency on the agent or action packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from troopai.adk.agents.agent_guardrails import AgentGuardrailSeverity
    from troopai.adk.types.guardrails.action import GuardrailAction, GuardrailSpan

GuardrailAuditLevel = Literal[
    "agent_input",
    "agent_output",
    "tool_input",
    "tool_output",
    "flow_pre",
    "flow_post",
]
"""Which guardrail surface produced the record."""


@dataclass(frozen=True, kw_only=True)
class GuardrailAuditRecord:
    """One guardrail decision, recorded for inspection after the run.

    Frozen + keyword-only: an audit entry is immutable once created and always
    constructed by name. Hashes stand in for the checked artifact so the trail
    never stores raw (possibly sensitive) payloads.

    Attributes:
        level: Which guardrail surface produced this record.
        guardrail_name: The guardrail's name.
        agent_name: The agent the guardrail ran for, or ``None``.
        action: The action the runner actually took for this verdict.
        severity: Agent-level severity when set; ``None`` at the tool/flow
            levels and for agent verdicts that carry no severity.
        triggered: Whether the verdict was anything other than a plain pass.
        output_hash: sha256 hex of the checked artifact, or ``None`` when there
            was nothing to hash.
        transformed_hash: sha256 hex of the replacement, set only when the
            action was a transform; ``None`` otherwise. A differing
            ``output_hash``/``transformed_hash`` pair marks a substitution.
        changed_spans: Observability ranges the guardrail reported. Empty when
            none were supplied.
        timestamp: UTC time the record was created.
    """

    level: GuardrailAuditLevel
    """Which guardrail surface produced this record."""

    guardrail_name: str
    """The guardrail's name."""

    agent_name: str | None
    """The agent the guardrail ran for, or ``None``."""

    action: GuardrailAction
    """The action the runner actually took for this verdict."""

    severity: AgentGuardrailSeverity | None
    """Agent-level severity when set; ``None`` elsewhere."""

    triggered: bool
    """Whether the verdict was anything other than a plain pass."""

    output_hash: str | None
    """sha256 hex of the checked artifact, or ``None`` when nothing was hashed."""

    transformed_hash: str | None
    """sha256 hex of the replacement; set only for a transform, else ``None``."""

    timestamp: datetime
    """UTC time the record was created."""

    changed_spans: tuple[GuardrailSpan, ...] = ()
    """Observability ranges reported by the guardrail; empty when none."""


__all__ = ["GuardrailAuditLevel", "GuardrailAuditRecord"]
