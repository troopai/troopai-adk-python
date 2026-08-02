"""Sandbox guardrails — typed verdicts for command policy + content filters.

``SandboxCommandGuardrail`` enforces per-rule command allow/deny
lists. Verdicts come back through the standard tool guardrail
surface so they appear in ``RunResult.guardrail_results`` and are
auditable end-to-end. Policy verdicts ALWAYS belong to a guardrail
type, never to plumbing middleware, so the audit trail stays
complete.
"""

from __future__ import annotations

from troopai.adk.sandbox.guardrails.command_guardrail import SandboxCommandGuardrail

__all__ = ["SandboxCommandGuardrail"]
