"""Per-tenant tool governance: allowlist gate + audit emit.

Executor-called logic, kept out of ``tools_executor.py`` (which is
already large) and mirroring ``run/cost.py``'s split between a pure
predicate and a policy-applying enforcer. Types live in ``audit/``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from troopai.adk.audit.event import AuditEvent, hash_payload
from troopai.adk.types.guardrails.action import GuardrailAction
from troopai.adk.types.run.guardrail_audit import GuardrailAuditRecord

if TYPE_CHECKING:
    from troopai.adk.agents.agent_guardrails import AgentGuardrailSeverity
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.guardrails.action import GuardrailSpan
    from troopai.adk.types.run.guardrail_audit import GuardrailAuditLevel

logger = logging.getLogger(__name__)

# Sentinel distinguishing "no result supplied" from "result is None".
_UNSET: Any = object()


def tenant_allowlist_permits(config: RunConfig, tenant_id: str | None, tool_name: str) -> bool:
    """Return whether ``tenant_id`` may call ``tool_name`` under ``config``.

    ``None`` allowlist -> feature off (permit). Untenanted run -> permit.
    Tenant in map -> membership test (empty set denies all). Tenant absent
    -> permit unless ``tenant_allowlist_default_deny``.
    """
    allowlist = config.tenant_tool_allowlist
    if allowlist is None:
        return True
    if tenant_id is None:
        return True
    allowed = allowlist.get(tenant_id)
    if allowed is None:
        return not config.tenant_allowlist_default_deny
    return tool_name in allowed


async def emit_audit(
    config: RunConfig,
    *,
    tenant_id: str | None,
    agent_name: str,
    tool_name: str,
    call_id: str,
    args: Any,
    outcome: Literal["ok", "denied", "error"],
    result: Any = _UNSET,
) -> None:
    """Record one audit event to ``config.audit_sink`` (no-op if unset).

    Best-effort: a sink failure is logged at warning unless
    ``config.audit_strict`` is set, in which case it re-raises.
    """
    sink = config.audit_sink
    if sink is None:
        return
    event = AuditEvent(
        tenant_id=tenant_id,
        agent_name=agent_name,
        tool_name=tool_name,
        tool_call_id=call_id,
        args_hash=hash_payload(args),
        result_hash=hash_payload(result) if result is not _UNSET else None,
        outcome=outcome,
        timestamp=datetime.now(UTC),
    )
    try:
        await sink.record(event)
    except Exception as exc:
        if config.audit_strict:
            raise
        logger.warning("audit sink failed for %s/%s: %s", tool_name, outcome, exc)


def emit_guardrail_audit(
    ctx: RunContext[Any],
    *,
    level: GuardrailAuditLevel,
    agent_name: str | None,
    guardrail_name: str,
    action: GuardrailAction,
    checked: Any,
    severity: AgentGuardrailSeverity | None = None,
    transformed: Any = None,
    changed_spans: tuple[GuardrailSpan, ...] = (),
) -> None:
    """Build one guardrail audit record and append it to the run context.

    Hashes the checked artifact (and any replacement) so the trail never stores
    raw payloads. ``triggered`` is derived from ``action``: anything other than a
    plain pass counts as triggered. ``transformed`` is the replacement value for a
    transform verdict (e.g. a tool rejection message); ``None`` otherwise, which
    leaves ``transformed_hash`` unset.
    """
    record = GuardrailAuditRecord(
        level=level,
        guardrail_name=guardrail_name,
        agent_name=agent_name,
        action=action,
        severity=severity,
        triggered=action is not GuardrailAction.PASS,
        output_hash=hash_payload(checked) if checked is not None else None,
        transformed_hash=hash_payload(transformed) if transformed is not None else None,
        changed_spans=changed_spans,
        timestamp=datetime.now(UTC),
    )
    ctx.record_guardrail_audit(record)


async def enforce_tenant_allowlist(
    config: RunConfig,
    *,
    tenant_id: str | None,
    agent_name: str,
    tool_name: str,
    call_id: str,
    raw_args: str,
) -> str | None:
    """Gate a tool call against the tenant allowlist.

    Returns ``None`` when permitted. On denial, emits a ``"denied"`` audit
    event, then either returns a denial message (soft mode) or raises
    :class:`ToolNotPermittedForTenant` (hard mode, the default).
    """
    from troopai.adk.exceptions import ToolNotPermittedForTenant
    from troopai.adk.run.config import get_messages

    if tenant_allowlist_permits(config, tenant_id, tool_name):
        return None
    if tenant_id is None:
        # Unreachable: permits() returns True for an untenanted run. This
        # explicit guard narrows tenant_id to str for the raise below
        # without an ``assert`` (which -O would strip).
        return None
    await emit_audit(
        config,
        tenant_id=tenant_id,
        agent_name=agent_name,
        tool_name=tool_name,
        call_id=call_id,
        args=raw_args,
        outcome="denied",
    )
    if config.tenant_allowlist_soft_deny:
        logger.info("tenant '%s' soft-denied tool '%s'", tenant_id, tool_name)
        return get_messages(config).tool_permission_denied(tool_name)
    raise ToolNotPermittedForTenant(tenant_id=tenant_id, tool_name=tool_name, agent_name=agent_name)
