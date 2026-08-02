"""Sandbox lifecycle honesty + observability — deltas OpenAI's Agents SDK lacks.

Synthetic (no Agent / no LLM / no provider credentials): drives the
public ``sandbox_run_context`` lifecycle bracket directly against the
local subprocess backend, demonstrating three TroopAI-ADK behaviours
that the OpenAI Agents SDK has no equivalent for:

1. Structured per-run lifecycle AUDIT — a pluggable ``AuditSink``
   receives typed start/stop/error ``SandboxAuditEvent``s.
2. HONEST rejection of a configured-but-unsupported snapshot store —
   ``config.snapshot_store`` raises ``UnsupportedSnapshotFeatureError``
   instead of silently discarding a persistence store (a
   data-durability lie).
3. LOUD discard of a configured snapshot — ``config.snapshot`` emits
   a backend-named ``WARNING`` (the session is still created;
   nothing is silently dropped).

Run: ``python examples/sandbox/snapshot_honesty_and_audit.py``
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from pathlib import Path

from troopai.adk.exceptions.exceptions import UnsupportedSnapshotFeatureError
from troopai.adk.sandbox.clients.local.subprocess_client import (
    LocalSandboxClientOptions,
    LocalSubprocessSandboxClient,
)
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.observability.audit_sink import AuditSink, SandboxAuditEvent
from troopai.adk.sandbox.runner_integration import sandbox_run_context
from troopai.adk.types.sandbox.snapshot import LocalSnapshotSpec

logger = logging.getLogger("snapshot_honesty_and_audit")


class CollectingAuditSink(AuditSink):
    """An ``AuditSink`` that records every lifecycle event in memory."""

    def __init__(self) -> None:
        self.events: list[SandboxAuditEvent] = []

    async def emit(self, event: SandboxAuditEvent) -> None:
        self.events.append(event)


def _local_config(
    *,
    audit_sink: AuditSink | None = None,
    snapshot_store: object | None = None,
    snapshot: LocalSnapshotSpec | None = None,
) -> SandboxRunConfig:
    return SandboxRunConfig(
        client=LocalSubprocessSandboxClient(warn_banner=False),
        options=LocalSandboxClientOptions(),
        audit_sink=audit_sink,
        snapshot_store=snapshot_store,
        snapshot=snapshot,
    )


async def demo_structured_audit() -> None:
    """A custom AuditSink receives typed start/stop events per run."""
    logger.info("--- Demo 1: structured lifecycle audit ---")
    sink = CollectingAuditSink()
    config = _local_config(audit_sink=sink)
    async with sandbox_run_context(
        config=config,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent_name="audit-demo",
    ) as handle:
        logger.info("session live: id=%s", handle.session.session_id)
    kinds = [e.event_type for e in sink.events]
    logger.info("audit events captured: %s", kinds)
    logger.info(
        "  backend_id=%s agent_name=%s",
        sink.events[0].backend_id,
        sink.events[0].agent_name,
    )
    assert kinds == ["start", "stop"], kinds


async def demo_snapshot_store_is_rejected_loudly() -> None:
    """A configured snapshot_store no backend implements raises (not silently dropped)."""
    logger.info("--- Demo 2: honest snapshot_store rejection ---")
    config = _local_config(snapshot_store=object())  # any non-None store
    try:
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
            agent_name="reject-demo",
        ):
            logger.error("UNREACHABLE — a store should have been rejected")
    except UnsupportedSnapshotFeatureError as exc:
        logger.info("rejected as designed: %s", exc)
        logger.info("  feature=%s backend_id=%s", exc.feature, exc.backend_id)


async def demo_snapshot_discard_is_loud() -> None:
    """A configured snapshot is discarded with a WARNING — never silently."""
    logger.info("--- Demo 3: loud snapshot discard (watch for the WARNING) ---")
    config = _local_config(snapshot=LocalSnapshotSpec(base_path=Path("/tmp/troopai-ex-snap")))
    async with sandbox_run_context(
        config=config,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent_name="discard-demo",
    ) as handle:
        logger.info(
            "session created (snapshot discarded, NOT restored): id=%s",
            handle.session.session_id,
        )


async def main() -> None:
    await demo_structured_audit()
    await demo_snapshot_store_is_rejected_loudly()
    await demo_snapshot_discard_is_loud()
    logger.info("All three sandbox-honesty demos completed.")


if __name__ == "__main__":
    asyncio.run(main())
