"""Audit sink Protocol + in-process and append-only-file backends.

Mirrors the cost-ledger idiom: ``@runtime_checkable`` Protocol, async
``record``. Heavy backends (S3, Postgres) live under ``sinks/`` and are
gated behind optional extras.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from troopai.adk.audit.event import AuditEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class AuditSink(Protocol):
    """Append-only destination for tool-call audit events."""

    async def record(self, event: AuditEvent) -> None:
        """Persist one audit event. Implementations must be append-only.

        Args:
            event: The audit event to persist.
        """
        ...


class InMemoryAuditSink:
    """Process-local sink. Default for single-process use and tests.

    Attributes:
        events: Ordered list of every event recorded in this process.
    """

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        """Append ``event`` to :attr:`events`.

        Args:
            event: The audit event to store.
        """
        self.events.append(event)
        logger.debug(
            "audit record tenant=%s tool=%s outcome=%s",
            event.tenant_id,
            event.tool_name,
            event.outcome,
        )


class JsonlFileAuditSink:
    """Append-only JSON-Lines file sink (one JSON object per line).

    File writes are blocking, so each append runs in a worker thread to
    avoid stalling the event loop.

    Attributes:
        path: Absolute or relative path to the ``.jsonl`` file.
    """

    def __init__(self, path: str | Path) -> None:
        """Create a JSONL file sink.

        Args:
            path: Filesystem path for the append-only ``.jsonl`` file.
                The file is created if it does not exist.
        """
        self.path = Path(path)

    async def record(self, event: AuditEvent) -> None:
        """Serialise ``event`` as JSON and append it to :attr:`path`.

        Args:
            event: The audit event to write.
        """
        payload = dataclasses.asdict(event)
        payload["timestamp"] = event.timestamp.isoformat()
        line = json.dumps(payload, ensure_ascii=False)
        try:
            await asyncio.to_thread(self._append, line)
        except Exception:
            # Governance's best-effort handler logs only a generic warning;
            # name the failed path here so a compliance team can locate it.
            logger.error("audit jsonl write FAILED path=%s", self.path)
            raise

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
