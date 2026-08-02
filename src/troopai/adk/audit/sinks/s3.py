"""S3-backed AuditSink. One object per event under a key prefix.

boto3 S3 docs: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any
from urllib.parse import quote

try:
    import boto3
except ImportError as exc:  # pragma: no cover
    raise ImportError("S3AuditSink requires boto3: pip install 'troopai-adk-python[audit-s3]'") from exc

from troopai.adk.audit.event import AuditEvent

logger = logging.getLogger(__name__)


def _key_segment(value: str) -> str:
    """Encode a value as a single, self-contained S3 key segment.

    ``tenant_id`` and ``tool_call_id`` flow into the object key. Leaving a
    raw ``/`` in either lets the value escape its per-tenant prefix — a
    ``tenant_id`` of ``a/b`` or ``../other`` would write under a different
    key path, breaking cross-tenant audit isolation (a ``ListObjectsV2``
    scoped to ``{prefix}/{tenant}/`` would miss those records). Percent-
    encoding with ``safe=""`` removes every ``/`` so the value stays one
    segment; no ``../`` traversal sequence can form without a literal ``/``.
    """
    return quote(value, safe="")


class S3AuditSink:
    """Append-only S3 audit sink: one JSON object per event.

    The key is ``{prefix}/{tenant_id}/{timestamp}-{tool_call_id}.json``,
    which is unique per call, so writes never overwrite each other. The
    ``tenant_id`` and ``tool_call_id`` segments are percent-encoded so a
    value containing ``/`` cannot escape its per-tenant prefix.

    Attributes:
        bucket: Target S3 bucket name.
        prefix: Key prefix for every stored object.

    Args:
        bucket: Target S3 bucket name.
        prefix: Key prefix for every object (default ``"audit"``).
        client: Optional pre-built boto3 S3 client (injected in tests);
            a default client is created when ``None``.
    """

    def __init__(self, bucket: str, prefix: str = "audit", client: Any = None) -> None:
        # boto3 clients are dynamically generated, so the type is Any.
        self.bucket = bucket
        self.prefix = prefix
        self._client: Any = client if client is not None else boto3.client("s3")

    async def record(self, event: AuditEvent) -> None:
        """Serialise ``event`` as JSON and PUT it to S3.

        Args:
            event: The audit event to upload.
        """
        payload = dataclasses.asdict(event)
        payload["timestamp"] = event.timestamp.isoformat()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        stamp = event.timestamp.strftime("%Y%m%dT%H%M%S%f")
        tenant_segment = _key_segment(event.tenant_id or "none")
        call_segment = _key_segment(event.tool_call_id)
        key = f"{self.prefix}/{tenant_segment}/{stamp}-{call_segment}.json"
        try:
            await asyncio.to_thread(self._client.put_object, Bucket=self.bucket, Key=key, Body=body)
        except Exception:
            logger.error("audit s3 put FAILED bucket=%s key=%s", self.bucket, key)
            raise
        logger.debug("audit s3 put bucket=%s key=%s", self.bucket, key)


__all__ = ["S3AuditSink"]
