from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

pytest.importorskip("boto3")

from troopai.adk.audit.event import AuditEvent
from troopai.adk.audit.sink import AuditSink
from troopai.adk.audit.sinks.s3 import S3AuditSink


class _RecordingS3:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803 — mirrors boto3 keyword names used in production call
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body})


def _event() -> AuditEvent:
    return AuditEvent(
        tenant_id="t1",
        agent_name="a",
        tool_name="search",
        tool_call_id="c1",
        args_hash="h",
        result_hash=None,
        outcome="denied",
        timestamp=datetime.now(UTC),
    )


def test_s3_is_an_audit_sink() -> None:
    assert isinstance(S3AuditSink(bucket="b", client=_RecordingS3()), AuditSink)


async def test_puts_one_object_per_event() -> None:
    client = _RecordingS3()
    sink = S3AuditSink(bucket="audit-bkt", prefix="logs", client=client)
    await sink.record(_event())
    assert len(client.puts) == 1
    put = client.puts[0]
    assert put["Bucket"] == "audit-bkt"
    assert put["Key"].startswith("logs/t1/")
    body = json.loads(put["Body"])
    assert body["outcome"] == "denied"
    assert "T" in body["timestamp"]  # ISO-8601, not space-separated


async def test_untenanted_event_uses_none_segment_and_default_prefix() -> None:
    client = _RecordingS3()
    sink = S3AuditSink(bucket="b", client=client)  # default prefix="audit"
    event = AuditEvent(
        tenant_id=None,
        agent_name="a",
        tool_name="t",
        tool_call_id="c1",
        args_hash="h",
        result_hash=None,
        outcome="ok",
        timestamp=datetime.now(UTC),
    )
    await sink.record(event)
    assert client.puts[0]["Key"].startswith("audit/none/")  # default prefix + 'none' segment


async def test_tenant_id_with_slash_cannot_escape_prefix() -> None:
    """A tenant_id / tool_call_id containing '/' must stay one key segment.

    Security regression: the object key interpolated tenant_id and
    tool_call_id raw, so a tenant_id like ``evil/../victim`` wrote records
    OUTSIDE its per-tenant prefix — breaking cross-tenant audit isolation
    (a ListObjectsV2 scoped to ``audit/<tenant>/`` would miss them, or a
    path-normalising reader would surface them under another tenant). Each
    dynamic segment is now percent-encoded, so no raw '/' survives.
    """
    client = _RecordingS3()
    sink = S3AuditSink(bucket="b", client=client)  # default prefix "audit"
    event = AuditEvent(
        tenant_id="evil/../victim",
        agent_name="a",
        tool_name="t",
        tool_call_id="call/../escape",
        args_hash="h",
        result_hash=None,
        outcome="ok",
        timestamp=datetime.now(UTC),
    )
    await sink.record(event)
    key = client.puts[0]["Key"]
    # Exactly audit/<tenant-seg>/<stamp>-<call-seg>.json → exactly two '/'.
    assert key.startswith("audit/")
    assert key.count("/") == 2, f"a slash from tenant_id/tool_call_id escaped its segment: {key!r}"
    assert "%2F" in key  # the raw slashes were percent-encoded


async def test_put_failure_logs_bucket_and_key_before_reraising(caplog) -> None:
    """A failed put_object logs the bucket+key at ERROR before propagating.

    Governance's best-effort handler only logs a generic warning; the sink
    must record its specific target so a compliance team can locate the
    failed write.
    """
    import logging

    class _FailingS3:
        def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
            del Bucket, Key, Body
            raise RuntimeError("s3 unavailable")

    sink = S3AuditSink(bucket="audit-bkt", client=_FailingS3())
    with (
        caplog.at_level(logging.ERROR, logger="troopai.adk.audit.sinks.s3"),
        pytest.raises(RuntimeError, match="s3 unavailable"),
    ):
        await sink.record(_event())
    assert "audit-bkt" in caplog.text
    assert "FAILED" in caplog.text
