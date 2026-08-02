"""``S3Checkpointer`` — archival, object-store graph-run persistence via AWS S3.

Each checkpoint is stored as a single JSON object at
``s3://{bucket}/{prefix}{thread_id}.json``. S3 is the **archival tier**:
last-write-wins semantics with no optimistic locking or conflict detection.
The intent is durable, long-term storage where a single writer per
``thread_id`` is the expected operational pattern.

boto3 is synchronous; every S3 call is wrapped in ``asyncio.to_thread``
to satisfy the async :class:`~troopai.adk.graphs.checkpointer.Checkpointer`
Protocol without blocking the event loop.

Requires ``boto3>=1.34.0``: ``pip install 'troopai-adk-python[checkpointer-s3]'``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, TypedDict, override

try:
    import boto3  # type: ignore[import-untyped]  # boto3 ships no type stubs
    from botocore.exceptions import ClientError  # type: ignore[import-untyped]  # botocore ships no type stubs
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "S3Checkpointer requires boto3>=1.34.0: pip install 'troopai-adk-python[checkpointer-s3]'"
    ) from exc

from troopai.adk.graphs.checkpointer import Checkpointer
from troopai.adk.graphs.checkpointers.hooks import CheckpointerHooks
from troopai.adk.graphs.hooks import HookRegistry

if TYPE_CHECKING:
    from troopai.adk.graphs.checkpointer import GraphCheckpoint
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.state import GraphState


logger = logging.getLogger(__name__)

# S3 error codes that indicate a missing object (key not found).
# "404" is intentionally excluded: AWS, moto, and MinIO all return "NoSuchKey";
# accepting "404" risks masking NoSuchBucket on some backends.
_MISS_CODES = ("NoSuchKey",)


class _CheckpointEnvelope(TypedDict):
    """JSON shape stored as the S3 object body."""

    graph_id: str
    superstep: int
    created_at: float
    updated_at: float
    state: dict[str, Any]


class S3Checkpointer(Checkpointer):
    """Archival, last-write-wins graphs checkpointer backed by AWS S3.

    Each ``thread_id`` maps to one S3 object at
    ``{prefix}{thread_id}.json``. Writes are unconditional — no fencing
    tokens and no :class:`~troopai.adk.exceptions.CheckpointConflictError`.
    This makes S3 the right choice for archival / audit workloads with a
    single writer per run, not for concurrent multi-writer coordination.

    The boto3 client is built eagerly in ``__init__``. Pass ``region``
    to target a specific AWS region; ``None`` delegates to the standard
    boto3 region resolution chain (environment variables, ``~/.aws/``,
    instance metadata, etc.).
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "graph-checkpoints/",
        region: str | None = None,
    ) -> None:
        """Initialise the checkpointer and build the boto3 S3 client.

        Args:
            bucket: S3 bucket name where checkpoints are stored.
            prefix: Key prefix prepended to every object key. Should
                end with ``"/"`` for a logical folder layout. Default
                ``"graph-checkpoints/"``.
            region: AWS region name (e.g. ``"us-east-1"``). ``None``
                uses the boto3 default resolution chain (environment
                variables, AWS config files, instance metadata).
        """
        self._bucket = bucket
        self._prefix = prefix
        # botocore client is dynamically typed; no stub ships with boto3.
        self._client: Any = boto3.client("s3", region_name=region)
        logger.debug(
            "S3Checkpointer initialised (bucket=%s, prefix=%s, region=%s).",
            bucket,
            prefix,
            region,
        )

    def _key(self, thread_id: str) -> str:
        return f"{self._prefix}{thread_id}.json"

    @override
    def register(self, registry: HookRegistry) -> None:
        """Subscribe to ``on_node_end`` / ``on_graph_end``."""
        registry.add(CheckpointerHooks(self))
        logger.debug("S3Checkpointer registered on HookRegistry.")

    @override
    async def save(self, checkpoint: GraphCheckpoint) -> None:
        """Persist ``checkpoint`` as an S3 object (last-write-wins).

        No locking is performed — the caller is responsible for ensuring
        that concurrent writes to the same ``thread_id`` are acceptable
        for their workload.

        Args:
            checkpoint: The snapshot to persist.
        """
        now = time.time()
        envelope: _CheckpointEnvelope = {
            "graph_id": checkpoint.graph_id,
            "superstep": checkpoint.superstep,
            "created_at": checkpoint.created_at,
            "updated_at": now,
            "state": checkpoint.state,
        }
        body = json.dumps(envelope, separators=(",", ":")).encode()
        key = self._key(checkpoint.thread_id)
        try:
            await asyncio.to_thread(
                self._client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=body,
            )  # put_object response (ETag, etc.) is unused
        except ClientError as exc:
            logger.error(
                "S3Checkpointer.save: failed thread_id=%s key=%s bucket=%s: %s",
                checkpoint.thread_id,
                key,
                self._bucket,
                exc,
            )
            raise
        logger.debug(
            "S3Checkpointer.save: thread_id=%s superstep=%s",
            checkpoint.thread_id,
            checkpoint.superstep,
        )

    @override
    async def load(self, thread_id: str, graph: Graph[Any]) -> GraphState[Any] | None:
        """Rehydrate the checkpoint for ``thread_id`` (``None`` if absent).

        Args:
            thread_id: The logical run key.
            graph: The :class:`Graph` the checkpoint belongs to. A stored
                ``graph_id`` that differs from ``graph.id`` raises
                ``ValueError``.

        Returns:
            A rehydrated :class:`GraphState`, or ``None`` when no object
            exists for ``thread_id``.

        Raises:
            ValueError: When the stored ``graph_id`` does not match
                ``graph.id``.
            ClientError: Re-raised for any S3 error other than missing key.
        """
        from troopai.adk.graphs.state import GraphState

        key = self._key(thread_id)

        def _get_bytes() -> bytes:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()  # type: ignore[no-any-return]

        try:
            raw = await asyncio.to_thread(_get_bytes)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _MISS_CODES:
                logger.debug("S3Checkpointer.load: no checkpoint for thread_id=%s", thread_id)
                return None
            logger.error(
                "S3Checkpointer.load: S3 error code=%s thread_id=%s key=%s bucket=%s",
                code,
                thread_id,
                key,
                self._bucket,
            )
            raise

        try:
            envelope: _CheckpointEnvelope = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(
                "S3Checkpointer.load: corrupt object for thread_id=%s key=%s bucket=%s",
                thread_id,
                key,
                self._bucket,
            )
            raise

        if envelope["graph_id"] != graph.id:
            raise ValueError(
                f"Checkpoint graph_id={envelope['graph_id']!r} does not match"
                f" supplied graph.id={graph.id!r}. Refusing to load."
            )
        logger.debug("S3Checkpointer.load: thread_id=%s graph_id=%s", thread_id, graph.id)
        return GraphState.from_dict(envelope["state"], graph)

    @override
    async def list_checkpoints(self) -> list[str]:
        """Return a sorted list of thread ids currently stored (paginated).

        ``list_objects_v2`` returns at most 1000 keys per call; this method
        follows ``NextContinuationToken`` until ``IsTruncated`` is false so
        that buckets with more than 1000 checkpoints are enumerated fully.
        The entire pagination loop runs inside a single ``asyncio.to_thread``
        callable to avoid crossing thread boundaries with boto3 state.
        """

        def _list_all() -> list[str]:
            ids: list[str] = []
            kwargs: dict[str, str] = {"Bucket": self._bucket, "Prefix": self._prefix}
            while True:
                resp = self._client.list_objects_v2(**kwargs)
                for obj in resp.get("Contents", []):
                    obj_key: str = obj["Key"]
                    if obj_key.endswith(".json"):
                        ids.append(obj_key[len(self._prefix) :].removesuffix(".json"))
                if not resp.get("IsTruncated"):
                    break
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            return sorted(ids)

        return await asyncio.to_thread(_list_all)

    @override
    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id`` (no-op if absent)."""
        await asyncio.to_thread(
            self._client.delete_object,
            Bucket=self._bucket,
            Key=self._key(thread_id),
        )  # delete_object response is unused
        logger.debug("S3Checkpointer.delete: thread_id=%s", thread_id)


__all__ = ["S3Checkpointer"]
