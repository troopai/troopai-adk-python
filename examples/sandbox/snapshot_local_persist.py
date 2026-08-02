"""Persist + restore a workspace snapshot via the local SnapshotStore.

Demonstrates the snapshot pipeline that backs every sandbox
backend's pause/resume: take a tar payload of the live workspace,
hash it, write it under the local store, and replay it into a
fresh session later.

This is the canonical local-disk path. ``S3SnapshotStore`` /
``GCSSnapshotStore`` share the same interface — swap the store
construction and nothing else changes.

No external API key required.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import io
import logging
import tarfile
import tempfile
from pathlib import Path

from troopai.adk.sandbox.snapshot.local_store import LocalSnapshotStore
from troopai.adk.types.sandbox.snapshot import SnapshotRef

logger = logging.getLogger(__name__)


def _build_demo_tar() -> bytes:
    """Build a tiny tar payload representing a workspace snapshot."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="src/app.py")
        payload = b"print('hello from restored snapshot')\n"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "snapshots"
        store = LocalSnapshotStore(base_path=base)

        logger.info("=== Persist snapshot ===")
        snapshot_id = "demo-run-001"
        payload = _build_demo_tar()
        metadata = await store.save(snapshot_id=snapshot_id, data=io.BytesIO(payload))
        logger.info("Saved %s (%d bytes)", metadata.ref.snapshot_id, metadata.size_bytes)
        ref = metadata.ref

        logger.info("=== Exists + list ===")
        assert await store.exists(ref)
        listing = await store.list()
        ids = [m.ref.snapshot_id for m in listing]
        logger.info("Store contains: %s", ids)
        assert snapshot_id in ids

        logger.info("=== Restore snapshot ===")
        handle = await store.load(ref)
        try:
            restored = handle.read()
        finally:
            handle.close()
        logger.info("Restored %d bytes", len(restored))
        assert restored == payload

        logger.info("=== Delete snapshot (idempotent) ===")
        await store.delete(ref)
        await store.delete(ref)  # second call must not raise
        assert not await store.exists(ref)
        _ = SnapshotRef  # imported for type-only context above

        logger.info("Snapshot round-trip complete.")


if __name__ == "__main__":
    asyncio.run(main())
