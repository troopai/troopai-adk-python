"""Tests for ``LocalSnapshotStore`` (P40)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from troopai.adk.exceptions.exceptions import (
    SnapshotError,
    SnapshotPersistError,
    SnapshotRestoreError,
)
from troopai.adk.sandbox.snapshot.local_store import LocalSnapshotStore
from troopai.adk.types.sandbox.snapshot import SnapshotRef


@pytest.fixture
def store(tmp_path: Path) -> LocalSnapshotStore:
    return LocalSnapshotStore(tmp_path / "snapshots")


class TestSaveAndLoadRoundTrip:
    @pytest.mark.asyncio
    async def test_basic_round_trip(self, store: LocalSnapshotStore) -> None:
        payload = b"hello world"
        metadata = await store.save(
            snapshot_id="snap-1",
            data=BytesIO(payload),
        )
        assert metadata.ref.snapshot_id == "snap-1"
        assert metadata.size_bytes == len(payload)

        stream = await store.load(metadata.ref)
        try:
            assert stream.read() == payload
        finally:
            stream.close()

    @pytest.mark.asyncio
    async def test_save_creates_base_path(self, tmp_path: Path) -> None:
        store = LocalSnapshotStore(tmp_path / "fresh")
        assert not (tmp_path / "fresh").exists()
        await store.save(snapshot_id="s", data=BytesIO(b"x"))
        assert (tmp_path / "fresh").exists()
        assert (tmp_path / "fresh" / "s.tar").exists()

    @pytest.mark.asyncio
    async def test_save_with_manifest_hash(self, store: LocalSnapshotStore) -> None:
        metadata = await store.save(
            snapshot_id="s",
            data=BytesIO(b"x"),
            manifest_hash="sha256:abc",
        )
        assert metadata.manifest_hash == "sha256:abc"

    @pytest.mark.asyncio
    async def test_load_missing_raises(self, store: LocalSnapshotStore) -> None:
        with pytest.raises(SnapshotRestoreError, match="not found"):
            await store.load(
                SnapshotRef(snapshot_id="nope", store_uri=store.store_uri),
            )


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, store: LocalSnapshotStore) -> None:
        meta = await store.save(snapshot_id="s", data=BytesIO(b"x"))
        await store.delete(meta.ref)
        assert await store.exists(meta.ref) is False

    @pytest.mark.asyncio
    async def test_delete_missing_is_idempotent(self, store: LocalSnapshotStore) -> None:
        # No raise.
        await store.delete(
            SnapshotRef(snapshot_id="never-saved", store_uri=store.store_uri),
        )


class TestList:
    @pytest.mark.asyncio
    async def test_empty_store_lists_empty(self, store: LocalSnapshotStore) -> None:
        assert await store.list() == []

    @pytest.mark.asyncio
    async def test_list_returns_all(self, store: LocalSnapshotStore) -> None:
        await store.save(snapshot_id="alpha", data=BytesIO(b"1"))
        await store.save(snapshot_id="beta", data=BytesIO(b"22"))
        result = await store.list()
        ids = {m.ref.snapshot_id for m in result}
        assert ids == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_list_prefix_filter(self, store: LocalSnapshotStore) -> None:
        await store.save(snapshot_id="run-1", data=BytesIO(b"a"))
        await store.save(snapshot_id="run-2", data=BytesIO(b"b"))
        await store.save(snapshot_id="other", data=BytesIO(b"c"))
        result = await store.list(prefix="run-")
        ids = {m.ref.snapshot_id for m in result}
        assert ids == {"run-1", "run-2"}


class TestExists:
    @pytest.mark.asyncio
    async def test_after_save(self, store: LocalSnapshotStore) -> None:
        meta = await store.save(snapshot_id="s", data=BytesIO(b"x"))
        assert await store.exists(meta.ref) is True

    @pytest.mark.asyncio
    async def test_never_saved(self, store: LocalSnapshotStore) -> None:
        ref = SnapshotRef(snapshot_id="nope", store_uri=store.store_uri)
        assert await store.exists(ref) is False


class TestStoreUri:
    def test_store_uri_is_file_scheme(self, store: LocalSnapshotStore) -> None:
        assert store.store_uri.startswith("file://")


class TestSnapshotIdTraversalRejected:
    """Regression: a snapshot_id with path separators / parent references must
    not escape base_path (path traversal)."""

    @pytest.mark.parametrize(
        "bad_id",
        ["", "../evil", "../../etc/cron.d/x", "sub/snap", "/abs/path"],
    )
    @pytest.mark.asyncio
    async def test_save_rejects_traversal_id(self, store: LocalSnapshotStore, bad_id: str) -> None:
        with pytest.raises(SnapshotError, match="invalid snapshot_id"):
            await store.save(snapshot_id=bad_id, data=BytesIO(b"payload"))

    @pytest.mark.asyncio
    async def test_traversal_writes_no_file_outside_base(
        self,
        store: LocalSnapshotStore,
        tmp_path: Path,
    ) -> None:
        # base_path is tmp_path/"snapshots"; "../evil" would land in tmp_path.
        sentinel = tmp_path / "evil.tar"
        with pytest.raises(SnapshotError):
            await store.save(snapshot_id="../evil", data=BytesIO(b"payload"))
        assert not sentinel.exists()

    @pytest.mark.asyncio
    async def test_load_rejects_traversal_id(self, store: LocalSnapshotStore) -> None:
        with pytest.raises(SnapshotError, match="invalid snapshot_id"):
            await store.load(SnapshotRef(snapshot_id="../evil", store_uri=store.store_uri))

    @pytest.mark.asyncio
    async def test_exists_rejects_traversal_id(self, store: LocalSnapshotStore) -> None:
        with pytest.raises(SnapshotError, match="invalid snapshot_id"):
            await store.exists(SnapshotRef(snapshot_id="../evil", store_uri=store.store_uri))

    @pytest.mark.asyncio
    async def test_valid_id_still_round_trips(self, store: LocalSnapshotStore) -> None:
        meta = await store.save(snapshot_id="snap-ok_1.2", data=BytesIO(b"payload"))
        assert await store.exists(meta.ref) is True


class TestSaveFailureCleanup:
    @pytest.mark.asyncio
    async def test_replace_failure_wraps_original_and_removes_temp(
        self,
        store: LocalSnapshotStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_error = OSError("disk full on replace")

        def _boom(src: object, dst: object) -> None:
            raise original_error

        monkeypatch.setattr("troopai.adk.sandbox.snapshot.local_store.os.replace", _boom)

        with pytest.raises(SnapshotPersistError) as exc_info:
            await store.save(snapshot_id="snap-fail", data=BytesIO(b"payload"))

        # Original failure is preserved as the cause, not masked.
        assert exc_info.value.__cause__ is original_error
        # The temp file written before the failed replace is cleaned up.
        leftovers = [p for p in store.base_path.iterdir() if ".tmp." in p.name]
        assert len(leftovers) == 0

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_mask_original(
        self,
        store: LocalSnapshotStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_error = OSError("disk full on replace")

        def _boom(src: object, dst: object) -> None:
            raise original_error

        def _unlink_boom(self: Path, *, missing_ok: bool = False) -> None:
            raise OSError("permission denied removing temp file")

        monkeypatch.setattr("troopai.adk.sandbox.snapshot.local_store.os.replace", _boom)
        monkeypatch.setattr(Path, "unlink", _unlink_boom)

        # A secondary OSError during cleanup must be swallowed: callers still
        # see SnapshotPersistError carrying the original write failure.
        with pytest.raises(SnapshotPersistError) as exc_info:
            await store.save(snapshot_id="snap-fail", data=BytesIO(b"payload"))
        assert exc_info.value.__cause__ is original_error
