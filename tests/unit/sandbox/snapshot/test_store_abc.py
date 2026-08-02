"""Tests for the ``SnapshotStore`` ABC contract.

``SnapshotStore`` is subclassed by the S3 / GCS / Local stores
(tested elsewhere), but the ABC itself was imported and asserted
by no test. These pin that it is abstract, exposes exactly its
five-method async contract, and is implementable.
"""

from __future__ import annotations

import abc
import inspect
from io import IOBase
from typing import override

import pytest

from troopai.adk.sandbox.snapshot import store as store_mod
from troopai.adk.sandbox.snapshot.store import SnapshotStore
from troopai.adk.types.sandbox.snapshot import SnapshotMetadata, SnapshotRef


class TestSnapshotStoreABC:
    def test_is_abstract_and_not_instantiable(self) -> None:
        assert issubclass(SnapshotStore, abc.ABC)
        assert inspect.isabstract(SnapshotStore)
        with pytest.raises(TypeError):
            SnapshotStore()  # type: ignore[abstract]

    def test_contract_is_the_five_async_methods(self) -> None:
        assert SnapshotStore.__abstractmethods__ == frozenset(
            {"save", "load", "delete", "list", "exists"},
        )
        for name in ("save", "load", "delete", "list", "exists"):
            assert inspect.iscoroutinefunction(getattr(SnapshotStore, name))

    def test_all_exports_only_the_abc(self) -> None:
        assert store_mod.__all__ == ["SnapshotStore"]

    def test_a_concrete_subclass_is_instantiable(self) -> None:
        # Overriding all five abstract methods removes the only
        # instantiation barrier — proves the contract is satisfiable.
        class _Concrete(SnapshotStore):
            @override
            async def save(
                self,
                *,
                snapshot_id: str,
                data: IOBase,
                manifest_hash: str | None = None,
            ) -> SnapshotMetadata:
                del data
                return SnapshotMetadata(
                    ref=SnapshotRef(snapshot_id=snapshot_id, store_uri="mem://test"),
                    created_at_iso="1970-01-01T00:00:00Z",
                    size_bytes=0,
                    manifest_hash=manifest_hash,
                )

            @override
            async def load(self, ref: SnapshotRef) -> IOBase:
                del ref
                return IOBase()

            @override
            async def delete(self, ref: SnapshotRef) -> None:
                del ref

            @override
            async def list(self, prefix: str | None = None) -> list[SnapshotMetadata]:
                del prefix
                return []

            @override
            async def exists(self, ref: SnapshotRef) -> bool:
                del ref
                return False

        assert isinstance(_Concrete(), SnapshotStore)
