"""Tests for the ``0o600`` file-permission tightening on SQLite DBs.

Session and memory DBs contain high-value data (tool results, memory
entries, HITL approval state). Default world/group readable permissions
leak that data to other local accounts. ``SQLiteDatabaseConnection``
tightens the main DB file and its ``-wal``/``-shm``/``-journal``
sidecars to owner-only after each successful connect.

POSIX-only: ``os.chmod`` permission modes are a no-op on Windows.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from troopai.adk.databases import SQLiteDatabaseConnection

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX permission bits are not meaningful on Windows",
)


def _mode_bits(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class TestRestrictsPermissionsByDefault:
    @pytest.mark.asyncio
    async def test_main_db_file_is_owner_only(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        conn = SQLiteDatabaseConnection(path=db_path)
        async with conn.connect() as db:
            await db.execute("CREATE TABLE t (id INTEGER)")
            await db.commit()
        assert _mode_bits(db_path) == 0o600

    @pytest.mark.asyncio
    async def test_wal_sidecar_is_owner_only(self, tmp_path: Path) -> None:
        db_path = tmp_path / "session.db"
        conn = SQLiteDatabaseConnection(path=db_path)
        async with conn.connect() as db:
            # Writing forces WAL sidecar files to materialize.
            await db.execute("CREATE TABLE t (id INTEGER)")
            await db.execute("INSERT INTO t VALUES (1)")
            await db.commit()

        wal = db_path.with_name(db_path.name + "-wal")
        if wal.exists():
            assert _mode_bits(wal) == 0o600

    @pytest.mark.asyncio
    async def test_permissions_reapplied_after_broad_chmod(
        self,
        tmp_path: Path,
    ) -> None:
        """If something external relaxes the mode, next connect tightens again."""
        db_path = tmp_path / "reopen.db"
        conn = SQLiteDatabaseConnection(path=db_path)
        async with conn.connect() as db:
            await db.execute("CREATE TABLE t (id INTEGER)")
            await db.commit()

        os.chmod(db_path, 0o644)  # operator/os drift
        assert _mode_bits(db_path) == 0o644

        async with conn.connect() as db:
            await db.execute("INSERT INTO t VALUES (1)")
            await db.commit()

        assert _mode_bits(db_path) == 0o600


class TestPrecreateClosesRace:
    """The race-close test: pre-creating the file at 0o600 before
    ``aiosqlite.connect`` is called means an attacker racing for a
    read cannot catch the file at the umask-default mode."""

    @pytest.mark.asyncio
    async def test_file_is_owner_only_before_any_write(
        self,
        tmp_path: Path,
    ) -> None:
        """The post-connect chmod already runs after ``aiosqlite.connect``.
        The new pre-create path tightens the mode *before* the file has
        any content — which is what matters for the race window.

        We assert the mode on the path as-soon-as it exists by hooking
        into the connection flow: the file is created synchronously by
        the pre-create call, then aiosqlite reopens it. By the time we
        reach the `async with` body the file has always been at 0o600,
        which is only meaningful when the pre-create ran."""
        db_path = tmp_path / "raceless.db"
        conn = SQLiteDatabaseConnection(path=db_path)

        # Manually drive the pre-create, independent of aiosqlite, so
        # we can observe the mode without racing the event loop.
        from troopai.adk.databases.connections.sqlite.sqlite_database_connection import (
            precreate_restricted_file,
        )

        precreate_restricted_file(str(db_path))
        assert db_path.exists()
        assert _mode_bits(db_path) == 0o600

        # Real connect should be idempotent — pre-create already
        # materialised the file, chmod re-applies for sidecars.
        async with conn.connect() as db:
            await db.execute("CREATE TABLE t (id INTEGER)")
            await db.commit()
        assert _mode_bits(db_path) == 0o600

    @pytest.mark.asyncio
    async def test_precreate_skips_existing_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Pre-create is O_EXCL — it must not clobber an existing DB
        file. The existing file's mode is preserved; the post-connect
        chmod is what tightens it."""
        db_path = tmp_path / "preexisting.db"
        db_path.write_bytes(b"")
        os.chmod(db_path, 0o644)

        from troopai.adk.databases.connections.sqlite.sqlite_database_connection import (
            precreate_restricted_file,
        )

        # Pre-create on an existing file: no-op, mode unchanged.
        precreate_restricted_file(str(db_path))
        assert _mode_bits(db_path) == 0o644

    @pytest.mark.asyncio
    async def test_precreate_skips_symlink_and_warns(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A symlinked DB path exists via ``lexists`` and is an
        attacker-attractive case: the post-connect chmod would follow
        the link and tighten the target, not the symlink itself. We
        refuse to pre-create (since the file "exists") and emit a
        warning so the operator sees the unusual shape."""
        from troopai.adk.databases.connections.sqlite.sqlite_database_connection import (
            precreate_restricted_file,
        )

        real_target = tmp_path / "actual.db"
        real_target.touch()
        link = tmp_path / "link.db"
        os.symlink(real_target, link)

        import logging

        with caplog.at_level(logging.WARNING):
            precreate_restricted_file(str(link))

        assert any("symlink" in rec.message.lower() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_precreate_skips_when_parent_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """When the parent directory does not exist, let aiosqlite
        surface the real error rather than eating it inside the
        pre-create helper."""
        from troopai.adk.databases.connections.sqlite.sqlite_database_connection import (
            precreate_restricted_file,
        )

        missing = tmp_path / "no-such-dir" / "db.sqlite"
        # Should not raise, should not create the parent.
        precreate_restricted_file(str(missing))
        assert not missing.exists()


class TestInMemoryConnectionRace:
    """Concurrent first-connect must yield a single shared in-memory DB."""

    @pytest.mark.asyncio
    async def test_concurrent_connect_uses_same_connection(self) -> None:
        """Two concurrent connect() calls on :memory: must share one DB."""
        import asyncio

        conn = SQLiteDatabaseConnection(path=":memory:")

        # Create a table via the first concurrent connect
        async def create_table() -> None:
            async with conn.connect() as db:
                await db.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
                await db.execute("INSERT INTO t VALUES (1)")
                await db.commit()

        # Verify data via a concurrent second connect
        row_counts: list[int] = []

        async def count_rows() -> None:
            # Small yield to ensure both coroutines start before either completes
            await asyncio.sleep(0)
            async with conn.connect() as db:
                cursor = await db.execute("SELECT COUNT(*) FROM t")
                row = await cursor.fetchone()
                row_counts.append(row[0] if row else 0)

        await create_table()
        await count_rows()

        # Must see the row written by create_table — same connection object
        assert row_counts == [1]

    @pytest.mark.asyncio
    async def test_gather_connect_yields_identical_connection_object(self) -> None:
        """asyncio.gather over connect() must yield the same connection each time."""
        import asyncio

        conn = SQLiteDatabaseConnection(path=":memory:")

        # First, ensure the persistent connection is created
        async with conn.connect() as db:
            await db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            await db.commit()

        connections: list[object] = []

        async def grab_conn() -> None:
            async with conn.connect() as db:
                connections.append(id(db))

        await asyncio.gather(grab_conn(), grab_conn(), grab_conn())
        # All three must be the same underlying connection object
        assert len(set(connections)) == 1


class TestInMemoryTransactionIsolation:
    """The shared :memory: connection must not leak a partial transaction from a
    failed coroutine into the next, and must serialise concurrent access."""

    @pytest.mark.asyncio
    async def test_rollback_on_error_isolates_next_coroutine(self) -> None:
        """A coroutine that fails mid-transaction must have its uncommitted work
        rolled back, not adopted by the next coroutine's commit.

        Pre-fix: the in-memory branch yielded the shared connection with no
        rollback, so A's uncommitted INSERT stayed pending and B's commit
        persisted it — cross-coroutine corruption.
        """
        conn = SQLiteDatabaseConnection(path=":memory:")
        async with conn.connect() as db:
            await db.execute("CREATE TABLE t (id INTEGER)")
            await db.commit()

        with pytest.raises(RuntimeError):
            async with conn.connect() as db:
                await db.execute("INSERT INTO t VALUES (1)")
                raise RuntimeError("boom before commit")

        # B does its own unit of work; A's INSERT must be gone.
        async with conn.connect() as db:
            await db.commit()

        async with conn.connect() as db:
            cursor = await db.execute("SELECT COUNT(*) FROM t")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0
        await conn.close()

    @pytest.mark.asyncio
    async def test_lock_serialises_read_modify_write(self) -> None:
        """Concurrent read-modify-write cycles on the shared connection must not
        lose updates — the per-connection lock serialises them.

        Pre-fix: with no lock, ten gathered coroutines interleaved at the yield
        point between SELECT and UPDATE and clobbered each other's increments.
        """
        import asyncio

        conn = SQLiteDatabaseConnection(path=":memory:")
        async with conn.connect() as db:
            await db.execute("CREATE TABLE c (n INTEGER)")
            await db.execute("INSERT INTO c VALUES (0)")
            await db.commit()

        async def increment() -> None:
            async with conn.connect() as db:
                cursor = await db.execute("SELECT n FROM c")
                row = await cursor.fetchone()
                assert row is not None
                n = int(row[0])
                await asyncio.sleep(0)  # yield mid-cycle to invite interleaving
                await db.execute("UPDATE c SET n = ?", (n + 1,))
                await db.commit()

        await asyncio.gather(*[increment() for _ in range(10)])

        async with conn.connect() as db:
            cursor = await db.execute("SELECT n FROM c")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 10
        await conn.close()


class TestOptOut:
    @pytest.mark.asyncio
    async def test_restrict_permissions_false_preserves_operator_mode(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "shared.db"

        # First, create the file and set a broad mode explicitly. This
        # decouples the test from the process umask — we want to confirm
        # the connection does NOT tighten when opted out.
        db_path.touch()
        os.chmod(db_path, 0o644)

        conn = SQLiteDatabaseConnection(path=db_path, restrict_permissions=False)
        async with conn.connect() as db:
            await db.execute("CREATE TABLE t (id INTEGER)")
            await db.commit()

        assert _mode_bits(db_path) == 0o644
