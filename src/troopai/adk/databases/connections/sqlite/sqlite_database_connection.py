"""Async SQLite connection via aiosqlite.

:class:`SQLiteDatabaseConnection` manages SQLite connections for both
file-based and in-memory databases.  For in-memory databases, it holds
a persistent connection (since in-memory DBs are destroyed when the
last connection closes).  For file-based databases, each ``connect()``
call opens a fresh connection.

See https://aiosqlite.omnilib.dev/ for aiosqlite documentation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

PRAGMA_FOREIGN_KEYS = "PRAGMA foreign_keys = ON"
PRAGMA_WAL = "PRAGMA journal_mode=WAL"

# Owner-only (0o600) default for DB files containing session state, memory
# entries, and auth tokens. Sidecars (``-wal``/``-shm``) inherit the same
# mode. Operators who need group/world access can opt out with
# ``restrict_permissions=False`` on construction.
_RESTRICTED_MODE = 0o600
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


class SQLiteDatabaseConnection:
    """Async SQLite connection manager.

    Handles both file-based and in-memory databases transparently.
    The ``connect()`` async context manager yields an
    ``aiosqlite.Connection`` with WAL mode and foreign keys enabled.

    For in-memory databases (``path=":memory:"``), a persistent
    connection is kept alive so data survives across ``connect()``
    calls.  For file-based databases, a new connection is opened
    each time.

    Example::

        db = SQLiteDatabaseConnection(path="data.db")
        async with db.connect() as conn:
            await conn.execute("CREATE TABLE ...")
            await conn.commit()
        await db.close()
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        restrict_permissions: bool = True,
    ) -> None:
        """Initialize the connection manager.

        Args:
            path: Path to the SQLite database file, or ``":memory:"``
                for an in-memory database.
            restrict_permissions: When ``True`` (the default), newly
                created database files and their WAL/SHM/journal
                sidecars are set to owner-only permissions (``0o600``).
                Pass ``False`` to leave file-mode management to the
                process umask.
        """
        self._db_path = str(path)
        self._in_memory = self._db_path == ":memory:"
        self._persistent_conn: aiosqlite.Connection | None = None
        self._restrict_permissions = restrict_permissions
        self._init_lock = asyncio.Lock()
        # Serialises access to the single shared in-memory connection so two
        # coroutines cannot interleave statements from different transactions.
        # Unused on the file-based path (each connect() opens its own).
        self._txn_lock = asyncio.Lock()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield an async SQLite connection.

        For file-based databases, opens a new connection per call.
        For in-memory databases, yields the single persistent connection,
        serialised behind a lock so concurrent coroutines cannot interleave
        statements from different transactions; a partial transaction is
        rolled back on error or cancellation so it cannot leak into the next
        borrower of the shared connection.

        Yields:
            An ``aiosqlite.Connection`` with foreign-key enforcement
            enabled.  File-based connections also have WAL journal mode
            activated.
        """
        if self._in_memory:
            conn = await self._ensure_persistent()
            async with self._txn_lock:
                try:
                    yield conn
                except BaseException:
                    # Undo any uncommitted statements so a failed/cancelled
                    # coroutine does not hand its open transaction to the next
                    # one. rollback() is a no-op when nothing is pending.
                    with suppress(Exception):
                        await conn.rollback()
                    raise
        else:
            # Close the file-create permission race *before* aiosqlite
            # sees the path: if the DB file does not yet exist and we
            # want owner-only mode, touch it with 0o600. Without this,
            # ``aiosqlite.connect()`` creates the file at the process
            # umask (often 0o644) and our post-connect chmod leaves a
            # small but real window where anyone on the host can read
            # session/memory state and auth tokens.
            if self._restrict_permissions:
                precreate_restricted_file(self._db_path)
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(PRAGMA_FOREIGN_KEYS)
                await db.execute(PRAGMA_WAL)
                if self._restrict_permissions:
                    # Sidecars (-wal/-shm) are only created after the
                    # first write; re-apply on every connect() so they
                    # inherit owner-only mode as they appear.
                    _restrict_db_permissions(self._db_path)
                yield db

    async def close(self) -> None:
        """Close the persistent connection (in-memory DBs only).

        For file-based databases, this is a no-op — connections are
        closed automatically when the ``connect()`` context exits.
        """
        if self._persistent_conn is not None:
            await self._persistent_conn.close()
            self._persistent_conn = None

    async def _ensure_persistent(self) -> aiosqlite.Connection:
        """Lazily open the persistent in-memory connection.

        ``PRAGMA journal_mode=WAL`` is intentionally omitted on the
        in-memory path: WAL journals write to companion files on
        disk, and ``:memory:`` has no on-disk backing, so the pragma
        is a no-op that only serves to generate log noise.

        Returns:
            The open ``aiosqlite.Connection`` for the in-memory
            database, creating it on the first call.
        """
        async with self._init_lock:
            if self._persistent_conn is None:
                self._persistent_conn = await aiosqlite.connect(":memory:")
                self._persistent_conn.row_factory = aiosqlite.Row
                await self._persistent_conn.execute(PRAGMA_FOREIGN_KEYS)
        return self._persistent_conn


def precreate_restricted_file(path: str) -> None:
    """Atomically create an empty DB file at owner-only mode.

    Closes the race between file creation and ``chmod`` by letting the
    kernel set the mode at ``open(2)`` time instead. If the file already
    exists we leave it alone — the subsequent ``_restrict_db_permissions``
    call still runs on the existing file.

    When the existing path is a symlink we log a warning but still
    skip the pre-create: an attacker with write access to the parent
    directory could pre-place a symlink to redirect SQLite to an
    unexpected target, and the post-connect ``chmod`` will follow the
    link and tighten the *target's* mode, not the symlink itself.
    Operators who intentionally symlink their DB path see only the
    warning; the app still opens as before.

    No-op on platforms where ``os.open`` cannot honour the mode (e.g.
    some Windows filesystems treat POSIX mode as advisory) or when the
    parent directory is missing — in the latter case we let
    ``aiosqlite.connect`` produce the canonical error.

    Args:
        path: Filesystem path at which the SQLite database file should
            be pre-created.  Must be a plain path string, not
            ``":memory:"``.
    """
    if os.path.lexists(path):
        if os.path.islink(path):
            logger.warning(
                "DB path %s is a symlink; pre-create skipped. "
                "Post-connect chmod will tighten the link target's "
                "permissions, not the symlink itself.",
                path,
            )
        return
    parent = os.path.dirname(path)
    if len(parent) > 0 and not os.path.isdir(parent):
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, _RESTRICTED_MODE)
    except FileExistsError:
        # Benign race: another connect() materialised the file
        # between the lexists check above and the open. The
        # post-connect chmod will still run on the winning file.
        return
    except OSError as exc:
        # PermissionError / ENOSPC / EROFS / bad-path — surface at
        # warning so the operator sees it when aiosqlite.connect
        # produces its own (usually less specific) error downstream.
        logger.warning(
            "Pre-create of restricted DB file %s failed: %s",
            path,
            exc,
        )
        return
    os.close(fd)


def _restrict_db_permissions(path: str) -> None:
    """Tighten the DB file and its WAL/SHM/journal sidecars to ``0o600``.

    Called after every successful ``aiosqlite.connect()`` so newly
    created sidecar files are brought back to owner-only. No-op on
    platforms where ``os.chmod`` is unsupported (e.g. Windows returns
    ``NotImplementedError`` for POSIX modes on some volumes).

    Args:
        path: Filesystem path of the primary SQLite database file.
            Sidecar paths are derived by appending ``-wal``, ``-shm``,
            and ``-journal`` suffixes.
    """
    for suffix in ("", *_SQLITE_SIDECAR_SUFFIXES):
        candidate = Path(path + suffix)
        if not candidate.exists():
            continue
        try:
            os.chmod(candidate, _RESTRICTED_MODE)
        except (OSError, NotImplementedError) as exc:
            logger.warning(
                "Could not restrict permissions on %s: %s",
                candidate,
                exc,
            )
