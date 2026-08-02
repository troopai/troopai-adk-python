"""Tests for materialization.metadata — chmod/chgrp via session.run."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from troopai.adk.exceptions.exceptions import ExecNonZeroError
from troopai.adk.sandbox.session.materialization.metadata import apply_entry_metadata
from troopai.adk.types.sandbox.entries import File
from troopai.adk.types.sandbox.permissions import Group, User


def _recording_session(*, run_exits: dict[str, int] | None = None) -> Any:
    exits = run_exits or {}

    class _Rec:
        def __init__(self) -> None:
            self.runs: list[tuple[str, ...]] = []

        async def run(
            self, *command: object, timeout: float | None = None, shell: bool = True, user: object = None
        ) -> Any:
            argv = tuple(str(c) for c in command)
            self.runs.append(argv)
            return SimpleNamespace(exit_code=exits.get(argv[0], 0), stdout=b"", stderr=b"boom")

    return _Rec()


class TestApplyEntryMetadata:
    async def test_chmod_only_when_no_group(self) -> None:
        session = _recording_session()
        await apply_entry_metadata(session, "f.txt", File(content=b""))
        # Permissions() default rw-r--r-- == 0o644 == "644".
        assert session.runs == [("chmod", "644", "f.txt")]

    async def test_chmod_then_chgrp_when_group(self) -> None:
        session = _recording_session()
        await apply_entry_metadata(session, "f.txt", File(content=b"", group=Group(name="staff")))
        assert session.runs == [
            ("chmod", "644", "f.txt"),
            ("chgrp", "staff", "f.txt"),
        ]

    async def test_chgrp_with_user_applied_as_same_name_group(self) -> None:
        # `entry.group` may be a User; it is applied via `chgrp <user.name>`
        # (the user's same-name primary group) — the upstream-faithful
        # contract pinned here so a future refactor cannot silently change it.
        session = _recording_session()
        await apply_entry_metadata(session, "f.txt", File(content=b"", group=User(name="devuser")))
        assert session.runs == [
            ("chmod", "644", "f.txt"),
            ("chgrp", "devuser", "f.txt"),
        ]

    async def test_chmod_nonzero_raises(self) -> None:
        session = _recording_session(run_exits={"chmod": 1})
        with pytest.raises(ExecNonZeroError, match="chmod 644 'f.txt' failed"):
            await apply_entry_metadata(session, "f.txt", File(content=b""))

    async def test_chgrp_nonzero_raises(self) -> None:
        session = _recording_session(run_exits={"chgrp": 2})
        with pytest.raises(ExecNonZeroError, match="chgrp 'staff' 'f.txt' failed"):
            await apply_entry_metadata(session, "f.txt", File(content=b"", group=Group(name="staff")))
