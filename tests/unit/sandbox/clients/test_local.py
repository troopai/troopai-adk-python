"""Tests for LocalSubprocessSandboxClient (P18)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    SandboxNetworkPolicyViolation,
    WorkspaceReadNotFoundError,
)
from troopai.adk.sandbox.clients.local import (
    LocalSandboxClientOptions,
    LocalSubprocessSandboxClient,
)
from troopai.adk.types.sandbox.network import NetworkPolicy


@pytest.fixture
def client() -> LocalSubprocessSandboxClient:
    return LocalSubprocessSandboxClient(warn_banner=False)


class TestCreateAndLifecycle:
    @pytest.mark.asyncio
    async def test_create_with_default_options(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            assert await session.running() is True
        assert await session.running() is False

    @pytest.mark.asyncio
    async def test_create_with_explicit_working_directory(
        self,
        client: LocalSubprocessSandboxClient,
        tmp_path: Path,
    ) -> None:
        session = await client.create(
            options=LocalSandboxClientOptions(working_directory=str(tmp_path)),
        )
        async with session:
            result = await session.run("pwd")
            assert tmp_path.name.encode() in result.stdout

    @pytest.mark.asyncio
    async def test_warn_banner_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        caplog.set_level(logging.WARNING)
        LocalSubprocessSandboxClient()
        assert any("NO isolation" in r.message for r in caplog.records)


class TestNetworkPolicy:
    @pytest.mark.asyncio
    async def test_deny_default_rejected_at_start(self) -> None:
        c = LocalSubprocessSandboxClient(
            network_policy=NetworkPolicy(deny_default=True),
            warn_banner=False,
        )
        session = await c.create(options=LocalSandboxClientOptions())
        with pytest.raises(SandboxNetworkPolicyViolation, match="cannot enforce"):
            await session.start()


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_basic_run(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            result = await session.run("echo", "hello", shell=False)
            assert b"hello" in result.stdout
            assert result.exit_code == 0
            assert result.ok is True

    @pytest.mark.asyncio
    async def test_non_zero_exit_surfaced(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            result = await session.run("false", shell=False)
            assert result.exit_code != 0
            assert result.ok is False

    @pytest.mark.asyncio
    async def test_timeout_raises_exec_timeout(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            with pytest.raises(ExecTimeoutError):
                await session.run("sleep", "5", shell=False, timeout=0.1)

    @pytest.mark.asyncio
    async def test_shell_pipeline(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            result = await session.run("echo hi | tr a-z A-Z", shell=True)
            assert b"HI" in result.stdout

    @pytest.mark.asyncio
    async def test_default_env_propagates(self) -> None:
        c = LocalSubprocessSandboxClient(warn_banner=False)
        opts = LocalSandboxClientOptions(default_env={"TROOPAI_TEST": "yes"})
        session = await c.create(options=opts)
        async with session:
            result = await session.run("echo $TROOPAI_TEST", shell=True)
            assert b"yes" in result.stdout


class TestFileOperations:
    @pytest.mark.asyncio
    async def test_write_then_read(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            await session.write("greet.txt", BytesIO(b"hi there"))
            stream = await session.read("greet.txt")
            try:
                content = stream.read()
            finally:
                stream.close()
            assert content == b"hi there"

    @pytest.mark.asyncio
    async def test_read_missing_raises(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            with pytest.raises(WorkspaceReadNotFoundError):
                await session.read("does-not-exist")

    @pytest.mark.asyncio
    async def test_mkdir_then_ls(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            await session.mkdir("sub/inner", parents=True)
            entries = await session.ls("sub")
            assert any(e.name == "inner" and e.is_directory for e in entries)

    @pytest.mark.asyncio
    async def test_rm_requires_recursive_for_dirs(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            await session.mkdir("d")
            with pytest.raises(IsADirectoryError):
                await session.rm("d")
            await session.rm("d", recursive=True)


class TestWorkspacePersistence:
    @pytest.mark.asyncio
    async def test_persist_then_hydrate_round_trip(self, client: LocalSubprocessSandboxClient) -> None:
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            await session.write("a.txt", BytesIO(b"alpha"))
            await session.mkdir("subdir")
            await session.write("subdir/b.txt", BytesIO(b"beta"))
            archive = await session.persist_workspace()
            # Wipe and hydrate from the archive.
            await session.rm("a.txt")
            await session.rm("subdir", recursive=True)
            assert await session.ls(".") == []
            await session.hydrate_workspace(archive)
            entries = await session.ls(".")
            names = {e.name for e in entries}
            assert "a.txt" in names
            assert "subdir" in names


class TestSessionStateRoundTrip:
    @pytest.mark.asyncio
    async def test_serialize_deserialize(self, client: LocalSubprocessSandboxClient) -> None:
        from troopai.adk.types.sandbox.session_state import SandboxSessionState

        state = SandboxSessionState(backend_id="unix_local")
        payload = client.serialize_session_state(state)
        restored = client.deserialize_session_state(payload)
        assert restored.backend_id == "unix_local"

    @pytest.mark.asyncio
    async def test_resume_rejects_wrong_backend(self, client: LocalSubprocessSandboxClient) -> None:
        from troopai.adk.types.sandbox.session_state import SandboxSessionState

        state = SandboxSessionState(backend_id="docker")
        with pytest.raises(ValueError, match="cannot resume"):
            await client.resume(state)


class TestCancellationSafety:
    """Regression tests for subprocess orphan on CancelledError."""

    async def test_cancelled_run_does_not_orphan_subprocess(self) -> None:
        """CancelledError during run() must kill + reap the subprocess."""
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            task = asyncio.create_task(session.run("sleep", "30", shell=False))
            # Allow the subprocess to actually start before cancelling.
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # If the subprocess was reaped, returncode is no longer None and
            # it is not lingering in the process table as a zombie.
            # We verify indirectly: another short run can succeed (thread pool
            # and event loop are still healthy).
            result = await session.run("echo", "alive", shell=False)
            assert result.ok is True


class TestPathTraversalPrevention:
    """Regression tests for _resolve_inside_workspace path-escape fixes."""

    async def test_absolute_path_raises_permission_error(self) -> None:
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            # An absolute path that is NOT under the workspace must be blocked.
            # (The session's working directory is a temp dir, so /tmp/secret
            # is outside it unless the temp dir IS /tmp exactly — use /etc/hosts
            # which is always outside any temp workspace.)
            with pytest.raises(PermissionError, match="escapes the sandbox workspace"):
                await session.read("/etc/hosts")

    async def test_relative_traversal_raises_permission_error(self) -> None:
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            with pytest.raises(PermissionError, match="escapes the sandbox workspace"):
                await session.read("../../etc/passwd")

    async def test_valid_relative_path_works(self) -> None:
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions())
        async with session:
            await session.write("safe.txt", BytesIO(b"ok"))
            stream = await session.read("safe.txt")
            try:
                assert stream.read() == b"ok"
            finally:
                stream.close()


class TestTarExtractFilter:
    """Regression tests for tarfile.extractall path-traversal via filter='data'."""

    def _make_traversal_tar(self) -> BytesIO:
        """Build a tar with a path-traversal member ``../../evil.txt``."""
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            content = b"hostile content"
            info = tarfile.TarInfo(name="../../evil.txt")
            info.size = len(content)
            tar.addfile(info, BytesIO(content))
        buf.seek(0)
        return buf

    async def test_extract_rejects_traversal_member(self, tmp_path: Path) -> None:
        """extract() must not write traversal members outside the workspace."""
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions(working_directory=str(tmp_path)))
        async with session:
            archive = self._make_traversal_tar()
            # filter='data' raises FilterError for members with .. components;
            # the exact exception class varies by Python version but it is a
            # subclass of Exception.
            with contextlib.suppress(Exception):
                await session.extract(".", archive)
            # The evil file must NOT have been written outside the workspace.
            evil_path = tmp_path.parent / "evil.txt"
            assert not evil_path.exists(), "traversal member must not escape the workspace"

    async def test_hydrate_workspace_rejects_traversal_member(self, tmp_path: Path) -> None:
        """hydrate_workspace() must not write traversal members outside the workspace."""
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions(working_directory=str(tmp_path)))
        async with session:
            archive = self._make_traversal_tar()
            with contextlib.suppress(Exception):
                await session.hydrate_workspace(archive)
            evil_path = tmp_path.parent / "evil.txt"
            assert not evil_path.exists(), "traversal member must not escape the workspace"


class TestArchiveResourceLimits:
    """extract()/hydrate_workspace() validate archives before extraction."""

    @staticmethod
    def _oversize_tar(member_count: int) -> BytesIO:
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for i in range(member_count):
                payload = b"x"
                info = tarfile.TarInfo(name=f"f{i}.txt")
                info.size = len(payload)
                tar.addfile(info, BytesIO(payload))
        buf.seek(0)
        return buf

    async def test_extract_rejects_archive_over_member_limit(self, tmp_path: Path) -> None:
        """A tar-bomb-shaped archive is rejected before any file lands."""
        from troopai.adk.sandbox.session import ArchiveResourceLimitError, SandboxArchiveLimits

        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(
            options=LocalSandboxClientOptions(
                working_directory=str(tmp_path),
                archive_limits=SandboxArchiveLimits(max_members=4),
            )
        )
        async with session:
            with pytest.raises(ArchiveResourceLimitError):
                await session.extract(".", self._oversize_tar(member_count=6))
            assert not (tmp_path / "f0.txt").exists(), "no member may be extracted from a rejected archive"

    async def test_hydrate_rejects_archive_over_member_limit(self, tmp_path: Path) -> None:
        from troopai.adk.sandbox.session import ArchiveResourceLimitError, SandboxArchiveLimits

        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(
            options=LocalSandboxClientOptions(
                working_directory=str(tmp_path),
                archive_limits=SandboxArchiveLimits(max_members=4),
            )
        )
        async with session:
            with pytest.raises(ArchiveResourceLimitError):
                await session.hydrate_workspace(self._oversize_tar(member_count=6))

    async def test_extract_within_limits_succeeds(self, tmp_path: Path) -> None:
        from troopai.adk.sandbox.session import SandboxArchiveLimits

        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(
            options=LocalSandboxClientOptions(
                working_directory=str(tmp_path),
                archive_limits=SandboxArchiveLimits(max_members=10),
            )
        )
        async with session:
            await session.extract(".", self._oversize_tar(member_count=6))
            assert (tmp_path / "f0.txt").exists()
            assert (tmp_path / "f5.txt").exists()


class TestPersistHydrateSymlinkRoundTrip:
    """persist_workspace -> hydrate_workspace must round-trip symlinks.

    Regression: tar.add stores symlink members verbatim, but the
    hydrate validation rejected them, so a snapshot the framework
    itself produced could not be restored.
    """

    async def test_symlink_round_trips(self, tmp_path: Path) -> None:
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions(working_directory=str(tmp_path)))
        async with session:
            await session.write("real.txt", BytesIO(b"payload"))
            os.symlink("real.txt", tmp_path / "link.txt")
            archive = await session.persist_workspace()
            await session.hydrate_workspace(archive)
            link = tmp_path / "link.txt"
            assert link.is_symlink()
            assert os.readlink(link) == "real.txt"
            assert (tmp_path / "real.txt").read_bytes() == b"payload"

    async def test_extract_still_rejects_symlink_members(self, tmp_path: Path) -> None:
        """extract() (untrusted archives) must keep rejecting symlink members."""
        from troopai.adk.sandbox.session import UnsafeTarMemberError

        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions(working_directory=str(tmp_path)))
        async with session:
            buf = BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name="link")
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                tar.addfile(info)
            buf.seek(0)
            with pytest.raises(UnsafeTarMemberError):
                await session.extract(".", buf)


class TestHydrateValidatesBeforeWipe:
    """hydrate_workspace must validate the archive BEFORE wiping the workspace.

    Regression: the wipe ran unconditionally before validation, so a
    rejected (corrupt / hostile / oversized) archive left the live
    workspace empty.
    """

    @staticmethod
    def _traversal_tar() -> BytesIO:
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            content = b"hostile"
            info = tarfile.TarInfo(name="../../evil.txt")
            info.size = len(content)
            tar.addfile(info, BytesIO(content))
        buf.seek(0)
        return buf

    async def test_rejected_archive_leaves_workspace_intact(self, tmp_path: Path) -> None:
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions(working_directory=str(tmp_path)))
        async with session:
            await session.write("keep.txt", BytesIO(b"precious"))
            with contextlib.suppress(Exception):
                await session.hydrate_workspace(self._traversal_tar())
            # The pre-existing file must survive a rejected hydrate.
            assert (tmp_path / "keep.txt").read_bytes() == b"precious"

    async def test_oversize_archive_leaves_workspace_intact(self, tmp_path: Path) -> None:
        from troopai.adk.sandbox.session import ArchiveResourceLimitError, SandboxArchiveLimits

        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(
            options=LocalSandboxClientOptions(
                working_directory=str(tmp_path),
                archive_limits=SandboxArchiveLimits(max_members=2),
            )
        )
        async with session:
            await session.write("keep.txt", BytesIO(b"precious"))
            buf = BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                for i in range(5):
                    payload = b"x"
                    info = tarfile.TarInfo(name=f"f{i}.txt")
                    info.size = len(payload)
                    tar.addfile(info, BytesIO(payload))
            buf.seek(0)
            with pytest.raises(ArchiveResourceLimitError):
                await session.hydrate_workspace(buf)
            assert (tmp_path / "keep.txt").read_bytes() == b"precious"


class TestLsDanglingSymlink:
    """ls() must not crash on a broken symlink (sandboxed code can create one)."""

    async def test_ls_dangling_symlink_returns_entry(self, tmp_path: Path) -> None:
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions(working_directory=str(tmp_path)))
        async with session:
            os.symlink("/nonexistent-target-xyz", tmp_path / "dangling")
            entries = await session.ls(".")
            names = {e.name for e in entries}
            assert "dangling" in names
            dangling = next(e for e in entries if e.name == "dangling")
            assert dangling.is_directory is False
            assert dangling.size_bytes >= 0


class TestApplyPatchPathWithSpaces:
    """apply_patch must apply through exec (no shell word-split) so a
    working_directory containing a space does not break the temp-file path."""

    async def test_apply_patch_in_directory_with_space(self, tmp_path: Path) -> None:
        if shutil.which("patch") is None:
            pytest.skip("`patch` binary not available")
        workdir = tmp_path / "my project"
        workdir.mkdir()
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(options=LocalSandboxClientOptions(working_directory=str(workdir)))
        async with session:
            await session.write("hello.txt", BytesIO(b"old\n"))
            diff = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n-old\n+new\n"
            summary = await session.apply_patch(diff)
            assert "OK" in summary
            stream = await session.read("hello.txt")
            try:
                assert stream.read() == b"new\n"
            finally:
                stream.close()
