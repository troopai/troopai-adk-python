"""Tests for DockerSandboxSession + DockerSandboxClient (TDK.1-TDK.12).

Uses unittest.mock to drive the docker SDK so the suite runs without
an actual Docker daemon.
"""

from __future__ import annotations

import sys
import tarfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    SandboxStartFailed,
    WorkspaceReadNotFoundError,
)


def _directory_tar_bytes() -> bytes:
    """A tar of a directory: a DIRTYPE entry plus a file inside it.

    Mirrors what ``get_archive`` streams for a directory path — more than
    one member — so a single-file ``read`` must reject it rather than
    silently returning the file found inside.
    """
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        dir_info = tarfile.TarInfo(name="mydir")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)
        payload = b"leaked"
        file_info = tarfile.TarInfo(name="mydir/secret.txt")
        file_info.size = len(payload)
        tar.addfile(file_info, BytesIO(payload))
    return buf.getvalue()


def _single_file_tar_bytes(payload: bytes) -> bytes:
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="foo.txt")
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))
    return buf.getvalue()


def _mock_container(
    *,
    container_id: str = "abc123def456",
    status: str = "running",
) -> MagicMock:
    container = MagicMock()
    container.id = container_id
    container.status = status
    container.attrs = {
        "NetworkSettings": {
            "Ports": {"80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}]},
            "IPAddress": "172.17.0.2",
        },
    }
    container.start = MagicMock()
    container.stop = MagicMock()
    container.remove = MagicMock()
    container.reload = MagicMock()
    # exec_run returns an object with exit_code + output (tuple when demux=True).
    exec_result = MagicMock()
    exec_result.exit_code = 0
    exec_result.output = (b"hello\n", b"")
    container.exec_run = MagicMock(return_value=exec_result)
    # File ops mocking.
    container.get_archive = MagicMock(return_value=(iter([b""]), {}))
    container.put_archive = MagicMock()
    return container


class TestDockerSessionLifecycle:
    @pytest.mark.asyncio
    async def test_start_polls_until_running(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container(status="created")

        # Simulate container reaching running after one reload.
        def _reload() -> None:
            container.status = "running"

        container.reload.side_effect = _reload
        session = DockerSandboxSession(container=container)
        # mkdir uses exec_run; just ensure no exception.
        await session.start()
        assert session._started is True
        container.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_re_idempotent(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        session._started = True  # simulate already-started
        await session.start()
        container.start.assert_not_called()

    async def test_start_installs_runtime_helpers(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        with patch(
            "troopai.adk.sandbox.session.runtime_helpers.install_runtime_helpers",
            new_callable=AsyncMock,
        ) as installer:
            await session.start()
        installer.assert_awaited_once()
        call = installer.call_args
        assert call.args[0] is session
        assert call.kwargs["backend_id"] == "docker"

    @pytest.mark.asyncio
    async def test_aclose_stops_and_removes(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        # Manually mark as started without going through start() so
        # the mkdir call inside start() is skipped.
        session._started = True
        await session.aclose()
        container.stop.assert_called_once()
        container.remove.assert_called_once()


class TestDockerSessionEnvironment:
    """Regression: _environment propagation must use explicit len() check, not 'or None'."""

    @pytest.mark.asyncio
    async def test_empty_environment_passes_none_to_exec(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container, environment={})
        await session.run("echo", "hi", shell=False)
        call_kwargs = container.exec_run.call_args.kwargs
        assert call_kwargs["environment"] is None

    @pytest.mark.asyncio
    async def test_nonempty_environment_passes_dict_to_exec(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        env = {"FOO": "bar", "BAZ": "qux"}
        container = _mock_container()
        session = DockerSandboxSession(container=container, environment=env)
        await session.run("env", shell=False)
        call_kwargs = container.exec_run.call_args.kwargs
        assert call_kwargs["environment"] == env


class TestDockerSessionRun:
    @pytest.mark.asyncio
    async def test_run_returns_exec_result(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        result = await session.run("echo", "hello", shell=True)
        assert result.exit_code == 0
        assert result.stdout == b"hello\n"

    @pytest.mark.asyncio
    async def test_run_with_timeout_raises_on_exceed(self) -> None:
        """Host-side backstop still raises when the image lacks 'timeout'."""
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()

        def _exec(argv: object, **kw: object) -> MagicMock:
            if isinstance(argv, list) and "command -v timeout" in argv:
                return MagicMock(exit_code=127, output=(b"", b""))  # binary absent
            import time

            time.sleep(0.2)
            return MagicMock(exit_code=0, output=(b"", b""))

        container.exec_run.side_effect = _exec
        session = DockerSandboxSession(container=container)
        with pytest.raises(ExecTimeoutError):
            await session.run("sleep", "5", shell=False, timeout=0.05)

    @pytest.mark.asyncio
    async def test_run_timeout_enforced_in_container_when_binary_present(self) -> None:
        """With 'timeout' in the image, the command is wrapped and exit 124 maps
        to ExecTimeoutError without waiting for the padded host backstop."""
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        run_argvs: list[list[str]] = []

        def _exec(argv: object, **kw: object) -> MagicMock:
            if isinstance(argv, list) and "command -v timeout" in argv:
                return MagicMock(exit_code=0, output=(b"/usr/bin/timeout\n", b""))
            assert isinstance(argv, list)
            run_argvs.append(argv)
            return MagicMock(exit_code=124, output=(b"", b""))  # killed by wrapper

        container.exec_run.side_effect = _exec
        session = DockerSandboxSession(container=container)
        with pytest.raises(ExecTimeoutError, match="in-container"):
            await session.run("sleep", "60", shell=False, timeout=2)
        assert run_argvs[0][:2] == ["timeout", "2"], "command must be wrapped with the in-container deadline"
        assert run_argvs[0][2:] == ["sleep", "60"]

    @pytest.mark.asyncio
    async def test_run_wrapped_success_within_deadline(self) -> None:
        """A wrapped command finishing in time returns its result unchanged."""
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()

        def _exec(argv: object, **kw: object) -> MagicMock:
            if isinstance(argv, list) and "command -v timeout" in argv:
                return MagicMock(exit_code=0, output=(b"/usr/bin/timeout\n", b""))
            return MagicMock(exit_code=0, output=(b"done\n", b""))

        container.exec_run.side_effect = _exec
        session = DockerSandboxSession(container=container)
        result = await session.run("echo", "done", shell=False, timeout=5)
        assert result.exit_code == 0
        assert result.stdout == b"done\n"

    @pytest.mark.asyncio
    async def test_timeout_probe_runs_once_per_session(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        probes: list[object] = []

        def _exec(argv: object, **kw: object) -> MagicMock:
            if isinstance(argv, list) and "command -v timeout" in argv:
                probes.append(argv)
                return MagicMock(exit_code=0, output=(b"", b""))
            return MagicMock(exit_code=0, output=(b"", b""))

        container.exec_run.side_effect = _exec
        session = DockerSandboxSession(container=container)
        await session.run("true", shell=False, timeout=1)
        await session.run("true", shell=False, timeout=1)
        assert len(probes) == 1

    @pytest.mark.asyncio
    async def test_timeout_logs_thread_leak_warning(self) -> None:
        """Regression: asyncio.wait_for cancels the asyncio task on timeout but
        the underlying OS thread running exec_run continues running. A WARNING
        must be emitted so operators know the thread leak occurred."""
        import logging

        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target_logger = logging.getLogger("troopai.adk.sandbox.clients.docker.docker_session")
        handler = _Capture()
        target_logger.addHandler(handler)
        try:
            container = _mock_container()

            def _slow(argv: object, **kw: object) -> object:
                if isinstance(argv, list) and "command -v timeout" in argv:
                    return MagicMock(exit_code=127, output=(b"", b""))  # force host-only fallback
                import time as _time

                _time.sleep(0.3)
                return MagicMock(exit_code=0, output=(b"", b""))

            container.exec_run.side_effect = _slow
            session = DockerSandboxSession(container=container)
            with pytest.raises(ExecTimeoutError):
                await session.run("sleep", "5", shell=False, timeout=0.05)
        finally:
            target_logger.removeHandler(handler)

        warning_records = [r for r in records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert any("thread" in r.getMessage().lower() for r in warning_records)


class TestDockerApplyPatch:
    @pytest.mark.asyncio
    async def test_patch_file_removed_when_run_raises(self) -> None:
        """Regression: apply_patch left .troopai_patch.diff when run() raised."""
        from unittest.mock import patch as mock_patch

        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        boom = RuntimeError("exec failed")
        removed_paths: list[str] = []

        async def fake_run(*args: object, **kwargs: object) -> object:
            raise boom

        async def fake_rm(path: object, **kwargs: object) -> None:
            removed_paths.append(str(path))

        async def fake_write(*args: object, **kwargs: object) -> None:
            return None

        with (
            mock_patch.object(session, "run", side_effect=fake_run),
            mock_patch.object(session, "rm", side_effect=fake_rm),
            mock_patch.object(session, "write", side_effect=fake_write),
            pytest.raises(RuntimeError, match="exec failed"),
        ):
            await session.apply_patch("--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-x\n+y\n")

        assert any(".troopai_patch.diff" in p for p in removed_paths), (
            f"Expected .troopai_patch.diff cleanup but rm was called with: {removed_paths}"
        )

    @pytest.mark.asyncio
    async def test_patch_file_removed_on_success(self) -> None:
        """apply_patch must also clean up the temp diff file on success."""
        from unittest.mock import patch as mock_patch

        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        removed_paths: list[str] = []
        fake_result = MagicMock()
        fake_result.exit_code = 0
        fake_result.stderr = b""

        async def fake_run(*args: object, **kwargs: object) -> object:
            return fake_result

        async def fake_rm(path: object, **kwargs: object) -> None:
            removed_paths.append(str(path))

        async def fake_write(*args: object, **kwargs: object) -> None:
            return None

        with (
            mock_patch.object(session, "run", side_effect=fake_run),
            mock_patch.object(session, "rm", side_effect=fake_rm),
            mock_patch.object(session, "write", side_effect=fake_write),
        ):
            await session.apply_patch("--- a/foo\n+++ b/foo\n")

        assert any(".troopai_patch.diff" in p for p in removed_paths)


class TestDockerSessionFileOps:
    @pytest.mark.asyncio
    async def test_resolve_path_relative(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(
            container=container,
            working_directory="/workspace",
        )
        assert session._resolve_path("foo.txt") == "/workspace/foo.txt"
        assert session._resolve_path("/etc/hosts") == "/etc/hosts"

    @pytest.mark.asyncio
    async def test_write_calls_put_archive(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        await session.write("foo.txt", BytesIO(b"hello"))
        container.put_archive.assert_called()

    @pytest.mark.asyncio
    async def test_read_missing_raises(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        container.get_archive.side_effect = Exception("not found")
        session = DockerSandboxSession(container=container)
        with pytest.raises(WorkspaceReadNotFoundError):
            await session.read("missing.txt")

    @pytest.mark.asyncio
    async def test_read_directory_raises_not_a_file(self) -> None:
        """read() on a directory must raise, not return the first file inside.

        Regression: get_archive on a directory yields a multi-member tar; the
        old code filtered to files and returned members[0] — leaking a file
        from inside the directory instead of failing.
        """
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        container.get_archive = MagicMock(return_value=(iter([_directory_tar_bytes()]), {}))
        session = DockerSandboxSession(container=container)
        with pytest.raises(WorkspaceReadNotFoundError, match="not a regular file"):
            await session.read("mydir")

    @pytest.mark.asyncio
    async def test_read_single_file_returns_content(self) -> None:
        """A normal single-file read still returns the file bytes."""
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        container.get_archive = MagicMock(return_value=(iter([_single_file_tar_bytes(b"hello")]), {}))
        session = DockerSandboxSession(container=container)
        result = await session.read("foo.txt")
        assert result.read() == b"hello"


class TestDockerSessionPortResolution:
    @pytest.mark.asyncio
    async def test_resolves_host_mapping(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        session = DockerSandboxSession(container=container)
        endpoint = await session.resolve_exposed_port(80)
        assert endpoint.host == "127.0.0.1"
        assert endpoint.port == 8080

    @pytest.mark.asyncio
    async def test_falls_back_to_container_ip(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        container.attrs["NetworkSettings"]["Ports"] = {}
        session = DockerSandboxSession(container=container)
        endpoint = await session.resolve_exposed_port(8080)
        assert endpoint.host == "172.17.0.2"


def _fake_docker_types() -> tuple[MagicMock, MagicMock]:
    """A fake ``docker`` / ``docker.types`` so the optional SDK need
    not be installed. ``Mount`` faithfully models docker-py's
    dict-subclass shape (keys ``Target``/``Source``/``Type``/
    ``ReadOnly``) so the materialization assertions are real.
    """
    mod = MagicMock()
    types_mod = MagicMock()

    def _mount(**kw: object) -> dict[str, object]:
        return {
            "Target": kw["target"],
            "Source": kw["source"],
            "Type": kw["type"],
            "ReadOnly": kw["read_only"],
        }

    types_mod.Mount = _mount
    mod.types = types_mod
    return mod, types_mod


class TestDockerClient:
    @pytest.mark.asyncio
    async def test_create_passes_image_and_command(self) -> None:
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )

        docker_client = MagicMock()
        container = _mock_container()
        docker_client.containers.run = MagicMock(return_value=container)
        client = DockerSandboxClient(docker_client=docker_client)
        await client.create(
            options=DockerSandboxClientOptions(image="python:3.12"),
        )
        docker_client.containers.run.assert_called_once()
        call_kwargs = docker_client.containers.run.call_args.kwargs
        assert call_kwargs["image"] == "python:3.12"
        assert call_kwargs["command"] == ["sleep", "infinity"]
        assert call_kwargs["detach"] is True

    @pytest.mark.asyncio
    async def test_create_translates_limits(self) -> None:
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        client = DockerSandboxClient(docker_client=docker_client)
        await client.create(
            options=DockerSandboxClientOptions(
                image="python:3.12",
                cpu_count=2.0,
                memory_mb=512,
                pid_limit=128,
            ),
        )
        kwargs = docker_client.containers.run.call_args.kwargs
        assert kwargs["nano_cpus"] == 2_000_000_000
        assert kwargs["mem_limit"] == "512m"
        assert kwargs["pids_limit"] == 128

    @pytest.mark.asyncio
    async def test_resume_finds_container(self) -> None:
        from troopai.adk.sandbox.clients.docker import DockerSandboxClient
        from troopai.adk.types.sandbox.session_state import SandboxSessionState

        docker_client = MagicMock()
        docker_client.containers.get = MagicMock(return_value=_mock_container())
        client = DockerSandboxClient(docker_client=docker_client)
        state = SandboxSessionState(
            backend_id="docker",
            provider_payload={"container_id": "abc123"},
        )
        session = await client.resume(state)
        assert session is not None
        docker_client.containers.get.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_resume_missing_payload_raises(self) -> None:
        from troopai.adk.sandbox.clients.docker import DockerSandboxClient
        from troopai.adk.types.sandbox.session_state import SandboxSessionState

        client = DockerSandboxClient(docker_client=MagicMock())
        state = SandboxSessionState(backend_id="docker")
        with pytest.raises(SandboxStartFailed, match="container_id"):
            await client.resume(state)

    async def test_create_docker_volume_spec_precreates_and_materializes(self) -> None:
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )
        from troopai.adk.types.sandbox.manifest import Manifest
        from troopai.adk.types.sandbox.mounts import DockerVolumeMountStrategy, S3Mount

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        docker_client.volumes.create = MagicMock(return_value=MagicMock())
        mount = S3Mount(
            bucket="b",
            mount_path="ds",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        manifest = Manifest(entries={"ds": mount})
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)
        with patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"), manifest=manifest)
        docker_client.volumes.create.assert_called_once()
        vk = docker_client.volumes.create.call_args.kwargs
        assert vk["driver"] == "rclone"
        assert vk["driver_opts"] == {"type": "s3"}
        run_kwargs = docker_client.containers.run.call_args.kwargs
        assert len(run_kwargs["mounts"]) == 1
        m = run_kwargs["mounts"][0]
        assert m["Type"] == "volume"
        assert m["Source"] == vk["name"]  # references the pre-created volume
        assert m["Target"] == "/workspace/ds"
        assert m["ReadOnly"] is True  # S3Mount.read_only default
        assert "strategy" not in m  # not a leaked spec dict

    async def test_create_in_container_spec_no_volume_create_keeps_sys_admin(self) -> None:
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )
        from troopai.adk.types.sandbox.manifest import Manifest
        from troopai.adk.types.sandbox.mounts import (
            InContainerMountStrategy,
            RcloneMountPattern,
            S3Mount,
        )

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        docker_client.volumes.create = MagicMock()
        mount = S3Mount(
            bucket="b",
            mount_path="m",
            mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern(remote_name="r")),
        )
        manifest = Manifest(entries={"m": mount})
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)
        with patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"), manifest=manifest)
        docker_client.volumes.create.assert_not_called()
        run_kwargs = docker_client.containers.run.call_args.kwargs
        m = run_kwargs["mounts"][0]
        assert m["Source"] is None  # docker-py anonymous-volume idiom
        assert m["ReadOnly"] is False  # in-container tool enforces ro itself
        assert "SYS_ADMIN" in run_kwargs["cap_add"]  # in-container FUSE mount needs it

    async def test_create_threads_working_directory_into_mount_target(self) -> None:
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )
        from troopai.adk.types.sandbox.manifest import Manifest
        from troopai.adk.types.sandbox.mounts import (
            InContainerMountStrategy,
            RcloneMountPattern,
            S3Mount,
        )

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        mount = S3Mount(
            bucket="b",
            mount_path="x",
            mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern(remote_name="r")),
        )
        manifest = Manifest(entries={"x": mount})
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)
        with patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}):
            await client.create(
                options=DockerSandboxClientOptions(image="python:3.12", working_directory="/custom"),
                manifest=manifest,
            )
        m = docker_client.containers.run.call_args.kwargs["mounts"][0]
        # Proves workspace_root=options.working_directory was threaded into
        # apply_mounts_to_docker → _resolve_target (else it'd be /workspace/x).
        assert m["Target"] == "/custom/x"

    async def test_create_unknown_mount_strategy_raises_sandbox_start_failed(self) -> None:
        # Defense-in-depth: if a malformed spec (unknown strategy) ever
        # reaches the materializer, it must fail loud as
        # SandboxStartFailed at session-create — never an opaque KeyError
        # or a silent docker_volume mis-route.
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)

        def _inject_bogus(_mounts: object, kwargs: dict[str, object], **_kw: object) -> dict[str, object]:
            return {**kwargs, "mounts": [{"strategy": "bogus", "target": "/x", "read_only": True}]}

        with (
            patch("troopai.adk.sandbox.policy.apply_mounts_to_docker", _inject_bogus),
            patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}),
            pytest.raises(SandboxStartFailed, match="unrecognized strategy"),
        ):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"))

    async def test_create_wraps_volume_create_failure_as_sandbox_start_failed(self) -> None:
        # A volume-driver failure (unknown driver / daemon down) must
        # surface as SandboxStartFailed like containers.run — not a raw
        # docker SDK exception that bypasses the create-failure contract.
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )
        from troopai.adk.types.sandbox.manifest import Manifest
        from troopai.adk.types.sandbox.mounts import DockerVolumeMountStrategy, S3Mount

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        docker_client.volumes.create = MagicMock(side_effect=Exception("volume driver rclone not found"))
        mount = S3Mount(
            bucket="b",
            mount_path="d",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        manifest = Manifest(entries={"d": mount})
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)
        with (
            patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}),
            pytest.raises(SandboxStartFailed, match="provisioning failed"),
        ):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"), manifest=manifest)

    async def test_create_docker_volume_empty_driver_options_passes_none(self) -> None:
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )
        from troopai.adk.types.sandbox.manifest import Manifest
        from troopai.adk.types.sandbox.mounts import DockerVolumeMountStrategy, S3Mount

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        docker_client.volumes.create = MagicMock(return_value=MagicMock())
        mount = S3Mount(
            bucket="b",
            mount_path="d",
            mount_strategy=DockerVolumeMountStrategy(driver="local"),  # driver_options defaults {}
        )
        manifest = Manifest(entries={"d": mount})
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)
        with patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"), manifest=manifest)
        # empty driver_options must pass driver_opts=None (not {}) — some
        # docker volume drivers reject an empty-dict opts payload.
        assert docker_client.volumes.create.call_args.kwargs["driver_opts"] is None

    async def test_create_malformed_docker_volume_spec_missing_key_raises(self) -> None:
        # A docker_volume spec missing a required key must fail loud as
        # SandboxStartFailed with an explicit "malformed mount spec"
        # cause — never a bare KeyError that create() re-wraps into an
        # opaque "provisioning failed: 'driver'" Docker-fault lookalike.
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)

        def _inject_missing_driver(_m: object, kwargs: dict[str, object], **_kw: object) -> dict[str, object]:
            # docker_volume strategy but no "driver"/"driver_options".
            return {**kwargs, "mounts": [{"strategy": "docker_volume", "target": "/x", "read_only": True}]}

        with (
            patch("troopai.adk.sandbox.policy.apply_mounts_to_docker", _inject_missing_driver),
            patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}),
            pytest.raises(SandboxStartFailed, match="malformed mount spec"),
        ):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"))

    async def test_create_spec_missing_strategy_key_raises(self) -> None:
        # A spec with no "strategy" key at all must still fail loud
        # (strategy resolves to None → the exhaustive final raise), not
        # KeyError on spec["strategy"].
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)

        def _inject_no_strategy(_m: object, kwargs: dict[str, object], **_kw: object) -> dict[str, object]:
            return {**kwargs, "mounts": [{"target": "/x", "read_only": True}]}

        with (
            patch("troopai.adk.sandbox.policy.apply_mounts_to_docker", _inject_no_strategy),
            patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}),
            pytest.raises(SandboxStartFailed, match="unrecognized strategy"),
        ):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"))

    async def test_create_docker_volume_logs_orphan_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # The orphan-able pre-created volume must be surfaced at WARNING
        # so it is findable in logs, not only in a source docstring
        # (the lifecycle is operator-managed, not silent).
        from troopai.adk.sandbox.clients.docker import (
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )
        from troopai.adk.types.sandbox.manifest import Manifest
        from troopai.adk.types.sandbox.mounts import DockerVolumeMountStrategy, S3Mount

        docker_client = MagicMock()
        docker_client.containers.run = MagicMock(return_value=_mock_container())
        docker_client.volumes.create = MagicMock(return_value=MagicMock())
        mount = S3Mount(
            bucket="b",
            mount_path="d",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        manifest = Manifest(entries={"d": mount})
        fake_docker, fake_types = _fake_docker_types()
        client = DockerSandboxClient(docker_client=docker_client)
        with (
            caplog.at_level("WARNING"),
            patch.dict(sys.modules, {"docker": fake_docker, "docker.types": fake_types}),
        ):
            await client.create(options=DockerSandboxClientOptions(image="python:3.12"), manifest=manifest)
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("operator-managed driver volume" in m for m in warnings)


class TestDockerSessionPty:
    @pytest.mark.asyncio
    async def test_pty_start_creates_exec(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        api_client = MagicMock()
        api_client.exec_create = MagicMock(return_value={"Id": "abcdef0123456789"})
        sock = MagicMock()
        api_client.exec_start = MagicMock(return_value=sock)
        container.client.api = api_client
        session = DockerSandboxSession(container=container)
        handle = await session.pty_start("bash", "-i")
        assert handle.session_id == session._session_id
        api_client.exec_create.assert_called_once()
        api_client.exec_start.assert_called_once()
        # backend_payload is typed `object` (opaque routing data); narrow
        # to dict before subscripting so pyright accepts the access.
        payload = handle.backend_payload
        assert isinstance(payload, dict)
        assert payload["impl"] == "docker_exec_socket"

    @pytest.mark.asyncio
    async def test_pty_write_stdin_sends_bytes(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        api_client = MagicMock()
        api_client.exec_create = MagicMock(return_value={"Id": "abc1234567890def"})
        sock = MagicMock()
        # Simulate docker-py's wrapped socket attribute.
        sock._sock = MagicMock()
        api_client.exec_start = MagicMock(return_value=sock)
        container.client.api = api_client
        session = DockerSandboxSession(container=container)
        handle = await session.pty_start("bash")
        await session.pty_write_stdin(handle, b"ls\n")
        sock._sock.send.assert_called_once_with(b"ls\n")

    @pytest.mark.asyncio
    async def test_pty_terminate_all_closes_sockets(self) -> None:
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        api_client = MagicMock()
        api_client.exec_create = MagicMock(return_value={"Id": "abcdef0123ee"})
        sock = MagicMock()
        api_client.exec_start = MagicMock(return_value=sock)
        container.client.api = api_client
        session = DockerSandboxSession(container=container)
        await session.pty_start("bash")
        await session.pty_terminate_all()
        sock.close.assert_called_once()
        assert len(session._pty_handles) == 0

    @pytest.mark.asyncio
    async def test_aclose_terminates_open_ptys(self) -> None:
        """aclose() must close open PTY sockets, not leak them until GC.

        Regression: aclose() called stop()/shutdown() but never
        pty_terminate_all(), so PTY sockets + container exec sessions leaked.
        """
        from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession

        container = _mock_container()
        api_client = MagicMock()
        api_client.exec_create = MagicMock(return_value={"Id": "abcdef01feed"})
        sock = MagicMock()
        api_client.exec_start = MagicMock(return_value=sock)
        container.client.api = api_client
        session = DockerSandboxSession(container=container)
        await session.pty_start("bash")
        await session.aclose()
        sock.close.assert_called_once()
        assert len(session._pty_handles) == 0
