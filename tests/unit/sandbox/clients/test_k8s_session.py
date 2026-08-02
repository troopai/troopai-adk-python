"""Tests for K8sPodSandboxSession + K8sPodSandboxClient.

Uses unittest.mock so the suite runs without an actual Kubernetes
cluster — kubernetes.stream.stream is replaced by a fake ws object.
"""

from __future__ import annotations

import asyncio
import tarfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.exceptions.exceptions import (
    SandboxStartFailed,
    WorkspaceReadNotFoundError,
)


def _directory_tar_bytes() -> bytes:
    """A tar of a directory: a DIRTYPE entry plus a file inside it.

    Mirrors what ``tar cf`` streams for a directory path — more than one
    member — so a single-file ``read`` must reject it rather than silently
    returning the file found inside.
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


def _mock_pod_status(phase: str = "Running") -> MagicMock:
    pod = MagicMock()
    pod.status = MagicMock(phase=phase, pod_ip="10.0.0.5")
    return pod


def _fake_ws(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
) -> MagicMock:
    """Build a fake kubernetes.stream stream object with one-shot output."""
    ws = MagicMock()
    is_open: list[bool] = [True]
    stdout_queue: list[bytes] = [stdout]
    stderr_queue: list[bytes] = [stderr]

    def _is_open() -> bool:
        return is_open[0]

    def _update(*, timeout: int = 1) -> None:
        del timeout
        is_open[0] = False

    def _peek_stdout() -> bool:
        return len(stdout_queue) > 0

    def _read_stdout() -> bytes:
        return stdout_queue.pop(0) if len(stdout_queue) > 0 else b""

    def _peek_stderr() -> bool:
        return len(stderr_queue) > 0 and stderr_queue[0] != b""

    def _read_stderr() -> bytes:
        return stderr_queue.pop(0) if len(stderr_queue) > 0 else b""

    ws.is_open.side_effect = _is_open
    ws.update.side_effect = _update
    ws.peek_stdout.side_effect = _peek_stdout
    ws.read_stdout.side_effect = _read_stdout
    ws.peek_stderr.side_effect = _peek_stderr
    ws.read_stderr.side_effect = _read_stderr
    ws.write_stdin = MagicMock()
    ws.close = MagicMock()
    ws.returncode = returncode
    return ws


def _make_session(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
    phase: str = "Running",
):
    from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession

    core_v1 = MagicMock()
    core_v1.read_namespaced_pod_status.return_value = _mock_pod_status(phase)
    core_v1.read_namespaced_pod.return_value = _mock_pod_status(phase)
    core_v1.delete_namespaced_pod = MagicMock()

    def _stream_fn(_fn, **_kwargs):
        return _fake_ws(stdout=stdout, stderr=stderr, returncode=returncode)

    session = K8sPodSandboxSession(
        core_v1=core_v1,
        pod_name="pod-x",
        namespace="ns",
        working_directory="/workspace",
        stream_api=_stream_fn,
    )
    return session, core_v1


class TestK8sSessionLifecycle:
    @pytest.mark.asyncio
    async def test_start_waits_until_running(self) -> None:
        session, core_v1 = _make_session(phase="Running")
        await session.start()
        assert session._started is True
        core_v1.read_namespaced_pod_status.assert_called()

    async def test_start_installs_runtime_helpers(self) -> None:
        session, _ = _make_session(phase="Running")
        with patch(
            "troopai.adk.sandbox.session.runtime_helpers.install_runtime_helpers",
            new_callable=AsyncMock,
        ) as installer:
            await session.start()
        installer.assert_awaited_once()
        call = installer.call_args
        assert call.args[0] is session
        assert call.kwargs["backend_id"] == "k8s_pod"

    @pytest.mark.asyncio
    async def test_start_raises_on_failed_phase(self) -> None:
        session, _core_v1 = _make_session(phase="Failed")
        with pytest.raises(SandboxStartFailed, match="terminal phase"):
            await session.start()

    @pytest.mark.asyncio
    async def test_stop_calls_delete_pod(self) -> None:
        session, core_v1 = _make_session()
        session._started = True
        await session.stop()
        core_v1.delete_namespaced_pod.assert_called_once()
        kwargs = core_v1.delete_namespaced_pod.call_args.kwargs
        assert kwargs["name"] == "pod-x"
        assert kwargs["namespace"] == "ns"

    @pytest.mark.asyncio
    async def test_aclose_idempotent(self) -> None:
        session, core_v1 = _make_session()
        session._started = True
        await session.aclose()
        await session.aclose()
        assert core_v1.delete_namespaced_pod.call_count == 1


class TestK8sSessionRun:
    @pytest.mark.asyncio
    async def test_run_returns_exec_result(self) -> None:
        session, _core_v1 = _make_session(stdout=b"hello\n", returncode=0)
        result = await session.run("echo", "hello", shell=True)
        assert result.exit_code == 0
        assert result.stdout == b"hello\n"

    @pytest.mark.asyncio
    async def test_run_nonzero_exit_returned(self) -> None:
        session, _core_v1 = _make_session(stderr=b"err\n", returncode=1)
        result = await session.run("false", shell=False)
        assert result.exit_code == 1
        assert result.stderr == b"err\n"


class TestK8sSessionFileOps:
    @pytest.mark.asyncio
    async def test_resolve_path_relative(self) -> None:
        session, _core_v1 = _make_session()
        assert session._resolve_path("foo.txt") == "/workspace/foo.txt"
        assert session._resolve_path("/etc/hosts") == "/etc/hosts"

    @pytest.mark.asyncio
    async def test_write_calls_tar_exec(self) -> None:
        session, _core_v1 = _make_session(returncode=0)
        # Just ensure no exception raised.
        await session.write("foo.txt", BytesIO(b"hello"))

    @pytest.mark.asyncio
    async def test_read_missing_raises(self) -> None:
        session, _core_v1 = _make_session(returncode=1, stderr=b"not found")
        with pytest.raises(WorkspaceReadNotFoundError):
            await session.read("missing.txt")

    @pytest.mark.asyncio
    async def test_read_directory_raises_not_a_file(self) -> None:
        """read() on a directory must raise, not return the first file inside.

        Regression: ``tar cf`` on a directory streams a multi-member archive;
        the old code filtered to files and returned members[0] — leaking a
        file from inside the directory instead of failing.
        """
        session, _core_v1 = _make_session(stdout=_directory_tar_bytes(), returncode=0)
        with pytest.raises(WorkspaceReadNotFoundError, match="not a regular file"):
            await session.read("mydir")

    @pytest.mark.asyncio
    async def test_read_single_file_returns_content(self) -> None:
        """A normal single-file read still returns the file bytes."""
        session, _core_v1 = _make_session(stdout=_single_file_tar_bytes(b"hi"), returncode=0)
        result = await session.read("foo.txt")
        assert result.read() == b"hi"


class TestK8sApplyPatch:
    @pytest.mark.asyncio
    async def test_patch_file_removed_when_run_raises(self) -> None:
        """Regression: k8s apply_patch left .troopai_patch.diff when run() raised."""
        from unittest.mock import patch as mock_patch

        from troopai.adk.exceptions.exceptions import SandboxStopFailed

        session, _core_v1 = _make_session()
        removed_paths: list[str] = []
        boom = SandboxStopFailed("exec failed")

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
            pytest.raises(SandboxStopFailed),
        ):
            await session.apply_patch("--- a/foo\n+++ b/foo\n")

        assert any(".troopai_patch.diff" in p for p in removed_paths), (
            f"Expected .troopai_patch.diff cleanup but rm was called with: {removed_paths}"
        )


class TestK8sSessionPort:
    @pytest.mark.asyncio
    async def test_resolves_pod_ip(self) -> None:
        session, _core_v1 = _make_session()
        endpoint = await session.resolve_exposed_port(8080)
        assert endpoint.host == "10.0.0.5"
        assert endpoint.port == 8080


class TestK8sClient:
    @pytest.mark.asyncio
    async def test_create_builds_pod_with_image(self) -> None:
        from troopai.adk.sandbox.clients.k8s import (
            K8sPodSandboxClient,
            K8sSandboxClientOptions,
        )

        core_v1 = MagicMock()
        core_v1.create_namespaced_pod = MagicMock()
        client = K8sPodSandboxClient(core_v1=core_v1)
        session = await client.create(
            options=K8sSandboxClientOptions(image="python:3.12", namespace="test-ns"),
        )
        assert session is not None
        core_v1.create_namespaced_pod.assert_called_once()
        call_kwargs = core_v1.create_namespaced_pod.call_args.kwargs
        assert call_kwargs["namespace"] == "test-ns"
        body = call_kwargs["body"]
        assert body["kind"] == "Pod"
        containers = body["spec"]["containers"]
        assert containers[0]["image"] == "python:3.12"
        assert containers[0]["command"] == ["sleep", "infinity"]

    @pytest.mark.asyncio
    async def test_netpol_without_networking_v1_raises_and_tears_down(self) -> None:
        """A configured NetworkPolicy with no NetworkingV1Api must not silently skip.

        Regression: the netpol block was guarded by ``self._networking_v1 is not
        None``, so injecting core_v1 without networking_v1 then configuring a
        policy started the pod with UNRESTRICTED network and skipped the policy
        with no error. It now tears the pod down and raises.
        """
        from troopai.adk.sandbox.clients.k8s import (
            K8sPodSandboxClient,
            K8sSandboxClientOptions,
        )
        from troopai.adk.types.sandbox.network import NetworkPolicy

        core_v1 = MagicMock()
        core_v1.create_namespaced_pod = MagicMock()
        core_v1.delete_namespaced_pod = MagicMock()
        # networking_v1 deliberately omitted.
        client = K8sPodSandboxClient(core_v1=core_v1)

        with pytest.raises(SandboxStartFailed, match="no NetworkingV1Api"):
            await client.create(
                options=K8sSandboxClientOptions(
                    image="python:3.12",
                    namespace="ns",
                    network_policy=NetworkPolicy(deny_default=True),
                ),
            )
        # The pod was torn down rather than left running unrestricted.
        core_v1.delete_namespaced_pod.assert_called_once()

    @pytest.mark.asyncio
    async def test_restricted_pss_sets_security_context(self) -> None:
        from troopai.adk.sandbox.clients.k8s import (
            K8sPodSandboxClient,
            K8sSandboxClientOptions,
        )

        core_v1 = MagicMock()
        client = K8sPodSandboxClient(core_v1=core_v1)
        await client.create(
            options=K8sSandboxClientOptions(image="python:3.12"),
        )
        body = core_v1.create_namespaced_pod.call_args.kwargs["body"]
        container = body["spec"]["containers"][0]
        assert container["securityContext"]["runAsNonRoot"] is True
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert "ALL" in container["securityContext"]["capabilities"]["drop"]

    @pytest.mark.asyncio
    async def test_resume_finds_pod(self) -> None:
        from troopai.adk.sandbox.clients.k8s import K8sPodSandboxClient
        from troopai.adk.types.sandbox.session_state import SandboxSessionState

        core_v1 = MagicMock()
        core_v1.read_namespaced_pod_status.return_value = _mock_pod_status("Running")
        client = K8sPodSandboxClient(core_v1=core_v1)
        state = SandboxSessionState(
            backend_id="k8s_pod",
            provider_payload={"pod_name": "p-1", "namespace": "ns-1"},
        )
        session = await client.resume(state)
        assert session is not None
        core_v1.read_namespaced_pod_status.assert_called_with(name="p-1", namespace="ns-1")

    @pytest.mark.asyncio
    async def test_resume_missing_payload_raises(self) -> None:
        from troopai.adk.sandbox.clients.k8s import K8sPodSandboxClient
        from troopai.adk.types.sandbox.session_state import SandboxSessionState

        client = K8sPodSandboxClient(core_v1=MagicMock())
        state = SandboxSessionState(backend_id="k8s_pod")
        with pytest.raises(SandboxStartFailed, match="pod_name"):
            await client.resume(state)


class TestK8sSessionPty:
    @pytest.mark.asyncio
    async def test_pty_start_returns_handle(self) -> None:
        session, _core_v1 = _make_session()
        handle = await session.pty_start("bash", "-i")
        assert handle.session_id == session._session_id
        # backend_payload is provider-opaque (typed object); narrow to read it.
        assert isinstance(handle.backend_payload, dict)
        assert handle.backend_payload["impl"] == "k8s_exec_ws"

    @pytest.mark.asyncio
    async def test_pty_write_stdin_writes_to_ws(self) -> None:
        session, _core_v1 = _make_session()
        # Replace stream_api to return a controllable ws.
        from unittest.mock import MagicMock as _Mock

        ws = _Mock()
        ws.write_stdin = _Mock()
        ws.close = _Mock()

        def _stream_fn(_fn, **_kwargs):
            return ws

        session._stream_api = _stream_fn
        handle = await session.pty_start("bash")
        await session.pty_write_stdin(handle, b"echo hi\n")
        # Bytes are delivered verbatim (binary frame) — never utf-8-decoded.
        ws.write_stdin.assert_called_once_with(b"echo hi\n")

    @pytest.mark.asyncio
    async def test_pty_terminate_all_closes_ws(self) -> None:
        session, _core_v1 = _make_session()
        from unittest.mock import MagicMock as _Mock

        ws = _Mock()

        def _stream_fn(_fn, **_kwargs):
            return ws

        session._stream_api = _stream_fn
        await session.pty_start("bash")
        await session.pty_terminate_all()
        ws.close.assert_called_once()
        assert len(session._pty_handles) == 0

    @pytest.mark.asyncio
    async def test_aclose_terminates_open_ptys(self) -> None:
        """aclose() must close open PTY websockets, not leak them until GC.

        Regression: aclose() called only shutdown(), never pty_terminate_all().
        """
        session, _core_v1 = _make_session()
        from unittest.mock import MagicMock as _Mock

        ws = _Mock()

        def _stream_fn(_fn: object, **_kwargs: object) -> object:
            return ws

        session._stream_api = _stream_fn
        await session.pty_start("bash")
        await session.aclose()
        ws.close.assert_called_once()
        assert len(session._pty_handles) == 0


class TestK8sClientCancellationSafety:
    """Regression tests for CancelledError pod/netpol leak in K8sPodSandboxClient.create()."""

    @pytest.mark.asyncio
    async def test_cancelled_during_pod_creation_triggers_cleanup(self) -> None:
        """If create_namespaced_pod is cancelled, best-effort delete is called."""
        from troopai.adk.sandbox.clients.k8s import K8sPodSandboxClient, K8sSandboxClientOptions

        core_v1 = MagicMock()
        deleted_pods: list[str] = []

        def _raise_cancelled(*args: object, **kwargs: object) -> object:
            raise asyncio.CancelledError()

        async def _fake_delete_best_effort(pod_name: str, namespace: str) -> None:
            deleted_pods.append(pod_name)

        core_v1.create_namespaced_pod.side_effect = _raise_cancelled
        client = K8sPodSandboxClient(core_v1=core_v1)
        client._delete_pod_best_effort = _fake_delete_best_effort  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await client.create(
                options=K8sSandboxClientOptions(image="python:3.12", namespace="test-ns"),
            )
        # At least one pod-name was passed to the cleanup helper.
        assert len(deleted_pods) == 1

    @pytest.mark.asyncio
    async def test_cancelled_during_netpol_creation_triggers_pod_cleanup(self) -> None:
        """If create_namespaced_network_policy is cancelled, the pod is deleted."""
        from troopai.adk.sandbox.clients.k8s import K8sPodSandboxClient, K8sSandboxClientOptions
        from troopai.adk.types.sandbox.network import NetworkPolicy

        core_v1 = MagicMock()
        networking_v1 = MagicMock()
        deleted_pods: list[str] = []

        def _raise_cancelled(*args: object, **kwargs: object) -> object:
            raise asyncio.CancelledError()

        async def _fake_delete_best_effort(pod_name: str, namespace: str) -> None:
            deleted_pods.append(pod_name)

        networking_v1.create_namespaced_network_policy.side_effect = _raise_cancelled
        client = K8sPodSandboxClient(core_v1=core_v1, networking_v1=networking_v1)
        client._delete_pod_best_effort = _fake_delete_best_effort  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await client.create(
                options=K8sSandboxClientOptions(
                    image="python:3.12",
                    namespace="test-ns",
                    network_policy=NetworkPolicy(deny_default=True),
                ),
            )
        # Pod cleanup was triggered despite CancelledError bypassing except-Exception.
        assert len(deleted_pods) == 1


class TestK8sSessionTimeoutForwarding:
    """Regression tests: run() timeout is forwarded to the thread-level _exec_sync."""

    @pytest.mark.asyncio
    async def test_timeout_forwarded_to_exec_sync(self) -> None:
        """_exec_sync must receive the caller's timeout as max_seconds."""
        session, _core_v1 = _make_session()
        received_max_seconds: list[float | None] = []

        def _fake_exec_sync(argv: list[str], *, max_seconds: float | None = None) -> tuple[int, bytes, bytes]:
            received_max_seconds.append(max_seconds)
            return 0, b"ok", b""

        session._exec_sync = _fake_exec_sync  # type: ignore[method-assign]
        await session.run("echo", "hi", timeout=5.0)
        assert received_max_seconds == [5.0], (
            f"Expected max_seconds=5.0 forwarded to _exec_sync, got {received_max_seconds}"
        )

    @pytest.mark.asyncio
    async def test_none_timeout_forwarded_as_none(self) -> None:
        """When timeout=None, None is forwarded so the class-level cap applies."""
        session, _core_v1 = _make_session()
        received_max_seconds: list[float | None] = []

        def _fake_exec_sync(argv: list[str], *, max_seconds: float | None = None) -> tuple[int, bytes, bytes]:
            received_max_seconds.append(max_seconds)
            return 0, b"", b""

        session._exec_sync = _fake_exec_sync  # type: ignore[method-assign]
        await session.run("echo", shell=False)
        assert received_max_seconds == [None]


class _RealisticWS:
    """Fake exec WS mirroring kubernetes ``WSClient`` semantics.

    Unlike the bare-MagicMock fake, ``returncode`` is a *property* that, on a
    closed socket whose error channel is empty, raises ``TypeError`` (the real
    client does ``yaml.safe_load("")['status']`` → ``None['status']``). This
    exercises the crash path that a plain ``ws.returncode = 0`` attribute masks.
    """

    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", binary: bool = False) -> None:
        self._open = True
        self._stdout: list[bytes] = [stdout] if len(stdout) > 0 else []
        self._stderr: list[bytes] = [stderr] if len(stderr) > 0 else []
        self.binary = binary
        self.written: list[bytes] = []
        self.closed_channels: list[int] = []
        self.close_calls = 0

    def write_stdin(self, data: bytes) -> None:
        self.written.append(data)

    def close_channel(self, channel: int) -> None:
        self.closed_channels.append(channel)

    def is_open(self) -> bool:
        return self._open

    def update(self, *, timeout: int = 1) -> None:
        del timeout
        # One drain tick then the server closes the socket (status frame was
        # NOT delivered, so the error channel stays empty — the crash case).
        self._open = False

    def peek_stdout(self) -> bool:
        return len(self._stdout) > 0

    def read_stdout(self) -> bytes:
        return self._stdout.pop(0)

    def peek_stderr(self) -> bool:
        return len(self._stderr) > 0

    def read_stderr(self) -> bytes:
        return self._stderr.pop(0)

    def close(self) -> None:
        self.close_calls += 1
        self._open = False

    @property
    def returncode(self) -> int:
        if self._open:
            raise AssertionError("returncode read while socket still open")
        # Empty error channel → real client raises TypeError here.
        raise TypeError("'NoneType' object is not subscriptable")


class TestK8sExecStdinDoesNotCrash:
    """Regression: stdin file ops crashed via early close + returncode TypeError."""

    @pytest.mark.asyncio
    async def test_write_with_stdin_drains_and_does_not_crash(self) -> None:
        from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession

        created: list[_RealisticWS] = []

        def _stream_fn(_fn: object, **kwargs: object) -> _RealisticWS:
            ws = _RealisticWS(stdout=b"", binary=bool(kwargs.get("binary")))
            created.append(ws)
            return ws

        session = K8sPodSandboxSession(
            core_v1=MagicMock(),
            pod_name="pod-x",
            namespace="ns",
            working_directory="/workspace",
            stream_api=_stream_fn,
        )
        # mkdir runs first (no stdin), then the tar-xf write (stdin). Neither
        # may raise TypeError from the returncode property.
        await session.write("foo.txt", BytesIO(b"hello"))
        # The stdin op signalled EOF via close_channel(0), NOT a full close()
        # that would abort the drain loop before output/status arrive.
        stdin_ws = created[-1]
        assert len(stdin_ws.written) == 1
        assert stdin_ws.closed_channels == [0]
        assert stdin_ws.close_calls == 0

    @pytest.mark.asyncio
    async def test_exec_returncode_swallows_empty_error_channel(self) -> None:
        from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession

        ws = _RealisticWS()
        ws.close()  # closed, empty error channel → property raises TypeError
        # The helper must treat the unreadable status as a clean exit (0).
        assert K8sPodSandboxSession._exec_returncode(ws) == 0


class TestK8sExecBinaryMode:
    """Regression: tar streams were utf-8-replace-decoded and corrupted."""

    @pytest.mark.asyncio
    async def test_exec_sync_requests_binary_stream(self) -> None:
        from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession

        captured_kwargs: list[dict[str, object]] = []

        def _stream_fn(_fn: object, **kwargs: object) -> _RealisticWS:
            captured_kwargs.append(kwargs)
            return _RealisticWS(stdout=b"\xff\x00\xfe")

        session = K8sPodSandboxSession(
            core_v1=MagicMock(),
            pod_name="pod-x",
            namespace="ns",
            stream_api=_stream_fn,
        )
        exit_code, stdout, _stderr = session._exec_sync(["tar", "cf", "-", "x"])
        assert exit_code == 0
        # Non-UTF-8 bytes survive intact (binary=True skips lossy decode).
        assert stdout == b"\xff\x00\xfe"
        assert captured_kwargs[0]["binary"] is True


class TestK8sNetworkPolicyCleanup:
    """Regression: per-pod NetworkPolicy CR was orphaned on teardown."""

    @pytest.mark.asyncio
    async def test_stop_deletes_network_policy(self) -> None:
        from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession
        from troopai.adk.types.sandbox.network import NetworkPolicy

        core_v1 = MagicMock()
        core_v1.delete_namespaced_pod = MagicMock()
        networking_v1 = MagicMock()
        networking_v1.delete_namespaced_network_policy = MagicMock()

        session = K8sPodSandboxSession(
            core_v1=core_v1,
            pod_name="pod-x",
            namespace="ns",
            network_policy=NetworkPolicy(deny_default=True),
            networking_v1=networking_v1,
        )
        session._started = True
        await session.stop()

        networking_v1.delete_namespaced_network_policy.assert_called_once()
        kwargs = networking_v1.delete_namespaced_network_policy.call_args.kwargs
        assert kwargs["name"] == "pod-x-egress-policy"
        assert kwargs["namespace"] == "ns"

    @pytest.mark.asyncio
    async def test_stop_without_policy_skips_netpol_delete(self) -> None:
        from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession

        core_v1 = MagicMock()
        core_v1.delete_namespaced_pod = MagicMock()
        networking_v1 = MagicMock()

        session = K8sPodSandboxSession(
            core_v1=core_v1,
            pod_name="pod-x",
            namespace="ns",
            network_policy=None,
            networking_v1=networking_v1,
        )
        session._started = True
        await session.stop()

        networking_v1.delete_namespaced_network_policy.assert_not_called()

    @pytest.mark.asyncio
    async def test_netpol_cleanup_failure_does_not_raise(self) -> None:
        from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession
        from troopai.adk.types.sandbox.network import NetworkPolicy

        core_v1 = MagicMock()
        core_v1.delete_namespaced_pod = MagicMock()
        networking_v1 = MagicMock()
        networking_v1.delete_namespaced_network_policy.side_effect = RuntimeError("boom")

        session = K8sPodSandboxSession(
            core_v1=core_v1,
            pod_name="pod-x",
            namespace="ns",
            network_policy=NetworkPolicy(deny_default=True),
            networking_v1=networking_v1,
        )
        session._started = True
        # A failed netpol cleanup must not mask the successful pod teardown.
        await session.stop()
        assert session._stopped is True

    @pytest.mark.asyncio
    async def test_netpol_deleted_even_when_pod_delete_raises(self) -> None:
        """A pod-delete failure must still reclaim the NetworkPolicy CR.

        Regression: netpol cleanup ran after the pod-delete try/except, so a
        pod-delete failure raised SandboxStopFailed and skipped the cleanup,
        orphaning the per-pod policy object. Cleanup now runs in a finally.
        """
        from troopai.adk.exceptions.exceptions import SandboxStopFailed
        from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession
        from troopai.adk.types.sandbox.network import NetworkPolicy

        core_v1 = MagicMock()
        core_v1.delete_namespaced_pod = MagicMock(side_effect=RuntimeError("pod delete failed"))
        networking_v1 = MagicMock()
        networking_v1.delete_namespaced_network_policy = MagicMock()

        session = K8sPodSandboxSession(
            core_v1=core_v1,
            pod_name="pod-x",
            namespace="ns",
            network_policy=NetworkPolicy(deny_default=True),
            networking_v1=networking_v1,
        )
        session._started = True
        with pytest.raises(SandboxStopFailed):
            await session.stop()
        # The NetworkPolicy CR is still reclaimed despite the pod-delete failure.
        networking_v1.delete_namespaced_network_policy.assert_called_once()


class TestK8sPtyWriteStdinBytes:
    """Regression: PTY stdin was utf-8-replace-decoded before resend."""

    @pytest.mark.asyncio
    async def test_pty_write_stdin_passes_raw_bytes(self) -> None:
        session, _core_v1 = _make_session()
        ws = MagicMock()
        ws.write_stdin = MagicMock()
        ws.close = MagicMock()

        def _stream_fn(_fn: object, **_kwargs: object) -> object:
            return ws

        session._stream_api = _stream_fn
        handle = await session.pty_start("bash")
        # Non-UTF-8 byte sequence must reach the PTY verbatim.
        await session.pty_write_stdin(handle, b"\xff\xfe\x00data")
        ws.write_stdin.assert_called_once_with(b"\xff\xfe\x00data")
