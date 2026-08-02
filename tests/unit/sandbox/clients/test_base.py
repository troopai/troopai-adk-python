"""Tests for ``BaseSandboxClient`` and ``BaseSandboxSession`` ABCs."""

from __future__ import annotations

from io import BytesIO, IOBase
from pathlib import Path
from typing import Any

import pytest

from troopai.adk.sandbox.clients import (
    BaseSandboxClient,
    BaseSandboxClientOptions,
    BaseSandboxSession,
    FileEntry,
    MaterializationResult,
)
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities
from troopai.adk.types.sandbox.exec_result import (
    ExecResult,
    ExposedPortEndpoint,
    PtyHandle,
)
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.session_state import SandboxSessionState

# --- Fake concrete subclasses for testing ABC behavior -----------------------


class _FakeOptions(BaseSandboxClientOptions):
    image: str = "fake:latest"


class _FakeSession(BaseSandboxSession):
    """Minimal concrete session - just enough to satisfy the ABC."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.shutdown_called = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def aclose(self) -> None:
        await self.stop()
        await self.shutdown()

    async def run(self, *command: Any, **kwargs: Any) -> ExecResult:
        del command, kwargs
        return ExecResult(stdout=b"", stderr=b"", exit_code=0)

    async def pty_start(self, *command: Any, **kwargs: Any) -> PtyHandle:
        del kwargs
        return PtyHandle(
            session_id="fake",
            command=" ".join(str(c) for c in command),
            backend_payload=None,
        )

    async def pty_write_stdin(self, handle: PtyHandle, data: bytes) -> None:
        del handle, data

    async def pty_terminate_all(self) -> None:
        pass

    async def read(self, path: Any, **kwargs: Any) -> IOBase:
        del path, kwargs
        return BytesIO(b"")

    async def write(self, path: Any, data: IOBase, **kwargs: Any) -> None:
        del path, data, kwargs

    async def ls(self, path: Any, **kwargs: Any) -> list[FileEntry]:
        del path, kwargs
        return []

    async def rm(self, path: Any, **kwargs: Any) -> None:
        del path, kwargs

    async def mkdir(self, path: Any, **kwargs: Any) -> None:
        del path, kwargs

    async def extract(self, path: Any, data: IOBase, **kwargs: Any) -> None:
        del path, data, kwargs

    async def persist_workspace(self) -> IOBase:
        return BytesIO(b"")

    async def hydrate_workspace(self, data: IOBase) -> None:
        del data

    async def apply_manifest(self, **kwargs: Any) -> MaterializationResult:
        del kwargs
        return MaterializationResult()

    async def apply_patch(self, patch: str, **kwargs: Any) -> str:
        del patch, kwargs
        return ""

    async def running(self) -> bool:
        return self.started and not self.stopped

    async def resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        return ExposedPortEndpoint(host="localhost", port=port)


class _FakeClient(BaseSandboxClient[_FakeOptions]):
    backend_id = "fake"

    async def create(self, *, options: _FakeOptions, **kwargs: Any) -> _FakeSession:
        del options, kwargs
        return _FakeSession()

    async def delete(self, session: _FakeSession) -> _FakeSession:
        await session.aclose()
        return session

    async def resume(self, state: SandboxSessionState) -> _FakeSession:
        del state
        return _FakeSession()

    def deserialize_session_state(self, payload: dict[str, Any]) -> SandboxSessionState:
        return SandboxSessionState.model_validate(payload)


# --- ABC enforcement tests --------------------------------------------------


class TestBaseSandboxClientABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseSandboxClient()  # type: ignore[abstract]

    def test_concrete_can_instantiate(self) -> None:
        client = _FakeClient()
        assert client.backend_id == "fake"


class TestBaseSandboxSessionABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseSandboxSession()  # type: ignore[abstract]

    def test_default_supports_pty_is_false(self) -> None:
        s = _FakeSession()
        assert s.supports_pty() is False

    def test_default_supports_docker_volume_mounts_is_false(self) -> None:
        s = _FakeSession()
        assert s.supports_docker_volume_mounts() is False

    def test_default_session_id_is_none(self) -> None:
        s = _FakeSession()
        assert s.session_id is None

    def test_normalize_path_default_unchanged(self) -> None:
        s = _FakeSession()
        assert s.normalize_path("/x/y") == Path("/x/y")


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_aenter_calls_start(self) -> None:
        s = _FakeSession()
        async with s:
            assert s.started is True

    @pytest.mark.asyncio
    async def test_aexit_calls_aclose(self) -> None:
        s = _FakeSession()
        async with s:
            pass
        assert s.stopped is True
        assert s.shutdown_called is True


class TestSessionStateRoundTrip:
    @pytest.mark.asyncio
    async def test_serialize_then_deserialize(self) -> None:
        client = _FakeClient()
        state = SandboxSessionState(
            backend_id="fake",
            provider_payload={"key": "value"},
        )
        payload = client.serialize_session_state(state)
        restored = client.deserialize_session_state(payload)
        assert restored.backend_id == "fake"
        assert restored.provider_payload == {"key": "value"}


class TestClientCreateAndDelete:
    @pytest.mark.asyncio
    async def test_create_returns_session(self) -> None:
        client = _FakeClient()
        session = await client.create(options=_FakeOptions(), manifest=Manifest())
        assert isinstance(session, _FakeSession)

    @pytest.mark.asyncio
    async def test_delete_aclosees_session(self) -> None:
        client = _FakeClient()
        session = await client.create(options=_FakeOptions())
        await session.start()
        returned = await client.delete(session)
        assert returned is session
        assert session.stopped is True
        assert session.shutdown_called is True


class TestFileEntry:
    def test_construction(self) -> None:
        e = FileEntry(name="foo.txt", is_directory=False, size_bytes=42)
        assert e.name == "foo.txt"
        assert e.is_directory is False
        assert e.size_bytes == 42

    def test_default_size_is_minus_one(self) -> None:
        e = FileEntry(name="dir", is_directory=True)
        assert e.size_bytes == -1


class TestMaterializationResult:
    def test_default_empty(self) -> None:
        r = MaterializationResult()
        assert r.files == []
        assert r.skipped_mounts == []

    def test_iteration_yields_files(self) -> None:
        from troopai.adk.types.sandbox.entries import MaterializedFile
        from troopai.adk.types.sandbox.permissions import Permissions

        mf = MaterializedFile(
            path="foo.txt",
            size_bytes=10,
            permissions=Permissions(),
        )
        r = MaterializationResult(files=[mf], skipped_mounts=["mounts/s3"])
        assert list(r) == [mf]
        assert r.skipped_mounts == ["mounts/s3"]


class TestBaseSandboxClientOptions:
    def test_frozen(self) -> None:
        opts = _FakeOptions(image="my:tag")
        with pytest.raises(Exception):
            opts.image = "other"  # type: ignore[misc]


class TestBaseSandboxClientCostCapabilitiesBilling:
    @pytest.mark.asyncio
    async def test_base_client_defaults_cost_capabilities_billing(self) -> None:
        # BaseSandboxClient is abstract; assert the class-level default contract.
        assert BaseSandboxClient.cost is None
        assert isinstance(BaseSandboxClient.capabilities, SandboxBackendCapabilities)
        assert BaseSandboxClient.capabilities.network is False
        # Default fetch_billing returns None (no live billing).
        client = _FakeClient()
        result = await client.fetch_billing(_FakeSession())
        assert result is None
