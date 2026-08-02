"""Tests for ``sandbox_run_context`` lifecycle ctx manager."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.exceptions.exceptions import (
    SandboxConcurrencyError,
    SandboxConfigurationError,
    UnsupportedSnapshotFeatureError,
)
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.runner_integration.concurrency_guard import (
    SandboxConcurrencyGuard,
)
from troopai.adk.sandbox.runner_integration.lifecycle import sandbox_run_context
from troopai.adk.types.sandbox.iac import IaCBundle
from troopai.adk.types.sandbox.session_state import SandboxSessionState

_APPLY = "troopai.adk.sandbox.runner_integration.lifecycle.apply_iac"
_DESTROY = "troopai.adk.sandbox.runner_integration.lifecycle.destroy_iac"


def _fake_session() -> Any:
    s = MagicMock()
    s.start = AsyncMock()
    s.aclose = AsyncMock()
    # Concrete session_id so audit-event assertions are real (a bare
    # MagicMock attr would flow silently into SandboxAuditEvent).
    s.session_id = "sess-test"
    return s


def _fake_client(session: Any) -> Any:
    c = MagicMock()
    c.create = AsyncMock(return_value=session)
    c.resume = AsyncMock(return_value=session)
    return c


class TestSessionResolutionOrder:
    @pytest.mark.asyncio
    async def test_explicit_session_takes_precedence(self) -> None:
        injected = _fake_session()
        client = _fake_client(_fake_session())
        config = SandboxRunConfig(session=injected, client=client)
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ) as handle:
            assert handle.session is injected
            assert handle.runner_owns_session is False
        # Injected session NOT closed by the runner.
        injected.aclose.assert_not_called()
        # Client.create + start NOT called either.
        client.create.assert_not_called()
        injected.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_state_resumes_via_client(self) -> None:
        resumed = _fake_session()
        client = _fake_client(resumed)
        state = SandboxSessionState(backend_id="unix_local")
        config = SandboxRunConfig(client=client, session_state=state)
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ) as handle:
            assert handle.session is resumed
            assert handle.runner_owns_session is True
        client.resume.assert_called_once()
        # Runner owns it — aclose called on exit.
        resumed.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_only_creates_fresh(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ) as handle:
            assert handle.session is session
            assert handle.runner_owns_session is True
        client.create.assert_called_once()
        session.start.assert_called_once()
        session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_state_without_client_raises(self) -> None:
        state = SandboxSessionState(backend_id="unix_local")
        # SandboxRunConfig itself validates that at least one source
        # is provided, but session_state alone is invalid because
        # resume requires a client.
        config = SandboxRunConfig.__new__(SandboxRunConfig)
        # Bypass __post_init__ to construct an illegal state.
        object.__setattr__(config, "client", None)
        object.__setattr__(config, "options", None)
        object.__setattr__(config, "session", None)
        object.__setattr__(config, "session_state", state)
        object.__setattr__(config, "manifest", None)
        object.__setattr__(config, "snapshot", None)
        object.__setattr__(config, "snapshot_store", None)
        object.__setattr__(config, "resource_limits", None)
        object.__setattr__(config, "network_policy", None)
        object.__setattr__(config, "command_policy", None)
        object.__setattr__(config, "audit_sink", None)
        object.__setattr__(config, "iac", None)
        object.__setattr__(config, "selector", None)
        object.__setattr__(config, "candidates", None)
        object.__setattr__(config, "requirements", None)
        object.__setattr__(config, "capture_live_cost", False)
        with pytest.raises(ValueError, match="requires a non-None client"):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                pass


class TestConcurrencyGuardIntegration:
    @pytest.mark.asyncio
    async def test_guard_acquired_and_released(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)
        guard = SandboxConcurrencyGuard()
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=guard,
        ):
            # Inside the block — second acquire raises.
            with pytest.raises(SandboxConcurrencyError):
                await guard.acquire()
        # After exit — can reacquire.
        await guard.acquire()
        guard.release()

    @pytest.mark.asyncio
    async def test_guard_released_on_exception(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)
        guard = SandboxConcurrencyGuard()
        with pytest.raises(RuntimeError, match="boom"):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=guard,
            ):
                raise RuntimeError("boom")
        # Guard released despite the exception.
        await guard.acquire()
        guard.release()


class TestSessionCleanup:
    @pytest.mark.asyncio
    async def test_runner_owned_session_closed_on_success(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ):
            pass
        session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_runner_owned_session_closed_on_exception(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)
        with pytest.raises(RuntimeError):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                raise RuntimeError("boom")
        session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_failure_does_not_mask_original(self) -> None:
        session = _fake_session()
        session.aclose.side_effect = RuntimeError("close-failed")
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)
        with pytest.raises(RuntimeError, match="original"):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                raise RuntimeError("original")


class TestIacWiring:
    async def test_iac_applied_and_env_surfaced_on_handle(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        config = SandboxRunConfig(client=client, iac=bundle)
        apply = AsyncMock(return_value={"DB_URL": "postgres://h/db"})
        with patch(_APPLY, apply), patch(_DESTROY, AsyncMock()):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ) as handle:
                assert handle.iac_env == {"DB_URL": "postgres://h/db"}
        apply.assert_awaited_once_with(bundle)

    async def test_no_iac_yields_empty_env_and_skips_apply(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)  # iac defaults None
        apply = AsyncMock()
        with patch(_APPLY, apply), patch(_DESTROY, AsyncMock()):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ) as handle:
                assert handle.iac_env == {}
        apply.assert_not_awaited()

    async def test_iac_apply_failure_propagates_and_session_closed(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        config = SandboxRunConfig(client=client, iac=bundle)
        apply = AsyncMock(side_effect=SandboxConfigurationError("iac boom"))
        destroy = AsyncMock()
        with (
            patch(_APPLY, apply),
            patch(_DESTROY, destroy),
            pytest.raises(SandboxConfigurationError, match="iac boom"),
        ):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                pass
        # Provisioning failed AFTER the runner-owned session started →
        # teardown still closes it (no orphaned session) AND still
        # attempts destroy_iac (best-effort cleanup of possibly
        # partially-provisioned infra — NOT skipped because apply
        # raised). Pins the failure-path teardown contract.
        session.aclose.assert_awaited_once()
        destroy.assert_awaited_once_with(bundle)

    async def test_destroy_iac_called_when_destroy_on_exit(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")  # destroy_on_exit defaults True
        config = SandboxRunConfig(client=client, iac=bundle)
        destroy = AsyncMock()
        with patch(_APPLY, AsyncMock(return_value={})), patch(_DESTROY, destroy):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                pass
        destroy.assert_awaited_once_with(bundle)

    async def test_destroy_iac_skipped_when_destroy_on_exit_false(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac", destroy_on_exit=False)
        config = SandboxRunConfig(client=client, iac=bundle)
        destroy = AsyncMock()
        with patch(_APPLY, AsyncMock(return_value={})), patch(_DESTROY, destroy):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                pass
        destroy.assert_not_awaited()

    async def test_teardown_closes_session_before_destroying_infra(self) -> None:
        # Order invariant: aclose BEFORE destroy_iac. A session may
        # flush/checkpoint against provisioned infra while closing, so
        # destroying infra first would silently break it (the failure
        # is swallowed by aclose's suppress). Pins the order so a
        # future _teardown reorder fails loud.
        order: list[str] = []

        async def _rec_aclose() -> None:
            order.append("aclose")

        async def _rec_destroy(*_a: object, **_k: object) -> None:
            order.append("destroy")

        session = _fake_session()
        session.aclose = AsyncMock(side_effect=_rec_aclose)
        client = _fake_client(session)
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        config = SandboxRunConfig(client=client, iac=bundle)
        with patch(_APPLY, AsyncMock(return_value={})), patch(_DESTROY, AsyncMock(side_effect=_rec_destroy)):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                pass
        assert order == ["aclose", "destroy"]


class TestCommandPolicyWiring:
    async def test_run_command_policy_reaches_shell_clone(self) -> None:
        # End-to-end: SandboxRunConfig.command_policy must land on the
        # per-run ShellCapability CLONE surfaced via the handle (so the
        # run_command tool enforces the run-level policy).
        session = _fake_session()
        client = _fake_client(session)
        sentinel = object()
        config = SandboxRunConfig(client=client, command_policy=sentinel)
        shell = ShellCapability()
        async with sandbox_run_context(
            config=config,
            capabilities=[shell],
            run_as=None,
            concurrency_guard=None,
        ) as handle:
            cloned_shell = handle.capabilities[0]
            assert isinstance(cloned_shell, ShellCapability)
            assert cloned_shell.command_policy is sentinel
        # The developer's ORIGINAL capability is never mutated.
        assert shell.command_policy is None

    async def test_no_run_policy_leaves_shell_clone_default(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)  # command_policy defaults None
        shell = ShellCapability()
        async with sandbox_run_context(
            config=config,
            capabilities=[shell],
            run_as=None,
            concurrency_guard=None,
        ) as handle:
            cloned_shell = handle.capabilities[0]
            assert isinstance(cloned_shell, ShellCapability)
            assert cloned_shell.command_policy is None


class _RecordingSink:
    """Minimal AuditSink stand-in: records every emitted event."""

    def __init__(self, *, raise_on_emit: bool = False) -> None:
        self.events: list[Any] = []
        self._raise = raise_on_emit

    async def emit(self, event: Any) -> None:
        self.events.append(event)
        if self._raise:
            raise RuntimeError("audit-sink-down")


class TestAuditWiring:
    async def test_start_and_stop_emitted_on_success(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        client.backend_id = "test-be"
        sink = _RecordingSink()
        config = SandboxRunConfig(client=client, audit_sink=sink)
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ):
            pass
        kinds = [e.event_type for e in sink.events]
        assert kinds == ["start", "stop"]
        assert sink.events[0].backend_id == "test-be"
        # session_id ties the audit event to its sandbox session —
        # core to audit-trail integrity; pin it, not a stray Mock.
        assert sink.events[0].session_id == "sess-test"
        # No agent_name passed (test path) → documented fallback.
        assert sink.events[0].agent_name == "<unknown>"

    async def test_error_event_emitted_on_exception(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        sink = _RecordingSink()
        config = SandboxRunConfig(client=client, audit_sink=sink)
        with pytest.raises(RuntimeError, match="boom"):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                raise RuntimeError("boom")
        kinds = [e.event_type for e in sink.events]
        assert kinds == ["start", "error"]
        assert "boom" in (sink.events[1].error or "")

    async def test_audit_emit_failure_does_not_mask_original(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        sink = _RecordingSink(raise_on_emit=True)
        config = SandboxRunConfig(client=client, audit_sink=sink)
        # The loop raises "original"; the audit sink ALSO raises on
        # emit. The caller must still see "original" — the best-effort
        # _emit_audit suppression must not mask it.
        with pytest.raises(RuntimeError, match="original"):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                raise RuntimeError("original")

    async def test_agent_name_threaded_into_events(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        sink = _RecordingSink()
        config = SandboxRunConfig(client=client, audit_sink=sink)
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
            agent_name="researcher",
        ):
            pass
        assert all(e.agent_name == "researcher" for e in sink.events)

    async def test_no_sink_runs_clean(self) -> None:
        # audit_sink defaults None → _emit_audit early-returns; the
        # lifecycle must still run end-to-end. The "no emit when None"
        # contract is pinned by CONTRAST with the start/stop/error
        # tests (which prove emission DOES happen when a sink is set)
        # — a never-wired recording sink would assert vacuously.
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)  # audit_sink defaults None
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ) as handle:
            assert handle.runner_owns_session is True
        # Non-vacuous: the lifecycle genuinely ran + tore down.
        session.start.assert_awaited_once()
        session.aclose.assert_awaited_once()


class TestSnapshotStoreForward:
    async def test_snapshot_store_forwarded_to_create(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        sentinel = object()
        config = SandboxRunConfig(client=client, snapshot_store=sentinel)
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ):
            pass
        assert client.create.await_args.kwargs["snapshot_store"] is sentinel

    async def test_snapshot_store_omitted_when_none(self) -> None:
        session = _fake_session()
        client = _fake_client(session)
        config = SandboxRunConfig(client=client)  # snapshot_store defaults None
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ):
            pass
        # Mirrors the snapshot / manifest conditional forward: a None
        # store is NOT passed, so a backend's default path is unchanged.
        assert "snapshot_store" not in client.create.await_args.kwargs

    async def test_unsupported_snapshot_store_propagates_loudly(self) -> None:
        # A backend that rejects snapshot_store (raises
        # UnsupportedSnapshotFeatureError) MUST surface OUT of
        # sandbox_run_context — never swallowed by teardown or
        # _emit_audit's suppress(Exception). Integration-level, not a
        # unit backend test.
        session = _fake_session()
        client = _fake_client(session)
        client.create = AsyncMock(
            side_effect=UnsupportedSnapshotFeatureError("snapshot_store", "fake"),
        )
        config = SandboxRunConfig(client=client, snapshot_store=object())
        with pytest.raises(UnsupportedSnapshotFeatureError):
            async with sandbox_run_context(
                config=config,
                capabilities=[],
                run_as=None,
                concurrency_guard=None,
            ):
                pass


class TestManifestMaterialization:
    """The run-lifecycle bracket materializes a configured manifest."""

    async def test_manifest_materialized_before_agent_loop_local(self, tmp_path: Any) -> None:
        # Real LocalSubprocessSandboxClient through the real
        # sandbox_run_context bracket (no agent / LLM): the declared
        # file MUST already exist on disk INSIDE the yield (the
        # agent-loop phase) — proving apply_manifest fires after
        # start() and before the loop.
        from troopai.adk.sandbox.clients.local.subprocess_client import (
            LocalSandboxClientOptions,
            LocalSubprocessSandboxClient,
        )
        from troopai.adk.types.sandbox.entries import File
        from troopai.adk.types.sandbox.manifest import Manifest

        manifest = Manifest(entries={"hello.txt": File(content=b"from-manifest")})
        config = SandboxRunConfig(
            client=LocalSubprocessSandboxClient(warn_banner=False),
            manifest=manifest,
            options=LocalSandboxClientOptions(working_directory=str(tmp_path)),
        )
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ):
            assert (tmp_path / "hello.txt").read_bytes() == b"from-manifest"

    async def test_apply_manifest_skipped_for_injected_session(self) -> None:
        # Injected session: runner_owns_session is False → the lifecycle
        # must NOT call apply_manifest (the caller owns that workspace).
        from troopai.adk.sandbox.clients.session import MaterializationResult
        from troopai.adk.types.sandbox.entries import File
        from troopai.adk.types.sandbox.manifest import Manifest

        injected = _fake_session()
        injected.apply_manifest = AsyncMock(return_value=MaterializationResult())
        config = SandboxRunConfig(
            session=injected,
            manifest=Manifest(entries={"x.txt": File(content=b"x")}),
        )
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ):
            pass
        injected.apply_manifest.assert_not_awaited()

    async def test_apply_manifest_skipped_on_resume_path(self) -> None:
        # Resume path (config.session_state set): per the Manifest
        # fresh-session-only contract the resumed workspace persists,
        # so the lifecycle must NOT call apply_manifest even when a
        # manifest is configured (no pointless no-op / spurious POST).
        from troopai.adk.types.sandbox.entries import File
        from troopai.adk.types.sandbox.manifest import Manifest

        resumed = _fake_session()
        resumed.apply_manifest = AsyncMock()
        client = _fake_client(resumed)
        config = SandboxRunConfig(
            client=client,
            session_state=SandboxSessionState(backend_id="unix_local"),
            manifest=Manifest(entries={"x.txt": File(content=b"x")}),
        )
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ):
            pass
        resumed.apply_manifest.assert_not_awaited()
