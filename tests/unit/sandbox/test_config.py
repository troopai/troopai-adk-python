"""Tests for ``SandboxRunConfig`` and its wiring into ``RunConfig``."""

from __future__ import annotations

from pathlib import Path

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.sandbox.clients.local.subprocess_client import LocalSubprocessSandboxClient
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.selector import CheapestFirstSelector, SandboxCandidate
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.network import NetworkPolicy
from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits
from troopai.adk.types.sandbox.session_state import SandboxSessionState
from troopai.adk.types.sandbox.snapshot import LocalSnapshotSpec


class _FakeBackendClient:
    """Stand-in for ``BaseSandboxClient``."""


class _FakeSession:
    """Stand-in for ``BaseSandboxSession``."""


class TestSandboxRunConfigConstruction:
    def test_session_only(self) -> None:
        session = _FakeSession()
        cfg = SandboxRunConfig(session=session)
        assert cfg.session is session
        assert cfg.client is None
        assert cfg.manifest is None

    def test_client_only(self) -> None:
        client = _FakeBackendClient()
        cfg = SandboxRunConfig(client=client)
        assert cfg.client is client

    def test_session_state_only(self) -> None:
        state = SandboxSessionState(backend_id="docker")
        cfg = SandboxRunConfig(session_state=state)
        assert cfg.session_state is state

    def test_full_fresh_session_config(self) -> None:
        client = _FakeBackendClient()
        manifest = Manifest(root="/workspace")
        snapshot = LocalSnapshotSpec(base_path=Path("/tmp/snaps"))
        cfg = SandboxRunConfig(
            client=client,
            manifest=manifest,
            snapshot=snapshot,
            resource_limits=SandboxResourceLimits(memory_mb=512),
            network_policy=NetworkPolicy(deny_default=True),
        )
        assert cfg.client is client
        assert cfg.manifest is manifest
        assert cfg.snapshot is snapshot
        assert cfg.resource_limits is not None
        assert cfg.resource_limits.memory_mb == 512
        assert cfg.network_policy is not None
        assert cfg.network_policy.deny_default is True


class TestSandboxRunConfigValidation:
    def test_all_none_rejected(self) -> None:
        with pytest.raises(ValueError, match="one of: session=, session_state="):
            SandboxRunConfig()


class TestRunConfigSandboxField:
    def test_default_is_none(self) -> None:
        cfg = RunConfig()
        assert cfg.sandbox is None

    def test_explicit_sandbox_wired(self) -> None:
        sb = SandboxRunConfig(client=_FakeBackendClient())
        cfg = RunConfig(sandbox=sb)
        assert cfg.sandbox is sb

    def test_run_config_unchanged_when_sandbox_unset(self) -> None:
        # Regression: adding the field must not break the existing
        # RunConfig defaults for any other attribute.
        cfg = RunConfig()
        assert cfg.tracing_enabled is False
        assert cfg.model is None


class TestSandboxRunConfigSelectorPath:
    def test_config_accepts_selector_plus_candidates(self) -> None:
        cfg = SandboxRunConfig(
            selector=CheapestFirstSelector(),
            candidates=[SandboxCandidate(client=LocalSubprocessSandboxClient(warn_banner=False))],
        )
        assert cfg.capture_live_cost is False  # cost-conservative default
        assert cfg.requirements is None

    def test_config_rejects_selector_without_candidates(self) -> None:
        # Another acquisition path (client) is set so the generic "no path"
        # guard passes and the dedicated selector-needs-candidates guard fires.
        with pytest.raises(ValueError, match="selector requires a non-empty candidates="):
            SandboxRunConfig(
                client=LocalSubprocessSandboxClient(warn_banner=False),
                selector=CheapestFirstSelector(),
            )

    def test_config_rejects_selector_with_empty_candidates_list(self) -> None:
        # The other branch of the guard: an explicitly empty list (len == 0),
        # distinct from candidates=None covered above.
        with pytest.raises(ValueError, match="selector requires a non-empty candidates="):
            SandboxRunConfig(
                client=LocalSubprocessSandboxClient(warn_banner=False),
                selector=CheapestFirstSelector(),
                candidates=[],
            )

    def test_config_still_rejects_completely_empty(self) -> None:
        with pytest.raises(ValueError, match="one of: session=, session_state="):
            SandboxRunConfig()
