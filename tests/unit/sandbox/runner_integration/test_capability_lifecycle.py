"""Tests for the capability-lifecycle helpers (P31)."""

from __future__ import annotations

from typing import Literal

import pytest

from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.runner_integration.capability_lifecycle import (
    apply_command_policy,
    bind_capabilities,
    clone_capabilities,
    collect_capability_instructions,
    collect_capability_tools,
    process_manifest_through_capabilities,
    validate_required_capability_types,
)
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.permissions import User


class _MarkerCapability(SandboxCapability):
    """Capability that appends a marker to manifest entries on process."""

    type: Literal["marker"] = "marker"
    marker: str = "M"
    instruction: str | None = None

    def process_manifest(self, manifest: Manifest) -> Manifest:
        # Append a synthetic File entry named after the marker so we
        # can verify ordering downstream.
        from troopai.adk.types.sandbox.entries import File

        new_entries = dict(manifest.entries)
        new_entries[f"{self.marker}.txt"] = File(content=self.marker.encode())
        return Manifest(
            root=manifest.root,
            entries=new_entries,
            environment=manifest.environment,
            users=manifest.users,
            groups=manifest.groups,
            extra_path_grants=manifest.extra_path_grants,
            remote_mount_command_allowlist=manifest.remote_mount_command_allowlist,
        )

    async def instructions(self, manifest: Manifest) -> str | None:
        del manifest
        return self.instruction


class _DependsOnMarker(SandboxCapability):
    """Capability that declares a structural dependency on _MarkerCapability."""

    type: Literal["depends_on_marker"] = "depends_on_marker"

    def required_capability_types(self) -> set[str]:
        return {"marker"}


class TestCloneCapabilities:
    def test_clone_preserves_order_and_count(self) -> None:
        caps = [_MarkerCapability(marker="A"), _MarkerCapability(marker="B")]
        cloned = clone_capabilities(caps)
        assert len(cloned) == 2
        assert cloned[0].marker == "A"  # type: ignore[attr-defined]
        assert cloned[1].marker == "B"  # type: ignore[attr-defined]

    def test_clone_independent_from_originals(self) -> None:
        caps = [_MarkerCapability()]
        caps[0].bind("session-X")
        cloned = clone_capabilities(caps)
        # Clone's session reset to None per the capability contract.
        assert cloned[0].session is None
        # Original still bound.
        assert caps[0].session == "session-X"


class TestBindCapabilities:
    def test_binds_session_and_run_as(self) -> None:
        caps = [_MarkerCapability(), _MarkerCapability()]
        user = User(name="alice")
        bind_capabilities(caps, session="S", run_as=user)
        for c in caps:
            assert c.session == "S"
            assert c.run_as == user

    def test_bind_with_no_run_as(self) -> None:
        caps = [_MarkerCapability()]
        bind_capabilities(caps, session="S", run_as=None)
        assert caps[0].run_as is None


class TestProcessManifestFold:
    def test_sequential_fold(self) -> None:
        caps: list[SandboxCapability] = [
            _MarkerCapability(marker="first"),
            _MarkerCapability(marker="second"),
        ]
        result = process_manifest_through_capabilities(caps, Manifest())
        # Both markers visible in final manifest.
        assert "first.txt" in result.entries
        assert "second.txt" in result.entries

    def test_empty_capabilities_returns_unchanged(self) -> None:
        m = Manifest(root="/workspace")
        result = process_manifest_through_capabilities([], m)
        assert result is m


class TestCollectTools:
    def test_empty(self) -> None:
        assert collect_capability_tools([]) == []

    def test_concatenates(self) -> None:
        caps: list[SandboxCapability] = [_MarkerCapability(), _MarkerCapability()]
        result = collect_capability_tools(caps)
        # _MarkerCapability has no tools — empty list.
        assert result == []


class TestCollectInstructions:
    @pytest.mark.asyncio
    async def test_filters_none_and_empty(self) -> None:
        caps: list[SandboxCapability] = [
            _MarkerCapability(instruction="rule A"),
            _MarkerCapability(instruction=None),
            _MarkerCapability(instruction=""),
            _MarkerCapability(instruction="rule B"),
        ]
        result = await collect_capability_instructions(caps, Manifest())
        assert result == ["rule A", "rule B"]

    @pytest.mark.asyncio
    async def test_empty_capabilities_returns_empty_list(self) -> None:
        result = await collect_capability_instructions([], Manifest())
        assert result == []


class TestValidateRequiredCapabilityTypes:
    def test_dependency_satisfied(self) -> None:
        caps: list[SandboxCapability] = [
            _MarkerCapability(),
            _DependsOnMarker(),
        ]
        # No raise.
        validate_required_capability_types(caps)

    def test_missing_dependency_raises(self) -> None:
        caps: list[SandboxCapability] = [_DependsOnMarker()]
        with pytest.raises(ValueError, match="requires.*marker"):
            validate_required_capability_types(caps)

    def test_empty_capabilities_passes(self) -> None:
        # No raise.
        validate_required_capability_types([])


class TestApplyCommandPolicy:
    def test_run_policy_applied_to_shell_clone(self) -> None:
        # Mirror the production use-site: lifecycle.py always calls
        # apply_command_policy on the per-run CLONE list. Assert the
        # clone gets the policy AND the original stays untouched.
        sentinel = object()
        shell = ShellCapability()
        clone = shell.clone()
        apply_command_policy([clone], sentinel)
        assert clone.command_policy is sentinel
        assert shell.command_policy is None

    def test_none_policy_is_noop_keeps_capability_own(self) -> None:
        own = object()
        shell = ShellCapability(command_policy=own)
        apply_command_policy([shell], None)
        # None config → capability keeps its own policy untouched.
        assert shell.command_policy is own

    def test_run_policy_overrides_capability_own(self) -> None:
        own = object()
        run = object()
        shell = ShellCapability(command_policy=own)
        apply_command_policy([shell], run)
        # Run-level config is the outer override — it wins.
        assert shell.command_policy is run

    def test_non_shell_capability_untouched(self) -> None:
        sentinel = object()
        marker = _MarkerCapability()
        # _MarkerCapability is not a ShellCapability → the isinstance
        # dispatch skips it; no attribute is set, no error raised.
        apply_command_policy([marker], sentinel)
        assert not hasattr(marker, "command_policy")
