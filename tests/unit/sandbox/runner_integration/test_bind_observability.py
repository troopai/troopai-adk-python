"""Tests for ``bind_observability`` on ``SandboxCapability`` and
``bind_capabilities`` observability threading."""

from __future__ import annotations

from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.runner_integration.capability_lifecycle import bind_capabilities


class TestBindObservability:
    def test_bind_observability_sets_field(self) -> None:
        cap = ShellCapability()
        sentinel = object()
        cap.bind_observability(sentinel)
        assert cap.observability is sentinel

    def test_bind_capabilities_threads_observability(self) -> None:
        caps = [ShellCapability()]
        sentinel = object()
        bind_capabilities(caps, session=object(), run_as=None, observability=sentinel)
        assert caps[0].observability is sentinel
        assert caps[0].session is not None

    def test_bind_capabilities_observability_defaults_to_none(self) -> None:
        caps = [ShellCapability()]
        bind_capabilities(caps, session=object(), run_as=None)
        assert caps[0].observability is None

    def test_clone_resets_observability(self) -> None:
        cap = ShellCapability()
        sentinel = object()
        cap.bind_observability(sentinel)
        cloned = cap.clone()
        assert cloned.observability is None
        # Original still bound.
        assert cap.observability is sentinel
