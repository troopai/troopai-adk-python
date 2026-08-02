"""Tests for the sandbox lifecycle building/binding observability + hooks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.runner_integration.lifecycle import sandbox_run_context
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities, SandboxCostDescriptor


def _client() -> tuple[MagicMock, AsyncMock]:
    session = AsyncMock()
    session.session_id = "sess-1"
    session.start = AsyncMock()
    session.apply_manifest = AsyncMock()
    session.aclose = AsyncMock()
    c = MagicMock()
    c.backend_id = "local"
    c.cost = SandboxCostDescriptor(free=True)
    c.capabilities = SandboxBackendCapabilities(network=True)
    c.create = AsyncMock(return_value=session)
    c.fetch_billing = AsyncMock(return_value=None)
    return c, session


async def test_lifecycle_binds_obs_and_fires_start_stop() -> None:
    client, _session = _client()
    hooks = AsyncMock()
    cfg = SandboxRunConfig(client=client)
    caps: list[SandboxCapability] = [ShellCapability()]
    async with sandbox_run_context(
        config=cfg,
        capabilities=caps,
        run_as=None,
        concurrency_guard=None,
        agent=MagicMock(),
        run_context=MagicMock(),
        hooks=hooks,
        tracing_enabled=False,
    ) as handle:
        assert handle.observability is not None
        assert handle.observability.backend_id == "local"
        assert handle.capabilities[0].observability is handle.observability
        usage = handle.observability.usage
    hooks.on_sandbox_start.assert_awaited_once()
    hooks.on_sandbox_stop.assert_awaited_once()
    # on_sandbox_stop receives the run's accumulated usage (4th positional arg).
    assert hooks.on_sandbox_stop.await_args.args[3] is usage
