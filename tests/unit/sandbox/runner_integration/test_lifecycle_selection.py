"""Selector-driven session acquisition through the public lifecycle bracket."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.runner_integration.lifecycle import sandbox_run_context
from troopai.adk.sandbox.selector import CheapestFirstSelector, SandboxCandidate
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities, SandboxCostDescriptor


def _client(backend_id: str, *, free: bool) -> MagicMock:
    session = AsyncMock()
    session.session_id = f"{backend_id}-sess"
    session.start = AsyncMock()
    session.apply_manifest = AsyncMock()
    session.aclose = AsyncMock()
    c = MagicMock()
    c.backend_id = backend_id
    c.cost = SandboxCostDescriptor(free=free)
    c.capabilities = SandboxBackendCapabilities(network=True)
    c.create = AsyncMock(return_value=session)
    return c


async def test_selector_chooses_cheapest_when_no_explicit_source() -> None:
    cheap = _client("cheap", free=True)
    pricey = _client("pricey", free=False)
    pricey.cost = SandboxCostDescriptor(usd_per_minute=9.0)
    cfg = SandboxRunConfig(
        selector=CheapestFirstSelector(),
        candidates=[SandboxCandidate(client=pricey), SandboxCandidate(client=cheap)],
    )

    async with sandbox_run_context(
        config=cfg,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent=MagicMock(),
        run_context=MagicMock(),
        hooks=AsyncMock(),
        tracing_enabled=False,
    ) as handle:
        assert handle.runner_owns_session is True
        assert handle.observability is not None
        assert handle.observability.backend_id == "cheap"

    cheap.create.assert_awaited_once()
    pricey.create.assert_not_awaited()


async def test_explicit_client_beats_selector() -> None:
    explicit = _client("explicit", free=True)
    selectable = _client("selectable", free=True)
    cfg = SandboxRunConfig(
        client=explicit,
        selector=CheapestFirstSelector(),
        candidates=[SandboxCandidate(client=selectable)],
    )

    async with sandbox_run_context(
        config=cfg,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent=MagicMock(),
        run_context=MagicMock(),
        hooks=AsyncMock(),
        tracing_enabled=False,
    ) as handle:
        assert handle.observability is not None
        assert handle.observability.backend_id == "explicit"

    explicit.create.assert_awaited_once()
    selectable.create.assert_not_awaited()
