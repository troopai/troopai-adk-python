# tests/unit/sandbox/test_selector.py
from unittest.mock import MagicMock

import pytest

from troopai.adk.exceptions.exceptions import SandboxSelectionError
from troopai.adk.sandbox.selector import CheapestFirstSelector, SandboxCandidate
from troopai.adk.types.sandbox.cost import (
    SandboxBackendCapabilities,
    SandboxCostDescriptor,
    SandboxRequirements,
)


def _candidate(
    backend_id: str,
    *,
    usd_per_minute: float | None = None,
    free: bool = False,
    network: bool = False,
) -> SandboxCandidate:
    """Build a candidate around a MagicMock client carrying real cost /
    capabilities objects, so the selector exercises real ``satisfies`` /
    ``rate_key`` logic while the mock stands in for ``BaseSandboxClient``."""
    cost = None
    if usd_per_minute is not None or free:
        rate = usd_per_minute if usd_per_minute is not None else 0.0
        cost = SandboxCostDescriptor(usd_per_minute=rate, free=free)
    client = MagicMock()
    client.backend_id = backend_id
    client.cost = cost
    client.capabilities = SandboxBackendCapabilities(network=network)
    return SandboxCandidate(client=client)


def test_cheapest_first_picks_lowest_rate():
    chosen = CheapestFirstSelector().select(
        [_candidate("pricey", usd_per_minute=9.0), _candidate("cheap", usd_per_minute=1.0)],
        SandboxRequirements(),
    )
    assert chosen.client.backend_id == "cheap"


def test_cheapest_first_prefers_free():
    chosen = CheapestFirstSelector().select(
        [_candidate("paid", usd_per_minute=1.0), _candidate("local", free=True)],
        SandboxRequirements(),
    )
    assert chosen.client.backend_id == "local"


def test_unpriced_client_sorts_last():
    chosen = CheapestFirstSelector().select(
        [_candidate("unpriced"), _candidate("priced", usd_per_minute=5.0)],
        SandboxRequirements(),
    )
    assert chosen.client.backend_id == "priced"


def test_filters_on_requirements():
    chosen = CheapestFirstSelector().select(
        [
            _candidate("cheap-no-net", usd_per_minute=1.0, network=False),
            _candidate("net", usd_per_minute=5.0, network=True),
        ],
        SandboxRequirements(network=True),
    )
    assert chosen.client.backend_id == "net"


def test_stable_tiebreak_keeps_first():
    chosen = CheapestFirstSelector().select(
        [_candidate("first", usd_per_minute=2.0), _candidate("second", usd_per_minute=2.0)],
        SandboxRequirements(),
    )
    assert chosen.client.backend_id == "first"


def test_raises_when_none_eligible():
    with pytest.raises(SandboxSelectionError):
        CheapestFirstSelector().select(
            [_candidate("no-net", usd_per_minute=1.0, network=False)],
            SandboxRequirements(network=True),
        )


def test_raises_on_empty_candidates():
    with pytest.raises(SandboxSelectionError):
        CheapestFirstSelector().select([], SandboxRequirements())
