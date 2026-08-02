# tests/unit/sandbox/test_cost_types.py
import pytest

from troopai.adk.types.sandbox.cost import (
    SandboxBackendCapabilities,
    SandboxBillingRecord,
    SandboxCostDescriptor,
    SandboxRequirements,
)


def test_cost_descriptor_rate_key_free_is_zero():
    assert SandboxCostDescriptor(usd_per_minute=5.0, free=True).rate_key() == 0.0
    assert SandboxCostDescriptor(usd_per_minute=2.0).rate_key() == 2.0


def test_cost_descriptor_cost_for_ms():
    desc = SandboxCostDescriptor(usd_per_minute=6.0)
    assert desc.cost_for_ms(60000) == pytest.approx(6.0)
    assert desc.cost_for_ms(30000) == pytest.approx(3.0)
    assert SandboxCostDescriptor(usd_per_minute=6.0, free=True).cost_for_ms(60000) == 0.0


def test_cost_for_ms_clamps_negative_duration():
    # A negative duration (clock-skew artifact) must never credit cost.
    assert SandboxCostDescriptor(usd_per_minute=6.0).cost_for_ms(-1000) == 0.0


def test_cost_descriptor_rejects_negative_rate():
    with pytest.raises(ValueError):
        SandboxCostDescriptor(usd_per_minute=-1.0)


def test_capabilities_satisfies_network_and_resources():
    caps = SandboxBackendCapabilities(network=True, persistent=True, max_cpu=4, max_memory_mb=8192)
    assert caps.satisfies(SandboxRequirements()) is True
    assert caps.satisfies(SandboxRequirements(network=True, min_cpu=2)) is True
    assert caps.satisfies(SandboxRequirements(min_cpu=8)) is False
    assert SandboxBackendCapabilities(network=False).satisfies(SandboxRequirements(network=True)) is False


def test_capabilities_region_match():
    caps = SandboxBackendCapabilities(regions=("us-east", "eu-west"))
    assert caps.satisfies(SandboxRequirements(region="eu-west")) is True
    assert caps.satisfies(SandboxRequirements(region="ap-south")) is False


def test_capabilities_rejects_unmet_persistent():
    caps = SandboxBackendCapabilities(network=True, persistent=False)
    assert caps.satisfies(SandboxRequirements(persistent=True)) is False
    assert caps.satisfies(SandboxRequirements(persistent=False)) is True


def test_capabilities_empty_regions_rejects_region_requirement():
    assert SandboxBackendCapabilities().satisfies(SandboxRequirements(region="us-east")) is False


def test_capabilities_unknown_capacity_rejects_resource_requirement():
    # max_cpu / max_memory_mb None means "unknown" — a concrete floor fails.
    assert SandboxBackendCapabilities(max_cpu=None).satisfies(SandboxRequirements(min_cpu=1)) is False
    assert SandboxBackendCapabilities(max_memory_mb=None).satisfies(SandboxRequirements(min_memory_mb=512)) is False


def test_billing_record_defaults():
    rec = SandboxBillingRecord(cost_usd=1.25)
    assert rec.cost_usd == 1.25
    assert rec.currency == "USD"
    assert rec.unit is None


def test_billing_record_round_trips_unit_and_raw():
    rec = SandboxBillingRecord(cost_usd=0.0, unit="compute-seconds", raw={"sessions": [1, 2]})
    assert rec.unit == "compute-seconds"
    assert rec.raw == {"sessions": [1, 2]}


def test_billing_record_rejects_negative_cost():
    # A negative provider cost must never read as a credit.
    with pytest.raises(ValueError):
        SandboxBillingRecord(cost_usd=-0.01)
