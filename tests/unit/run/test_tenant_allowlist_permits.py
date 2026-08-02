from __future__ import annotations

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.run.governance import tenant_allowlist_permits


def test_feature_off_permits_everything() -> None:
    assert tenant_allowlist_permits(RunConfig(), "t1", "any_tool") is True


def test_untenanted_run_is_not_governed() -> None:
    config = RunConfig(tenant_tool_allowlist={"t1": {"search"}})
    assert tenant_allowlist_permits(config, None, "search") is True


def test_untenanted_run_bypasses_default_deny() -> None:
    # Security invariant: the None-tenant guard fires before default_deny,
    # so an untenanted run is never tenant-governed even under fail-closed.
    config = RunConfig(
        tenant_tool_allowlist={"t1": {"search"}},
        tenant_allowlist_default_deny=True,
    )
    assert tenant_allowlist_permits(config, None, "anything") is True


def test_tool_in_set_permitted_and_out_of_set_denied() -> None:
    config = RunConfig(tenant_tool_allowlist={"t1": {"search"}})
    assert tenant_allowlist_permits(config, "t1", "search") is True
    assert tenant_allowlist_permits(config, "t1", "delete") is False


def test_empty_set_denies_all_tools() -> None:
    config = RunConfig(tenant_tool_allowlist={"t1": set()})
    assert tenant_allowlist_permits(config, "t1", "search") is False


@pytest.mark.parametrize(("default_deny", "expected"), [(False, True), (True, False)])
def test_absent_tenant_follows_default_deny(default_deny: bool, expected: bool) -> None:
    config = RunConfig(
        tenant_tool_allowlist={"t1": {"search"}},
        tenant_allowlist_default_deny=default_deny,
    )
    assert tenant_allowlist_permits(config, "t2", "search") is expected
