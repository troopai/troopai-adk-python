"""Tests for :mod:`troopai.adk.workflows.temporal.determinism`."""

from __future__ import annotations

import pytest

temporalio = pytest.importorskip("temporalio")

from troopai.adk.workflows.temporal.determinism import (
    DEFAULT_PASSTHROUGH_MODULES,
    build_sandbox_restrictions,
)


class TestDefaultPassthroughModules:
    def test_contains_required_modules(self) -> None:
        required = {"pydantic", "pydantic_core", "litellm", "openai", "anthropic", "google", "troopai"}
        assert required == set(DEFAULT_PASSTHROUGH_MODULES)

    def test_is_tuple(self) -> None:
        assert isinstance(DEFAULT_PASSTHROUGH_MODULES, tuple)


class TestBuildSandboxRestrictions:
    def test_returns_sandbox_restrictions(self) -> None:
        from temporalio.worker.workflow_sandbox import SandboxRestrictions

        result = build_sandbox_restrictions()
        assert isinstance(result, SandboxRestrictions)

    def test_includes_extra_modules(self) -> None:
        result_default = build_sandbox_restrictions()
        result_extra = build_sandbox_restrictions(extra_passthrough_modules=("numpy",))
        assert result_default is not result_extra

    def test_empty_extras_succeeds(self) -> None:
        result = build_sandbox_restrictions(extra_passthrough_modules=())
        assert result is not None


class TestBuildSandboxRestrictionsImportNotificationPolicy:
    """``build_sandbox_restrictions`` exposes the sandbox import-notification policy.

    The installed temporalio SDK exposes
    ``SandboxRestrictions.with_import_notification_policy(policy)``
    (``_restrictions.py`` line 189) which returns a new ``SandboxRestrictions``
    with ``import_notification_policy`` replaced.  ``SandboxImportNotificationPolicy``
    is a ``Flag`` enum defined in ``temporalio/workflow/_sandbox.py``.
    """

    def test_default_leaves_policy_untouched(self) -> None:
        """Omitting ``import_notification_policy`` preserves the SDK default."""
        from temporalio.worker.workflow_sandbox import SandboxRestrictions

        default_policy = SandboxRestrictions.default.import_notification_policy
        result = build_sandbox_restrictions()

        assert result.import_notification_policy == default_policy

    def test_none_leaves_policy_untouched(self) -> None:
        """Explicitly passing ``None`` is identical to the no-argument call."""
        from temporalio.worker.workflow_sandbox import SandboxRestrictions

        default_policy = SandboxRestrictions.default.import_notification_policy
        result = build_sandbox_restrictions(import_notification_policy=None)

        assert result.import_notification_policy == default_policy

    def test_stricter_policy_is_reflected(self) -> None:
        """A non-``None`` policy is stored on the returned restrictions object."""
        from temporalio.workflow import SandboxImportNotificationPolicy

        policy = SandboxImportNotificationPolicy.RAISE_ON_UNINTENTIONAL_PASSTHROUGH
        result = build_sandbox_restrictions(import_notification_policy=policy)

        assert result.import_notification_policy == policy

    def test_silent_policy_is_reflected(self) -> None:
        """``SILENT`` policy disables all import notifications."""
        from temporalio.workflow import SandboxImportNotificationPolicy

        policy = SandboxImportNotificationPolicy.SILENT
        result = build_sandbox_restrictions(import_notification_policy=policy)

        assert result.import_notification_policy == policy

    def test_policy_combined_with_extra_modules(self) -> None:
        """Passing both ``extra_passthrough_modules`` and a policy works together."""
        from temporalio.workflow import SandboxImportNotificationPolicy

        policy = SandboxImportNotificationPolicy.WARN_ON_UNINTENTIONAL_PASSTHROUGH
        result = build_sandbox_restrictions(
            extra_passthrough_modules=("numpy",),
            import_notification_policy=policy,
        )

        assert result.import_notification_policy == policy
