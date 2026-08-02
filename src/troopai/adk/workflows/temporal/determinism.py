"""Temporal sandbox passthrough configuration for TroopAI ADK.

Temporal's workflow sandbox intercepts module imports to enforce
deterministic replay.  Libraries that are safe to import without
sandboxing (because they have no non-deterministic side-effects at
import time) must be registered as passthrough modules.

This module defines the default set of passthrough modules for
TroopAI workflows and provides a factory that produces a configured
:class:`~temporalio.worker.workflow_sandbox.SandboxRestrictions` object.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from temporalio.worker.workflow_sandbox import SandboxRestrictions
    from temporalio.workflow import SandboxImportNotificationPolicy

logger = logging.getLogger(__name__)


DEFAULT_PASSTHROUGH_MODULES: tuple[str, ...] = (
    "pydantic",
    "pydantic_core",
    "litellm",
    "openai",
    "anthropic",
    "google",
    "troopai",
)
"""Modules that are always passed through the Temporal sandbox.

These libraries are safe to import in a deterministic workflow
context — they do not perform I/O or read wall-clock time at module
level.  Callers can extend this set via ``build_sandbox_restrictions``.
"""


def build_sandbox_restrictions(
    extra_passthrough_modules: tuple[str, ...] = (),
    *,
    import_notification_policy: SandboxImportNotificationPolicy | None = None,
) -> SandboxRestrictions:
    """Return a :class:`~temporalio.worker.workflow_sandbox.SandboxRestrictions` for TroopAI workflows.

    Starts from :attr:`SandboxRestrictions.default` and registers each
    module in :data:`DEFAULT_PASSTHROUGH_MODULES` plus any caller-supplied
    extras as sandbox passthrough entries.

    Args:
        extra_passthrough_modules: Additional module names to pass through
            the sandbox beyond :data:`DEFAULT_PASSTHROUGH_MODULES`.
        import_notification_policy: Optional
            :class:`~temporalio.workflow.SandboxImportNotificationPolicy`
            controlling how the sandbox reacts to dynamic or unintentional
            imports.  When ``None`` (the default), the policy from
            :attr:`~temporalio.worker.workflow_sandbox.SandboxRestrictions.default`
            is preserved unchanged — currently
            ``SandboxImportNotificationPolicy.WARN_ON_DYNAMIC_IMPORT``.
            Pass a stricter value such as
            ``SandboxImportNotificationPolicy.RAISE_ON_UNINTENTIONAL_PASSTHROUGH``
            to catch missing passthrough registrations as errors rather than
            warnings.

    Returns:
        A fully configured ``SandboxRestrictions`` instance.
    """
    from temporalio.worker.workflow_sandbox import SandboxRestrictions

    all_modules: tuple[str, ...] = DEFAULT_PASSTHROUGH_MODULES + extra_passthrough_modules
    logger.debug("Configuring sandbox with %d passthrough modules", len(all_modules))

    restrictions = SandboxRestrictions.default.with_passthrough_modules(*all_modules)
    if import_notification_policy is not None:
        restrictions = restrictions.with_import_notification_policy(import_notification_policy)
        logger.debug("Sandbox import_notification_policy set to %s", import_notification_policy)
    return restrictions
