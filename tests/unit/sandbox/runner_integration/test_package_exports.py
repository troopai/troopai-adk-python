"""The ``runner_integration`` package re-exports the per-run entry points.

These are the symbols callers are expected to import from the package
rather than reaching into submodules; the test pins the public
surface so a missing re-export (or an ``__all__`` drift) fails loudly.
"""

from __future__ import annotations

import troopai.adk.sandbox.runner_integration as ri

_EXPECTED_EXPORTS = (
    "SandboxConcurrencyGuard",
    "SandboxLifecycleHandle",
    "apply_iac",
    "compose_sandbox_prompt",
    "destroy_iac",
    "sandbox_run_context",
)


def test_public_surface_is_importable() -> None:
    for name in _EXPECTED_EXPORTS:
        assert hasattr(ri, name), f"{name} is not importable from the package"


def test_all_is_exact_and_sorted() -> None:
    # RUF022 grouped order: the two CamelCase classes first, then the
    # lowercase functions, each block sorted.
    assert ri.__all__ == [
        "SandboxConcurrencyGuard",
        "SandboxLifecycleHandle",
        "apply_iac",
        "compose_sandbox_prompt",
        "destroy_iac",
        "sandbox_run_context",
    ]


def test_reexports_are_the_submodule_objects() -> None:
    from troopai.adk.sandbox.runner_integration.iac_runner import (
        apply_iac,
        destroy_iac,
    )
    from troopai.adk.sandbox.runner_integration.instructions_composer import (
        compose_sandbox_prompt,
    )
    from troopai.adk.sandbox.runner_integration.lifecycle import (
        SandboxLifecycleHandle,
        sandbox_run_context,
    )

    assert ri.SandboxLifecycleHandle is SandboxLifecycleHandle
    assert ri.sandbox_run_context is sandbox_run_context
    assert ri.compose_sandbox_prompt is compose_sandbox_prompt
    assert ri.apply_iac is apply_iac
    assert ri.destroy_iac is destroy_iac
