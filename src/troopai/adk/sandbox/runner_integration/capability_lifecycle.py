"""Capability-lifecycle helpers consumed by the sandbox runner integration.

A single agent loop turn against a SandboxAgent goes through five
capability-lifecycle steps:

1. ``clone_capabilities`` — per-run copies so concurrent runs cannot
   race on shared state.
2. ``bind_capabilities`` — attach the live session, run-as user, and
   observability handle to every cloned capability.
3. ``process_manifest_through_capabilities`` — fold each capability's
   ``process_manifest`` over the manifest sequentially.
4. ``collect_capability_tools`` — flatten ``cap.tools()`` from each
   capability into a single tool list for the agent loop.
5. ``collect_capability_instructions`` — gather + join async
   instruction fragments.

The Runner integration composes these into the bracketing
context manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from troopai.adk.sandbox.capabilities.base import SandboxCapability
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.permissions import User

__all__ = [
    "apply_command_policy",
    "bind_capabilities",
    "clone_capabilities",
    "collect_capability_instructions",
    "collect_capability_tools",
    "process_manifest_through_capabilities",
    "validate_required_capability_types",
]


def apply_command_policy(
    capabilities: list[SandboxCapability],
    command_policy: object | None,
) -> None:
    """Apply a run-level command policy onto every Shell clone.

    ``SandboxRunConfig.command_policy`` is a run-scoped override:
    when set, it replaces each cloned ``ShellCapability``'s
    ``command_policy`` so the run_command tool enforces the run's
    policy. ``None`` is a no-op — each per-run clone keeps a DEEP
    COPY of its capability's own policy (``clone_capabilities``
    deep-copies fields for per-run isolation, so the tool enforces a
    copy, not the developer's exact object; a stateless
    ``SandboxCommandGuardrail`` is unaffected, while a stateful
    custom policy sees independent per-run state). Only the per-run
    CLONES are ever mutated — the developer's original capability
    list is never touched, so a later run with no override is
    unaffected.
    """
    if command_policy is None:
        return
    from troopai.adk.sandbox.capabilities.shell import ShellCapability

    for cap in capabilities:
        if isinstance(cap, ShellCapability):
            cap.command_policy = command_policy


def clone_capabilities(
    capabilities: list[SandboxCapability],
) -> list[SandboxCapability]:
    """Return per-run clones, preserving order.

    Each clone has a fresh asyncio.Lock / Event / Semaphore /
    Condition; sandbox session references reset to None.
    """
    return [c.clone() for c in capabilities]


def bind_capabilities(
    capabilities: Sequence[SandboxCapability],
    *,
    session: object,
    run_as: User | None,
    observability: object | None = None,
) -> None:
    """Bind ``session``, ``run_as``, and ``observability`` on every capability.

    Separate passes so a capability that overrides ``bind_run_as`` to read
    ``self.session`` sees a populated session before later passes run.
    """
    for cap in capabilities:
        cap.bind(session)
    for cap in capabilities:
        cap.bind_run_as(run_as)
    for cap in capabilities:
        cap.bind_observability(observability)


def process_manifest_through_capabilities(
    capabilities: list[SandboxCapability],
    manifest: Manifest,
) -> Manifest:
    """Fold every capability's ``process_manifest`` sequentially.

    Later capabilities see earlier capabilities' mutations.
    """
    current = manifest
    for cap in capabilities:
        current = cap.process_manifest(current)
    return current


def collect_capability_tools(
    capabilities: list[SandboxCapability],
) -> list[FunctionTool]:
    """Concatenate ``cap.tools()`` from every capability in order."""
    result: list[FunctionTool] = []
    for cap in capabilities:
        result.extend(cap.tools())
    return result


async def collect_capability_instructions(
    capabilities: list[SandboxCapability],
    manifest: Manifest | None,
) -> list[str]:
    """Return the non-None instruction fragments in order.

    Async so capabilities that read live workspace state can fetch
    it before the prompt is finalized. ``manifest`` is ``None`` when no
    workspace contract was configured; capabilities whose primer does
    not depend on the manifest (e.g. the shell run_command guidance)
    still contribute their fragment.
    """
    fragments: list[str] = []
    for cap in capabilities:
        fragment = await cap.instructions(manifest)
        if fragment is not None and len(fragment) > 0:
            fragments.append(fragment)
    return fragments


def validate_required_capability_types(
    capabilities: list[SandboxCapability],
) -> None:
    """Raise ``ValueError`` if any capability's required types are missing.

    Called by the Runner integration AFTER ``clone_capabilities`` and
    BEFORE ``bind_capabilities`` so structural deps are checked once
    per run.
    """
    present_types = {cap.type for cap in capabilities}
    for cap in capabilities:
        required = cap.required_capability_types()
        missing = required - present_types
        if len(missing) > 0:
            raise ValueError(
                f"{type(cap).__name__} (type={cap.type!r}) requires "
                f"capabilities {sorted(required)}; missing {sorted(missing)}. "
                f"Add them to SandboxAgent.capabilities."
            )
