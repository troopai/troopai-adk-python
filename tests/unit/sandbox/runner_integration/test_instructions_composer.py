"""Tests for the sandbox system-prompt composer.

Covers ``compose_sandbox_prompt`` (placeholder resolution, capability
fragment layering, manifest-tree rendering, and the empty/None
short-circuits) plus the ``run/loop.py`` delegation wrapper
``_maybe_compose_sandbox_prompt``.
"""

from __future__ import annotations

from typing import Literal, override

from troopai.adk.run.loop import _maybe_compose_sandbox_prompt
from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.runner_integration.instructions_composer import (
    DEFAULT_SANDBOX_PROMPT,
    SANDBOX_PLACEHOLDER_SYSTEM_PROMPT,
    compose_sandbox_prompt,
)
from troopai.adk.types.sandbox.entries import File
from troopai.adk.types.sandbox.manifest import Manifest


class _InstrCapability(SandboxCapability):
    """Capability whose instruction fragment is configurable per instance."""

    type: Literal["instr"] = "instr"
    fragment: str | None = None

    @override
    async def instructions(self, manifest: Manifest | None) -> str | None:
        del manifest
        return self.fragment


class _FakeHandle:
    """Minimal stand-in for SandboxLifecycleHandle (duck-typed)."""

    def __init__(self, capabilities: list[SandboxCapability], manifest: Manifest | None) -> None:
        self.capabilities = capabilities
        self.manifest = manifest


class _FakeCtx:
    """Run-context stand-in; ``_sandbox_handle`` set only when provided."""

    def __init__(self, handle: _FakeHandle | None = None) -> None:
        if handle is not None:
            self._sandbox_handle = handle


class TestComposeSandboxPrompt:
    async def test_placeholder_resolves_to_default(self) -> None:
        out = await compose_sandbox_prompt(SANDBOX_PLACEHOLDER_SYSTEM_PROMPT, capabilities=[], manifest=None)
        assert out == DEFAULT_SANDBOX_PROMPT

    async def test_placeholder_resolves_to_base_instructions(self) -> None:
        out = await compose_sandbox_prompt(
            SANDBOX_PLACEHOLDER_SYSTEM_PROMPT,
            capabilities=[],
            manifest=None,
            base_instructions="Custom base prompt",
        )
        assert out == "Custom base prompt"

    async def test_placeholder_empty_base_falls_back_to_default(self) -> None:
        # An empty string base_instructions must NOT win over the
        # default (the len(str(...)) > 0 guard).
        out = await compose_sandbox_prompt(
            SANDBOX_PLACEHOLDER_SYSTEM_PROMPT,
            capabilities=[],
            manifest=None,
            base_instructions="",
        )
        assert out == DEFAULT_SANDBOX_PROMPT

    async def test_explicit_prompt_passthrough_ignores_base(self) -> None:
        # A non-placeholder prompt is returned verbatim; base_instructions
        # is irrelevant when the prompt is explicit.
        out = await compose_sandbox_prompt(
            "Explicit operator prompt",
            capabilities=[],
            manifest=None,
            base_instructions="should be ignored",
        )
        assert out == "Explicit operator prompt"

    async def test_manifest_none_still_appends_capability_fragments(self) -> None:
        # Capability instruction fragments are manifest-independent: a
        # ShellCapability (here a stand-in) surfaces its primer even
        # when no workspace manifest was configured. Only the workspace
        # tree is suppressed when manifest is None.
        caps: list[SandboxCapability] = [_InstrCapability(fragment="run_command primer")]
        out = await compose_sandbox_prompt("BASE", capabilities=caps, manifest=None)
        assert out == "BASE\n\nrun_command primer"
        assert "Workspace layout:" not in out

    async def test_manifest_none_no_fragments_returns_prompt_unchanged(self) -> None:
        # With no manifest AND no fragment-producing capability, the
        # prompt is returned verbatim — no spurious blank lines.
        caps: list[SandboxCapability] = [_InstrCapability(fragment=None)]
        out = await compose_sandbox_prompt("BASE", capabilities=caps, manifest=None)
        assert out == "BASE"

    async def test_capability_fragments_appended_filtering_none(self) -> None:
        caps: list[SandboxCapability] = [
            _InstrCapability(fragment="frag-A"),
            _InstrCapability(fragment=None),  # filtered (None)
            _InstrCapability(fragment=""),  # filtered (empty — the len(...) > 0 half)
            _InstrCapability(fragment="frag-B"),
        ]
        out = await compose_sandbox_prompt("BASE", capabilities=caps, manifest=Manifest(entries={}))
        # collect_capability_instructions drops BOTH None and empty
        # fragments (`fragment is not None and len(fragment) > 0`);
        # survivors joined by blank lines.
        assert out == "BASE\n\nfrag-A\n\nfrag-B"

    async def test_manifest_tree_appended(self) -> None:
        manifest = Manifest(entries={"a.txt": File(content=b"x"), "d/b.txt": File(content=b"y")})
        out = await compose_sandbox_prompt("BASE", capabilities=[], manifest=manifest)
        assert out.startswith("BASE\n\nWorkspace layout:")
        assert "  /a.txt" in out
        assert "  /d/b.txt" in out

    async def test_fragments_and_tree_combined(self) -> None:
        manifest = Manifest(entries={"only.txt": File(content=b"z")})
        caps: list[SandboxCapability] = [_InstrCapability(fragment="cap-frag")]
        out = await compose_sandbox_prompt("BASE", capabilities=caps, manifest=manifest)
        assert "BASE\n\ncap-frag\n\n" in out
        assert "Workspace layout:" in out
        assert "  /only.txt" in out

    async def test_empty_manifest_no_fragments_no_tree(self) -> None:
        # Empty manifest + no fragment-producing caps → just the prompt.
        out = await compose_sandbox_prompt("BASE", capabilities=[], manifest=Manifest(entries={}))
        assert out == "BASE"

    async def test_shell_primer_appended_when_no_manifest(self) -> None:
        # A real ShellCapability surfaces its run_command primer even
        # with manifest=None — the manifest-less client= config mode is
        # valid and the model must still learn how to use run_command.
        caps: list[SandboxCapability] = [ShellCapability()]
        out = await compose_sandbox_prompt(DEFAULT_SANDBOX_PROMPT, capabilities=caps, manifest=None)
        assert out.startswith(DEFAULT_SANDBOX_PROMPT)
        assert "run_command" in out
        assert "Workspace layout:" not in out


class TestMaybeComposeSandboxPrompt:
    async def test_no_handle_returns_prompt_unchanged(self) -> None:
        # Escape-hatch rationale (applies to the suppression on the
        # next line): _maybe_compose_sandbox_prompt's ONLY ctx access
        # is getattr(ctx, "_sandbox_handle", None) — no isinstance
        # gate — so a duck-typed _FakeCtx is the sanctioned
        # test-scaffolding boundary; constructing a real RunContext
        # here would add nothing the function reads. No _sandbox_handle
        # attr set → real getattr-None branch → passthrough.
        out = await _maybe_compose_sandbox_prompt("untouched", _FakeCtx())  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        assert out == "untouched"

    async def test_with_handle_delegates_to_composer(self) -> None:
        # A bound handle → delegates; the manifest tree proves the
        # composer actually ran (not a bare passthrough).
        caps: list[SandboxCapability] = [_InstrCapability(fragment="bound-frag")]
        handle = _FakeHandle(
            capabilities=caps,
            manifest=Manifest(entries={"f.txt": File(content=b"q")}),
        )
        # Escape-hatch rationale (suppression on the next line): same
        # invariant as test_no_handle — the function only
        # getattr-probes ctx._sandbox_handle, so _FakeCtx is the
        # sanctioned duck-typed test-scaffolding boundary.
        out = await _maybe_compose_sandbox_prompt("P", _FakeCtx(handle))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        assert "P\n\nbound-frag" in out
        assert "Workspace layout:" in out
        assert "  /f.txt" in out

    async def test_with_handle_placeholder_resolves_to_default(self) -> None:
        # troopai contract pin: SandboxAgent has NO base_instructions
        # (it uses system_prompt). The placeholder sentinel arriving
        # via the wrapper resolves to DEFAULT_SANDBOX_PROMPT — the
        # wrapper deliberately does NOT plumb base_instructions (a
        # vestigial OpenAI-port param on compose_sandbox_prompt).
        # Override by setting an explicit system_prompt, never
        # base_instructions. This pins the real troopai behavior so a
        # future "plumb base_instructions" regression fails loud.
        handle = _FakeHandle(capabilities=[], manifest=None)
        # Escape-hatch rationale (suppression on the next line): same
        # getattr-only duck-typed boundary as the tests above.
        out = await _maybe_compose_sandbox_prompt(SANDBOX_PLACEHOLDER_SYSTEM_PROMPT, _FakeCtx(handle))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        assert out == DEFAULT_SANDBOX_PROMPT
