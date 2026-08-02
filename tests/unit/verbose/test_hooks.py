"""Tests for :class:`VerboseHooks` lifecycle rendering.

These tests wire :class:`VerboseHooks` directly (bypassing the
:class:`~troopai.adk.run.runner.Runner`) and invoke lifecycle methods
to verify events land on the configured stream with the right prefix/
icon, and that the per-agent override mechanism wins over the
run-level default.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pytest

from troopai.adk.verbose.config import (
    EVENT_TOOL_START,
    EventStyle,
    VerboseConfig,
)
from troopai.adk.verbose.hooks import VerboseHooks


@dataclass
class _FakeAgent:
    name: str
    verbose: VerboseConfig | None = None
    hooks: None = None


@dataclass
class _FakeResult:
    final_output: str = "done"


@pytest.mark.asyncio
async def test_on_agent_start_renders_headline() -> None:
    stream = io.StringIO()
    run_cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(run_cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_agent_start(None, agent)  # type: ignore[arg-type]

    out = stream.getvalue()
    assert "Alice" in out
    assert "[agent]" in out


@pytest.mark.asyncio
async def test_on_agent_start_line_mode_lists_capabilities() -> None:
    """Line mode shows the agent's tools / skills / handoffs as payload rows."""

    @dataclass
    class _Named:
        name: str

    @dataclass
    class _HandoffDecl:
        agent_name: str

    stream = io.StringIO()
    run_cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(run_cfg)
    agent = _FakeAgent(name="Alice")
    agent.description = "Answers billing questions."  # type: ignore[attr-defined]
    agent.tools = [_Named("lookup_invoice"), _Named("refund")]  # type: ignore[attr-defined]
    agent.skills = [_Named("cite_policy")]  # type: ignore[attr-defined]
    agent.handoffs = [_HandoffDecl("Escalations")]  # type: ignore[attr-defined]

    await hooks.on_agent_start(None, agent)  # type: ignore[arg-type]

    out = stream.getvalue()
    assert "Alice" in out
    assert "Description: Answers billing questions." in out
    assert "Tools: lookup_invoice, refund" in out
    assert "Skills: cite_policy" in out
    assert "Handoffs: Escalations" in out


@pytest.mark.asyncio
async def test_on_agent_start_silent_when_run_cfg_disabled() -> None:
    stream = io.StringIO()
    run_cfg = VerboseConfig(enabled=False, output=stream, use_rich=False)
    hooks = VerboseHooks(run_cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_agent_start(None, agent)  # type: ignore[arg-type]

    assert stream.getvalue() == ""


@pytest.mark.asyncio
async def test_per_agent_override_wins() -> None:
    run_stream = io.StringIO()
    agent_stream = io.StringIO()
    run_cfg = VerboseConfig(output=run_stream, use_rich=False)
    agent_cfg = VerboseConfig(output=agent_stream, use_rich=False)
    agent = _FakeAgent(name="Alice", verbose=agent_cfg)

    hooks = VerboseHooks(run_cfg)
    await hooks.on_agent_start(None, agent)  # type: ignore[arg-type]

    # Only the agent-level stream should have received output.
    assert "Alice" in agent_stream.getvalue()
    assert run_stream.getvalue() == ""


@pytest.mark.asyncio
async def test_per_agent_override_can_silence() -> None:
    run_stream = io.StringIO()
    run_cfg = VerboseConfig(output=run_stream, use_rich=False)
    silent_agent = _FakeAgent(
        name="Bob",
        verbose=VerboseConfig(enabled=False),
    )

    hooks = VerboseHooks(run_cfg)
    await hooks.on_agent_start(None, silent_agent)  # type: ignore[arg-type]

    assert run_stream.getvalue() == ""


@pytest.mark.asyncio
async def test_on_tool_start_includes_args() -> None:
    stream = io.StringIO()
    run_cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(run_cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_tool_start(None, agent, "search", {"q": "anthropic"})  # type: ignore[arg-type]

    out = stream.getvalue()
    assert "search" in out
    assert "anthropic" in out


@pytest.mark.asyncio
async def test_on_handoff_renders_arrow() -> None:
    stream = io.StringIO()
    run_cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(run_cfg)
    a = _FakeAgent(name="Coordinator")
    b = _FakeAgent(name="Specialist")

    await hooks.on_handoff(None, a, b)  # type: ignore[arg-type]

    out = stream.getvalue()
    assert "Coordinator" in out and "Specialist" in out
    assert "→" in out


@pytest.mark.asyncio
async def test_null_run_cfg_with_no_agent_cfg_emits_nothing() -> None:
    hooks = VerboseHooks(None)
    agent = _FakeAgent(name="Alice")

    # Should not raise even though there is no config anywhere.
    await hooks.on_agent_start(None, agent)  # type: ignore[arg-type]
    await hooks.on_tool_start(None, agent, "x", {})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_agent_end_includes_final_output() -> None:
    stream = io.StringIO()
    run_cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(run_cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_agent_end(None, agent, _FakeResult(final_output="hi there"))  # type: ignore[arg-type]

    assert "hi there" in stream.getvalue()


@pytest.mark.asyncio
async def test_on_tool_start_redacts_secret_keys() -> None:
    stream = io.StringIO()
    run_cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(run_cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_tool_start(
        None,  # type: ignore[arg-type]
        agent,
        "http_request",
        {"url": "https://example.com", "api_key": "sk-abc123", "Authorization": "Bearer xyz"},
    )

    out = stream.getvalue()
    # Secret values must be masked.
    assert "sk-abc123" not in out
    assert "Bearer xyz" not in out
    assert "[REDACTED]" in out
    # Non-secret fields survive.
    assert "example.com" in out


@pytest.mark.asyncio
async def test_mute_event_via_style_override() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    # Replace tool.start style with an all-empty one (no icon, no prefix).
    cfg.styles[EVENT_TOOL_START] = EventStyle()
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_tool_start(None, agent, "search", {"q": "x"})  # type: ignore[arg-type]

    out = stream.getvalue()
    # Body still shows since payload is enabled by default; headline
    # has no icon/prefix but still carries the tool name.
    assert "search" in out
    assert "[tool]" not in out


# ---------------------------------------------------------------------------
# V5 — tool-level guardrails (ADK enrichment)
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolGuardrail:
    name: str


@dataclass
class _FakeToolGuardrailOutput:
    behavior: dict


@dataclass
class _FakeToolGuardrailResult:
    guardrail: _FakeToolGuardrail
    output: _FakeToolGuardrailOutput


@pytest.mark.asyncio
async def test_on_tool_input_guardrail_start_renders() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_tool_input_guardrail_start(
        None,  # type: ignore[arg-type]
        agent,
        "search",
        "pii_check",
    )

    out = stream.getvalue()
    assert "pii_check" in out
    assert "search" in out


@pytest.mark.asyncio
async def test_on_tool_input_guardrail_end_pass_verdict() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    result = _FakeToolGuardrailResult(
        guardrail=_FakeToolGuardrail(name="pii_check"),
        output=_FakeToolGuardrailOutput(behavior={"type": "allow"}),
    )
    await hooks.on_tool_input_guardrail_end(
        None,
        agent,
        "search",
        result,  # type: ignore[arg-type]
    )

    out = stream.getvalue()
    assert "allow" in out
    assert "search" in out


@pytest.mark.asyncio
async def test_on_tool_input_guardrail_end_reject_verdict() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    result = _FakeToolGuardrailResult(
        guardrail=_FakeToolGuardrail(name="pii_check"),
        output=_FakeToolGuardrailOutput(
            behavior={"type": "reject_content", "message": "blocked"},
        ),
    )
    await hooks.on_tool_input_guardrail_end(
        None,
        agent,
        "search",
        result,  # type: ignore[arg-type]
    )

    out = stream.getvalue()
    assert "reject_content" in out


@pytest.mark.asyncio
async def test_on_tool_output_guardrail_end_raise_verdict() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    result = _FakeToolGuardrailResult(
        guardrail=_FakeToolGuardrail(name="schema_check"),
        output=_FakeToolGuardrailOutput(behavior={"type": "raise_exception"}),
    )
    await hooks.on_tool_output_guardrail_end(
        None,
        agent,
        "search",
        result,  # type: ignore[arg-type]
    )

    out = stream.getvalue()
    assert "raise_exception" in out
    assert "schema_check" in out


@pytest.mark.asyncio
async def test_tool_guardrail_silent_when_disabled() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(enabled=False, output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    await hooks.on_tool_input_guardrail_start(
        None,
        agent,
        "search",
        "pii_check",  # type: ignore[arg-type]
    )
    assert stream.getvalue() == ""


# ---------------------------------------------------------------------------
# V5 — ADK enrichment emit helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150


def test_emit_turn_start_end_renders() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_turn_start(agent, 1)
    hooks.emit_turn_end(agent, 1)

    out = stream.getvalue()
    assert "turn 1" in out


def test_emit_hitl_approval_requested_with_nested_path() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Outer")

    hooks.emit_hitl_approval_requested(
        agent,
        "book_trip",
        "call-123",
        nested_path=["Outer", "book_trip", "Inner", "charge_card"],
    )

    out = stream.getvalue()
    assert "book_trip" in out
    assert "charge_card" in out
    assert "Inner" in out


def test_emit_hitl_approval_granted_closes_gate() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Outer")

    hooks.emit_hitl_approval_requested(agent, "book_trip", "call-1")
    hooks.emit_hitl_approval_granted(
        agent,
        "book_trip",
        "call-1",
        approver_id="user@x",
        reason="ok",
    )

    out = stream.getvalue()
    assert "approved" in out.lower() or "user@x" in out


def test_emit_hitl_approval_rejected_closes_gate() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Outer")

    hooks.emit_hitl_approval_rejected(
        agent,
        "book_trip",
        "call-1",
        message="policy violation",
    )

    out = stream.getvalue()
    assert "rejected" in out.lower() or "policy violation" in out


def test_emit_budget_warning() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_budget_warning(agent, 0.8, "total_tokens")

    out = stream.getvalue()
    assert "80%" in out
    assert "total_tokens" in out


def test_emit_budget_exceeded() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_budget_exceeded(agent, "total_tokens")

    out = stream.getvalue()
    assert "exceeded" in out.lower()
    assert "total_tokens" in out


def test_emit_cache_hit_and_miss() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_cache_hit(agent, "search")
    hooks.emit_cache_miss(agent, "fetch")

    out = stream.getvalue()
    assert "search" in out
    assert "fetch" in out
    assert "hit" in out.lower()
    assert "miss" in out.lower()


def test_emit_context_compacted_reports_tokens() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_context_compacted(agent, 10_000, 2_500)

    out = stream.getvalue()
    assert "10000" in out
    assert "2500" in out
    assert "7500" in out  # saved


def test_emit_context_edited() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_context_edited(agent, "cleared 3 old tool results")

    out = stream.getvalue()
    assert "cleared 3" in out


def test_emit_retry() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_retry(agent, "search", 2, "rate_limit")

    out = stream.getvalue()
    assert "retry 2" in out
    assert "rate_limit" in out


def test_emit_stream_start_end() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_stream_start(agent)
    hooks.emit_stream_end(agent)

    out = stream.getvalue()
    assert "streaming started" in out
    assert "streaming ended" in out


def test_emit_tool_error_closes_with_error_payload() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    err = ValueError("boom")
    hooks.emit_tool_error(agent, "search", err)

    out = stream.getvalue()
    assert "search" in out
    assert "ValueError" in out
    assert "boom" in out


def test_emit_usage_recorded() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    hooks.emit_usage_recorded(agent, _FakeUsage())

    out = stream.getvalue()
    assert "100" in out
    assert "50" in out
    assert "150" in out


def test_emit_helpers_silent_when_no_config() -> None:
    hooks = VerboseHooks(None)
    agent = _FakeAgent(name="Alice")

    # None of these should raise even though there is no config.
    hooks.emit_turn_start(agent, 1)
    hooks.emit_cache_hit(agent, "search")
    hooks.emit_hitl_approval_requested(agent, "t", "c1")
    hooks.emit_budget_warning(agent, 0.5, "tokens")
    hooks.emit_context_compacted(agent, 100, 50)
    hooks.emit_retry(agent, "t", 1, "timeout")
    hooks.emit_stream_start(agent)
    hooks.emit_tool_error(agent, "t", RuntimeError("x"))
    hooks.emit_usage_recorded(agent, _FakeUsage())


# ---------------------------------------------------------------------------
# Agent-level guardrail name resolution (unnamed guardrail → function name)
# ---------------------------------------------------------------------------
#
# An ``AgentInputGuardrail`` / ``AgentOutputGuardrail`` declares
# ``name: str | None = None``. The start handler resolves the display
# name via ``guardrail.get_name()`` (falls back to the function
# ``__name__``). The end handlers must resolve identically — reading
# ``.name`` directly would yield ``None``, rendering "guardrail None" in
# the headline and building a close key that never matches the start key.


@dataclass
class _FakeAgentGuardrail:
    """Mirrors the ``get_name()`` / ``.name`` shape of an agent guardrail.

    ``name`` is ``None`` (the common "no explicit name" case);
    ``get_name()`` falls back to the supplied function name exactly like
    the real ``AgentInputGuardrail.get_name()``.
    """

    func_name: str
    name: str | None = None

    def get_name(self) -> str:
        if self.name is not None and len(self.name) > 0:
            return self.name
        return self.func_name


@dataclass
class _FakeAgentGuardrailResult:
    guardrail: _FakeAgentGuardrail


async def test_input_guardrail_end_uses_function_name_when_unnamed() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    result = _FakeAgentGuardrailResult(guardrail=_FakeAgentGuardrail(func_name="check_content"))
    await hooks.on_input_guardrail_end(None, agent, result)  # type: ignore[arg-type]

    out = stream.getvalue()
    # The function name must surface in the headline; the literal "None"
    # (the pre-fix behaviour of reading ``.name`` directly) must not.
    assert "check_content" in out
    assert "guardrail None" not in out


async def test_output_guardrail_end_uses_function_name_when_unnamed() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    result = _FakeAgentGuardrailResult(guardrail=_FakeAgentGuardrail(func_name="no_secrets"))
    await hooks.on_output_guardrail_end(None, agent, result)  # type: ignore[arg-type]

    out = stream.getvalue()
    assert "no_secrets" in out
    assert "guardrail None" not in out


async def test_input_guardrail_end_honours_explicit_name() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    result = _FakeAgentGuardrailResult(
        guardrail=_FakeAgentGuardrail(func_name="check_content", name="pii_filter"),
    )
    await hooks.on_input_guardrail_end(None, agent, result)  # type: ignore[arg-type]

    out = stream.getvalue()
    assert "pii_filter" in out
    assert "check_content" not in out


async def test_guardrail_end_key_matches_start_name_when_unnamed() -> None:
    """The end-handler close key must use the same name as the start key.

    The start handler builds its key from ``guardrail.get_name()`` (the
    function ``__name__`` for an unnamed guardrail). The end handler must
    resolve the identical string so the panel-mode close key matches the
    start key. Asserting the rendered headline carries the function name
    (not ``None``) proves the close key is built from the same source —
    both are derived from the single resolved ``name_str``.
    """
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    hooks = VerboseHooks(cfg)
    agent = _FakeAgent(name="Alice")

    func_name = "check_content"
    await hooks.on_input_guardrail_start(None, agent, func_name)  # type: ignore[arg-type]
    result = _FakeAgentGuardrailResult(guardrail=_FakeAgentGuardrail(func_name=func_name))
    await hooks.on_input_guardrail_end(None, agent, result)  # type: ignore[arg-type]

    out = stream.getvalue()
    # The start line and the end line must reference the SAME name. A
    # divergence (start: function name, end: ``None``) is the defect.
    assert out.count(func_name) >= 2
    assert "None" not in out


# ---------------------------------------------------------------------------
# Panel-mode markup escaping for atomic events (bracket-bearing names)
# ---------------------------------------------------------------------------


async def test_atomic_panel_escapes_bracket_bearing_agent_name() -> None:
    """A ``[/]``-bearing agent name must not silently drop the panel.

    Atomic events promote their headline into the panel body, which the
    panel backend parses as Rich markup. An unescaped ``[/]`` is a
    dangling close tag that raises ``MarkupError`` — swallowed at DEBUG,
    dropping the whole panel. The headline must be markup-escaped first.
    """
    from rich.console import Console

    cfg = VerboseConfig(mode="panel", use_rich=True, use_color=True)
    hooks = VerboseHooks(cfg)
    renderer = hooks._get_panel_renderer(cfg)
    console = Console(file=io.StringIO(), record=True, force_terminal=True, width=120)
    renderer._console = console

    a = _FakeAgent(name="Agent[/]")
    b = _FakeAgent(name="Worker")
    await hooks.on_handoff(None, a, b)  # type: ignore[arg-type]

    text = console.export_text()
    # The panel must render (not be silently dropped) and the literal
    # bracket name must survive rather than being parsed as a tag.
    assert "Agent[/]" in text
    assert "Worker" in text
