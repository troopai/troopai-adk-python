"""Tests for :mod:`troopai.adk.verbose.panel_renderer`.

Validates the CrewAI-faithful Rich-Panel backend using
``rich.console.Console(record=True)`` so assertions run against
captured text rather than real stdout. Covers:

* block lifecycle (open → append → close → flush);
* atomic one-shot rendering;
* cleanup on interrupt via :meth:`PanelRenderer.close_all`;
* event-kind border colours (cyan crew, yellow task / tool, magenta
  agent, green completion, red failure) — verdict-driven borders are
  no longer used for the canonical CrewAI events;
* per-tool iteration counter (``"(#N)"`` suffix on tool panels);
* task-boundary panels (``📋 Task Started`` /
  ``📋 Task Completed`` / ``❌ Task Failed``);
* graceful behaviour when rich is missing / console returns None.

The Live-streaming surface is exercised in
``tests/unit/verbose/test_live_streaming.py``.
"""

from __future__ import annotations

import io
from unittest import mock

import pytest
from rich.console import Console

from troopai.adk.verbose.config import (
    EVENT_AGENT_END,
    EVENT_AGENT_FINISH,
    EVENT_AGENT_START,
    EVENT_HANDOFF,
    EVENT_TASK_END,
    EVENT_TASK_FAILED,
    EVENT_TASK_START,
    EVENT_TOOL_START,
    VerboseConfig,
)
from troopai.adk.verbose.hooks import VerboseHooks
from troopai.adk.verbose.panel_renderer import (
    PanelRenderer,
    escape_markup,
    format_tool_payload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recording_console() -> Console:
    """Return a Rich console that records output to an in-memory buffer.

    ``record=True`` lets us call ``export_text()`` for assertions.
    ``force_terminal=True`` + ``no_color=False`` forces Rich to emit
    panel borders even though the underlying file is not a TTY.
    """
    return Console(
        file=io.StringIO(),
        record=True,
        force_terminal=True,
        width=100,
        no_color=False,
    )


def _renderer_with_recording_console() -> tuple[PanelRenderer, Console]:
    cfg = VerboseConfig(mode="panel", use_color=True)
    renderer = PanelRenderer(cfg)
    console = _recording_console()
    # Prime the renderer's cached console so _get_console returns ours.
    renderer._console = console
    return renderer, console


@pytest.fixture(autouse=True)
def clear_tool_counter() -> None:
    """Reset the class-level per-tool counter between tests.

    ``PanelRenderer._tool_usage_counts`` is a ``ClassVar`` so the
    ``(#N)`` suffix matches CrewAI's cross-agent counter. Test order
    must not leak between cases. Autouse — pytest collects this
    automatically; no direct caller appears.
    """
    PanelRenderer._tool_usage_counts.clear()


# ---------------------------------------------------------------------------
# PanelRenderer — block lifecycle
# ---------------------------------------------------------------------------


class TestPanelRendererLifecycle:
    def test_open_block_appends_to_tree(self) -> None:
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        node = renderer.open_block(
            EVENT_AGENT_START,
            ("agent", "r1"),
            "coordinator",
        )
        assert renderer.depth() == 1
        assert node.event == EVENT_AGENT_START
        assert node.headline == "coordinator"

    def test_open_block_accepts_initial_payload(self) -> None:
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        node = renderer.open_block(
            EVENT_TOOL_START,
            ("tool", "t1"),
            "search_web",
            payload="[white]Args:[/] [yellow]query='llama 4'[/]",
        )
        assert node.payload == ["[white]Args:[/] [yellow]query='llama 4'[/]"]

    def test_append_payload_on_open_block(self) -> None:
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        renderer.open_block(EVENT_TOOL_START, ("tool", "t1"), "search")
        renderer.append_payload(("tool", "t1"), "first line")
        renderer.append_payload(("tool", "t1"), "second line")
        node = renderer._find_open(("tool", "t1"))
        assert node is not None
        assert node.payload == ["first line", "second line"]

    def test_append_payload_on_missing_block_is_noop(self) -> None:
        """Appending to a non-existent key should not raise."""
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        renderer.append_payload(("tool", "nonexistent"), "orphan line")
        assert renderer.depth() == 0

    def test_close_block_marks_verdict_and_flushes(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "coordinator")
        closed = renderer.close_block(("agent", "r1"), verdict="ok")
        assert closed is not None
        assert closed.verdict == "ok"
        assert closed.is_open() is False
        output = console.export_text()
        assert len(output) > 0
        # The CrewAI-faithful title uses the agent-start panel label.
        assert "Agent Started" in output

    def test_close_block_returns_none_for_unknown_key(self) -> None:
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        result = renderer.close_block(("tool", "never-opened"))
        assert result is None

    def test_close_block_appends_final_payload(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.open_block(EVENT_TOOL_START, ("tool", "t1"), "search")
        renderer.append_payload(("tool", "t1"), "[white]Args:[/] [yellow]q=foo[/]")
        closed = renderer.close_block(
            ("tool", "t1"),
            verdict="ok",
            final_payload="[white]Output:[/] [bright_green]42[/]",
        )
        assert closed is not None
        # Payload list keeps both rows in order.
        assert len(closed.payload) == 2
        output = console.export_text()
        assert "Args" in output
        assert "Output" in output
        assert "42" in output

    def test_close_block_tracks_elapsed(self) -> None:
        renderer, _ = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "coordinator")
        closed = renderer.close_block(("agent", "r1"), verdict="ok")
        assert closed is not None
        assert closed.elapsed() >= 0.0


# ---------------------------------------------------------------------------
# PanelRenderer — nested blocks (no indentation now; flat panels)
# ---------------------------------------------------------------------------


class TestPanelRendererNesting:
    def test_nested_open_close(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "outer")
        renderer.open_block(EVENT_TOOL_START, ("tool", "t1"), "inner")
        renderer.close_block(("tool", "t1"), verdict="ok")
        assert renderer.depth() == 1
        renderer.close_block(("agent", "r1"), verdict="ok")
        assert renderer.depth() == 0
        output = console.export_text()
        # Both panels appear; CrewAI uses titles from EventStyle.
        assert "Agent Started" in output
        assert "Tool Execution Started" in output


# ---------------------------------------------------------------------------
# PanelRenderer.close_all
# ---------------------------------------------------------------------------


class TestPanelRendererCloseAll:
    def test_close_all_flushes_every_open_block(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "outer agent")
        renderer.open_block(EVENT_TOOL_START, ("tool", "t1"), "inner tool")
        renderer.close_all(verdict="interrupted")
        assert renderer.depth() == 0
        output = console.export_text()
        assert "Agent Started" in output
        assert "Tool Execution Started" in output

    def test_close_all_on_empty_is_noop(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.close_all()  # must not raise
        assert len(console.export_text()) == 0


# ---------------------------------------------------------------------------
# PanelRenderer.render_atomic
# ---------------------------------------------------------------------------


class TestPanelRendererAtomic:
    def test_atomic_renders_standalone_panel(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.render_atomic(
            EVENT_HANDOFF,
            "coordinator → specialist",
            payload="reason: expertise",
        )
        output = console.export_text()
        assert "🔗 Agent Handoff" in output
        assert "reason: expertise" in output

    def test_atomic_does_not_affect_tree(self) -> None:
        renderer, _ = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "outer")
        renderer.render_atomic(EVENT_HANDOFF, "a → b")
        # Atomic events don't touch the block stack.
        assert renderer.depth() == 1


# ---------------------------------------------------------------------------
# PanelRenderer — Rich-unavailable path
# ---------------------------------------------------------------------------


class TestPanelRendererRichMissing:
    def test_console_returns_none_when_rich_import_fails(self) -> None:
        """If rich is not importable, _get_console returns None."""
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        with mock.patch.dict("sys.modules", {"rich.console": None}):
            renderer._console = None
            result = renderer._get_console()
        assert result is None

    def test_close_block_does_not_raise_without_console(self) -> None:
        """A block close with no console just skips the flush — no crash."""
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "coordinator")
        with mock.patch.object(renderer, "_get_console", return_value=None):
            result = renderer.close_block(("agent", "r1"), verdict="ok")
        assert result is not None

    def test_open_stream_panel_noop_without_console(self) -> None:
        """Live setup skips silently when Rich is unavailable."""
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        with mock.patch.object(renderer, "_get_console", return_value=None):
            renderer.open_stream_panel("coordinator", "text")
        assert renderer._streaming_live is None


# ---------------------------------------------------------------------------
# PanelRenderer — colour / NO_COLOR
# ---------------------------------------------------------------------------


class TestPanelRendererColor:
    def test_no_color_disables_border_style(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With use_color=False, borders degrade — no colour codes in output."""
        monkeypatch.setenv("NO_COLOR", "1")
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        console = Console(
            file=io.StringIO(),
            record=True,
            force_terminal=True,
            width=100,
            no_color=True,
        )
        renderer._console = console
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "coordinator")
        renderer.close_block(("agent", "r1"), verdict="ok")
        assert len(console.export_text()) > 0


# ---------------------------------------------------------------------------
# format_tool_payload helper
# ---------------------------------------------------------------------------


class TestFormatToolPayload:
    def test_input_only(self) -> None:
        text = format_tool_payload({"query": "llama"})
        assert "Args:" in text
        assert "llama" in text
        assert "Output:" not in text

    def test_input_and_output(self) -> None:
        text = format_tool_payload({"query": "llama"}, tool_output="42")
        assert "Args:" in text
        assert "Output:" in text
        assert "42" in text

    def test_secrets_redacted(self) -> None:
        """Secret-looking keys MUST be replaced with [REDACTED]."""
        text = format_tool_payload({"api_key": "sk-1234", "query": "safe"})
        assert "REDACTED" in text
        assert "sk-1234" not in text
        assert "safe" in text


# ---------------------------------------------------------------------------
# PanelRenderer — event-kind border integration
# ---------------------------------------------------------------------------


class TestEventBorderIntegration:
    """End-to-end check: each canonical CrewAI event gets the right border colour.

    Uses ``export_html`` so we can scan for Rich colour styles —
    plain ``export_text`` strips them.
    """

    def _render_event(self, event: str) -> str:
        with mock.patch.dict("os.environ", {"NO_COLOR": ""}):
            cfg = VerboseConfig(mode="panel", use_color=True)
            renderer = PanelRenderer(cfg)
            console = Console(
                file=io.StringIO(),
                record=True,
                force_terminal=True,
                width=100,
                no_color=False,
                color_system="truecolor",
            )
            renderer._console = console
            renderer.open_block(event, ("k", event), "test")
            renderer.close_block(("k", event), verdict="ok")
            return console.export_html(inline_styles=True)

    def test_task_start_border_is_yellow(self) -> None:
        html = self._render_event(EVENT_TASK_START)
        assert "yellow" in html.lower() or "808000" in html.lower()

    def test_task_end_border_is_green(self) -> None:
        html = self._render_event(EVENT_TASK_END)
        assert "green" in html.lower() or "008000" in html.lower()

    def test_task_failed_border_is_red(self) -> None:
        html = self._render_event(EVENT_TASK_FAILED)
        assert "red" in html.lower() or "800000" in html.lower()

    def test_agent_start_border_is_magenta(self) -> None:
        html = self._render_event(EVENT_AGENT_START)
        # Magenta in rich's named-colour scheme maps to #800080 (purple) or
        # the literal "magenta" word in inline style names.
        assert "magenta" in html.lower() or "800080" in html.lower()

    def test_tool_start_border_is_yellow(self) -> None:
        html = self._render_event(EVENT_TOOL_START)
        assert "yellow" in html.lower() or "808000" in html.lower()


# ---------------------------------------------------------------------------
# PanelRenderer — CrewAI-style title composition
# ---------------------------------------------------------------------------


class TestCrewAIPanelTitles:
    """The panel title comes from ``EventStyle.panel_title`` (set in
    ``_default_styles``) for canonical CrewAI events. ADK-only events
    without a ``panel_title`` fall back to ``f"{icon} [{prefix}]"``.
    """

    def test_agent_start_title_matches_crewai(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_START, ("agent", "r1"), "coordinator")
        renderer.close_block(("agent", "r1"), verdict="ok")
        output = console.export_text()
        assert "🤖 Agent Started" in output

    def test_tool_start_title_includes_counter(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.render_tool_started("search_web", "{'q': 'llama'}")
        output = console.export_text()
        assert "🔧 Tool Execution Started (#1)" in output

    def test_task_start_renders_via_render_task_start(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.render_task_start("Summarise the docs", "ab12cd34")
        output = console.export_text()
        assert "📋 Task Started" in output
        assert "Summarise the docs" in output
        assert "ab12cd34" in output

    def test_task_end_success_uses_completed_title(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.render_task_end("Summarise", "ab12cd34", success=True)
        output = console.export_text()
        assert "📋 Task Completed" in output

    def test_task_end_failure_uses_failed_title(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.render_task_end("Summarise", "ab12cd34", success=False, error="boom")
        output = console.export_text()
        assert "❌ Task Failed" in output
        assert "boom" in output


# ---------------------------------------------------------------------------
# PanelRenderer — per-tool iteration counter
# ---------------------------------------------------------------------------


class TestToolUsageCounter:
    """The ``(#N)`` suffix on tool panels is driven by a class-level
    counter so the count persists across renderer instances (CrewAI's
    cross-agent behaviour). Each test gets a clean counter via the
    autouse fixture at the top of this file.
    """

    def test_counter_increments_per_tool_name(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.render_tool_started("search_web", None)
        renderer.render_tool_started("search_web", None)
        renderer.render_tool_started("search_web", None)
        output = console.export_text()
        assert "(#1)" in output
        assert "(#2)" in output
        assert "(#3)" in output

    def test_counter_is_per_tool_name(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.render_tool_started("search_web", None)
        renderer.render_tool_started("fetch_url", None)
        # Each tool starts at 1 because the counter keys on tool name.
        assert PanelRenderer._tool_usage_counts == {"search_web": 1, "fetch_url": 1}
        output = console.export_text()
        # Both panels render (#1).
        assert output.count("(#1)") == 2

    def test_counter_shared_across_renderer_instances(self) -> None:
        """Two renderers see the same counter — matches CrewAI's
        class-level ``tool_usage_counts``."""
        r1, _ = _renderer_with_recording_console()
        r1.render_tool_started("search_web", None)
        r2, console = _renderer_with_recording_console()
        r2.render_tool_started("search_web", None)
        # Second call appears as (#2) even though it's a different renderer.
        output = console.export_text()
        assert "(#2)" in output

    async def test_counter_keyed_by_tool_name_not_headline(self) -> None:
        """Two agents calling the same tool share one per-tool count.

        The hooks used to key the class-level counter by the composed
        headline (``"<agent> calling <tool>"``), so the same tool driven
        by two different agents rendered ``(#1)`` twice and the dict
        grew one entry per (agent, tool) pair. The counter must key on
        the tool name alone — CrewAI's ``tool_usage_counts[tool_name]``.
        """
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        alice = _StubAgent("Alice", None)
        bob = _StubAgent("Bob", None)
        await hooks.on_tool_start(None, alice, "search_web", {"q": "a"})  # type: ignore[arg-type]
        await hooks.on_tool_start(None, bob, "search_web", {"q": "b"})  # type: ignore[arg-type]
        output = console.export_text()
        assert "(#1)" in output
        assert "(#2)" in output
        assert set(PanelRenderer._tool_usage_counts.keys()) == {"search_web"}


# ---------------------------------------------------------------------------
# PanelRenderer — agent finish suppression after Live final answer
# ---------------------------------------------------------------------------


class TestJustStreamedFlagConsumption:
    """The just-streamed-final-answer flag is consumed by the
    ``VerboseHooks._dispatch_close`` path. The renderer exposes
    :meth:`consume_just_streamed_flag` so the dispatch layer can read
    and reset it in one call.
    """

    def test_flag_starts_false(self) -> None:
        renderer, _ = _renderer_with_recording_console()
        assert renderer.consume_just_streamed_flag() is False

    def test_flag_consume_resets_state(self) -> None:
        renderer, _ = _renderer_with_recording_console()
        renderer._just_streamed_final_answer = True
        assert renderer.consume_just_streamed_flag() is True
        # One-shot — second consume returns False.
        assert renderer.consume_just_streamed_flag() is False

    def test_close_agent_finish_renders_via_close_block(self) -> None:
        """``close_block`` flushes a block-tree panel unconditionally —
        the dispatch layer is where suppression happens, not here."""
        renderer, console = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_FINISH, ("agent", "r1"), "coordinator")
        renderer.close_block(
            ("agent", "r1"),
            verdict="ok",
            final_payload="[bright_green]final[/]",
        )
        output = console.export_text()
        assert "Agent Final Answer" in output


# ---------------------------------------------------------------------------
# Cleanup safety: agent_end (non-finish) still renders normally
# ---------------------------------------------------------------------------


class TestAgentEndRenders:
    """``agent.end`` is distinct from ``agent.finish`` and always renders
    so users still see a confirmation panel when streaming is disabled
    or when the run did not stream a final answer."""

    def test_agent_end_renders(self) -> None:
        renderer, console = _renderer_with_recording_console()
        renderer.open_block(EVENT_AGENT_END, ("agent", "r1"), "coordinator")
        renderer.close_block(("agent", "r1"), verdict="ok")
        output = console.export_text()
        # ``EVENT_AGENT_END`` falls back to the icon+prefix title.
        assert len(output) > 0


# ---------------------------------------------------------------------------
# escape_markup — Rich-markup metacharacter escaping
# ---------------------------------------------------------------------------


class TestEscapeMarkup:
    """``escape_markup`` neutralises ``[`` so plain-text payloads with
    brackets (dict ``repr`` list values, paths, markdown footnotes) are
    rendered literally instead of being interpreted as Rich markup."""

    def test_open_bracket_is_escaped(self) -> None:
        assert escape_markup("see [bold]") == "see \\[bold]"

    def test_round_trips_through_from_markup(self) -> None:
        from rich.text import Text

        # A dangling close tag would raise ``MarkupError`` unescaped;
        # escaped, it renders the literal characters.
        raw = "{'q': 'a[/]b'}"
        rendered = Text.from_markup(escape_markup(raw))
        assert rendered.plain == raw

    def test_string_without_brackets_unchanged(self) -> None:
        assert escape_markup("plain text") == "plain text"


# ---------------------------------------------------------------------------
# Regression: panel bodies must escape attacker/repr brackets
# ---------------------------------------------------------------------------


class _StubAgent:
    """Minimal stand-in for :class:`Agent` for dispatch-path tests.

    ``VerboseHooks`` dispatch reads ``verbose`` (config override) and
    ``name`` off the agent — plus the optional capability attributes
    for the agent-start banner — so a lightweight stub avoids the full
    Agent construction (which requires instructions, tools, etc.).
    """

    def __init__(
        self,
        name: str,
        verbose: VerboseConfig | None,
        *,
        description: str | None = None,
        tools: list[object] | None = None,
        skills: list[object] | None = None,
        handoffs: object = None,
    ) -> None:
        self.name = name
        self.verbose = verbose
        self.description = description
        self.tools = tools if tools is not None else []
        self.skills = skills if skills is not None else []
        self.handoffs = handoffs


class _NamedThing:
    """Duck-typed tool / skill carrying only a ``name``."""

    def __init__(self, name: str) -> None:
        self.name = name


class _HandoffStub:
    """Duck-typed handoff declaration carrying only ``agent_name``."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name


def _hooks_with_recording_console(
    cfg: VerboseConfig,
) -> tuple[VerboseHooks, Console]:
    hooks = VerboseHooks(run_config_verbose=cfg)
    renderer = hooks._get_panel_renderer(cfg)
    console = _recording_console()
    renderer._console = console
    return hooks, console


class TestPanelPayloadMarkupEscaping:
    """Tool / agent / HITL panel bodies are plain text, but the panel
    backend parses every payload line as Rich markup. Without escaping,
    a ``repr`` that contains ``[bold]`` is silently stripped (content
    corruption) and a dangling ``[/]`` raises ``MarkupError`` — caught
    at DEBUG in ``_flush_panel``, dropping the whole panel. These tests
    pin that bracketed payloads render literally and the panel survives.
    """

    async def test_tool_start_args_with_brackets_render_literally(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        await hooks.on_tool_start(None, agent, "search", {"items": [1, 2, 3], "note": "see [bold]"})  # type: ignore[arg-type]
        output = console.export_text()
        # The bracket substrings survive verbatim — not stripped as markup.
        assert "[1, 2, 3]" in output
        assert "[bold]" in output

    async def test_tool_start_dangling_close_tag_does_not_drop_panel(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        # ``[/]`` is a dangling close tag — unescaped it raises
        # MarkupError and the panel is silently dropped.
        await hooks.on_tool_start(None, agent, "search", {"q": "a[/]b"})  # type: ignore[arg-type]
        output = console.export_text()
        assert "Tool Execution Started" in output
        assert "a[/]b" in output

    async def test_hitl_payload_with_brackets_render_literally(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        # The HITL request panel renders atomically at emit time; the
        # tool-input args carry brackets that must render literally.
        hooks.emit_hitl_approval_requested(
            agent,  # type: ignore[arg-type]
            "delete_user",
            "call-1",
            tool_input={"ids": [1, 2], "note": "see [bold]"},
        )
        hooks.emit_hitl_approval_granted(agent, "delete_user", "call-1")  # type: ignore[arg-type]
        output = console.export_text()
        assert "[1, 2]" in output
        assert "[bold]" in output

    async def test_agent_end_final_output_with_brackets_render_literally(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)

        class _Result:
            final_output = "Answer with footnote [1] and a [/] tag"

        await hooks.on_agent_end(None, agent, _Result())  # type: ignore[arg-type]
        output = console.export_text()
        assert "Agent Final Answer" in output
        assert "[1]" in output
        assert "[/]" in output

    async def test_atomic_handoff_payload_with_brackets_render_literally(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        hooks._dispatch_atomic(agent, EVENT_HANDOFF, "to [specialist]")  # type: ignore[arg-type]
        output = console.export_text()
        # The headline is promoted into the body and must render literally.
        assert "[specialist]" in output

    def test_escape_markup_doubles_pre_bracket_backslashes(self) -> None:
        r"""Backslash-bearing payloads must round-trip, not re-form tags.

        ``escape_markup`` used to insert ``\`` before ``[`` without
        doubling a pre-existing backslash run, so an input containing
        ``\[tag]`` escaped to ``\\[tag]`` — a literal backslash followed
        by a *live* tag. A valid tag name was silently swallowed
        (content corruption); a dangling ``\[/]`` raised ``MarkupError``
        and dropped the whole panel.
        """
        from rich.text import Text

        for raw in (r"C:\[temp]\file", r"tail \[/] done", "a\\b", r"footnote \[1]"):
            rendered = Text.from_markup(escape_markup(raw))
            assert rendered.plain == raw

    def test_escape_markup_trailing_backslash_keeps_close_tag_live(self) -> None:
        r"""A value ending in an odd backslash run must not eat the
        caller's closing tag when embedded in a labeled markup row."""
        from rich.text import Text

        row = f"[yellow]{escape_markup('end' + chr(92))}[/]"
        rendered = Text.from_markup(row)
        # The closing tag stays a tag — no literal "[/]" in the output.
        assert "[/]" not in rendered.plain
        assert rendered.plain.startswith("end")


# ---------------------------------------------------------------------------
# Regression: panel mode must render agent capabilities + close-side events
# ---------------------------------------------------------------------------


class TestPanelAgentStartCapabilities:
    """The agent-start banner must convey the agent and its capabilities
    the way CrewAI's agent panel does — labeled ``Agent:`` /
    ``Description:`` / ``Tools:`` / ``Skills:`` / ``Handoffs:`` rows —
    instead of a bare ``Agent: <name>`` body."""

    async def test_agent_start_lists_tools_skills_handoffs(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent(
            "Research Analyst",
            None,
            description="Digs through filings.",
            tools=[_NamedThing("search_web"), _NamedThing("calculator")],
            skills=[_NamedThing("cite_sources")],
            handoffs=[_HandoffStub("Report Writer")],
        )
        await hooks.on_agent_start(None, agent)  # type: ignore[arg-type]
        output = console.export_text()
        assert "🤖 Agent Started" in output
        assert "Agent: Research Analyst" in output
        assert "Description: Digs through filings." in output
        assert "Tools: search_web, calculator" in output
        assert "Skills: cite_sources" in output
        assert "Handoffs: Report Writer" in output

    async def test_agent_start_omits_empty_capability_rows(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("Minimal", None)
        await hooks.on_agent_start(None, agent)  # type: ignore[arg-type]
        output = console.export_text()
        assert "Agent: Minimal" in output
        assert "Tools:" not in output
        assert "Skills:" not in output
        assert "Handoffs:" not in output

    async def test_agent_end_renders_final_answer_shape(self) -> None:
        """Panel mode renders CrewAI's final-answer grammar: an
        ``Agent:`` identity row above a ``Final Answer:`` label."""
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("Research Analyst", None)

        class _Result:
            final_output = "Revenue grew 12%."

        await hooks.on_agent_end(None, agent, _Result())  # type: ignore[arg-type]
        output = console.export_text()
        assert "✅ Agent Final Answer" in output
        assert "Agent: Research Analyst" in output
        assert "Final Answer:" in output
        assert "Revenue grew 12%." in output


class TestStreamedFinalAnswerSuppression:
    """``on_agent_end`` owns the panel-mode duplicate-final-answer
    suppression: when the streaming Live widget already painted the
    final answer, the ``✅ Agent Final Answer`` panel is skipped once."""

    async def test_agent_end_suppressed_after_streamed_text(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        renderer = hooks._get_panel_renderer(cfg)
        agent = _StubAgent("coordinator", None)

        class _Result:
            final_output = "streamed already"

        # Simulate a text stream that just closed (the Live painted it).
        renderer._just_streamed_final_answer = True
        await hooks.on_agent_end(None, agent, _Result())  # type: ignore[arg-type]
        assert "Agent Final Answer" not in console.export_text()

        # One-shot: the next agent end renders normally.
        await hooks.on_agent_end(None, agent, _Result())  # type: ignore[arg-type]
        assert "Agent Final Answer" in console.export_text()


class TestPanelCloseSideEventsRender:
    """Close-side lifecycle events must produce panels in panel mode.

    The generic dispatch used to route ``tool.end`` / ``tool.error`` /
    guardrail-end events to ``close_block`` — but their start events
    render atomically (or are muted), so no matching open block ever
    existed and the close was silently dropped: no tool results, no
    tool errors, no guardrail verdicts.
    """

    async def test_tool_end_renders_completed_panel(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        await hooks.on_tool_start(None, agent, "search_web", {"q": "acme"})  # type: ignore[arg-type]
        await hooks.on_tool_end(None, agent, "search_web", "42 results")  # type: ignore[arg-type]
        output = console.export_text()
        assert "✅ Tool Execution Completed (#1)" in output
        assert "Tool: search_web" in output
        assert "Output: 42 results" in output

    async def test_tool_error_renders_error_panel(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        await hooks.on_tool_start(None, agent, "calculator", {"expr": "1/0"})  # type: ignore[arg-type]
        hooks.emit_tool_error(agent, "calculator", ZeroDivisionError("division by zero"))  # type: ignore[arg-type]
        output = console.export_text()
        assert "🔧 Tool Error (#1)" in output
        assert "Tool: calculator" in output
        assert "ZeroDivisionError" in output

    async def test_agent_guardrail_verdict_renders(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Guardrail:
            name: str

            def get_name(self) -> str:
                return self.name

        @dataclass
        class _Output:
            tripwire_triggered: bool

        @dataclass
        class _Result:
            guardrail: _Guardrail
            output: _Output

        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        await hooks.on_input_guardrail_start(None, agent, "check_topic")  # type: ignore[arg-type]
        await hooks.on_input_guardrail_end(
            None,  # type: ignore[arg-type]
            agent,  # type: ignore[arg-type]
            _Result(_Guardrail("check_topic"), _Output(tripwire_triggered=False)),  # type: ignore[arg-type]
        )
        await hooks.on_output_guardrail_end(
            None,  # type: ignore[arg-type]
            agent,  # type: ignore[arg-type]
            _Result(_Guardrail("no_pii"), _Output(tripwire_triggered=True)),  # type: ignore[arg-type]
        )
        output = console.export_text()
        assert "🛡️ Guardrail Success" in output
        assert "Name: check_topic" in output
        assert "🛡️ Guardrail Failed" in output
        assert "Name: no_pii" in output
        assert "Tripwire triggered" in output

    async def test_tool_guardrail_verdict_renders_with_tool_row(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Guardrail:
            name: str

        @dataclass
        class _Output:
            behavior: dict

        @dataclass
        class _Result:
            guardrail: _Guardrail
            output: _Output

        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        await hooks.on_tool_input_guardrail_end(
            None,  # type: ignore[arg-type]
            agent,  # type: ignore[arg-type]
            "search_web",
            _Result(_Guardrail("pii_check"), _Output(behavior={"type": "reject_content"})),  # type: ignore[arg-type]
        )
        output = console.export_text()
        assert "🛡️ Guardrail Failed" in output
        assert "Name: pii_check" in output
        assert "Tool: search_web" in output
        assert "reject_content" in output


class TestPanelHitlRendersImmediately:
    """HITL approval panels must render at emit time in panel mode.

    The request used to open a block that only flushed at the matching
    close — so the operator saw nothing before the approval prompt, and
    a run resumed with fresh hooks (out-of-band approval in another
    process) never rendered anything at all.
    """

    def test_request_renders_before_any_close(self) -> None:
        cfg = VerboseConfig(mode="panel", use_color=True)
        hooks, console = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        hooks.emit_hitl_approval_requested(
            agent,  # type: ignore[arg-type]
            "delete_user",
            "call-1",
            tool_input={"user_id": 42},
        )
        output = console.export_text()
        assert "🙋 Human Approval Required" in output
        assert "Tool: delete_user" in output
        assert "42" in output

    def test_verdict_renders_on_fresh_hooks(self) -> None:
        """Approval verdicts render even when the request was emitted by
        a different hooks instance (the resume-after-approval flow)."""
        cfg = VerboseConfig(mode="panel", use_color=True)
        request_hooks, _ = _hooks_with_recording_console(cfg)
        agent = _StubAgent("coordinator", None)
        request_hooks.emit_hitl_approval_requested(agent, "delete_user", "call-1")  # type: ignore[arg-type]

        resume_cfg = VerboseConfig(mode="panel", use_color=True)
        resume_hooks, resume_console = _hooks_with_recording_console(resume_cfg)
        resume_hooks.emit_hitl_approval_granted(
            agent,  # type: ignore[arg-type]
            "delete_user",
            "call-1",
            approver_id="ops@example.com",
        )
        resume_hooks.emit_hitl_approval_rejected(
            agent,  # type: ignore[arg-type]
            "charge_card",
            "call-2",
            message="over limit",
        )
        output = resume_console.export_text()
        assert "✅ Approval Granted" in output
        assert "Approved by: ops@example.com" in output
        assert "🚫 Approval Rejected" in output
        assert "over limit" in output
