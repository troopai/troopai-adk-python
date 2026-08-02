"""CrewAI-faithful Rich-Panel backend for verbose output.

:class:`PanelRenderer` emits Rich Panels that match CrewAI's
``ConsoleFormatter`` visual grammar 1:1:

* full-terminal-width panels (``Console(width=None)``);
* ``padding=(1, 2)`` everywhere;
* event-kind border colours (cyan crew start, yellow task / tool,
  magenta agent, green completion, red failure) — NOT verdict-driven;
* bold-title content rows composed with Rich markup;
* a ``rich.live.Live`` widget that grows the streaming panel
  token-by-token (CrewAI's signature visual).

Surfaces
--------

* **Block primitives** (``open_block`` / ``append_payload`` /
  ``close_block`` / ``render_atomic``) — kept for ADK-only events that
  have no CrewAI counterpart (HITL approval gates, budget warnings,
  prompt-cache hits, context compaction, MCP lifecycle). Those panels
  still flow through :class:`~troopai.adk.verbose.state.RunTree` and
  render with the same CrewAI visual chrome.
* **Task boundary** (``render_task_start`` / ``render_task_end``) —
  one-shot ``📋 Task`` panels emitted from
  :meth:`troopai.adk.run.runner.Runner.arun` entry / exit.
* **Live streaming** (``open_stream_panel`` / ``update_stream_panel`` /
  ``close_stream_panel``) — Rich ``Live`` widget driven by per-chunk
  LLM stream emission via the
  :mod:`troopai.adk.verbose.run_bridge` ``ContextVar`` bridge.
* **HITL coordination** (``pause_live_updates`` /
  ``resume_live_updates``) — temporarily stop the ``Live`` so an HITL
  approval prompt's stdin read does not race the refresh loop.

Per-tool iteration counter (``(#N)`` suffix on tool panels) lives on
``ClassVar``-level state guarded by a ``threading.Lock`` so the counter
spans every renderer instance — matching CrewAI's class-level
``tool_usage_counts``. Test code MUST clear it between cases.

The renderer never raises. Rich-layer exceptions are caught and
logged at DEBUG — verbose output is a convenience, never a
correctness requirement.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from troopai.adk.verbose.config import (
    EVENT_AGENT_END,
    EVENT_AGENT_FINISH,
    EVENT_AGENT_START,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_TASK_END,
    EVENT_TASK_FAILED,
    EVENT_TASK_START,
    EVENT_TOOL_END,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_START,
    EventStyle,
    VerboseConfig,
)
from troopai.adk.verbose.renderer import redact_secrets
from troopai.adk.verbose.state import BlockKey, BlockNode, RunTree

if TYPE_CHECKING:
    from rich.console import Console as _RichConsole
    from rich.live import Live as _RichLive

logger = logging.getLogger(__name__)


StreamCallType = Literal["text", "tool_call"]
"""Discriminator for the Live streaming surface.

``"text"`` → green ``✅ Agent Final Answer`` panel; ``"tool_call"`` →
yellow ``🔧 Tool Arguments`` panel. Matches CrewAI's
``handle_llm_stream_chunk`` branching on ``LLMCallType``.
"""


_EVENT_BORDER: dict[str, str] = {
    EVENT_RUN_START: "cyan",
    EVENT_RUN_END: "green",
    EVENT_TASK_START: "yellow",
    EVENT_TASK_END: "green",
    EVENT_TASK_FAILED: "red",
    EVENT_AGENT_START: "magenta",
    EVENT_AGENT_END: "green",
    EVENT_AGENT_FINISH: "green",
    EVENT_TOOL_START: "yellow",
    EVENT_TOOL_END: "green",
    EVENT_TOOL_ERROR: "red",
}
"""Border colour per canonical CrewAI event. Unknown events fall back
to :attr:`EventStyle.color` so ADK-only enrichments (HITL, budget,
cache, context, MCP) keep their configured palette."""


_STREAM_MAX_LINES = 20
"""Trailing-line cap for the Live widget panel — CrewAI truncates
streaming output to the last 20 lines with a ``"...\\n"`` prefix."""


_MARKUP_BRACKET_RE = re.compile(r"(\\*)(\[[a-z#/@][^\[]*?\])|(\[)")
"""One ``[`` occurrence with its escape context.

First alternative: a tag-shaped bracket (``[bold]``, ``[/]``) together
with the backslash run preceding it — the shape ``rich.markup.escape``
matches, where the parser *halves* the run. Second alternative: any
other ``[`` (``[1]``, ``[1, 2, 3]``), where the parser consumes exactly
one preceding backslash. :func:`escape_markup` substitutes each shape
accordingly."""


class PanelRenderer:
    """CrewAI-faithful Rich-Panel renderer.

    One instance per :class:`VerboseConfig` (cached on
    :class:`~troopai.adk.verbose.hooks.VerboseHooks`). Owns its own
    :class:`RunTree` so concurrent runs stay isolated.

    The per-tool iteration counter (:attr:`_tool_usage_counts`) is
    class-level — CrewAI does the same so ``(#N)`` keeps incrementing
    across multiple agents inside one swarm. Tests MUST clear it in
    ``setup_method`` / ``teardown_method``.

    Instantiating does not import Rich — the Console is built lazily so
    a renderer constructed for a disabled config pays zero Rich import
    cost.

    Attributes:
        _tool_usage_counts: Per-tool invocation counter keyed by tool
            name. Class-level so the ``(#N)`` suffix matches CrewAI's
            cross-agent counting. Concurrent access is guarded by
            :attr:`_tool_count_lock`.
        _tool_count_lock: ``threading.Lock`` protecting
            :attr:`_tool_usage_counts` against concurrent increments
            from swarm agents running on different threads.
    """

    _tool_usage_counts: ClassVar[dict[str, int]] = {}
    """Per-tool invocation counter. Class-level so ``(#N)`` matches
    CrewAI's cross-agent counting. Concurrent access guarded by
    :attr:`_tool_count_lock`."""

    _tool_count_lock: ClassVar[threading.Lock] = threading.Lock()
    """Lock guarding :attr:`_tool_usage_counts`."""

    def __init__(self, config: VerboseConfig) -> None:
        """Bind the renderer to *config*.

        Args:
            config: The :class:`VerboseConfig` that owns this renderer.
                Style lookups go through ``config.get_style(event)``.
                The output stream is resolved from
                ``config.resolve_output()`` on first Console
                construction.
        """
        self._config = config
        self._tree = RunTree()
        self._console: _RichConsole | None = None
        self._streaming_live: _RichLive | None = None
        self._live_call_type: StreamCallType = "text"
        self._just_streamed_final_answer: bool = False
        logger.debug("PanelRenderer initialised (config id=%s)", id(config))

    # ------------------------------------------------------------------
    # Console (lazy)
    # ------------------------------------------------------------------

    def _get_console(self) -> _RichConsole | None:
        """Lazy-construct the Rich Console pinned to the config's stream.

        Returns ``None`` when Rich is not importable.

        ``force_terminal`` is left unset so Rich auto-detects whether
        the target stream is a TTY — required by the ``Live`` widget,
        which needs accurate detection to refresh in place. Tests that
        want deterministic captured output should inject their own
        ``Console(force_terminal=True, record=True)`` via
        ``renderer._console = ...``.
        """
        if self._console is not None:
            return self._console
        try:
            from rich.console import Console as _Console
        except ImportError:
            logger.debug("Rich unavailable; PanelRenderer will emit nothing")
            return None
        self._console = _Console(
            file=self._config.resolve_output(),
            no_color=not self._config.resolve_use_color(),
            width=None,
            highlight=False,
        )
        return self._console

    # ------------------------------------------------------------------
    # Block primitives (ADK-only events: HITL, budget, cache, context,
    # MCP) plus CrewAI-canonical agent / tool blocks
    # ------------------------------------------------------------------

    def open_block(
        self,
        event: str,
        key: BlockKey,
        headline: str,
        payload: str | None = None,
    ) -> BlockNode:
        """Open a new block.

        Returns the new :class:`BlockNode`. *payload*, if given, is
        appended immediately as one Rich-markup line.

        Args:
            event: Dotted event name for the opening event
                (e.g. ``"tool.start"``).
            key: Identity tuple used to match this block on a later
                :meth:`close_block` call.
            headline: Short one-line header for the panel title area.
            payload: Optional initial payload line. Appended to the
                new node's buffer immediately when non-empty.

        Returns:
            The newly-created :class:`BlockNode`.
        """
        node = self._tree.open(event, key, headline)
        if payload is not None and len(payload) > 0:
            node.append_payload(payload)
        return node

    def _next_tool_iteration(self, tool_name: str) -> int:
        """Increment and return the per-tool invocation counter.

        Keyed by the *tool name* (never a composed headline) so the
        ``(#N)`` suffix counts each tool across every agent and
        renderer instance — matching CrewAI's class-level
        ``tool_usage_counts[tool_name]`` — and so the class-level dict
        stays bounded by the run's tool vocabulary.
        """
        with self._tool_count_lock:
            iteration = self._tool_usage_counts.get(tool_name, 0) + 1
            self._tool_usage_counts[tool_name] = iteration
            return iteration

    def _tool_iteration(self, tool_name: str) -> int:
        """Return the current counter for *tool_name* without incrementing.

        Used by the completed / error panels so they display the same
        ``(#N)`` as the started panel of the invocation they close.
        Defaults to ``1`` for an unknown tool — same fallback as
        CrewAI's ``tool_usage_counts.get(tool_name, 1)``.
        """
        with self._tool_count_lock:
            return self._tool_usage_counts.get(tool_name, 1)

    def consume_just_streamed_flag(self) -> bool:
        """Return the just-streamed-final-answer flag and clear it.

        Used by :class:`VerboseHooks` to suppress the duplicate
        ``agent.end`` / ``agent.finish`` panel when the streaming Live
        widget already painted the final answer. One-shot semantics:
        next caller sees ``False``.

        Returns:
            The value of the flag at call time. Always ``False`` on
            the next call after a ``True`` return.
        """
        flag = self._just_streamed_final_answer
        self._just_streamed_final_answer = False
        return flag

    def append_payload(self, key: BlockKey, line: str) -> None:
        """Append *line* to the payload of the block matching *key*.

        Silent no-op when no matching block is open — lets emitters
        append mid-block payload (streaming tokens, retry attempts)
        without checking first.

        Args:
            key: Identity tuple identifying the target open block.
            line: Text line (may contain Rich markup) to append.
        """
        node = self._find_open(key)
        if node is None:
            return
        node.append_payload(line)

    def close_block(
        self,
        key: BlockKey,
        verdict: str = "ok",
        final_payload: str | None = None,
    ) -> BlockNode | None:
        """Close the block matching *key* and flush its panel.

        Returns the closed :class:`BlockNode`, or ``None`` if no block
        matched (stale close — logged at DEBUG by :class:`RunTree`).

        When the block's event is :data:`EVENT_AGENT_FINISH` and the
        streaming Live widget already painted the final answer
        (:attr:`_just_streamed_final_answer` is True), the panel is
        suppressed to avoid the duplicate-final-answer regression.

        Args:
            key: Identity tuple identifying the block to close.
            verdict: Close state string passed to the block tree and
                used by the renderer for panel border colouring.
            final_payload: Optional final line appended to the node's
                payload buffer before flushing.

        Returns:
            The closed :class:`BlockNode`, or ``None`` on a stale
            close.
        """
        node = self._tree.close(key, verdict=verdict)
        if node is None:
            return None
        if final_payload is not None and len(final_payload) > 0:
            node.append_payload(final_payload)
        # The streaming-flag suppression lives on the dispatch path
        # (:meth:`VerboseHooks._dispatch_close`) so it runs even for
        # CrewAI-canonical events that bypass the block tree.
        self._flush_panel(node)
        return node

    def close_all(self, verdict: str = "interrupted") -> None:
        """Close every still-open block and flush each panel.

        Blocks flush in unwind order (deepest first). Called from the
        hooks teardown path so an exception-interrupted run does not
        leave panels visually open.
        """
        closed = self._tree.close_all(verdict=verdict)
        for node in closed:
            self._flush_panel(node)

    def render_atomic(
        self,
        event: str,
        headline: str,
        payload: str | None = None,
        verdict: str = "ok",
    ) -> None:
        """Render a one-shot event (handoff, skill, warning, task).

        Composes a single Panel with the event's title, border, and
        payload — no entry on the block tree. Tool lifecycle events
        have dedicated methods (:meth:`render_tool_started`,
        :meth:`render_tool_finished`, :meth:`render_tool_error`) that
        carry the tool name for the ``(#N)`` counter; this generic
        path renders whatever style the event resolves to.

        Args:
            event: Dotted event name used for style and border
                lookups (e.g. ``"handoff"``).
            headline: Short one-line description placed in the panel
                header area.
            payload: Optional body text rendered inside the panel.
            verdict: Close state used by the renderer for border
                colour (``"ok"`` → event-kind colour; ``"error"`` →
                red; ``"warn"`` → yellow). Defaults to ``"ok"``.
        """
        temp = BlockNode(event=event, key=("__atomic__", event), headline=headline, verdict=verdict)
        if payload is not None and len(payload) > 0:
            temp.append_payload(payload)
        self._flush_panel(temp)

    # ------------------------------------------------------------------
    # Task-boundary surface
    # ------------------------------------------------------------------

    def render_task_start(self, task_name: str, task_id: str) -> None:
        """Emit the ``📋 Task Started`` panel.

        Body shape matches CrewAI's ``handle_task_started``: bold
        "Task Started" line followed by labeled "Name:" and "ID:" rows.

        The task identity stored in :class:`TaskOutput` / hook callbacks
        is a full ``str(uuid.uuid4())`` (36 chars); the panel truncates
        to the first 8 characters so the bordered display row stays
        compact. Truncation is presentation-only — the underlying
        identity propagates intact through hooks, tracing, and session
        events.

        Args:
            task_name: Human-readable task name shown in the
                "Name:" row.
            task_id: Full task identifier. Displayed truncated to its
                first 8 characters in the "ID:" row.
        """
        payload = (
            "[bold yellow]Task Started[/]\n"
            f"[white]Name:[/] [yellow]{escape_markup(task_name)}[/]\n"
            f"[white]ID:[/] [yellow]{escape_markup(task_id[:8])}[/]"
        )
        self.render_atomic(EVENT_TASK_START, headline="", payload=payload, verdict="ok")

    def render_task_end(
        self,
        task_name: str,
        task_id: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Emit the ``📋 Task Completed`` or ``❌ Task Failed`` panel.

        The task ID is truncated to its first 8 characters for the
        display row, mirroring :meth:`render_task_start`. The full
        identity remains addressable on :class:`TaskOutput.task_id`
        and via the ``on_task_end`` hook callback.

        Args:
            task_name: Same task name passed to
                :meth:`render_task_start`.
            task_id: Same task ID passed to :meth:`render_task_start`.
            success: True → green "Task Completed"; False → red
                "Task Failed".
            error: Optional error string; appended as a red row when
                *success* is False.
        """
        if success is True:
            event = EVENT_TASK_END
            payload = (
                "[bold green]Task Completed[/]\n"
                f"[white]Name:[/] [green]{escape_markup(task_name)}[/]\n"
                f"[white]ID:[/] [green]{escape_markup(task_id[:8])}[/]"
            )
            verdict = "ok"
        else:
            event = EVENT_TASK_FAILED
            payload = (
                "[bold red]Task Failed[/]\n"
                f"[white]Name:[/] [red]{escape_markup(task_name)}[/]\n"
                f"[white]ID:[/] [red]{escape_markup(task_id[:8])}[/]"
            )
            if error is not None and len(error) > 0:
                payload = payload + f"\n[white]Error:[/] [red]{escape_markup(error)}[/]"
            verdict = "error"
        self.render_atomic(event, headline="", payload=payload, verdict=verdict)

    # ------------------------------------------------------------------
    # Agent / tool / guardrail lifecycle surface (CrewAI-canonical)
    # ------------------------------------------------------------------
    #
    # These methods compose the labeled-row bodies CrewAI's
    # ``ConsoleFormatter`` prints (white ``Label:`` prefixes, value text
    # in the panel's status colour) and are called by ``VerboseHooks``
    # in panel mode instead of the generic block-tree dispatch. Titles
    # come from the event's ``EventStyle.panel_title`` so they stay
    # user-overridable.

    def render_agent_started(
        self,
        agent_name: str,
        *,
        description: str | None = None,
        tool_names: list[str] | None = None,
        skill_names: list[str] | None = None,
        handoff_names: list[str] | None = None,
    ) -> None:
        """Emit the ``🤖 Agent Started`` panel with the agent's capabilities.

        Body shape follows CrewAI's ``handle_agent_logs_started``
        (``Agent:`` identity row, blank line, contextual rows) and its
        ``create_status_content`` labeled-row grammar with tools
        simplified to a comma-separated name list (CrewAI's
        ``_simplify_tools_field``). Rows with nothing to show are
        omitted rather than rendered as ``None``.

        Args:
            agent_name: Agent identity for the ``Agent:`` row.
            description: Optional agent description; rendered as a
                ``Description:`` row when the agent-start style has
                ``show_payload=True``.
            tool_names: Tool names for the ``Tools:`` row.
            skill_names: Skill names for the ``Skills:`` row.
            handoff_names: Delegation target names for the
                ``Handoffs:`` row.
        """
        style = self._config.get_style(EVENT_AGENT_START)
        title = style.panel_title if len(style.panel_title) > 0 else self._fallback_title(style)
        lines = [f"[white]Agent:[/] [bright_green bold]{escape_markup(agent_name)}[/]"]
        detail: list[str] = []
        if style.show_payload is True and description is not None and len(description) > 0:
            detail.append(f"[white]Description:[/] [bright_green]{escape_markup(description)}[/]")
        for label, names in (
            ("Tools", tool_names),
            ("Skills", skill_names),
            ("Handoffs", handoff_names),
        ):
            if names is not None and len(names) > 0:
                joined = escape_markup(", ".join(names))
                detail.append(f"[white]{label}:[/] [bright_green]{joined}[/]")
        if len(detail) > 0:
            lines.append("")
            lines.extend(detail)
        self._print_panel_markup("\n".join(lines), title, self._border_for(EVENT_AGENT_START))

    def render_agent_finished(self, agent_name: str, output_text: str | None) -> None:
        """Emit the ``✅ Agent Final Answer`` panel.

        Body shape follows the ``AgentFinish`` branch of CrewAI's
        ``handle_agent_logs_execution``: an ``Agent:`` identity row, a
        blank line, then a ``Final Answer:`` label above the output.

        Args:
            agent_name: Agent identity for the ``Agent:`` row.
            output_text: Pre-truncated final output. Omitted when
                empty or when the agent-end style has
                ``show_payload=False``.
        """
        style = self._config.get_style(EVENT_AGENT_END)
        title = style.panel_title if len(style.panel_title) > 0 else self._fallback_title(style)
        lines = [f"[white]Agent:[/] [bright_green bold]{escape_markup(agent_name)}[/]"]
        if style.show_payload is True and output_text is not None and len(output_text) > 0:
            lines.append("")
            lines.append("[white]Final Answer:[/]")
            lines.append(f"[bright_green]{escape_markup(output_text)}[/]")
        self._print_panel_markup("\n".join(lines), title, self._border_for(EVENT_AGENT_END))

    def render_tool_started(self, tool_name: str, args_text: str | None) -> None:
        """Emit the ``🔧 Tool Execution Started (#N)`` panel.

        Matches CrewAI's ``handle_tool_usage_started``: the per-tool
        iteration counter is incremented here (keyed by *tool_name*)
        and the body carries ``Tool:`` and ``Args:`` rows.

        Args:
            tool_name: Name of the tool being invoked.
            args_text: Pre-formatted, pre-redacted argument preview.
                Omitted when empty or when the tool-start style has
                ``show_payload=False``.
        """
        iteration = self._next_tool_iteration(tool_name)
        style = self._config.get_style(EVENT_TOOL_START)
        base = style.panel_title if len(style.panel_title) > 0 else self._fallback_title(style)
        lines = [f"[white]Tool:[/] [bold yellow]{escape_markup(tool_name)}[/]"]
        if style.show_payload is True and args_text is not None and len(args_text) > 0:
            lines.append(f"[white]Args:[/] [yellow]{escape_markup(args_text)}[/]")
        self._print_panel_markup(
            "\n".join(lines),
            f"{base} (#{iteration})",
            self._border_for(EVENT_TOOL_START),
        )

    def render_tool_finished(self, tool_name: str, output_text: str | None) -> None:
        """Emit the ``✅ Tool Execution Completed (#N)`` panel.

        Matches CrewAI's ``handle_tool_usage_finished``: bold
        ``Tool Completed`` headline row, ``Tool:`` row, and an
        ``Output:`` row when a result preview is available.

        Args:
            tool_name: Name of the tool that finished.
            output_text: Pre-formatted, pre-redacted result preview.
                Omitted when empty or when the tool-end style has
                ``show_payload=False``.
        """
        iteration = self._tool_iteration(tool_name)
        style = self._config.get_style(EVENT_TOOL_END)
        base = style.panel_title if len(style.panel_title) > 0 else self._fallback_title(style)
        lines = [
            "[bold green]Tool Completed[/]",
            f"[white]Tool:[/] [bold green]{escape_markup(tool_name)}[/]",
        ]
        if style.show_payload is True and output_text is not None and len(output_text) > 0:
            lines.append(f"[white]Output:[/] [green]{escape_markup(output_text)}[/]")
        self._print_panel_markup(
            "\n".join(lines),
            f"{base} (#{iteration})",
            self._border_for(EVENT_TOOL_END),
        )

    def render_tool_error(self, tool_name: str, error_text: str) -> None:
        """Emit the ``🔧 Tool Error (#N)`` panel.

        Matches CrewAI's ``handle_tool_usage_error``: bold
        ``Tool Failed`` headline row, ``Tool:`` row, ``Error:`` row.

        Args:
            tool_name: Name of the tool that failed.
            error_text: Pre-truncated error description.
        """
        iteration = self._tool_iteration(tool_name)
        style = self._config.get_style(EVENT_TOOL_ERROR)
        base = style.panel_title if len(style.panel_title) > 0 else self._fallback_title(style)
        lines = [
            "[bold red]Tool Failed[/]",
            f"[white]Tool:[/] [bold red]{escape_markup(tool_name)}[/]",
            f"[white]Error:[/] [red]{escape_markup(error_text)}[/]",
        ]
        self._print_panel_markup(
            "\n".join(lines),
            f"{base} (#{iteration})",
            self._border_for(EVENT_TOOL_ERROR),
        )

    def render_guardrail_verdict(
        self,
        guardrail_name: str,
        *,
        passed: bool,
        status_text: str,
        tool_name: str | None = None,
    ) -> None:
        """Emit the ``🛡️ Guardrail Success`` / ``🛡️ Guardrail Failed`` panel.

        Matches CrewAI's ``handle_guardrail_completed`` grammar: a bold
        ``Guardrail Passed`` / ``Guardrail Failed`` headline row plus
        labeled ``Name:`` / ``Status:`` rows (green on pass, red on
        trip). Tool-level guardrails add a ``Tool:`` row — an ADK
        enrichment CrewAI has no counterpart for.

        Args:
            guardrail_name: Display name of the guardrail.
            passed: ``True`` renders the green success panel; ``False``
                the red failure panel.
            status_text: Verdict detail for the ``Status:`` row (e.g.
                ``"✅ Validated"``, ``"❌ Tripwire triggered"``, or a
                tool-guardrail behavior type).
            tool_name: Optional tool the guardrail wraps; adds a
                ``Tool:`` row when set.
        """
        color = "green" if passed is True else "red"
        title = "🛡️ Guardrail Success" if passed is True else "🛡️ Guardrail Failed"
        headline = "Guardrail Passed" if passed is True else "Guardrail Failed"
        lines = [
            f"[bold {color}]{headline}[/]",
            f"[white]Name:[/] [{color}]{escape_markup(guardrail_name)}[/]",
        ]
        if tool_name is not None and len(tool_name) > 0:
            lines.append(f"[white]Tool:[/] [{color}]{escape_markup(tool_name)}[/]")
        lines.append(f"[white]Status:[/] [{color}]{escape_markup(status_text)}[/]")
        self._print_panel_markup("\n".join(lines), title, color)

    def _print_panel_markup(self, body_markup: str, title: str, border: str) -> None:
        """Print one panel composed from a Rich-markup body. Never raises.

        Shared chrome for the dedicated lifecycle methods: CrewAI's
        ``create_panel`` shape (``padding=(1, 2)``, left-aligned title).
        ``use_color=False`` (or ``NO_COLOR``) degrades the border to
        ``"dim"``. Rich-layer errors are swallowed at DEBUG — verbose
        output is telemetry, never a correctness requirement.
        """
        console = self._get_console()
        if console is None:
            return
        try:
            from rich.panel import Panel
            from rich.text import Text

            if self._config.resolve_use_color() is False:
                border = "dim"
            panel = Panel(
                Text.from_markup(body_markup),
                title=title,
                title_align="left",
                border_style=border,
                padding=(1, 2),
            )
            console.print(panel)
        except Exception as exc:
            logger.debug("PanelRenderer panel print failed on %r: %s", title, exc)

    # ------------------------------------------------------------------
    # Live streaming surface
    # ------------------------------------------------------------------

    def open_stream_panel(self, agent_name: str, call_type: StreamCallType = "text") -> None:
        """Open a ``rich.live.Live`` panel that grows token-by-token.

        Args:
            agent_name: Recorded only in the DEBUG breadcrumb — CrewAI
                panels do not include the agent name in the streaming
                title.
            call_type: ``"text"`` → green "✅ Agent Final Answer";
                ``"tool_call"`` → yellow "🔧 Tool Arguments".

        If a Live widget is already running (e.g. concurrent swarm
        agents sharing one Console), stop it first — the newer stream
        wins. Matches CrewAI's single-Live-per-Console invariant.
        """
        console = self._get_console()
        if console is None:
            return
        if self._streaming_live is not None:
            self._stop_live_quietly()
        try:
            from rich.live import Live
            from rich.panel import Panel
            from rich.text import Text
        except ImportError:
            return
        title, border = _stream_title_and_border(call_type)
        panel = Panel(
            Text(""),
            title=title,
            title_align="left",
            border_style=border,
            padding=(1, 2),
        )
        try:
            live = Live(panel, console=console, refresh_per_second=10)
            live.start()
        except Exception as exc:
            logger.debug("PanelRenderer.open_stream_panel failed: %s (agent=%s)", exc, agent_name)
            return
        self._streaming_live = live
        self._live_call_type = call_type
        self._just_streamed_final_answer = False
        logger.debug("Live opened (call_type=%s agent=%s)", call_type, agent_name)

    def update_stream_panel(self, accumulated_text: str, call_type: StreamCallType = "text") -> None:
        """Update the running Live panel with the latest accumulated text.

        No-op when no Live is running (Rich unavailable, non-TTY, or
        :meth:`pause_live_updates` is in effect).

        Args:
            accumulated_text: Full text accumulated since the stream
                opened, not just the latest delta. The last
                :data:`_STREAM_MAX_LINES` lines are displayed; earlier
                lines are replaced by ``"...\\n"``.
            call_type: ``"text"`` → green panel; ``"tool_call"`` →
                yellow panel. Defaults to ``"text"``.
        """
        if self._streaming_live is None:
            return
        try:
            from rich.panel import Panel
            from rich.text import Text
        except ImportError:
            return
        display = _truncate_stream_text(accumulated_text, max_lines=_STREAM_MAX_LINES)
        title, border = _stream_title_and_border(call_type)
        text_style = "bright_green" if call_type == "text" else "yellow"
        panel = Panel(
            Text(display, style=text_style),
            title=title,
            title_align="left",
            border_style=border,
            padding=(1, 2),
        )
        try:
            self._streaming_live.update(panel, refresh=True)
        except Exception as exc:
            logger.debug("PanelRenderer.update_stream_panel failed: %s", exc)
        self._live_call_type = call_type

    def close_stream_panel(self) -> None:
        """Stop the running Live panel and clear streaming state.

        Sets :attr:`_just_streamed_final_answer` so a subsequent
        :data:`EVENT_AGENT_FINISH` block close suppresses its duplicate
        panel — only when the last call_type was ``"text"`` (i.e. the
        Live actually showed the final answer; a tool-call stream does
        not count).
        """
        if self._streaming_live is None:
            return
        self._stop_live_quietly()
        if self._live_call_type == "text":
            self._just_streamed_final_answer = True
        else:
            self._just_streamed_final_answer = False

    def pause_live_updates(self) -> None:
        """Stop the running Live widget so an HITL prompt can read stdin.

        Idempotent; safe to call when no Live is active. Subsequent
        :meth:`update_stream_panel` calls become no-ops until the next
        :meth:`open_stream_panel` reopens a fresh Live.
        """
        if self._streaming_live is None:
            return
        self._stop_live_quietly()

    def resume_live_updates(self) -> None:
        """Reset the suppression flag after an HITL pause.

        We do not preserve the paused panel because the streaming text
        accumulator lives on the streaming emit path
        (:mod:`troopai.adk.run.llm_calls`). The next chunk emission will
        construct a fresh Live via :meth:`open_stream_panel`.
        """
        return

    # ------------------------------------------------------------------
    # Queries (tests + integration)
    # ------------------------------------------------------------------

    def tree(self) -> RunTree:
        """Return the underlying :class:`RunTree` (read-only use).

        Returns:
            The :class:`RunTree` instance owned by this renderer.
        """
        return self._tree

    def depth(self) -> int:
        """Return the current number of open blocks (0 when idle).

        Returns:
            Non-negative integer; ``0`` when no blocks are open.
        """
        return self._tree.depth()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_open(self, key: BlockKey) -> BlockNode | None:
        """Locate the most recent open block with matching *key*.

        Delegates to :meth:`RunTree.find_open` — no cross-module
        access to the tree's private stack.
        """
        return self._tree.find_open(key)

    def _flush_panel(self, node: BlockNode) -> None:
        """Compose and print the Rich Panel for *node*.

        Swallows any Rich-layer exception at DEBUG.
        """
        console = self._get_console()
        if console is None:
            return
        try:
            self._render_panel(console, node)
        except Exception as exc:
            logger.debug("PanelRenderer flush failed on %s: %s", node.event, exc)

    def _render_panel(self, console: _RichConsole, node: BlockNode) -> None:
        """Build and print the Rich Panel for *node*.

        Visual rules (CrewAI parity):

        * Title from :attr:`EventStyle.panel_title` when set, else
          ``f"{icon} ({prefix})"`` (ADK-only events without an explicit
          CrewAI counterpart).
        * Border from :data:`_EVENT_BORDER`, falling back to the
          ``EventStyle.color`` for unmapped events.
        * Content parsed as Rich markup so callers can embed
          ``[white]Label:[/]`` rows for CrewAI-style content shape.
        * ``padding=(1, 2)``, ``title_align="left"``. No subtitle. No
          indentation.
        """
        from rich.panel import Panel
        from rich.text import Text

        style = self._config.get_style(node.event)
        title = style.panel_title if len(style.panel_title) > 0 else self._fallback_title(style)
        if style.show_payload is True and len(node.payload) > 0:
            body_markup = "\n".join(node.payload)
            body: Text = Text.from_markup(body_markup)
        else:
            body = Text("")
        panel = Panel(
            body,
            title=title,
            title_align="left",
            border_style=self._border_for(node.event),
            padding=(1, 2),
        )
        console.print(panel)

    def _border_for(self, event: str) -> str:
        """Resolve the border style for *event*.

        Canonical CrewAI events use the fixed per-event colours in
        :data:`_EVENT_BORDER`; other events fall back to their
        :attr:`EventStyle.color`. ``use_color=False`` (or ``NO_COLOR``)
        degrades every border to ``"dim"``.
        """
        if self._config.resolve_use_color() is False:
            return "dim"
        style = self._config.get_style(event)
        return _EVENT_BORDER.get(event, style.color if len(style.color) > 0 else "dim")

    def _fallback_title(self, style: EventStyle) -> str:
        """Fallback title for events without an explicit ``panel_title``.

        Used by ADK-only events (HITL, budget, cache, context, MCP) so
        their panels still get a recognisable header. Rich strips
        ``[name]`` sequences as markup, so we use a parenthesised
        prefix instead.
        """
        icon_part = f"{style.icon} " if len(style.icon) > 0 else ""
        prefix_part = f"({style.prefix})" if len(style.prefix) > 0 else ""
        composed = f"{icon_part}{prefix_part}".strip()
        return composed

    def _stop_live_quietly(self) -> None:
        """Stop the running Live and clear state. Never raises."""
        live = self._streaming_live
        self._streaming_live = None
        if live is None:
            return
        try:
            live.stop()
        except Exception as exc:
            logger.debug("PanelRenderer._stop_live_quietly: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _stream_title_and_border(call_type: StreamCallType) -> tuple[str, str]:
    """Return ``(title, border_style)`` for the Live panel of *call_type*.

    Matches CrewAI's ``handle_llm_stream_chunk`` selection: tool-arg
    streams render yellow with the ``🔧 Tool Arguments`` title; text
    streams render green with the ``✅ Agent Final Answer`` title.

    Args:
        call_type: ``"text"`` or ``"tool_call"``.

    Returns:
        A 2-tuple of ``(panel_title, border_style_string)``.
    """
    if call_type == "tool_call":
        return ("🔧 Tool Arguments", "yellow")
    return ("✅ Agent Final Answer", "green")


def _truncate_stream_text(text: str, *, max_lines: int) -> str:
    """Keep at most *max_lines* trailing lines.

    Prefixes ``"...\\n"`` when truncation happened. Mirrors CrewAI's
    20-line tail in ``handle_llm_stream_chunk`` — keeps the Live panel
    from growing unbounded on long completions.

    Args:
        text: Accumulated stream text to truncate.
        max_lines: Maximum number of trailing lines to keep.
            Non-positive values return *text* unchanged.

    Returns:
        The (possibly truncated) text string.
    """
    if max_lines <= 0:
        return text
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "...\n" + "\n".join(lines[-max_lines:])


def escape_markup(value: str) -> str:
    """Escape Rich markup metacharacters in user-supplied strings.

    Task and tool panel bodies are composed with Rich markup, and the
    panel renderer parses every payload line through
    :meth:`rich.text.Text.from_markup`. Plain-text payloads (tool args,
    tool results, agent final answers, exception messages) routinely
    contain ``[`` characters — a Python ``repr`` of a dict with list
    values, a path like ``C:[temp]``, or markdown footnotes like
    ``[1]``. Left unescaped those substrings are interpreted as markup:
    ``[bold]`` is silently stripped (corrupting the displayed value) and
    a dangling close tag like ``[/]`` raises ``MarkupError`` (which the
    renderer swallows at DEBUG, dropping the whole panel). Escaping via
    a leading ``"\\["`` makes the brackets render literally.

    Backslashes need care too, and Rich's parser treats the two bracket
    shapes differently (mirrored here from ``rich.markup.escape`` plus
    observed ``Text.from_markup`` behaviour):

    * Before a *tag-shaped* bracket (``[bold]``, ``[/]``) the parser
      halves the backslash run — so the run is doubled before the
      escape is inserted, exactly as ``rich.markup.escape`` does.
      Without the doubling, a payload already containing ``\\[tag]``
      would escape into a literal backslash followed by a *live* tag:
      a valid tag name is silently swallowed, and a dangling close tag
      raises ``MarkupError`` and drops the panel.
    * Before any *other* bracket (``[1]``) the parser consumes exactly
      one backslash, so a single inserted escape suffices and the
      existing run must be left alone.
    * A trailing run of odd length gains one backslash so that a
      closing tag appended by the caller (``f"[yellow]{escaped}[/]"``
      rows) is not escaped into literal ``[/]`` text.

    Backslashes anywhere else are left alone — Rich renders them
    literally, so doubling them would corrupt the displayed value.

    Args:
        value: Arbitrary string that may contain ``\\`` or ``[``
            characters.

    Returns:
        A copy of *value* that ``Text.from_markup`` renders as the
        original text (with live markup neutralised), safe to embed
        before a closing tag.
    """
    value = _MARKUP_BRACKET_RE.sub(_escape_bracket_match, value)
    trailing = len(value) - len(value.rstrip("\\"))
    if trailing % 2 == 1:
        value = value + "\\"
    return value


def _escape_bracket_match(match: re.Match[str]) -> str:
    """Substitution callback for one :data:`_MARKUP_BRACKET_RE` match."""
    backslashes = match.group(1)
    tag = match.group(2)
    if tag is not None:
        return f"{backslashes}{backslashes}\\{tag}"
    return "\\["


def format_tool_payload(tool_input: Any, tool_output: Any = None) -> str:
    """Render tool input and optional output as CrewAI-style payload rows.

    Output shape (Rich markup) — the caller's panel title is expected to
    carry ``"🔧 Tool Execution Started (#N)"`` already, so the body only
    contains ``Args:`` and optionally ``Output:`` rows::

        [white]Args:[/] [yellow]<truncated input>[/]
        [white]Output:[/] [bright_green]<truncated output>[/]

    Secret-looking keys in BOTH *tool_input* and *tool_output* are
    redacted via :func:`troopai.adk.verbose.renderer.redact_secrets`
    before display — tool results commonly echo credentials back in
    "user record retrieved" / OAuth refresh response shapes, so the
    output side MUST sanitize identically to the input side.

    Args:
        tool_input: The tool's input arguments (typically a ``dict``).
            Secret-key values are redacted before display.
        tool_output: Optional tool result. When given, appended as an
            ``Output:`` row. Also redacted for secret-key values.

    Returns:
        A Rich-markup string containing one or two labeled rows,
        ready to pass as a panel body.
    """
    from troopai.adk.verbose.renderer import format_payload

    safe_input = redact_secrets(tool_input)
    lines = [
        f"[white]Args:[/] [yellow]{escape_markup(format_payload(safe_input, max_chars=600))}[/]",
    ]
    if tool_output is not None:
        safe_output = redact_secrets(tool_output)
        lines.append(f"[white]Output:[/] [bright_green]{escape_markup(format_payload(safe_output, max_chars=600))}[/]")
    return "\n".join(lines)
