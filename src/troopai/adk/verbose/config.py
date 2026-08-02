"""Verbose-output configuration.

``VerboseConfig`` is the user-facing knob for colorful, event-driven
execution output: a declarative mapping from event-name to render style.

The model has two levels (per-agent overrides per-run):

- ``RunConfig.verbose: Optional[VerboseConfig]`` — run-wide default.
- ``Agent.verbose: Optional[VerboseConfig]`` — per-agent override that
  the renderer resolves at emit time against the current agent, so a
  single run can have a loud coordinator and a silent summariser.

Design notes
------------

The style table (:attr:`VerboseConfig.styles`) is a plain ``dict`` keyed
by string event names rather than a closed ``Enum``. This is on
purpose: future ``Agent`` attributes that want visibility (e.g. a
hypothetical ``memory_access``, ``reasoning.step``, ``plan.revised``)
can publish their own default styles and register them on an existing
``VerboseConfig`` via :meth:`register_event`, without modifying the
renderer.

Colour / icon choices are inspired by CrewAI's ``ConsoleFormatter`` but
differ in one deliberate way: CrewAI hard-codes ``border_style="cyan"``
etc. at each handler site, so users cannot reskin output without
editing the framework. Here every style is user-overridable and the
``NO_COLOR`` standard is honoured.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal, TextIO

VerboseMode = Literal["auto", "line", "panel", "off"]
"""Developer-facing backend selector for :class:`VerboseConfig`.

``"auto"`` defers to :func:`troopai.adk.verbose.mode.resolve_mode`
which inspects the environment (TTY, ``NO_COLOR``, ``FORCE_COLOR``,
``CI``, rich availability) and picks the right backend.
``"line"`` forces the stateless line renderer (CI / non-TTY / piped output).
``"panel"`` forces the Rich-backed panel renderer.
``"off"`` emits nothing.
"""


# ---------------------------------------------------------------------------
# Event-name constants
# ---------------------------------------------------------------------------
#
# Using dotted string keys (not an Enum) so third-party Agent attributes
# can register their own events at runtime. The constants below are
# the framework-built-in events; custom ones follow the same format.

# Event constants emitted by ``VerboseHooks``.
EVENT_AGENT_START = "agent.start"
EVENT_AGENT_END = "agent.end"
EVENT_LLM_START = "llm.start"
EVENT_LLM_END = "llm.end"
EVENT_TOOL_START = "tool.start"
EVENT_TOOL_END = "tool.end"
EVENT_TOOL_ERROR = "tool.error"
EVENT_HANDOFF = "handoff"
EVENT_GUARDRAIL_INPUT_START = "guardrail.input.start"
EVENT_GUARDRAIL_INPUT_END = "guardrail.input.end"
EVENT_GUARDRAIL_OUTPUT_START = "guardrail.output.start"
EVENT_GUARDRAIL_OUTPUT_END = "guardrail.output.end"
EVENT_SKILL_ACTIVATED = "skill.activated"
EVENT_SESSION_LOAD = "session.load"
EVENT_SESSION_SAVE = "session.save"
EVENT_STATE_SAVE = "state.save"

# Reserved event constants. Not emitted by ``VerboseHooks`` yet, but
# pre-registered in the default style table so downstream modules
# (memory, reasoning, MCP, flow, HITL, budget, cache, context mgmt)
# that want to render via ``renderer.render_line(...)`` inherit a
# coherent, CrewAI-aligned (and enriched) vocabulary without each
# module inventing its own icon palette.
#
# Events in this section mirror CrewAI's vocabulary plus extensions
# for features CrewAI lacks: HITL approval gates, per-run token
# budget ceilings, prompt-cache hit/miss, context compaction, and
# explicit per-turn boundaries from our loop model.

# CrewAI parity (aligned to console_formatter.py)
EVENT_RUN_START = "run.start"
EVENT_RUN_END = "run.end"
EVENT_TASK_START = "task.start"
EVENT_TASK_END = "task.end"
EVENT_TASK_FAILED = "task.failed"
# ``EVENT_AGENT_FINISH`` is a reserved style entry for a streaming-only
# finish panel separate from ``agent.end``. The runner emits
# ``EVENT_AGENT_END`` for both streaming and non-streaming runs and does not
# fire ``EVENT_AGENT_FINISH``; ``_default_styles`` carries it for completeness.
EVENT_AGENT_FINISH = "agent.finish"
EVENT_REASONING_START = "reasoning.start"
EVENT_REASONING_END = "reasoning.end"
EVENT_MEMORY_READ = "memory.read"
EVENT_MEMORY_WRITE = "memory.write"
EVENT_MEMORY_ERROR = "memory.error"
EVENT_KNOWLEDGE_QUERY = "knowledge.query"
EVENT_KNOWLEDGE_RESULT = "knowledge.result"
EVENT_MCP_CONNECT = "mcp.connect"
EVENT_MCP_CONNECTED = "mcp.connected"
EVENT_MCP_ERROR = "mcp.error"
EVENT_FLOW_START = "flow.start"
EVENT_FLOW_END = "flow.end"
EVENT_FLOW_PAUSED = "flow.paused"
EVENT_PLAN_REFINED = "plan.refined"
EVENT_REPLAN = "plan.replan"
EVENT_GOAL_ACHIEVED = "goal.achieved"
EVENT_WARNING = "warning"
EVENT_RETRY = "retry"

# Enrichment beyond CrewAI — HITL (first-class in TroopAI, limited in
# CrewAI) so approval gates in the run appear as proper lifecycle
# lines rather than buried log messages.
EVENT_HITL_APPROVAL_REQUESTED = "hitl.approval.requested"
EVENT_HITL_APPROVAL_GRANTED = "hitl.approval.granted"
EVENT_HITL_APPROVAL_REJECTED = "hitl.approval.rejected"

# Enrichment — token / cost budgets. ``LLMUsageLimits``,
# ``max_total_turns``, and ``HandoffConfig.budget`` all enforce
# ceilings; these events surface when we're close to one or when
# the run was truncated because we hit one.
EVENT_BUDGET_WARNING = "budget.warning"
EVENT_BUDGET_EXCEEDED = "budget.exceeded"
EVENT_USAGE_RECORDED = "usage.recorded"

# Enrichment — prompt-cache observability (LiteLLM / Anthropic /
# Gemini caching). CrewAI has no equivalent surface.
EVENT_CACHE_HIT = "cache.hit"
EVENT_CACHE_MISS = "cache.miss"

# Enrichment — context management. Automatic compaction and editing
# fire when the context window is under pressure; without a visible
# event users see "their tokens disappear" with no explanation.
EVENT_CONTEXT_COMPACTED = "context.compacted"
EVENT_CONTEXT_EDITED = "context.edited"

# Enrichment — explicit turn boundaries. The loop model exposes
# turns as a first-class concept; rendering them makes it obvious
# where ``max_total_turns`` is being consumed.
EVENT_TURN_START = "turn.start"
EVENT_TURN_END = "turn.end"

# Enrichment — streaming. ``Runner.arun_streamed`` emits deltas;
# these markers let the renderer show when a stream opens/closes
# without overwhelming the console with per-delta lines.
EVENT_STREAM_START = "stream.start"
EVENT_STREAM_END = "stream.end"


@dataclass(frozen=True)
class EventStyle:
    """Per-event render style.

    Attributes:
        color: Colour name. Passed to the active renderer (ANSI or
            Rich). ``""`` disables colour for this event. The Rich
            renderer accepts any ``rich.style.Style`` string
            (``"bold red"``, ``"cyan on grey10"``); the ANSI renderer
            understands the short set below and ignores the rest.
        icon: Optional single-glyph prefix (e.g. ``"→"``, ``"✓"``).
            Empty string disables the icon for this event.
        prefix: Short text tag (e.g. ``"tool"``, ``"handoff"``) emitted
            after the icon. Empty string disables the prefix.
        show_payload: If ``False``, the renderer emits the headline
            (``prefix`` + ``icon`` + agent name) but omits any bulky
            payload — LLM messages, tool inputs/outputs, guardrail
            details. Useful for keeping a compact trace in CI logs.
        panel_title: Pre-formatted panel title used by the Rich-Panel
            renderer (e.g. ``"🚀 Crew Execution Started"``,
            ``"🤖 Agent Started"``). When empty the panel renderer
            falls back to ``f"{icon} {prefix}"``. The line renderer
            ignores this field. Set it to align the panel layout with
            CrewAI's ``ConsoleFormatter`` vocabulary verbatim, or to
            customise titles without touching the renderer source.
    """

    color: str = ""
    """Colour tag. Empty string = no colour for this event."""

    icon: str = ""
    """Leading glyph. Empty string = no icon."""

    prefix: str = ""
    """Short text tag. Empty string = no prefix."""

    show_payload: bool = True
    """Whether to include event payload (message body, tool args, etc.)."""

    panel_title: str = ""
    """CrewAI-style panel title for the Rich-Panel renderer. Empty falls back to icon+prefix."""


def _default_styles() -> dict[str, EventStyle]:
    """Default style table.

    Icons follow CrewAI's ``ConsoleFormatter`` vocabulary (🚀 kickoff,
    🧠 thinking, 🔧 tool, ✅ success, ❌ error, 🔗 handoff, 🛡 guardrail,
    …) so output is immediately legible to operators moving between
    frameworks. Colours use Rich's named-style vocabulary and degrade
    to the closest ANSI code in the fallback renderer.

    Users on minimal TTYs that don't render emoji can swap individual
    glyphs via ``cfg.styles[EVENT_*] = EventStyle(icon="●", …)``. The
    ASCII-friendly geometric-shape palette (●, ✓, ◇, ⟳, ⇄) is
    available for direct substitution.
    """
    return {
        # Run / crew lifecycle
        EVENT_RUN_START: EventStyle(
            color="bold cyan",
            icon="🚀",
            prefix="run",
            panel_title="🚀 Crew Execution Started",
        ),
        EVENT_RUN_END: EventStyle(
            color="bold green",
            icon="🎯",
            prefix="run",
            panel_title="Crew Completion",
        ),
        # Task lifecycle
        EVENT_TASK_START: EventStyle(
            color="cyan",
            icon="📋",
            prefix="task",
            panel_title="📋 Task Started",
        ),
        EVENT_TASK_END: EventStyle(
            color="green",
            icon="✅",
            prefix="task",
            panel_title="📋 Task Completed",
        ),
        EVENT_TASK_FAILED: EventStyle(
            color="red",
            icon="❌",
            prefix="task",
            panel_title="❌ Task Failed",
        ),
        # Agent lifecycle — start banner mirrors CrewAI's magenta-bordered
        # "🤖 Agent Started" panel; finish is a separate constant so the
        # streaming Live widget's final-answer panel suppresses the
        # duplicate ``agent.finish`` panel at end-of-stream.
        EVENT_AGENT_START: EventStyle(
            color="bold cyan",
            icon="🤖",
            prefix="agent",
            panel_title="🤖 Agent Started",
        ),
        EVENT_AGENT_END: EventStyle(
            color="bold green",
            icon="✅",
            prefix="agent",
            panel_title="✅ Agent Final Answer",
        ),
        EVENT_AGENT_FINISH: EventStyle(
            color="bright_green",
            icon="✅",
            prefix="agent",
            panel_title="✅ Agent Final Answer",
        ),
        # LLM calls — muted by default (noisy) but icon set to brain
        # so users who enable show_payload get CrewAI-style thinking
        # annotation.
        EVENT_LLM_START: EventStyle(color="blue", icon="🧠", prefix="llm", show_payload=False),
        EVENT_LLM_END: EventStyle(color="blue", icon="💭", prefix="llm", show_payload=False),
        # Reasoning (reserved — style entries provided; VerboseHooks does not emit these).
        EVENT_REASONING_START: EventStyle(color="magenta", icon="✏️", prefix="reason"),
        EVENT_REASONING_END: EventStyle(color="bold green", icon="💡", prefix="reason"),
        EVENT_PLAN_REFINED: EventStyle(color="yellow", icon="✏️", prefix="plan"),
        EVENT_REPLAN: EventStyle(color="yellow", icon="🔄", prefix="plan"),
        EVENT_GOAL_ACHIEVED: EventStyle(color="bold green", icon="🎯", prefix="goal"),
        # Tools — panel titles get the per-tool iteration counter
        # appended at render time by the dedicated
        # ``PanelRenderer.render_tool_*`` methods (CrewAI's ``(#N)``
        # suffix). Started / Completed / Error titles mirror CrewAI's
        # ``handle_tool_usage_started`` / ``_finished`` / ``_error``.
        EVENT_TOOL_START: EventStyle(
            color="yellow",
            icon="🔧",
            prefix="tool",
            panel_title="🔧 Tool Execution Started",
        ),
        EVENT_TOOL_END: EventStyle(
            color="green",
            icon="✅",
            prefix="tool",
            panel_title="✅ Tool Execution Completed",
        ),
        EVENT_TOOL_ERROR: EventStyle(
            color="red",
            icon="❌",
            prefix="tool",
            panel_title="🔧 Tool Error",
        ),
        EVENT_RETRY: EventStyle(color="yellow", icon="🔁", prefix="retry", panel_title="🔁 Tool Retry"),
        # Handoffs / delegation
        EVENT_HANDOFF: EventStyle(color="magenta", icon="🔗", prefix="handoff", panel_title="🔗 Agent Handoff"),
        # Guardrails
        EVENT_GUARDRAIL_INPUT_START: EventStyle(color="cyan", icon="🛡️", prefix="guardrail-in", show_payload=False),
        EVENT_GUARDRAIL_INPUT_END: EventStyle(color="cyan", icon="🛡️", prefix="guardrail-in"),
        EVENT_GUARDRAIL_OUTPUT_START: EventStyle(color="cyan", icon="🛡️", prefix="guardrail-out", show_payload=False),
        EVENT_GUARDRAIL_OUTPUT_END: EventStyle(color="cyan", icon="🛡️", prefix="guardrail-out"),
        # Skills
        EVENT_SKILL_ACTIVATED: EventStyle(
            color="bright_blue",
            icon="⚡",
            prefix="skill",
            panel_title="⚡ Skill Activated",
        ),
        # Memory (reserved)
        EVENT_MEMORY_READ: EventStyle(color="blue", icon="🧠", prefix="memory"),
        EVENT_MEMORY_WRITE: EventStyle(color="blue", icon="💾", prefix="memory"),
        EVENT_MEMORY_ERROR: EventStyle(color="red", icon="❌", prefix="memory"),
        # Knowledge retrieval (reserved)
        EVENT_KNOWLEDGE_QUERY: EventStyle(color="cyan", icon="🔎", prefix="knowledge"),
        EVENT_KNOWLEDGE_RESULT: EventStyle(color="cyan", icon="📚", prefix="knowledge"),
        # MCP (reserved)
        EVENT_MCP_CONNECT: EventStyle(color="blue", icon="🔌", prefix="mcp", show_payload=False),
        EVENT_MCP_CONNECTED: EventStyle(color="green", icon="✅", prefix="mcp", show_payload=False),
        EVENT_MCP_ERROR: EventStyle(color="red", icon="❌", prefix="mcp", panel_title="❌ MCP Error"),
        # Flow (reserved — mirrors CrewAI's flow panels)
        EVENT_FLOW_START: EventStyle(color="bright_cyan", icon="🌊", prefix="flow"),
        EVENT_FLOW_END: EventStyle(color="bright_green", icon="✅", prefix="flow"),
        EVENT_FLOW_PAUSED: EventStyle(color="yellow", icon="⏳", prefix="flow"),
        # Session / state
        EVENT_SESSION_LOAD: EventStyle(color="dim", icon="📥", prefix="session", show_payload=False),
        EVENT_SESSION_SAVE: EventStyle(color="dim", icon="💾", prefix="session", show_payload=False),
        EVENT_STATE_SAVE: EventStyle(color="dim", icon="💾", prefix="state", show_payload=False),
        # HITL (enrichment — first-class in TroopAI). Panel titles
        # make these atomic in panel mode: the request panel MUST print
        # the moment the gate opens (before any stdin prompt), and the
        # verdict panels MUST print even when the run resumes in a new
        # process where no matching open block exists.
        EVENT_HITL_APPROVAL_REQUESTED: EventStyle(
            color="yellow",
            icon="🙋",
            prefix="hitl",
            panel_title="🙋 Human Approval Required",
        ),
        EVENT_HITL_APPROVAL_GRANTED: EventStyle(
            color="green",
            icon="✅",
            prefix="hitl",
            panel_title="✅ Approval Granted",
        ),
        EVENT_HITL_APPROVAL_REJECTED: EventStyle(
            color="red",
            icon="🚫",
            prefix="hitl",
            panel_title="🚫 Approval Rejected",
        ),
        # Budgets / cost (enrichment)
        EVENT_BUDGET_WARNING: EventStyle(color="yellow", icon="⚠️", prefix="budget", panel_title="⚠️ Budget Warning"),
        EVENT_BUDGET_EXCEEDED: EventStyle(color="red", icon="🛑", prefix="budget", panel_title="🛑 Budget Exceeded"),
        EVENT_USAGE_RECORDED: EventStyle(color="dim", icon="📊", prefix="usage", show_payload=False),
        # Prompt cache (enrichment)
        EVENT_CACHE_HIT: EventStyle(color="green", icon="⚡", prefix="cache", show_payload=False),
        EVENT_CACHE_MISS: EventStyle(color="dim", icon="⏭", prefix="cache", show_payload=False),
        # Context management (enrichment)
        EVENT_CONTEXT_COMPACTED: EventStyle(
            color="cyan", icon="🗜", prefix="context", panel_title="🗜 Context Compacted"
        ),
        EVENT_CONTEXT_EDITED: EventStyle(color="cyan", icon="✂️", prefix="context", panel_title="✂️ Context Edited"),
        # Turn boundaries (enrichment)
        EVENT_TURN_START: EventStyle(color="dim", icon="🔄", prefix="turn", show_payload=False),
        EVENT_TURN_END: EventStyle(color="dim", icon="✅", prefix="turn", show_payload=False),
        # Streaming (enrichment)
        EVENT_STREAM_START: EventStyle(color="blue", icon="📡", prefix="stream", show_payload=False),
        EVENT_STREAM_END: EventStyle(color="blue", icon="📴", prefix="stream", show_payload=False),
        # Miscellaneous
        EVENT_WARNING: EventStyle(color="yellow", icon="⚠️", prefix="warn", panel_title="⚠️ Warning"),
    }


@dataclass(eq=False)
class VerboseConfig:
    """Configuration for colourful, event-driven execution output.

    .. warning::
        Tool inputs and outputs are rendered verbatim (subject to the
        ``max_chars`` truncation and secret-key redaction applied at
        emit time). Do NOT enable verbose output in production paths
        where tools receive or return credentials, API keys, or PII
        — the configured stream (default ``sys.stderr``) is typically
        captured by container orchestrators and forwarded to log
        aggregators.

    ``eq=False`` keeps identity-based ``__hash__`` so instances are
    usable as ``WeakKeyDictionary`` keys (the renderer cache on
    :class:`~troopai.adk.verbose.hooks.VerboseHooks` relies on this).

    Plug into a run via::

        from troopai.adk import RunConfig
        from troopai.adk.verbose import VerboseConfig

        config = RunConfig(verbose=VerboseConfig())
        result = await Runner.arun(agent, "Hello", run_config=config)

    Or per-agent::

        Agent(name="Coordinator", verbose=VerboseConfig())  # loud
        Agent(name="Summariser", verbose=VerboseConfig(enabled=False))  # silenced

    Agent-level configs fully override run-level configs when both
    are set for the same agent — an agent speaking with its own
    ``VerboseConfig`` is styled by that instance, full stop.

    Attributes:
        enabled: Master switch. When ``False`` the renderer emits
            nothing regardless of individual styles.
        mode: Render backend selector. ``"auto"`` picks a backend
            based on the environment (TTY, ``NO_COLOR``,
            ``FORCE_COLOR``, ``CI``, rich availability); ``"line"``
            forces the stateless line renderer; ``"panel"`` forces the
            Rich-backed panel renderer; ``"off"`` emits nothing.
        use_color: Honour colour codes. When ``False``, or when the
            ``NO_COLOR`` environment variable is set to a non-empty
            value, the renderer emits plain text.
        use_rich: Prefer the Rich-backed renderer (panels, smart
            wrapping). Falls back to plain ANSI when ``rich`` is not
            installed — no error.
        show_timestamps: Prefix each line with an ISO-8601-ish
            ``HH:MM:SS`` stamp. Useful for triaging slow runs.
        output: Destination stream. ``None`` resolves to ``sys.stderr``
            at render time so that verbose output does not pollute
            stdout pipelines.
        styles: Map from event name to :class:`EventStyle`. Modify or
            replace individual entries to recolour / rename / mute
            specific events. New event types (from future Agent
            attributes) can be added via :meth:`register_event`.

    Example — recolour tools and silence LLM events::

        from troopai.adk.verbose import VerboseConfig, EventStyle
        from troopai.adk.verbose.config import (
            EVENT_LLM_START,
            EVENT_LLM_END,
            EVENT_TOOL_START,
            EVENT_TOOL_END,
        )

        cfg = VerboseConfig()
        cfg.styles[EVENT_LLM_START] = EventStyle()  # empty style = mute
        cfg.styles[EVENT_LLM_END] = EventStyle()
        cfg.styles[EVENT_TOOL_START] = EventStyle(
            color="bright_magenta",
            icon="▶",
            prefix="tool",
        )

    Example — register a future Agent attribute's events::

        cfg.register_event(
            "memory.read",
            EventStyle(color="blue", icon="⇲", prefix="memory"),
        )
    """

    enabled: bool = True
    """Master switch. False disables all output from this config."""

    mode: VerboseMode = "auto"
    """Render backend selector. ``"auto"`` picks a backend based on the
    environment (TTY, ``NO_COLOR``, ``FORCE_COLOR``, ``CI``, rich
    availability); ``"line"`` forces the stateless line renderer;
    ``"panel"`` forces the Rich-backed panel renderer; ``"off"``
    emits nothing. See :func:`troopai.adk.verbose.mode.resolve_mode`
    for the full precedence ladder."""

    use_color: bool = True
    """Whether to emit colour. Overridden to False by ``NO_COLOR``."""

    use_rich: bool = True
    """Prefer the Rich renderer. Falls back to ANSI if Rich is missing."""

    show_timestamps: bool = False
    """Emit an ``HH:MM:SS`` prefix on each line."""

    output: TextIO | None = None
    """Destination stream. ``None`` resolves to ``sys.stderr`` at render time."""

    styles: dict[str, EventStyle] = field(default_factory=_default_styles)
    """Per-event render styles. Mutable — edit in place to customise."""

    def get_style(self, event: str) -> EventStyle:
        """Return the style for *event*, or a neutral style if unknown.

        Unknown events are not an error — they degrade to a plain text
        line with no colour, icon, or prefix. This keeps forward
        compatibility with future Agent attributes that publish new
        event types the user has not yet seen.

        Args:
            event: Dotted event name (e.g. ``"tool.start"``).

        Returns:
            The registered :class:`EventStyle`, or a default-constructed
            (all-empty) ``EventStyle`` when *event* is not in the table.
        """
        return self.styles.get(event, EventStyle())

    def register_event(self, event: str, style: EventStyle) -> None:
        """Register or overwrite a style for *event*.

        Intended extension point for future ``Agent`` attributes that
        want their own visibility. An attribute's module should call
        ``verbose_cfg.register_event("attr.name", EventStyle(...))``
        during agent construction so the renderer picks it up on
        first emit.

        Args:
            event: Dotted event name to register
                (e.g. ``"memory.read"``).
            style: The :class:`EventStyle` to associate with *event*.
                Overwrites any previously-registered style silently.
        """
        self.styles[event] = style

    def resolve_output(self) -> TextIO:
        """Return the effective output stream (defaults to stderr).

        Returns:
            The explicitly-configured :attr:`output` stream, or
            ``sys.stderr`` when :attr:`output` is ``None``.
        """
        if self.output is not None:
            return self.output
        return sys.stderr

    def resolve_use_color(self) -> bool:
        """Return the effective colour flag with ``NO_COLOR`` applied.

        Honours the `NO_COLOR standard <https://no-color.org/>`_ —
        any non-empty value disables colour universally. CrewAI does
        not do this; we do.

        Returns:
            ``True`` when colour is enabled and the ``NO_COLOR``
            environment variable is absent or empty; ``False``
            otherwise.
        """
        if not self.use_color:
            return False
        no_color = os.environ.get("NO_COLOR", "")
        return not len(no_color) > 0
