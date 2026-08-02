"""``VerboseHooks`` — runtime bridge from lifecycle events to a renderer.

Implements :class:`~troopai.adk.hooks.RunHooks`. Wired automatically by
the runner when ``RunConfig.verbose`` is a :class:`VerboseConfig` with
``enabled=True``.

Resolution order at each event: the current agent's
``Agent.verbose`` takes precedence over the run-level
``RunConfig.verbose``. A per-agent :class:`VerboseConfig` can therefore
silence or recolour one agent in a handoff chain without touching the
run-level config.

Two backends are dispatched via :func:`troopai.adk.verbose.mode.resolve_mode`:

* **line** — :class:`~troopai.adk.verbose.renderer.VerboseRenderer`,
  stateless, one line per event. Today's behaviour, preserved bit-
  for-bit. This is also the safe fallback in non-TTY, CI, or
  Rich-missing environments.
* **panel** — :class:`~troopai.adk.verbose.panel_renderer.PanelRenderer`,
  stateful. Paired ``*_start`` / ``*_end`` events open and close
  bordered Rich panels; atomic events (handoff, skill activation)
  flush standalone panels.

The hooks never raise. Any rendering exception is caught and logged at
DEBUG — verbose output is a convenience, not a correctness
requirement.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.verbose.config import (
    EVENT_AGENT_END,
    EVENT_AGENT_START,
    EVENT_BUDGET_EXCEEDED,
    EVENT_BUDGET_WARNING,
    EVENT_CACHE_HIT,
    EVENT_CACHE_MISS,
    EVENT_CONTEXT_COMPACTED,
    EVENT_CONTEXT_EDITED,
    EVENT_GUARDRAIL_INPUT_END,
    EVENT_GUARDRAIL_INPUT_START,
    EVENT_GUARDRAIL_OUTPUT_END,
    EVENT_GUARDRAIL_OUTPUT_START,
    EVENT_HANDOFF,
    EVENT_HITL_APPROVAL_GRANTED,
    EVENT_HITL_APPROVAL_REJECTED,
    EVENT_HITL_APPROVAL_REQUESTED,
    EVENT_LLM_END,
    EVENT_LLM_START,
    EVENT_MCP_CONNECT,
    EVENT_MCP_CONNECTED,
    EVENT_MCP_ERROR,
    EVENT_RETRY,
    EVENT_SESSION_LOAD,
    EVENT_SESSION_SAVE,
    EVENT_SKILL_ACTIVATED,
    EVENT_STATE_SAVE,
    EVENT_STREAM_END,
    EVENT_STREAM_START,
    EVENT_TASK_END,
    EVENT_TASK_FAILED,
    EVENT_TASK_START,
    EVENT_TOOL_END,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_START,
    EVENT_TURN_END,
    EVENT_TURN_START,
    EVENT_USAGE_RECORDED,
    VerboseConfig,
)
from troopai.adk.verbose.mode import ResolvedMode, resolve_mode
from troopai.adk.verbose.panel_renderer import PanelRenderer, StreamCallType, escape_markup, format_tool_payload
from troopai.adk.verbose.renderer import VerboseRenderer, format_payload, redact_secrets
from troopai.adk.verbose.state import BlockKey

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.agents.agent_guardrails import AgentInputGuardrailResult, AgentOutputGuardrailResult
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.session.session_event import SessionEvent
    from troopai.adk.session.state import State
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.tools.tool_guardrails import (
        ToolInputGuardrailResult,
        ToolOutputGuardrailResult,
    )
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.run.run_result import RunResult
    from troopai.adk.types.session import SessionStore
    from troopai.adk.types.tokens.llm_usage import LLMUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block-key helpers
# ---------------------------------------------------------------------------
#
# For panel mode we need a key that uniquely identifies an open block.
# Without access to a run identifier at hook-dispatch time we use
# agent-name-scoped keys — sufficient because each ``RunContext`` owns
# its own ``VerboseHooks`` instance (and therefore its own block tree),
# so concurrent runs of the same :class:`VerboseConfig` never share
# keyspace at the renderer level.


def _agent_key(agent_name: str) -> BlockKey:
    return ("agent", agent_name)


def _llm_key(agent_name: str) -> BlockKey:
    return ("llm", agent_name)


def _tool_key(agent_name: str, tool_name: str) -> BlockKey:
    return ("tool", agent_name, tool_name)


def _guardrail_key(agent_name: str, kind: str, name: str) -> BlockKey:
    return ("guardrail", kind, agent_name, name)


def _tool_guardrail_key(agent_name: str, kind: str, tool_name: str, name: str) -> BlockKey:
    # Distinct from agent-level guardrails so two guardrails with the
    # same name on different tools never collide on the open-block
    # table. Composite key: ("tool_guardrail", "input"/"output",
    # agent_name, tool_name, guardrail_name).
    return ("tool_guardrail", kind, agent_name, tool_name, name)


def _hitl_key(agent_name: str, tool_name: str, call_id: str) -> BlockKey:
    return ("hitl", agent_name, tool_name, call_id)


def _resolve_guardrail_name(result: Any) -> str:
    # A guardrail result wraps the ``ToolInputGuardrail`` /
    # ``ToolOutputGuardrail`` instance on ``.guardrail``. Both classes
    # expose ``get_name()`` which falls back to the wrapped function's
    # ``__name__`` when no explicit ``name`` is set — exactly the
    # behaviour the verbose panel wants. The ``getattr`` double-guard
    # tolerates duck-typed test doubles that set ``name`` directly.
    guardrail = getattr(result, "guardrail", None)
    getter = getattr(guardrail, "get_name", None)
    if callable(getter):
        resolved = getter()
        if isinstance(resolved, str) and len(resolved) > 0:
            return resolved
    raw = getattr(guardrail, "name", None)
    if isinstance(raw, str) and len(raw) > 0:
        return raw
    return "guardrail"


def _tool_guardrail_verdict(behavior_type: str) -> str:
    # Map ToolGuardrailFunctionOutput.behavior["type"] to a PanelRenderer
    # verdict. ``allow`` is a clean pass. ``reject_content`` is a soft
    # trip — execution continues but the model sees a rejection message.
    # ``raise_exception`` is a hard trip. Both non-allow outcomes render
    # with a red border; the headline explains which kind.
    if behavior_type == "allow":
        return "pass"
    if behavior_type == "reject_content":
        return "trip"
    if behavior_type == "raise_exception":
        return "trip"
    return "warn"


def _tool_guardrail_status(verdict: str, behavior_type: str) -> str:
    # Status-row text for the tool-guardrail verdict panel. Carries the
    # behavior type so the operator can distinguish a soft rejection
    # (``reject_content`` — the model sees a rejection message and may
    # retry) from a hard stop (``raise_exception``).
    if verdict == "pass":
        return "✅ Validated"
    if verdict == "trip":
        return f"❌ {behavior_type}"
    return f"⚠️ {behavior_type}"


def _capability_label(item: Any) -> str:
    # Prefer the item's ``name`` (FunctionTool, hosted tools, Skill,
    # Agent); fall back to the class name for containers that expand
    # lazily (a Toolset resolves its member tools at run time, so its
    # type is the only stable label available at agent start). Mirrors
    # CrewAI's tools-field simplification (``getattr(t, "name", str(t))``)
    # with a class-name fallback instead of a noisy ``repr``.
    name = getattr(item, "name", None)
    if isinstance(name, str) and len(name) > 0:
        return name
    return type(item).__name__


def _handoff_labels(handoffs: Any) -> list[str]:
    """Best-effort target names for an agent's ``handoffs`` attribute.

    Accepts the three shapes ``Agent.handoffs`` can take — ``None``, a
    list of agents / handoff declarations, or a routing DSL object with
    ``all_targets()`` — and degrades to an empty list on anything else.
    """
    if handoffs is None:
        return []
    if isinstance(handoffs, list):
        labels: list[str] = []
        for entry in handoffs:
            # A Handoff declaration exposes the destination as
            # ``agent_name``; a bare Agent target exposes ``name``.
            agent_name = getattr(entry, "agent_name", None)
            if isinstance(agent_name, str) and len(agent_name) > 0:
                labels.append(agent_name)
            else:
                labels.append(_capability_label(entry))
        return labels
    all_targets = getattr(handoffs, "all_targets", None)
    if not callable(all_targets):
        return []
    # HandoffRoute — every registered HandoffTarget wraps its
    # destination agent on ``.target``. ``Any``-typed local because
    # ``callable()`` narrows the duck-typed attribute to a callable
    # returning ``object``, which would not be iterable.
    targets: Any = all_targets()
    if not isinstance(targets, list):
        return []
    return [_capability_label(getattr(target, "target", target)) for target in targets]


def _agent_capabilities(agent: Agent) -> tuple[str | None, list[str], list[str], list[str]]:
    """Extract ``(description, tool names, skill names, handoff names)``.

    Verbose output is telemetry: the agent-start banner must render for
    any object that quacks like an :class:`Agent` (tests drive the hooks
    with lightweight stand-ins), so missing attributes degrade to empty
    values instead of raising.
    """
    description = getattr(agent, "description", None)
    if not isinstance(description, str) or len(description) == 0:
        description = None
    tools = getattr(agent, "tools", None)
    tool_names = [_capability_label(tool) for tool in tools] if isinstance(tools, list) else []
    skills = getattr(agent, "skills", None)
    skill_names = [_capability_label(skill) for skill in skills] if isinstance(skills, list) else []
    handoff_names = _handoff_labels(getattr(agent, "handoffs", None))
    return description, tool_names, skill_names, handoff_names


class VerboseHooks(RunHooks):
    """Render lifecycle events via a mode-aware renderer.

    Constructed internally by the runner when verbose output is
    enabled. Not intended to be attached directly by users — set
    ``RunConfig.verbose=VerboseConfig(...)`` instead.

    Caches a :class:`VerboseRenderer` *and* a :class:`PanelRenderer`
    per :class:`VerboseConfig` so that a mode switch across
    per-agent configs does not repay the cost of building a console
    or importing Rich.
    """

    def __init__(self, run_config_verbose: VerboseConfig | None) -> None:
        """Bind to a run-wide :class:`VerboseConfig` default.

        Args:
            run_config_verbose: The ``RunConfig.verbose`` instance, or
                ``None`` if only per-agent configs will apply. When
                ``None``, events from agents that have no own config
                produce no output.
        """
        self._run_config_verbose = run_config_verbose
        # One cache per backend type. Weakly-keyed so entries die with
        # their configs — avoids leaks on runs that build one config
        # per request.
        self._line_cache: weakref.WeakKeyDictionary[VerboseConfig, VerboseRenderer] = weakref.WeakKeyDictionary()
        self._panel_cache: weakref.WeakKeyDictionary[VerboseConfig, PanelRenderer] = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------
    # Resolution & helpers
    # ------------------------------------------------------------------

    def _resolve_config(self, agent: Agent) -> VerboseConfig | None:
        """Pick the effective config for *agent*.

        Per-agent override wins when present. Returns ``None`` if
        neither level is configured, in which case the caller short-
        circuits and emits nothing.
        """
        if agent.verbose is not None:
            return agent.verbose
        return self._run_config_verbose

    def _get_line_renderer(self, config: VerboseConfig) -> VerboseRenderer:
        cached = self._line_cache.get(config)
        if cached is not None:
            return cached
        renderer = VerboseRenderer(config)
        self._line_cache[config] = renderer
        return renderer

    def _get_panel_renderer(self, config: VerboseConfig) -> PanelRenderer:
        cached = self._panel_cache.get(config)
        if cached is not None:
            return cached
        renderer = PanelRenderer(config)
        self._panel_cache[config] = renderer
        return renderer

    def _backend_for(
        self,
        config: VerboseConfig,
    ) -> VerboseRenderer | PanelRenderer | None:
        """Return a lazily-cached renderer for *config*.

        Resolves the mode via :func:`resolve_mode`, returns the matching
        renderer instance, or ``None`` when mode is ``"off"`` — the
        caller short-circuits without touching a console.
        """
        mode: ResolvedMode = resolve_mode(config)
        if mode == "off":
            return None
        if mode == "line":
            return self._get_line_renderer(config)
        return self._get_panel_renderer(config)

    def _panel_backend(self, agent: Agent) -> PanelRenderer | None:
        """Return the panel backend for *agent*, or ``None``.

        ``None`` covers every non-panel outcome — no config, disabled,
        ``"off"``, or line mode — so hook methods can route
        CrewAI-canonical events to a dedicated ``PanelRenderer`` method
        and fall back to the generic line dispatch otherwise.
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return None
        backend = self._backend_for(config)
        if isinstance(backend, PanelRenderer):
            return backend
        return None

    def _render_guardrail_panel(
        self,
        agent: Agent,
        guardrail_name: str,
        *,
        passed: bool,
        status_text: str | None = None,
        tool_name: str | None = None,
    ) -> bool:
        """Render a guardrail verdict panel in panel mode.

        Shared by the four guardrail end handlers (agent-level and
        tool-level, input and output). Returns ``True`` when the panel
        backend consumed the event — the caller then skips the generic
        line dispatch. Never raises.

        Args:
            agent: The agent whose guardrail finished.
            guardrail_name: Resolved guardrail display name.
            passed: ``True`` renders the green success panel.
            status_text: Verdict detail for the ``Status:`` row.
                Defaults to ``"✅ Validated"`` / ``"❌ Tripwire
                triggered"``.
            tool_name: Optional tool the guardrail wraps (tool-level
                guardrails only).
        """
        panel = self._panel_backend(agent)
        if panel is None:
            return False
        if status_text is None:
            status_text = "✅ Validated" if passed is True else "❌ Tripwire triggered"
        try:
            panel.render_guardrail_verdict(
                guardrail_name,
                passed=passed,
                status_text=status_text,
                tool_name=tool_name,
            )
        except Exception as exc:
            logger.debug("VerboseHooks guardrail render failed on %s: %s", guardrail_name, exc)
        return True

    # ------------------------------------------------------------------
    # Dispatch primitives (shared by all hook methods)
    # ------------------------------------------------------------------

    def _dispatch_open(
        self,
        agent: Agent,
        event: str,
        key: BlockKey,
        headline: str,
        payload: str | None = None,
    ) -> None:
        """Open a panel block (panel mode) or emit a line (line mode).

        Panel-mode policy (CrewAI parity):

        * Events with ``EventStyle.panel_title`` set → atomic emission
          immediately on open (matches CrewAI's per-event panels; the
          HITL approval request fires the moment the gate opens, not
          at the matched close). Agent and tool lifecycle events do
          not reach this path in panel mode — their hook methods call
          the dedicated ``PanelRenderer.render_*`` methods directly.
        * Events without a panel title and with ``show_payload=False``
          and no payload → silenced in panel mode entirely. These are
          internal markers (``llm.start``, ``turn.start``) that would
          otherwise render as empty bordered boxes.
        * Otherwise → open a block on the tree; the panel renders when
          :meth:`_dispatch_close` flushes it (kept for ADK-only paired
          events where the open→close pairing is meaningful).
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                style = config.get_style(event)
                # Hook payloads are plain text; the panel backend parses
                # every payload line as Rich markup, so brackets in a
                # ``repr`` (dict list values), a path, or markdown must be
                # escaped or they corrupt the body / drop the whole panel.
                panel_payload = escape_markup(payload) if payload is not None else None
                if len(style.panel_title) > 0:
                    # CrewAI-canonical event — atomic emission. The
                    # tool-counter bump happens inside ``render_atomic``.
                    backend.render_atomic(event, headline, payload=panel_payload, verdict="ok")
                    return
                if style.show_payload is False and (payload is None or len(payload) == 0):
                    # Muted event — panel backend skips entirely.
                    return
                backend.open_block(event, key, headline, payload=panel_payload)
            else:
                backend.render_line(event, headline, payload)
        except Exception as exc:
            logger.debug("VerboseHooks open-dispatch failed on %s: %s", event, exc)

    def _dispatch_close(
        self,
        agent: Agent,
        event: str,
        key: BlockKey,
        headline: str,
        verdict: str = "ok",
        payload: str | None = None,
    ) -> None:
        """Close a panel block (panel mode) or emit a line (line mode).

        Panel-mode policy:

        * Events with ``EventStyle.panel_title`` set → atomic close
          emission (a separate "Tool Output" / "Agent Final Answer"
          panel rather than retroactively flushing an open block).
        * Muted events (``show_payload=False`` + no payload + no
          panel_title) → silenced.
        * Otherwise → close the block on the tree, which flushes the
          panel composed from accumulated payload.
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                style = config.get_style(event)
                # Hook payloads are plain text; escape Rich-markup
                # metacharacters before the panel backend parses them as
                # markup (see :meth:`_dispatch_open`).
                panel_payload = escape_markup(payload) if payload is not None else None
                if len(style.panel_title) > 0:
                    # Titled close events render atomically — a
                    # standalone panel rather than a retroactive flush
                    # of an open block. (The duplicate final-answer
                    # suppression after a streamed answer lives in
                    # :meth:`on_agent_end`, which owns the panel-mode
                    # agent-end rendering.)
                    backend.render_atomic(event, headline, payload=panel_payload, verdict=verdict)
                    return
                if style.show_payload is False and (payload is None or len(payload) == 0):
                    return
                backend.close_block(key, verdict=verdict, final_payload=panel_payload)
            else:
                backend.render_line(event, headline, payload)
        except Exception as exc:
            logger.debug("VerboseHooks close-dispatch failed on %s: %s", event, exc)

    def _dispatch_atomic(
        self,
        agent: Agent,
        event: str,
        headline: str,
        payload: str | None = None,
        verdict: str = "ok",
    ) -> None:
        """Emit an atomic event (handoff, skill activated, warning).

        Panel-mode policy mirrors :meth:`_dispatch_open` /
        :meth:`_dispatch_close`: events with ``show_payload=False`` and
        no payload AND no ``panel_title`` are silenced (otherwise they
        render as empty bordered boxes — uninformative noise). The
        line backend always emits.
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                style = config.get_style(event)
                if (
                    len(style.panel_title) == 0
                    and style.show_payload is False
                    and (payload is None or len(payload) == 0)
                ):
                    return
                # When no payload was supplied but the headline carries
                # information (handoff arrow, skill activation, budget
                # figure), promote the headline into the body so the
                # panel shows it — titled panels would otherwise render
                # an empty bordered box.
                if (payload is None or len(payload) == 0) and len(headline) > 0:
                    payload = headline
                # Hook payloads are plain text; escape Rich-markup
                # metacharacters before the panel backend parses them as
                # markup (see :meth:`_dispatch_open`).
                panel_payload = escape_markup(payload) if payload is not None else None
                backend.render_atomic(event, headline, panel_payload, verdict=verdict)
            else:
                backend.render_line(event, headline, payload)
        except Exception as exc:
            logger.debug("VerboseHooks atomic-dispatch failed on %s: %s", event, exc)

    def _dispatch_run_level(
        self,
        event: str,
        headline: str,
        payload: str | None = None,
    ) -> None:
        """Render an event with no current-agent context (session/state)."""
        config = self._run_config_verbose
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                # Plain-text payload — escape Rich-markup metacharacters
                # before the panel backend parses them as markup (see
                # :meth:`_dispatch_open`).
                panel_payload = escape_markup(payload) if payload is not None else None
                backend.render_atomic(event, headline, panel_payload)
            else:
                backend.render_line(event, headline, payload)
        except Exception as exc:
            logger.debug("VerboseHooks run-level dispatch failed on %s: %s", event, exc)

    # ------------------------------------------------------------------
    # Cleanup — called by runner teardown to close orphan blocks
    # ------------------------------------------------------------------

    def close_all_panels(self, verdict: str = "interrupted") -> None:
        """Flush every still-open panel across cached PanelRenderers.

        Called from the runner's teardown path (e.g. a ``try/finally``
        wrapping ``Runner.arun``) to guarantee no block stays visually
        open if the run ends by exception. Safe to call when no panel
        renderers have been built — it is a no-op in that case.
        """
        for renderer in list(self._panel_cache.values()):
            try:
                renderer.close_all(verdict=verdict)
            except Exception as exc:
                logger.debug("VerboseHooks close_all failed: %s", exc)

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    @override
    async def on_agent_start(
        self,
        context: RunContext[Any],
        agent: Agent,
    ) -> None:
        del context
        try:
            description, tool_names, skill_names, handoff_names = _agent_capabilities(agent)
        except Exception as exc:
            logger.debug("VerboseHooks capability extraction failed on %s: %s", agent.name, exc)
            description, tool_names, skill_names, handoff_names = None, [], [], []
        if description is not None:
            # Bound the banner row — descriptions are developer-authored
            # prose and must not dominate the panel.
            description = format_payload(description, max_chars=300)
        panel = self._panel_backend(agent)
        if panel is not None:
            try:
                panel.render_agent_started(
                    agent.name,
                    description=description,
                    tool_names=tool_names,
                    skill_names=skill_names,
                    handoff_names=handoff_names,
                )
            except Exception as exc:
                logger.debug("VerboseHooks agent-start render failed on %s: %s", agent.name, exc)
            return
        # Line backend: plain-text capability rows under the headline
        # (Rich markup would leak as literal tags on this path).
        lines: list[str] = []
        if description is not None:
            lines.append(f"Description: {description}")
        for label, names in (("Tools", tool_names), ("Skills", skill_names), ("Handoffs", handoff_names)):
            if len(names) > 0:
                lines.append(f"{label}: {', '.join(names)}")
        payload = "\n".join(lines) if len(lines) > 0 else None
        self._dispatch_open(
            agent,
            EVENT_AGENT_START,
            _agent_key(agent.name),
            f"{agent.name} starting",
            payload=payload,
        )

    @override
    async def on_agent_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        result: RunResult | RunResultStreaming,
    ) -> None:
        del context
        output_preview = format_payload(getattr(result, "final_output", None), max_chars=400)
        panel = self._panel_backend(agent)
        if panel is not None:
            try:
                # The streaming Live widget may have already painted the
                # final answer — consume the one-shot flag and skip the
                # duplicate panel in that case.
                if panel.consume_just_streamed_flag() is True:
                    return
                panel.render_agent_finished(agent.name, output_preview)
            except Exception as exc:
                logger.debug("VerboseHooks agent-end render failed on %s: %s", agent.name, exc)
            return
        self._dispatch_close(
            agent,
            EVENT_AGENT_END,
            _agent_key(agent.name),
            headline=f"{agent.name} finished",
            verdict="ok",
            payload=output_preview,
        )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    @override
    async def on_llm_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        messages: list[LLMInputContentItem],
    ) -> None:
        del context
        self._dispatch_open(
            agent,
            EVENT_LLM_START,
            _llm_key(agent.name),
            f"{agent.name} calling LLM ({len(messages)} messages)",
        )

    @override
    async def on_llm_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        response: Any,
    ) -> None:
        del context, response
        self._dispatch_close(
            agent,
            EVENT_LLM_END,
            _llm_key(agent.name),
            headline=f"{agent.name} received LLM response",
            verdict="ok",
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @override
    async def on_tool_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        del context
        # Shallow-redact well-known secret field names before rendering.
        # Tool arguments that carry credentials (``authorization``,
        # ``api_key``) are a real exposure vector when the stream is
        # captured by a log aggregator.
        safe_input = redact_secrets(tool_input)
        args_text = format_payload(safe_input)
        panel = self._panel_backend(agent)
        if panel is not None:
            try:
                panel.render_tool_started(tool_name, args_text)
            except Exception as exc:
                logger.debug("VerboseHooks tool-start render failed on %s: %s", tool_name, exc)
            return
        self._dispatch_open(
            agent,
            EVENT_TOOL_START,
            _tool_key(agent.name, tool_name),
            f"{agent.name} calling {tool_name}",
            payload=args_text,
        )

    @override
    async def on_tool_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        tool_name: str,
        tool_output: Any,
    ) -> None:
        del context
        # Symmetric with ``on_tool_start``: redact well-known secret
        # field names before rendering. Tool results can echo back
        # authentication tokens or API responses that embed credentials,
        # and the verbose stream is a legitimate operator audit channel —
        # but only for non-secret payloads.
        safe_output = redact_secrets(tool_output)
        output_text = format_payload(safe_output)
        panel = self._panel_backend(agent)
        if panel is not None:
            try:
                panel.render_tool_finished(tool_name, output_text)
            except Exception as exc:
                logger.debug("VerboseHooks tool-end render failed on %s: %s", tool_name, exc)
            return
        self._dispatch_close(
            agent,
            EVENT_TOOL_END,
            _tool_key(agent.name, tool_name),
            headline=f"{agent.name} got {tool_name} result",
            verdict="ok",
            payload=output_text,
        )

    # ------------------------------------------------------------------
    # Handoffs
    # ------------------------------------------------------------------

    @override
    async def on_handoff(
        self,
        context: RunContext[Any],
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        del context
        # Render under the outgoing agent's style so a "loud" source is
        # the one announcing the transition.
        self._dispatch_atomic(
            from_agent,
            EVENT_HANDOFF,
            f"{from_agent.name} → {to_agent.name}",
        )

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    @override
    async def on_input_guardrail_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        guardrail_name: str,
    ) -> None:
        del context
        self._dispatch_open(
            agent,
            EVENT_GUARDRAIL_INPUT_START,
            _guardrail_key(agent.name, "input", guardrail_name),
            f"{agent.name} running input guardrail {guardrail_name}",
        )

    @override
    async def on_input_guardrail_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        result: AgentInputGuardrailResult,
    ) -> None:
        del context
        tripped = getattr(getattr(result, "output", None), "tripwire_triggered", False)
        # Resolve the name exactly as the start handler does — via
        # ``guardrail.get_name()``, which falls back to the guardrail
        # function's ``__name__`` when no explicit ``name`` is set. Reading
        # ``.name`` directly returns ``None`` for an unnamed guardrail, which
        # both renders "None" in the headline and builds a close key that
        # never matches the ``get_name()``-derived start key in panel mode.
        name_str = _resolve_guardrail_name(result)
        verdict = "trip" if tripped else "pass"
        if self._render_guardrail_panel(agent, name_str, passed=verdict == "pass") is True:
            return
        self._dispatch_close(
            agent,
            EVENT_GUARDRAIL_INPUT_END,
            _guardrail_key(agent.name, "input", name_str),
            headline=f"{agent.name} input guardrail {name_str}: {verdict}",
            verdict=verdict,
        )

    @override
    async def on_output_guardrail_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        guardrail_name: str,
    ) -> None:
        del context
        self._dispatch_open(
            agent,
            EVENT_GUARDRAIL_OUTPUT_START,
            _guardrail_key(agent.name, "output", guardrail_name),
            f"{agent.name} running output guardrail {guardrail_name}",
        )

    @override
    async def on_output_guardrail_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        result: AgentOutputGuardrailResult,
    ) -> None:
        del context
        tripped = getattr(getattr(result, "output", None), "tripwire_triggered", False)
        # Mirror ``on_input_guardrail_end``: resolve via ``get_name()`` so the
        # close key and headline match the start side for unnamed guardrails.
        name_str = _resolve_guardrail_name(result)
        verdict = "trip" if tripped else "pass"
        if self._render_guardrail_panel(agent, name_str, passed=verdict == "pass") is True:
            return
        self._dispatch_close(
            agent,
            EVENT_GUARDRAIL_OUTPUT_END,
            _guardrail_key(agent.name, "output", name_str),
            headline=f"{agent.name} output guardrail {name_str}: {verdict}",
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Tool-level guardrails (ADK enrichment — CrewAI has no equivalent)
    # ------------------------------------------------------------------
    #
    # Tool-level guardrails are wired through ``RunHooks`` so the Panel
    # renderer shows one block per guardrail scoped by
    # ``(tool_name, guardrail_name)``. Verdicts map from
    # ``ToolGuardrailFunctionOutput.behavior["type"]`` via
    # :func:`_tool_guardrail_verdict`.

    @override
    async def on_tool_input_guardrail_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        tool_name: str,
        guardrail_name: str,
    ) -> None:
        del context
        self._dispatch_open(
            agent,
            EVENT_GUARDRAIL_INPUT_START,
            _tool_guardrail_key(agent.name, "input", tool_name, guardrail_name),
            f"{agent.name} running input guardrail {guardrail_name} on {tool_name}",
        )

    @override
    async def on_tool_input_guardrail_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        tool_name: str,
        result: ToolInputGuardrailResult,
    ) -> None:
        del context
        output = getattr(result, "output", None)
        behavior = getattr(output, "behavior", None)
        behavior_type = behavior.get("type", "allow") if isinstance(behavior, dict) else "allow"
        guardrail_name = _resolve_guardrail_name(result)
        verdict = _tool_guardrail_verdict(behavior_type)
        handled = self._render_guardrail_panel(
            agent,
            guardrail_name,
            passed=verdict == "pass",
            status_text=_tool_guardrail_status(verdict, behavior_type),
            tool_name=tool_name,
        )
        if handled is True:
            return
        self._dispatch_close(
            agent,
            EVENT_GUARDRAIL_INPUT_END,
            _tool_guardrail_key(agent.name, "input", tool_name, guardrail_name),
            headline=(f"{agent.name} input guardrail {guardrail_name} on {tool_name}: {behavior_type}"),
            verdict=verdict,
        )

    @override
    async def on_tool_output_guardrail_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        tool_name: str,
        guardrail_name: str,
    ) -> None:
        del context
        self._dispatch_open(
            agent,
            EVENT_GUARDRAIL_OUTPUT_START,
            _tool_guardrail_key(agent.name, "output", tool_name, guardrail_name),
            f"{agent.name} running output guardrail {guardrail_name} on {tool_name}",
        )

    @override
    async def on_tool_output_guardrail_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        tool_name: str,
        result: ToolOutputGuardrailResult,
    ) -> None:
        del context
        output = getattr(result, "output", None)
        behavior = getattr(output, "behavior", None)
        behavior_type = behavior.get("type", "allow") if isinstance(behavior, dict) else "allow"
        guardrail_name = _resolve_guardrail_name(result)
        verdict = _tool_guardrail_verdict(behavior_type)
        handled = self._render_guardrail_panel(
            agent,
            guardrail_name,
            passed=verdict == "pass",
            status_text=_tool_guardrail_status(verdict, behavior_type),
            tool_name=tool_name,
        )
        if handled is True:
            return
        self._dispatch_close(
            agent,
            EVENT_GUARDRAIL_OUTPUT_END,
            _tool_guardrail_key(agent.name, "output", tool_name, guardrail_name),
            headline=(f"{agent.name} output guardrail {guardrail_name} on {tool_name}: {behavior_type}"),
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    @override
    async def on_skill_activated(
        self,
        context: RunContext[Any],
        agent: Agent,
        skill_name: str,
    ) -> None:
        del context
        self._dispatch_atomic(
            agent,
            EVENT_SKILL_ACTIVATED,
            f"{agent.name} activated skill {skill_name}",
        )

    # ------------------------------------------------------------------
    # Session / state (no agent attribution — use run-level dispatch)
    # ------------------------------------------------------------------

    @override
    async def on_session_load(
        self,
        context: RunContext[Any],
        session: SessionStore,
        events: list[SessionEvent],
    ) -> None:
        del context, session
        self._dispatch_run_level(
            EVENT_SESSION_LOAD,
            f"session loaded ({len(events)} events)",
        )

    @override
    async def on_session_save(
        self,
        context: RunContext[Any],
        session: SessionStore,
        events: list[SessionEvent],
    ) -> None:
        del context, session
        self._dispatch_run_level(
            EVENT_SESSION_SAVE,
            f"session saved ({len(events)} events)",
        )

    @override
    async def on_state_save(
        self,
        context: RunContext[Any],
        session: SessionStore,
        state: State,
    ) -> None:
        del context, session, state
        self._dispatch_run_level(EVENT_STATE_SAVE, "state persisted")

    @override
    async def on_mcp_connect(
        self,
        context: RunContext[Any],
        server_name: str,
    ) -> None:
        del context
        self._dispatch_run_level(EVENT_MCP_CONNECT, f"connecting MCP server {server_name!r}")

    @override
    async def on_mcp_connected(
        self,
        context: RunContext[Any],
        server_name: str,
    ) -> None:
        del context
        self._dispatch_run_level(EVENT_MCP_CONNECTED, f"MCP server {server_name!r} ready")

    @override
    async def on_mcp_error(
        self,
        context: RunContext[Any],
        server_name: str,
        error: BaseException,
    ) -> None:
        del context
        self._dispatch_run_level(
            EVENT_MCP_ERROR,
            f"MCP server {server_name!r} error: {error}",
        )

    # ==================================================================
    # ADK enrichment emit helpers
    # ==================================================================
    #
    # These are NOT ``RunHooks`` method overrides — they are explicit
    # emission entry points called by the runner for events that CrewAI
    # cannot render but ADK naturally produces (HITL gates, cache
    # hit/miss, context compaction, budget alerts, turn boundaries,
    # streaming markers, typed retries, usage recording).
    #
    # The runner wires these into ``run/loop.py``, ``run/tools_executor.py``,
    # ``run/resumption.py``, and ``run/guardrails_executor.py``. Each helper
    # no-ops when no verbose config is active for the supplied agent.

    # -- Turn boundaries --------------------------------------------------

    def emit_turn_start(self, agent: Agent, turn_number: int) -> None:
        """Open a per-turn block.

        Args:
            agent: The agent whose turn is starting.
            turn_number: 1-based turn counter within the current
                agent loop invocation.
        """
        self._dispatch_open(
            agent,
            EVENT_TURN_START,
            ("turn", agent.name, turn_number),
            f"{agent.name} turn {turn_number}",
        )

    def emit_turn_end(
        self,
        agent: Agent,
        turn_number: int,
        verdict: str = "ok",
    ) -> None:
        """Close a per-turn block.

        Args:
            agent: The agent whose turn just ended.
            turn_number: 1-based turn counter matching the
                corresponding :meth:`emit_turn_start` call.
            verdict: Close state (``"ok"``, ``"error"``, etc.).
                Defaults to ``"ok"``.
        """
        self._dispatch_close(
            agent,
            EVENT_TURN_END,
            ("turn", agent.name, turn_number),
            headline=f"{agent.name} turn {turn_number} complete",
            verdict=verdict,
        )

    # -- HITL approval gates (ADK-specific) ------------------------------

    def emit_hitl_approval_requested(
        self,
        agent: Agent,
        tool_name: str,
        call_id: str,
        *,
        tool_input: dict[str, Any] | None = None,
        nested_path: list[str] | None = None,
    ) -> None:
        """Render the pending HITL approval panel.

        Panel mode emits the panel *immediately* (the approval-requested
        style carries a ``panel_title``, which routes the dispatch to an
        atomic emission): the operator must see what they are approving
        before any stdin prompt blocks, and the matching verdict often
        arrives in a *different* process (the run pauses, the approval
        happens out-of-band, and resumption builds fresh hooks), where a
        deferred open-block would never be flushed.

        Args:
            agent: The outer agent that deferred the tool call.
            tool_name: The tool awaiting approval.
            call_id: Unique identifier for this approval gate (used to
                match the later granted/rejected event).
            tool_input: The tool arguments the operator is being asked
                to approve. Rendered as an ``Args:`` row so reviewers
                can see *what* is being approved (e.g.
                ``delete_user(user_id=42)``). Shallow-redacted for
                well-known secret field names before rendering.
            nested_path: When the approval bubbles up through an
                ``as_tool()`` boundary, the full agent/tool breadcrumb
                — e.g. ``["outer", "book_trip", "inner", "charge_card"]``
                — gets rendered as a ``Path:`` row. ADK-specific;
                CrewAI has no nested HITL.
        """
        headline = f"HITL approval required: {agent.name} → {tool_name}"
        lines: list[str] = [f"Agent: {agent.name}", f"Tool: {tool_name}"]
        if nested_path is not None and len(nested_path) > 0:
            lines.append(f"Path: {' → '.join(nested_path)}")
        if tool_input is not None:
            safe_input = redact_secrets(tool_input)
            lines.append(f"Args: {format_payload(safe_input)}")
        payload = "\n".join(lines)
        # Pause the streaming Live widget so the approval prompt's stdin
        # read does not race the refresh loop. Resume happens on the
        # granted / rejected paths below.
        self.pause_live_for_hitl(agent)
        self._dispatch_open(
            agent,
            EVENT_HITL_APPROVAL_REQUESTED,
            _hitl_key(agent.name, tool_name, call_id),
            headline,
            payload=payload,
        )

    def emit_hitl_approval_granted(
        self,
        agent: Agent,
        tool_name: str,
        call_id: str,
        *,
        approver_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Close the HITL gate with an approved verdict.

        Panel mode renders a standalone green panel (the granted style
        carries a ``panel_title``) — it must not depend on a matching
        open block, because resumption after an out-of-band approval
        runs with fresh hooks where no such block exists.

        ``reason`` is operator-supplied free text and is truncated
        through :func:`format_payload` so a malformed audit string
        cannot blow the panel or log stream.

        Args:
            agent: The agent that owns the pending approval gate.
            tool_name: The tool that was approved for execution.
            call_id: Unique identifier matching the corresponding
                :meth:`emit_hitl_approval_requested` call.
            approver_id: Optional identifier of the human or system
                that granted approval (e.g. a user ID or service
                account name).
            reason: Optional free-text rationale supplied by the
                approver. Truncated before display.
        """
        lines: list[str] = [f"Tool: {tool_name}"]
        if approver_id is not None and len(approver_id) > 0:
            lines.append(f"Approved by: {approver_id}")
        if reason is not None and len(reason) > 0:
            lines.append(f"Reason: {format_payload(reason, max_chars=200)}")
        payload = "\n".join(lines)
        self._dispatch_close(
            agent,
            EVENT_HITL_APPROVAL_GRANTED,
            _hitl_key(agent.name, tool_name, call_id),
            headline=f"HITL approved: {agent.name} → {tool_name}",
            verdict="approved",
            payload=payload,
        )
        self.resume_live_for_hitl(agent)

    def emit_hitl_approval_rejected(
        self,
        agent: Agent,
        tool_name: str,
        call_id: str,
        *,
        approver_id: str | None = None,
        message: str | None = None,
    ) -> None:
        """Close the HITL gate with a rejected verdict.

        Panel mode renders a standalone red panel (the rejected style
        carries a ``panel_title``) for the same resume-safety reason as
        :meth:`emit_hitl_approval_granted`.

        ``message`` is operator-supplied free text (shown to the LLM
        to guide retry) and is truncated through :func:`format_payload`
        for the same reason as ``reason`` on the grant path.

        Args:
            agent: The agent that owns the pending approval gate.
            tool_name: The tool whose execution was rejected.
            call_id: Unique identifier matching the corresponding
                :meth:`emit_hitl_approval_requested` call.
            approver_id: Optional identifier of the human or system
                that rejected the request.
            message: Optional rejection message forwarded to the LLM
                to help it decide how to recover. Truncated before
                display.
        """
        lines: list[str] = [f"Tool: {tool_name}"]
        if approver_id is not None and len(approver_id) > 0:
            lines.append(f"Rejected by: {approver_id}")
        if message is not None and len(message) > 0:
            lines.append(f"Message: {format_payload(message, max_chars=200)}")
        payload = "\n".join(lines)
        self._dispatch_close(
            agent,
            EVENT_HITL_APPROVAL_REJECTED,
            _hitl_key(agent.name, tool_name, call_id),
            headline=f"HITL rejected: {agent.name} → {tool_name}",
            verdict="rejected",
            payload=payload,
        )
        self.resume_live_for_hitl(agent)

    # -- Budget (ADK-specific: LLMUsageLimits) ---------------------------

    def emit_budget_warning(
        self,
        agent: Agent,
        percent_used: float,
        limit_type: str,
    ) -> None:
        """Atomic warning panel — budget approaching limit.

        Args:
            agent: The agent attributed with the budget consumption.
            percent_used: Fraction of the budget consumed, expressed
                as a value in ``[0.0, 1.0]`` (e.g. ``0.85`` for 85%).
            limit_type: Short name identifying which ceiling is
                approaching (e.g. ``"input_tokens"``,
                ``"total_cost"``).
        """
        self._dispatch_atomic(
            agent,
            EVENT_BUDGET_WARNING,
            f"budget warning: {limit_type} at {percent_used:.0%}",
            verdict="warn",
        )

    def emit_budget_exceeded(
        self,
        agent: Agent,
        limit_type: str,
    ) -> None:
        """Atomic error panel — budget exceeded. Run usually halts.

        Args:
            agent: The agent attributed with the budget exhaustion.
            limit_type: Short name identifying which ceiling was
                exceeded (e.g. ``"output_tokens"``, ``"total_cost"``).
        """
        self._dispatch_atomic(
            agent,
            EVENT_BUDGET_EXCEEDED,
            f"budget exceeded: {limit_type}",
            verdict="error",
        )

    # -- Provider-aware prompt cache -------------------------------------

    def emit_cache_hit(self, agent: Agent, tool_name: str) -> None:
        """Atomic green panel — tool result served from local cache.

        Args:
            agent: The agent that triggered the cached tool call.
            tool_name: Name of the tool whose result was served from
                cache.
        """
        self._dispatch_atomic(
            agent,
            EVENT_CACHE_HIT,
            f"{agent.name} cache hit: {tool_name}",
            verdict="ok",
        )

    def emit_cache_miss(self, agent: Agent, tool_name: str) -> None:
        """Atomic dim panel — tool result not cached, will execute.

        Args:
            agent: The agent that triggered the cache lookup.
            tool_name: Name of the tool whose result was not found
                in cache.
        """
        self._dispatch_atomic(
            agent,
            EVENT_CACHE_MISS,
            f"{agent.name} cache miss: {tool_name}",
        )

    # -- Context compaction / editing ------------------------------------

    def emit_context_compacted(
        self,
        agent: Agent,
        original_tokens: int,
        final_tokens: int,
    ) -> None:
        """Atomic dim panel — context was summarized by the LLM.

        Args:
            agent: The agent whose context window was compacted.
            original_tokens: Token count before compaction.
            final_tokens: Token count after compaction.
        """
        saved = max(original_tokens - final_tokens, 0)
        self._dispatch_atomic(
            agent,
            EVENT_CONTEXT_COMPACTED,
            (f"context compacted: {original_tokens} → {final_tokens} tokens (saved {saved})"),
        )

    def emit_context_edited(self, agent: Agent, reason: str) -> None:
        """Atomic dim panel — old tool results were cleared from context.

        Args:
            agent: The agent whose context was edited.
            reason: Short human-readable description of why the context
                was edited (e.g. ``"tool_results_pruned"``).
        """
        self._dispatch_atomic(
            agent,
            EVENT_CONTEXT_EDITED,
            f"context edited: {reason}",
        )

    # -- Retry (typed by reason) -----------------------------------------

    def emit_retry(
        self,
        agent: Agent,
        tool_name: str,
        attempt: int,
        reason: str,
    ) -> None:
        """Atomic yellow panel — tool being retried after a typed failure.

        Args:
            agent: The agent that triggered the tool call being retried.
            tool_name: Name of the tool being retried.
            attempt: 1-based retry attempt number.
            reason: Short description of why the retry was triggered
                (e.g. ``"timeout"``, ``"schema_error"``).
        """
        self._dispatch_atomic(
            agent,
            EVENT_RETRY,
            f"retry {attempt}: {agent.name} → {tool_name} ({reason})",
            verdict="warn",
        )

    # -- Streaming markers + Live widget lifecycle -----------------------

    def emit_stream_start(self, agent: Agent, call_type: str = "text") -> None:
        """Open the streaming Live panel (panel mode) or emit a marker line.

        Panel backend: opens a :class:`rich.live.Live` widget via
        :meth:`PanelRenderer.open_stream_panel`. The widget will receive
        per-chunk updates from :meth:`emit_stream_chunk`.
        Line backend: emits one atomic line for CI log compatibility.

        Args:
            agent: The agent whose LLM call is starting to stream.
            call_type: ``"text"`` → green "✅ Agent Final Answer";
                ``"tool_call"`` → yellow "🔧 Tool Arguments".
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                resolved_call_type: StreamCallType = "tool_call" if call_type == "tool_call" else "text"
                backend.open_stream_panel(agent.name, resolved_call_type)
            else:
                backend.render_line(
                    EVENT_STREAM_START,
                    f"{agent.name} streaming started",
                    None,
                )
        except Exception as exc:
            logger.debug("emit_stream_start failed on %s: %s", agent.name, exc)

    def emit_stream_chunk(
        self,
        agent: Agent,
        accumulated_text: str,
        call_type: str = "text",
    ) -> None:
        """Update the running Live panel with the latest accumulated text.

        Panel backend: delegates to
        :meth:`PanelRenderer.update_stream_panel`.
        Line backend: no-op (line renderer cannot live-update).

        This is the only hook method called per LLM stream chunk so it
        is on a hot path. ``CompositeRunHooks`` does NOT fan it out —
        the module-level :func:`emit_stream_chunk` walks the chain
        directly to ``VerboseHooks``, keeping user hooks unwoken on
        every token.

        Args:
            agent: The agent whose LLM call is streaming.
            accumulated_text: Full text accumulated since the stream
                opened, not just the latest delta.
            call_type: ``"text"`` → green panel; ``"tool_call"`` →
                yellow panel. Defaults to ``"text"``.
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        if not isinstance(backend, PanelRenderer):
            return
        try:
            resolved_call_type: StreamCallType = "tool_call" if call_type == "tool_call" else "text"
            backend.update_stream_panel(accumulated_text, resolved_call_type)
        except Exception as exc:
            logger.debug("emit_stream_chunk failed on %s: %s", agent.name, exc)

    def emit_stream_end(self, agent: Agent) -> None:
        """Close the streaming Live panel (panel mode) or emit a marker line.

        Panel backend: stops the Live via
        :meth:`PanelRenderer.close_stream_panel` and sets the
        ``_just_streamed_final_answer`` flag so a subsequent
        ``agent.finish`` block close suppresses its duplicate panel.
        Line backend: emits one atomic line for CI log compatibility.
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                backend.close_stream_panel()
            else:
                backend.render_line(
                    EVENT_STREAM_END,
                    f"{agent.name} streaming ended",
                    None,
                )
        except Exception as exc:
            logger.debug("emit_stream_end failed on %s: %s", agent.name, exc)

    # -- Task boundary (per outer Runner.arun / arun_swarm / arun_graph call) -

    def _resolve_task_panel_config(
        self,
        scope: Agent | Swarm | Graph,
    ) -> tuple[VerboseConfig | None, str]:
        """Resolve the effective verbose config + debug label for *scope*.

        Task panels bracket whole-run scopes (an :class:`Agent`, a
        :class:`Swarm`, or a :class:`Graph`). Each type carries the
        verbose toggle differently:

        * :class:`Agent` — delegates to :meth:`_resolve_config`, which
          honours the per-instance ``agent.verbose`` override.
        * :class:`Swarm` — delegates to :meth:`_resolve_config` on the
          entry agent. There is no swarm-level toggle, and the first
          turn renders through that agent, so its config is the right
          proxy.
        * :class:`Graph` — no per-graph toggle exists; falls back to
          the run-config value.

        The returned label is used only in the ``logger.debug`` error
        path; rendering itself uses ``task_name`` / ``task_id`` only.

        Args:
            scope: The whole-run unit whose verbose config should be
                resolved. May be an :class:`Agent`, :class:`Swarm`,
                or :class:`Graph`.

        Returns:
            A 2-tuple of ``(config_or_none, debug_label)`` where
            *config_or_none* is the effective :class:`VerboseConfig`
            (``None`` when none is configured) and *debug_label* is
            a short string used only in error log lines.
        """
        from troopai.adk.agents.agent import Agent
        from troopai.adk.swarms.swarm import Swarm

        if isinstance(scope, Agent):
            return self._resolve_config(scope), scope.name
        if isinstance(scope, Swarm):
            return self._resolve_config(scope.entry), f"swarm:{scope.entry.name}"
        return self._run_config_verbose, f"graph:{scope.id}"

    def emit_task_start(
        self,
        scope: Agent | Swarm | Graph,
        task_name: str,
        task_id: str,
    ) -> None:
        """Render the ``📋 Task Started`` panel at the start of a run.

        Called from ``Runner.arun`` / ``arun_swarm`` / ``arun_graph`` /
        streamed-run entry points before the agent loop runs. Line
        backend emits a single descriptive line; panel backend renders
        the bordered CrewAI-style panel via
        :meth:`PanelRenderer.render_task_start`.

        *scope* is the whole-run unit bracketed by the panel — an
        :class:`Agent`, :class:`Swarm`, or :class:`Graph`. The widened
        union lets graph runs (which lack an entry-Agent) participate
        in the same panel lifecycle as single-agent and swarm runs;
        see :meth:`_resolve_task_panel_config` for how the effective
        :class:`VerboseConfig` is selected per scope type.

        Args:
            scope: The whole-run unit bracketed by the panel. May be
                an :class:`Agent`, :class:`Swarm`, or :class:`Graph`.
            task_name: Human-readable task name shown in the panel
                body.
            task_id: Full task identifier shown truncated in the panel
                body.
        """
        config, scope_label = self._resolve_task_panel_config(scope)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                backend.render_task_start(task_name, task_id)
            else:
                backend.render_line(
                    EVENT_TASK_START,
                    f"task started: {task_name}",
                    f"id={task_id}",
                )
        except Exception as exc:
            logger.debug("emit_task_start failed on %s: %s", scope_label, exc)

    def emit_task_end(
        self,
        scope: Agent | Swarm | Graph,
        task_name: str,
        task_id: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Render the ``📋 Task Completed`` or ``❌ Task Failed`` panel.

        Called from the success / exception arms of ``Runner.arun``,
        ``arun_swarm``, and ``arun_graph``. *error* (when *success* is
        False) is appended to the panel as a red row. See
        :meth:`emit_task_start` for the *scope* contract.

        Args:
            scope: The whole-run unit whose verbose config is resolved
                via :meth:`_resolve_task_panel_config`. Same value
                passed to the corresponding :meth:`emit_task_start`.
            task_name: Human-readable task name.
            task_id: Full task identifier.
            success: ``True`` emits a green "Task Completed" panel;
                ``False`` emits a red "Task Failed" panel.
            error: Optional error description appended as a red row
                when *success* is ``False``.
        """
        config, scope_label = self._resolve_task_panel_config(scope)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if backend is None:
            return
        try:
            if isinstance(backend, PanelRenderer):
                backend.render_task_end(task_name, task_id, success=success, error=error)
            else:
                event = EVENT_TASK_END if success is True else EVENT_TASK_FAILED
                label = "completed" if success is True else "failed"
                payload = f"id={task_id}"
                if error is not None and len(error) > 0:
                    payload = payload + f" error={error}"
                backend.render_line(event, f"task {label}: {task_name}", payload)
        except Exception as exc:
            logger.debug("emit_task_end failed on %s: %s", scope_label, exc)

    # -- HITL coordination (pause Live during approval prompts) ----------

    def pause_live_for_hitl(self, agent: Agent) -> None:
        """Stop the running Live widget so an HITL prompt can read stdin.

        Called by HITL approval emitters before opening their own
        block. No-op for the line backend.
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if not isinstance(backend, PanelRenderer):
            return
        try:
            backend.pause_live_updates()
        except Exception as exc:
            logger.debug("pause_live_for_hitl failed on %s: %s", agent.name, exc)

    def resume_live_for_hitl(self, agent: Agent) -> None:
        """Reset Live state after the HITL prompt completes.

        Subsequent ``emit_stream_chunk`` calls on a new stream will
        construct a fresh Live via :meth:`PanelRenderer.open_stream_panel`.
        """
        config = self._resolve_config(agent)
        if config is None or not config.enabled:
            return
        backend = self._backend_for(config)
        if not isinstance(backend, PanelRenderer):
            return
        try:
            backend.resume_live_updates()
        except Exception as exc:
            logger.debug("resume_live_for_hitl failed on %s: %s", agent.name, exc)

    # -- Tool error (first real call site) -------------------------------

    def emit_tool_error(
        self,
        agent: Agent,
        tool_name: str,
        error: BaseException,
    ) -> None:
        """Render the tool failure (red panel / error line).

        Called from ``run/tools_executor.py`` in the exception path so
        the operator sees a red ``🔧 Tool Error (#N)`` panel naming the
        tool and the failure.

        Third-party exception messages can embed connection strings,
        file paths, or endpoint URLs; the payload is truncated through
        ``format_payload`` so a runaway stack trace never dominates the
        panel stream.
        """
        message = format_payload(f"{type(error).__name__}: {error}", max_chars=200)
        panel = self._panel_backend(agent)
        if panel is not None:
            try:
                panel.render_tool_error(tool_name, message)
            except Exception as exc:
                logger.debug("VerboseHooks tool-error render failed on %s: %s", tool_name, exc)
            return
        self._dispatch_close(
            agent,
            EVENT_TOOL_ERROR,
            _tool_key(agent.name, tool_name),
            headline=f"{agent.name} tool {tool_name} failed",
            verdict="error",
            payload=message,
        )

    # -- Usage recording -------------------------------------------------

    def emit_usage_recorded(
        self,
        agent: Agent,
        usage: LLMUsage,
    ) -> None:
        """Atomic info panel — cumulative token usage after an LLM call.

        Reads the framework-canonical :class:`LLMUsage` field names
        (``input_tokens`` / ``output_tokens`` / ``total_tokens``); uses
        ``getattr`` with zero-defaults so providers that populate a
        subset (or a mock for tests) degrade gracefully.
        """
        input_t = getattr(usage, "input_tokens", 0) or 0
        output_t = getattr(usage, "output_tokens", 0) or 0
        total_t = getattr(usage, "total_tokens", input_t + output_t) or (input_t + output_t)
        self._dispatch_atomic(
            agent,
            EVENT_USAGE_RECORDED,
            (f"{agent.name} usage: {input_t} input + {output_t} output = {total_t} total"),
        )


# ---------------------------------------------------------------------------
# Runner-facing module helpers
# ---------------------------------------------------------------------------
#
# The runner holds a ``RunHooks`` reference that is commonly a
# ``CompositeRunHooks`` wrapping user hooks plus ``VerboseHooks``.
# Rather than teach every emission site how to navigate that structure,
# expose module-level free functions that walk the hook chain, locate
# any ``VerboseHooks`` instance, and invoke the matching ``emit_*``
# method. Runners stay oblivious to the verbose layer — if none is
# installed, every call is a zero-cost no-op.


def find_verbose_hooks(hooks: Any) -> list[VerboseHooks]:
    """Return every :class:`VerboseHooks` reachable from *hooks*.

    Understands :class:`CompositeRunHooks`' public ``members`` list so
    a run with both a user hook and a framework-installed verbose hook
    emits once per event (the verbose member is still called even when
    nested behind the composite).
    """
    if hooks is None:
        return []
    found: list[VerboseHooks] = []
    if isinstance(hooks, VerboseHooks):
        found.append(hooks)
        return found
    members = getattr(hooks, "members", None)
    if members is None:
        return found
    for m in members:
        if isinstance(m, VerboseHooks):
            found.append(m)
    return found


def _for_each_verbose(hooks: Any, label: str, action: Callable[[VerboseHooks], None]) -> None:
    """Run *action* on every reachable :class:`VerboseHooks`.

    Silent no-op when no verbose hooks are installed. Per-instance
    errors are caught and logged at DEBUG (or WARNING for HITL emit
    paths — silently-dropped approval panels matter for audit). Never
    raises — verbose output is a telemetry concern; runner correctness
    takes precedence.

    Callers pass a direct method call inside a lambda (rather than
    ``getattr(vh, method_name)``) so the call is type-checked against
    ``VerboseHooks`` and avoids computed-name attribute access on a
    typed object.
    """
    severity_warning = label.startswith("emit_hitl_")
    for vh in find_verbose_hooks(hooks):
        try:
            action(vh)
        except Exception as exc:
            if severity_warning:
                logger.warning("verbose %s failed: %s", label, exc)
            else:
                logger.debug("verbose %s failed: %s", label, exc)


def emit_turn_start(hooks: Any, agent: Agent, turn_number: int) -> None:
    """Open a per-turn block on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain (may be a
            ``CompositeRunHooks`` or a bare :class:`VerboseHooks`).
        agent: The agent whose turn is starting.
        turn_number: 1-based turn counter within the current agent
            loop invocation.
    """
    _for_each_verbose(hooks, "emit_turn_start", lambda vh: vh.emit_turn_start(agent, turn_number))


def emit_turn_end(
    hooks: Any,
    agent: Agent,
    turn_number: int,
    verdict: str = "ok",
) -> None:
    """Close a per-turn block on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose turn just ended.
        turn_number: 1-based turn counter matching the corresponding
            :func:`emit_turn_start` call.
        verdict: Close state (``"ok"``, ``"error"``, etc.). Defaults
            to ``"ok"``.
    """
    _for_each_verbose(
        hooks,
        "emit_turn_end",
        lambda vh: vh.emit_turn_end(agent, turn_number, verdict=verdict),
    )


def emit_hitl_approval_requested(
    hooks: Any,
    agent: Agent,
    tool_name: str,
    call_id: str,
    *,
    tool_input: dict[str, Any] | None = None,
    nested_path: list[str] | None = None,
) -> None:
    """Open a pending HITL approval panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent that deferred the tool call.
        tool_name: The tool awaiting approval.
        call_id: Unique identifier for this approval gate.
        tool_input: Optional tool arguments to display. Secret-key
            values are redacted before rendering.
        nested_path: Optional breadcrumb list when the approval
            bubbles through an ``as_tool()`` boundary.
    """
    _for_each_verbose(
        hooks,
        "emit_hitl_approval_requested",
        lambda vh: vh.emit_hitl_approval_requested(
            agent,
            tool_name,
            call_id,
            tool_input=tool_input,
            nested_path=nested_path,
        ),
    )


def emit_hitl_approval_granted(
    hooks: Any,
    agent: Agent,
    tool_name: str,
    call_id: str,
    *,
    approver_id: str | None = None,
    reason: str | None = None,
) -> None:
    """Close the HITL gate with an approved verdict on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent that owns the approval gate.
        tool_name: The tool that was approved.
        call_id: Identifier matching the corresponding
            :func:`emit_hitl_approval_requested` call.
        approver_id: Optional identifier of the approver.
        reason: Optional free-text rationale. Truncated before display.
    """
    _for_each_verbose(
        hooks,
        "emit_hitl_approval_granted",
        lambda vh: vh.emit_hitl_approval_granted(
            agent,
            tool_name,
            call_id,
            approver_id=approver_id,
            reason=reason,
        ),
    )


def emit_hitl_approval_rejected(
    hooks: Any,
    agent: Agent,
    tool_name: str,
    call_id: str,
    *,
    approver_id: str | None = None,
    message: str | None = None,
) -> None:
    """Close the HITL gate with a rejected verdict on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent that owns the approval gate.
        tool_name: The tool whose execution was rejected.
        call_id: Identifier matching the corresponding
            :func:`emit_hitl_approval_requested` call.
        approver_id: Optional identifier of the rejector.
        message: Optional rejection message forwarded to the LLM.
            Truncated before display.
    """
    _for_each_verbose(
        hooks,
        "emit_hitl_approval_rejected",
        lambda vh: vh.emit_hitl_approval_rejected(
            agent,
            tool_name,
            call_id,
            approver_id=approver_id,
            message=message,
        ),
    )


def emit_budget_warning(
    hooks: Any,
    agent: Agent,
    percent_used: float,
    limit_type: str,
) -> None:
    """Emit a budget-approaching warning panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent attributed with the budget consumption.
        percent_used: Fraction consumed, in ``[0.0, 1.0]``.
        limit_type: Name of the ceiling being approached.
    """
    _for_each_verbose(
        hooks,
        "emit_budget_warning",
        lambda vh: vh.emit_budget_warning(agent, percent_used, limit_type),
    )


def emit_budget_exceeded(hooks: Any, agent: Agent, limit_type: str) -> None:
    """Emit a budget-exceeded error panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent attributed with the budget exhaustion.
        limit_type: Name of the ceiling that was exceeded.
    """
    _for_each_verbose(
        hooks,
        "emit_budget_exceeded",
        lambda vh: vh.emit_budget_exceeded(agent, limit_type),
    )


def emit_cache_hit(hooks: Any, agent: Agent, tool_name: str) -> None:
    """Emit a cache-hit panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose tool result was served from cache.
        tool_name: Name of the tool with a cache hit.
    """
    _for_each_verbose(hooks, "emit_cache_hit", lambda vh: vh.emit_cache_hit(agent, tool_name))


def emit_cache_miss(hooks: Any, agent: Agent, tool_name: str) -> None:
    """Emit a cache-miss panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose tool result was not in cache.
        tool_name: Name of the tool with a cache miss.
    """
    _for_each_verbose(hooks, "emit_cache_miss", lambda vh: vh.emit_cache_miss(agent, tool_name))


def emit_context_compacted(
    hooks: Any,
    agent: Agent,
    original_tokens: int,
    final_tokens: int,
) -> None:
    """Emit a context-compacted panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose context window was compacted.
        original_tokens: Token count before compaction.
        final_tokens: Token count after compaction.
    """
    _for_each_verbose(
        hooks,
        "emit_context_compacted",
        lambda vh: vh.emit_context_compacted(agent, original_tokens, final_tokens),
    )


def emit_context_edited(hooks: Any, agent: Agent, reason: str) -> None:
    """Emit a context-edited panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose context was edited.
        reason: Short description of the edit reason.
    """
    _for_each_verbose(hooks, "emit_context_edited", lambda vh: vh.emit_context_edited(agent, reason))


def emit_retry(
    hooks: Any,
    agent: Agent,
    tool_name: str,
    attempt: int,
    reason: str,
) -> None:
    """Emit a retry panel on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent that triggered the retried tool call.
        tool_name: Name of the tool being retried.
        attempt: 1-based retry attempt number.
        reason: Short description of the failure that triggered the
            retry.
    """
    _for_each_verbose(
        hooks,
        "emit_retry",
        lambda vh: vh.emit_retry(agent, tool_name, attempt, reason),
    )


def emit_stream_start(hooks: Any, agent: Agent, call_type: str = "text") -> None:
    """Open the Live streaming panel (panel backend) or emit a marker line.

    *call_type* picks the panel's title and border colour: ``"text"`` →
    green "✅ Agent Final Answer"; ``"tool_call"`` → yellow
    "🔧 Tool Arguments".

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose LLM call is starting to stream.
        call_type: ``"text"`` or ``"tool_call"``. Defaults to
            ``"text"``.
    """
    _for_each_verbose(
        hooks,
        "emit_stream_start",
        lambda vh: vh.emit_stream_start(agent, call_type),
    )


def emit_stream_chunk(
    hooks: Any,
    agent: Agent,
    accumulated_text: str,
    call_type: str = "text",
) -> None:
    """Per-chunk Live update during LLM streaming.

    Called from ``run/llm_calls.py`` on every ``part_delta`` event of the
    streaming iterator. Routed directly to :class:`VerboseHooks` (NOT
    fan-out through ``CompositeRunHooks``) so user hooks are not woken
    on every token.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose LLM call is streaming.
        accumulated_text: Full text accumulated since stream start.
        call_type: ``"text"`` or ``"tool_call"``. Defaults to
            ``"text"``.
    """
    _for_each_verbose(
        hooks,
        "emit_stream_chunk",
        lambda vh: vh.emit_stream_chunk(agent, accumulated_text, call_type),
    )


def emit_stream_end(hooks: Any, agent: Agent) -> None:
    """Close the Live streaming panel (panel backend) or emit a marker line.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose LLM stream just finished.
    """
    _for_each_verbose(hooks, "emit_stream_end", lambda vh: vh.emit_stream_end(agent))


def emit_task_start(
    hooks: Any,
    scope: Agent | Swarm | Graph,
    task_name: str,
    task_id: str,
) -> None:
    """Render the ``📋 Task Started`` panel at the start of a run.

    Called from ``Runner.arun`` / ``Runner.arun_swarm`` /
    ``Runner.arun_graph`` / the streamed-run entry point before the
    agent loop runs. *scope* is the whole-run unit bracketed by the
    panel — see :meth:`VerboseHooks.emit_task_start` for the contract.

    Args:
        hooks: The active ``RunHooks`` chain.
        scope: The whole-run unit bracketed by the panel. May be an
            :class:`Agent`, :class:`Swarm`, or :class:`Graph`.
        task_name: Human-readable task name.
        task_id: Full task identifier.
    """
    _for_each_verbose(
        hooks,
        "emit_task_start",
        lambda vh: vh.emit_task_start(scope, task_name, task_id),
    )


def emit_task_end(
    hooks: Any,
    scope: Agent | Swarm | Graph,
    task_name: str,
    task_id: str,
    *,
    success: bool,
    error: str | None = None,
) -> None:
    """Render the ``📋 Task Completed`` (success) or ``❌ Task Failed`` panel.

    Called from the success / exception arms of ``Runner.arun`` /
    ``arun_swarm`` / ``arun_graph``. *scope* is the same value passed
    to :func:`emit_task_start` for the open of this bracket.

    Args:
        hooks: The active ``RunHooks`` chain.
        scope: The whole-run unit passed to :func:`emit_task_start`.
        task_name: Human-readable task name.
        task_id: Full task identifier.
        success: ``True`` emits a green "Task Completed" panel;
            ``False`` emits a red "Task Failed" panel.
        error: Optional error description appended as a red row
            when *success* is ``False``.
    """
    _for_each_verbose(
        hooks,
        "emit_task_end",
        lambda vh: vh.emit_task_end(scope, task_name, task_id, success=success, error=error),
    )


def pause_live_for_hitl(hooks: Any, agent: Agent) -> None:
    """Stop the streaming Live widget while an HITL prompt reads stdin.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose streaming widget should be paused.
    """
    _for_each_verbose(hooks, "pause_live_for_hitl", lambda vh: vh.pause_live_for_hitl(agent))


def resume_live_for_hitl(hooks: Any, agent: Agent) -> None:
    """Reset Live state after the HITL prompt completes.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose streaming widget was paused.
    """
    _for_each_verbose(hooks, "resume_live_for_hitl", lambda vh: vh.resume_live_for_hitl(agent))


def emit_tool_error(
    hooks: Any,
    agent: Agent,
    tool_name: str,
    error: BaseException,
) -> None:
    """Close the tool block with an error verdict on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent whose tool call raised the error.
        tool_name: Name of the failed tool.
        error: The exception raised by the tool. The message is
            truncated before display.
    """
    _for_each_verbose(
        hooks,
        "emit_tool_error",
        lambda vh: vh.emit_tool_error(agent, tool_name, error),
    )


def emit_usage_recorded(hooks: Any, agent: Agent, usage: LLMUsage) -> None:
    """Emit cumulative token-usage info on every reachable :class:`VerboseHooks`.

    Args:
        hooks: The active ``RunHooks`` chain.
        agent: The agent attributed with the token usage.
        usage: The :class:`~troopai.adk.types.tokens.llm_usage.LLMUsage`
            instance populated by the LLM provider after the call.
    """
    _for_each_verbose(hooks, "emit_usage_recorded", lambda vh: vh.emit_usage_recorded(agent, usage))


# ``format_tool_payload`` is re-exported here purely to flag that
# PanelRenderer-scoped helpers are available on the hooks module for
# advanced customisation (custom VerboseHooks subclasses composing
# tool-call payload differently). The import is kept at module load
# time so users who grep for it find it.
__all__ = [
    "VerboseHooks",
    "emit_budget_exceeded",
    "emit_budget_warning",
    "emit_cache_hit",
    "emit_cache_miss",
    "emit_context_compacted",
    "emit_context_edited",
    "emit_hitl_approval_granted",
    "emit_hitl_approval_rejected",
    "emit_hitl_approval_requested",
    "emit_retry",
    "emit_stream_chunk",
    "emit_stream_end",
    "emit_stream_start",
    "emit_task_end",
    "emit_task_start",
    "emit_tool_error",
    "emit_turn_end",
    "emit_turn_start",
    "emit_usage_recorded",
    "find_verbose_hooks",
    "format_tool_payload",
    "pause_live_for_hitl",
    "resume_live_for_hitl",
]
