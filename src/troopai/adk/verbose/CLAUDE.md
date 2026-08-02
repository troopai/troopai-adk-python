# Verbose Module

Configurable, colourful event-driven output for agent runs. The
panel backend mirrors CrewAI's `ConsoleFormatter` visual grammar
1:1 (panels with fixed per-event border colours, padding `(1, 2)`,
full-terminal width, and a live-updating streaming widget).

## Files

| File | Purpose |
|---|---|
| `config.py` | `VerboseConfig`, `EventStyle` (now with `panel_title`), event-name constants, default style table. |
| `hooks.py` | `VerboseHooks(RunHooks)` + module-level emit free functions. Installed automatically by the runner. |
| `renderer.py` | `VerboseRenderer` — the stateless line backend (CI / non-TTY / piped output). Unchanged. |
| `panel_renderer.py` | `PanelRenderer` — CrewAI-faithful panel backend with `Live` streaming surface and task-boundary panels. |
| `state.py` | `BlockNode` / `RunTree` — pure data block-tree used by panel mode for ADK-only events. |
| `mode.py` | `resolve_mode()` — environment-aware backend selection. |
| `run_bridge.py` | `ContextVar` bridge for per-chunk streaming emission. Loop sets, `call_llm_streamed` reads. |

## Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Style table is `dict[str, EventStyle]` keyed by dotted event name | Open for extension — future `Agent` attributes can `register_event()` without touching the renderer. |
| 2 | Two-level resolution: `Agent.verbose` → `RunConfig.verbose` | Matches CrewAI's per-agent `verbose=True` ergonomics, with added per-agent styling. |
| 3 | `NO_COLOR` env honoured | CrewAI ignores it; we comply with the standard. |
| 4 | Rich is a soft-import; ANSI is the line backend's fallback | Framework stays rich-less at baseline. Optional `[rich]` extra. |
| 5 | Renderer exceptions never raise | Caught + `logger.debug`. Verbose is telemetry, not correctness. |
| 6 | `VerboseHooks` installed by the runner | Single source of truth for enablement; `Agent.hooks` stays free for user metrics/tracing. |
| 7 | Panel backend mirrors CrewAI's `ConsoleFormatter` 1:1 | Same Rich vocabulary (`Console`, `Panel`, `Text`, `Live`), same titles, same fixed-per-event border colours, same `padding=(1, 2)`. |
| 8 | Live streaming via `ContextVar` bridge | Avoids threading `hooks` / `agent` through `call_llm_streamed` (hot path). Matches `mcp/run_hooks_bridge.py` pattern. |
| 9 | Per-tool `(#N)` counter at class level on `PanelRenderer`, keyed by tool name | Counter persists across renderer instances in a swarm — matches CrewAI's class-level `tool_usage_counts[tool_name]`. Keying by tool name (never a composed headline) keeps counts shared across agents and the dict bounded by the tool vocabulary. Tests MUST clear it in setup/teardown. |
| 10 | `EVENT_AGENT_FINISH` reserved but not emitted today | Constant + style kept for a future streaming-only finish event. The Live-stream suppression flag (`_just_streamed_final_answer`) is consumed by `VerboseHooks.on_agent_end`'s panel branch — that's the only close event the runner fires. Drop `EVENT_AGENT_FINISH` once we either wire it or commit to never wiring it. |
| 11 | CrewAI-canonical lifecycle events render via dedicated `PanelRenderer` methods, never the block tree | `render_agent_started` (capabilities banner), `render_agent_finished`, `render_tool_started/finished/error`, `render_guardrail_verdict`, `render_task_start/end`, `open_stream_panel`. Close-side events must not depend on a matching open block (starts render atomically, so none exists). The `state.py` tree remains for generic paired ADK-only blocks. |
| 12 | HITL approval panels are atomic (styles carry `panel_title`) | The request must print before any stdin prompt blocks, and verdicts arrive in a fresh process after out-of-band approval — a deferred open-block would never flush. |
| 13 | Agent-start banner enumerates capabilities | `Agent:` identity row plus `Description:`/`Tools:`/`Skills:`/`Handoffs:` labeled rows (CrewAI's agent panel + status-content grammar, tools simplified to comma-separated names). Empty rows are omitted. Line mode prints the same rows as plain payload. |

## Wiring

In `run/runner.py`:
* `wrap_hooks_with_verbose()` composes `VerboseHooks(config.verbose)` with user hooks via `compose_run_hooks`.
* The 📋 Task panel is emitted at every `arun()` / `arun_swarm()` / streamed-run entry, closed in the `finally` arm.

In `run/loop.py`:
* `emit_stream_start` / `emit_stream_end` bracket each streamed LLM call.
* Inside that bracket, the `run_bridge` ContextVar is set so the streaming
  loop in `call_llm_streamed` can locate the active hooks chain.

In `run/llm_calls.py`:
* `call_llm_streamed`'s streaming loop emits `fire_stream_chunk` on every
  `part_delta`, distinguishing text vs tool-call deltas via two index sets.

## Extending to new Agent attributes

Add a new event constant + style entry; pass `panel_title=...` for the
panel backend label. Emit via the appropriate `open_block` /
`close_block` / `render_atomic` primitive (ADK-only) or by adding a
dedicated method on `PanelRenderer` (CrewAI-style top-level panel).

See `docs/verbose/verbose.md` for usage patterns and `examples/verbose/`
for runnable examples.
