# Verbose Output

Colourful, event-driven tracing of agent runs. Inspired by CrewAI's
console output, but with **two backends** (stateless line + stateful
panel), **developer-chosen mode resolution**, and **ADK-first-class
event vocabulary** (HITL, budgets, cache, context, turn boundaries,
streaming markers, typed retries).

TroopAI = "work of art". Verbose output is where that promise lives
on the terminal.

## Quick start

```python
import asyncio
import logging

from troopai.adk import Agent, Runner, RunConfig, VerboseConfig

logger = logging.getLogger(__name__)

agent = Agent(
    name="Assistant",
    llm="gpt-4o-mini",
    system_prompt="Answer concisely.",
)

async def main() -> None:
    result = await Runner.arun(
        agent,
        "What is the capital of France?",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Final: %s", result.final_output)

asyncio.run(main())
```

Default `VerboseConfig()` resolves to `mode="auto"`: Rich panels on an
interactive TTY, plain lines in CI or when stdout is piped / redirected
to a file. No code changes required across environments.

## Backends

| Backend | When it runs | What it looks like | Stability |
|---|---|---|---|
| `line` | Always safe; picked in non-TTY, CI, `NO_COLOR`, `TERM=dumb`, or when Rich isn't installed | One coloured line per event on stderr | Byte-for-byte stable — the original TroopAI verbose output |
| `panel` | Interactive TTY + Rich installed + not CI | Bordered Rich panels per logical block (📋 Task, 🤖 Agent, 🔧 Tool, ✅ Final Answer) with **event-kind border colours** and a live-updating streaming panel | CrewAI-faithful — mirrors `ConsoleFormatter` verbatim |

Select explicitly with `VerboseConfig(mode=...)`:

```python
VerboseConfig(mode="auto")     # default, environment-aware
VerboseConfig(mode="line")     # force line renderer
VerboseConfig(mode="panel")    # force Rich panels (requires Rich)
VerboseConfig(mode="off")      # emit nothing
```

## Mode resolution ladder

`resolve_mode(config)` walks this ladder top-to-bottom and returns the
first match:

1. `config.enabled is False` → `off`
2. `mode == "off"` → `off`
3. `mode == "line"` → `line`
4. `mode == "panel"` + Rich missing → `line` (with DEBUG log)
5. `mode == "panel"` → `panel`
6. **auto mode only** — `NO_COLOR` env → `line`
7. **auto mode only** — `FORCE_COLOR` env → `panel` if Rich else `line`
8. **auto mode only** — output is not a TTY → `line`
9. **auto mode only** — `CI` / `TERM=dumb` → `line`
10. **auto mode only** — Rich not installed → `line`
11. **auto mode default** → `panel`

Every downgrade logs at `DEBUG` (`configs/logging/default_logger.yaml`
routes this to the `.log` file), so operators can verify the resolved
mode without adding instrumentation.

## Scenario matrix

| Scenario | `mode="auto"` picks | Why |
|---|---|---|
| Dev laptop, interactive terminal | `panel` | TTY + Rich + not CI |
| `python script.py > run.log` | `line` | stdout redirected, no TTY |
| `python script.py \| cat` | `line` | pipe, no TTY |
| GitHub Actions / GitLab CI | `line` | `CI=1` env |
| `NO_COLOR=1 python script.py` | `line` | NO_COLOR standard |
| `TERM=dumb python script.py` | `line` | dumb terminal |
| Docker container, Rich installed, interactive | `panel` | TTY + Rich |
| Docker container, Rich missing | `line` | Rich not importable |

See `examples/verbose/ci_safe.py` for a script that prints the
resolved mode + its inputs at startup.

## Interaction with the classic logger

`VerboseConfig` writes to a `TextIO` stream (default `sys.stderr`) — it
does **not** go through Python's `logging` module. The two channels are
fully independent by design.

The default logger configuration (`configs/logging/default_logger.yaml`)
attaches **no console handler**: `logger.*` records go only to the
rotating `.log` file. The terminal therefore belongs entirely to the
verbose event stream — there is no collision to manage in a standard
setup.

These are two independent channels:

| Channel | Source | Destination | Config |
|---|---|---|---|
| Verbose output | `VerboseRenderer.render_line()` / `PanelRenderer.close_block()` | `stderr` (or explicit `TextIO`) | `VerboseConfig` |
| Structured logs | `logger.info/debug/...` | Rotating `.log` file (file-only by default) | `configs/logging/*.yaml` |

The programmatic fallback — used when the YAML file is absent or PyYAML
is not installed — installs a `logging.NullHandler`, following standard
library practice. No log records reach the terminal in that path either.

**Re-enabling console log output.** If you want `logger.*` records on
the terminal as well, uncomment the `console` handler definition in
`configs/logging/default_logger.yaml` and add `console` to the
`handlers:` list for each logger you want it on. Accept that log lines
and verbose event lines will interleave on the same terminal; this is a
deliberate operator choice, not the default.

If you want verbose output in the same file as your structured logs,
open the file yourself and pass it via `VerboseConfig(output=...)`.

## Per-event control

Every event carries an `EventStyle` — customise colour, icon, prefix,
and whether the payload (tool args, LLM message bodies, guardrail
results) is shown.

```python
from troopai.adk.verbose import VerboseConfig, EventStyle
from troopai.adk.verbose.config import EVENT_TOOL_START, EVENT_LLM_START

cfg = VerboseConfig()
cfg.styles[EVENT_TOOL_START] = EventStyle(
    color="bright_magenta", icon="▶", prefix="tool",
)
cfg.styles[EVENT_LLM_START] = EventStyle()  # empty style = mute
```

## Per-agent override

Set `Agent.verbose` to override the run-level config for one agent.
Useful in multi-agent swarms to make one agent loud and another silent.

```python
from troopai.adk import Agent
from troopai.adk.verbose import VerboseConfig

coordinator = Agent(name="Coordinator", llm="gpt-4o", verbose=VerboseConfig())
summariser = Agent(
    name="Summariser",
    llm="gpt-4o-mini",
    verbose=VerboseConfig(enabled=False),
)
```

The renderer resolves at emit time: `Agent.verbose` wins over
`RunConfig.verbose`. A per-agent config with `enabled=False` silences
that agent while the rest of the run keeps its styling.

## Event-kind border colour (panel mode)

Border colours are fixed per **event kind**, not derived from a verdict
string. This matches CrewAI's `ConsoleFormatter`: each panel type has a
recognisable colour signature so the user can scan output without
reading titles:

| Event | Panel title | Border |
|---|---|---|
| `task.start` | 📋 Task Started | yellow |
| `task.end` | 📋 Task Completed | green |
| `task.failed` | ❌ Task Failed | red |
| `agent.start` | 🤖 Agent Started | magenta |
| `agent.finish` | ✅ Agent Final Answer | green |
| `tool.start` | 🔧 Tool Execution Started (#N) | yellow |
| `tool.error` | (inherits `EventStyle.color`) | red |
| `run.start` | 🚀 Crew Execution Started | cyan |
| `run.end` | Crew Completion | green |

ADK-only events (HITL, budget, cache, context, MCP, guardrails) without
an entry in `_EVENT_BORDER` fall through to their configured
`EventStyle.color` — they stay extensible and the rest of the palette is
locked.

## Live streaming

When an LLM call streams (e.g. `Runner.arun(..., stream=True)`), the
panel backend opens a `rich.live.Live` widget that updates in place as
tokens arrive:

* Final-answer text streams render in the green **✅ Agent Final Answer**
  panel.
* Tool-call argument deltas (function-call JSON) stream in the yellow
  **🔧 Tool Arguments** panel.

The Live widget refreshes at 10 Hz internally — chunk emission is
cheap. Per-chunk events go straight to `VerboseHooks` (not through
`CompositeRunHooks` fan-out) so user-installed hooks are not woken on
every token.

After a text stream finishes, the renderer suppresses the duplicate
`agent.finish` block panel (the Live widget already painted the answer)
via a `_just_streamed_final_answer` flag.

`pause_live_for_hitl` / `resume_live_for_hitl` stop and restart the
widget around HITL approval prompts so the stdin read does not race
the refresh loop.

## Task boundary

Every outer `Runner.arun()` / `Runner.arun_swarm()` call emits a
📋 Task panel pair:

1. **`📋 Task Started`** — yellow border, contains the user prompt
   (truncated to 80 chars) and an 8-char task ID.
2. **`📋 Task Completed`** (green) or **`❌ Task Failed`** (red, with
   error string) — fires once the run returns or raises.

The line backend emits a `task started: …` / `task completed: …` line
for CI log compatibility.

## Event reference

### Currently emitted

| Constant | Event name | Fires at |
|---|---|---|
| `EVENT_AGENT_START` | `agent.start` | Each agent turn begins |
| `EVENT_AGENT_END` | `agent.end` | Each agent turn ends |
| `EVENT_LLM_START` | `llm.start` | Before every LLM call |
| `EVENT_LLM_END` | `llm.end` | After every LLM call |
| `EVENT_TOOL_START` | `tool.start` | Before every tool invocation |
| `EVENT_TOOL_END` | `tool.end` | After every tool invocation |
| `EVENT_TOOL_ERROR` | `tool.error` | Tool raised; panel closes red |
| `EVENT_HANDOFF` | `handoff` | On each agent-to-agent handoff |
| `EVENT_GUARDRAIL_INPUT_START/END` | `guardrail.input.*` | Agent-level input guardrail |
| `EVENT_GUARDRAIL_OUTPUT_START/END` | `guardrail.output.*` | Agent-level output guardrail |
| `EVENT_SKILL_ACTIVATED` | `skill.activated` | When a skill activates |
| `EVENT_SESSION_LOAD/SAVE` | `session.*` | Session history load/save |
| `EVENT_TURN_START/END` | `turn.*` | Agent-loop turn boundaries |
| `EVENT_USAGE_RECORDED` | `usage.recorded` | Cumulative tokens after each LLM call |
| `EVENT_CACHE_HIT/MISS` | `cache.*` | Prompt/tool cache |
| `EVENT_RETRY` | `retry` | Tool `ToolRetry` caught |
| `EVENT_CONTEXT_COMPACTED` | `context.compacted` | Context manager summarized history |
| `EVENT_STREAM_START/END` | `stream.*` | Streaming window |
| `EVENT_HITL_APPROVAL_REQUESTED` | `hitl.approval.requested` | Tool deferred for approval |
| `EVENT_HITL_APPROVAL_GRANTED` | `hitl.approval.granted` | `RunState.approve()` resume |
| `EVENT_HITL_APPROVAL_REJECTED` | `hitl.approval.rejected` | `RunState.reject()` resume |
| `EVENT_BUDGET_EXCEEDED` | `budget.exceeded` | `UsageLimitExceeded` raised |

### Tool-level guardrails

Four additional `RunHooks` methods that fire around each tool's
input/output guardrail chain:

* `on_tool_input_guardrail_start(ctx, tool, guardrail)`
* `on_tool_input_guardrail_end(ctx, tool, guardrail, result)`
* `on_tool_output_guardrail_start(ctx, tool, guardrail)`
* `on_tool_output_guardrail_end(ctx, tool, guardrail, result)`

`VerboseHooks` overrides each to emit a scoped guardrail panel keyed
by `(tool_name, guardrail_name, kind)`, so agent-level and tool-level
guardrails render as distinct panels without colliding. Verdicts are
derived from the `ToolGuardrailFunctionOutput.behavior["type"]`:

| `behavior["type"]` | Verdict | Border |
|---|---|---|
| `allow` | `pass` | green |
| `reject_content` | `trip` | red |
| `raise_exception` | `trip` | red |

### Task Boundary Events

Task boundary panels add these events around each top-level run:

| Constant | Event name | Fires at |
|---|---|---|
| `EVENT_TASK_START` | `task.start` | `Runner.arun()` / `arun_swarm()` entry |
| `EVENT_TASK_END` | `task.end` | `Runner.arun()` clean exit |
| `EVENT_TASK_FAILED` | `task.failed` | `Runner.arun()` exception path |

Per-chunk LLM streaming drives the Live widget via the
`emit_stream_chunk` free function plus a `ContextVar` bridge in
`troopai.adk.verbose.run_bridge`.

### Reserved Style Entries

Events listed in `VerboseConfig.styles` but without a call site in the
runner. These correspond to feature-specific event surfaces such as
Flow API, Memory read/write, Knowledge retrieval, MCP lifecycle, and
reasoning tiers.

`EVENT_RUN_START`, `EVENT_RUN_END`, `EVENT_REASONING_START`,
`EVENT_REASONING_END`, `EVENT_PLAN_REFINED`, `EVENT_REPLAN`,
`EVENT_GOAL_ACHIEVED`, `EVENT_MEMORY_READ`, `EVENT_MEMORY_WRITE`,
`EVENT_MEMORY_ERROR`, `EVENT_KNOWLEDGE_QUERY`,
`EVENT_KNOWLEDGE_RESULT`, `EVENT_MCP_CONNECT`,
`EVENT_MCP_CONNECTED`, `EVENT_MCP_ERROR`, `EVENT_FLOW_START`,
`EVENT_FLOW_END`, `EVENT_FLOW_PAUSED`, `EVENT_BUDGET_WARNING`,
`EVENT_CONTEXT_EDITED`, `EVENT_STATE_SAVE`, `EVENT_WARNING`.

## Extending to new Agent attributes

The event registry is a plain `dict[str, EventStyle]`. New attributes
that want visibility (hypothetical `memory_access`, `plan_revised`,
etc.) can register their own events at runtime without changing the
renderer:

```python
cfg = VerboseConfig()
cfg.register_event(
    "memory.read",
    EventStyle(color="blue", icon="⇲", prefix="memory"),
)
```

Any emit path that calls `renderer.render_line("memory.read", headline, payload)`
will pick up the style. Unknown events render as plain text without
colour — forward compatibility is automatic.

## `NO_COLOR` support

Honours the `NO_COLOR` standard (https://no-color.org/). Any non-empty
value disables colour universally, independent of
`VerboseConfig.use_color`.

```bash
NO_COLOR=1 python my_script.py
```

## Examples

See `examples/verbose/`:

| File | Demonstrates |
|---|---|
| `basic.py` | Default config, single agent |
| `basic_panel.py` | `mode="panel"` with Rich panels |
| `ci_safe.py` | `mode="auto"` reporting the resolved mode + inputs |
| `custom_styles.py` | Recolouring and muting events |
| `per_agent.py` | Per-agent overrides in a multi-agent flow |
| `hitl.py` | HITL approval gate visualization |
| `nested_hitl.py` | Approvals bubbling through `as_tool()` |
| `multi_level_guardrails.py` | Tool-level + agent-level guardrail panels |
| `streaming.py` | `stream.start` / `stream.end` markers |
