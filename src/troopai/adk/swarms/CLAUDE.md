# Swarms Module

Iterative multi-agent collaboration. Unlike `Handoff` (one-shot transfer)
and `Agent.as_tool()` (delegate-and-resume), a `Swarm` supports **cycles**:
A → B → A until an explicit stop signal.

## Files

- `swarm.py` — `Swarm` frozen dataclass (roster + policy + termination + config + hooks + metadata); `Swarm.new()` builder entry point; `get_member(name)` lookup; `DEFAULT_TERMINATION` / `DEFAULT_MAX_TURNS`
- `builder.py` — `SwarmBuilder` fluent API (mirrors `graphs/builder.py`); `.compile()` constructs the frozen `Swarm` so all `__post_init__` validation fires at build time
- `config.py` — `SwarmConfig` budgets (`max_handoffs`, `max_total_tokens`) and `SharedContextConfig`. (`per_turn_timeout` / `retry_on_throttle` were deliberately NOT shipped — see `config.py` docstring)
- `policy.py` — `SwarmPolicy` ABC + four implementations (`LLMHandoffPolicy`, `RoundRobinPolicy`, `StructuredRoutingPolicy`, `CustomPolicy`)
- `termination.py` — `TerminationCondition` ABC with `__and__` / `__or__` + `validate_roster` hook; five built-ins (`MaxTurnsTermination`, `TokenBudgetTermination`, `HandoffToTermination`, `ExplicitDoneTermination`, `TextMentionTermination`)
- `shared_context.py` — `prepare_turn_input(state, next_agent, yield_signal, config)` → `list[LLMInputContentItem]`
- `shared_context_strategy.py` — `SharedContextStrategy` enum (`SCOPED` default, `LAST_N`, `FULL_BROADCAST`, `SUMMARIZED`)
- `state.py` — `SwarmState` (JSON-serializable via plain `to_dict`/`from_dict`; no version field)
- `stop_reason.py` — `StopReason(kind, detail)` tagging why the swarm stopped
- `yield_signal.py` — `SwarmHandoff` / `SwarmDone` signals + `SWARM_DONE_TOOL_NAME`
- `events.py` — `SwarmEvent` union (`SwarmStartEvent`, `SwarmTurnStartEvent`, `SwarmHandoffEvent`, `SwarmTurnEndEvent`, `SwarmTurnInterruptEvent`, `SwarmDoneEvent`)
- `hooks.py` — `SwarmHooks` lifecycle callbacks (complement `RunHooks` + `AgentHooks`) + `HookRegistry` fan-out
- `interrupt.py` — `SwarmResume` + `request_human_input_in_swarm()` (HITL)
- `checkpointer.py` — `SwarmCheckpointer` / `SwarmHookRegistry` protocols + `SwarmCheckpoint`; backends live in `checkpointers/`
- `result.py` — `SwarmRunResult` (one-line `repr` run summary) / `SwarmRunResultStreaming`
- `swarm_prompt.py` — opt-in `prompt_with_swarm_instructions()` + `RECOMMENDED_SWARM_PROMPT_PREFIX`

## Public API Style (mirrors graphs/handoffs)

- **Builder-first definition**: `Swarm.new("name", description=...).members(...).entry("name").llm_handoff().terminate_on(...).compile()` — the readability-first surface.
- **Defaults for the common case**: `Swarm(members=(...), entry="name")` works — `policy` defaults to `LLMHandoffPolicy()`, `termination` to `DEFAULT_TERMINATION` (`ExplicitDoneTermination() | MaxTurnsTermination(25)`).
- **Entry by name**: `entry` accepts a member name string, resolved in `Swarm.__init__` (unknown name → `ValueError` listing valid names).
- **`handoff_descriptions`**: per-member routing hints consumed by `LLMHandoffPolicy` as `transfer_to_<name>` tool descriptions (mirrors OpenAI Agents SDK `handoff_description`); keys validated against the roster.
- **Fail at compile time**: roster-aware conditions (`TextMentionTermination(member=...)`) are validated by `TerminationCondition.validate_roster` from `Swarm.__post_init__`.

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Swarm = config, Runner = execution** | No `Swarm.run()`. Driver lives at `run/swarm_loop.py` |
| **Explicit termination only** | `swarm_done` tool MUST be called — never terminate by absence. `DEFAULT_TERMINATION` ships this pre-wired (+ 25-turn net) |
| **No auto-injection** | System prompts not mutated; opt-in via `prompt_with_swarm_instructions()` |
| **Tool injection at dispatch, not construction** | Agents stay reusable outside swarms; `transfer_to_<name>` + `swarm_done` are added per turn |
| **No provider-specific types** | Shared context is Layer 1 (`LLMInputContentItem`) or Layer 3 (`RunItem`) only — never Layer 2 |
| **Composable termination** | `MaxTurnsTermination(20) | ExplicitDoneTermination()` via operator overloads |
| **Tolerant serialized state** | `SwarmState.to_json()` = `json.dumps(to_dict())`; no version field (mirrors `RunState`) for HITL-style persistence |
| **`SCOPED` default** | Each agent sees its own scratch + the explicit handoff message — no hidden cross-agent broadcast |

## Integration Seams (read-only pointers)

- `run/next_step.py` — `NextStepSwarmYield` variant on the `NextStep` union
- `run/loop.py` — `match` arm surfaces the yield back to the driver
- `run/turn_resolution.py` — detects `swarm_done` / `transfer_to_<name>` tool calls and builds the `NextStepSwarmYield`
- `run/runner.py` — `Runner.arun_swarm`, `Runner.arun_swarm_streamed`, `Runner.configure`
- `run/profile.py` — `RunnerProfile` and `SwarmRunner`
- `run/stream.py` — `SwarmEvent` variants are accepted on `StreamEvent`
- `run/swarm_loop.py` — the driver itself (`run_swarm_loop`, `run_swarm_loop_streamed`)

## Policy Table

| Policy | Who chooses next agent | LLM routing tokens | Typical use |
|--------|------------------------|---------------------|-------------|
| `LLMHandoffPolicy` | LLM via injected `transfer_to_<name>` tool | Yes | Open-ended collaboration |
| `RoundRobinPolicy` | Deterministic rotation | No | Debates, fixed pipelines, tests |
| `StructuredRoutingPolicy` | Agent's structured `Intent` output + `HandoffRoute` DSL | No | Typed triage / classification |
| `CustomPolicy` | User-supplied selector callable | No | Escape hatch |

See `docs/swarms/policies.md` for the decision tree.

## Cost Levers (five, composable)

1. `FunctionTool.max_result_tokens` — per-tool result cap (existing)
2. `HandoffConfig.budget` — per-handoff history transfer cap (existing)
3. `SwarmConfig.max_handoffs` — swarm-wide switch cap
4. `SwarmConfig.max_total_tokens` — swarm-wide cumulative tokens
5. `SharedContextStrategy` — per-turn context size (SCOPED / LAST_N / SUMMARIZED / FULL_BROADCAST)

Absolute safety net: `RunConfig.max_total_turns` — default `500` per the cost-conservative-defaults rule. Production deployments can override (raise or lower). Set to `None` explicitly to disable the safety net. See `docs/swarms/cost_optimization.md`.

See `docs/swarms/cost_optimization.md` for how they interact.

## When to Use What

| Pattern | Primitive |
|---------|-----------|
| One-shot transfer (agent A → agent B, no return) | `Handoff` |
| Delegate-and-resume (A asks B a question, continues with answer) | `Agent.as_tool()` |
| Iterative collaboration with cycles (A ↔ B ↔ C until done) | `Swarm` |
| Parallel fan-out with join | `asyncio.gather` over multiple `Runner.arun(...)` |

See `docs/swarms/swarms.md` for the decision tree in detail. See
`examples/swarms/` for runnable code.
