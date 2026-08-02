(concepts/index)=

# Concepts

> Every concept in this ADK, what it does, and — critically — what it
> is **not**. Nearby concepts that get confused live side-by-side here
> so the distinctions are explicit.

This page is a glossary plus a series of compare-and-contrast tables.
Architecture pages go deeper on individual mechanisms; this page is the
orientation map.

## `Agent` vs `Runner`

| Aspect           | `Agent`                            | `Runner`                                                   |
| ---------------- | ---------------------------------- | ---------------------------------------------------------- |
| What it is       | **Configuration** (data).          | **Execution** (behaviour).                                 |
| Has methods like | `as_tool()`, `clone()`             | `arun()`, `arun_graph()`, `arun_swarm()`, streamed variants |
| Holds            | Name, instructions, tools, handoffs, guardrails. | Loop state, budgets, telemetry, checkpointer wiring. |
| Identity         | Stateless. Two Agents with the same config are equal. | Stateful while a run is in flight.        |

**Rule:** there is **no** `agent.arun()`. Every execution path goes
through the `Runner`. An `Agent` instance is reused across many runs.

## Guardrails vs Middleware vs Hooks vs Sandbox

All four intercept agent behaviour, but at different stages and for
different purposes. Confusing them is the most common architectural
mistake.

| Concept    | Where it sits                          | What it sees                  | What it does                            | Example                       |
| ---------- | -------------------------------------- | ----------------------------- | --------------------------------------- | ----------------------------- |
| Guardrail  | Before stage 3 (input) / after stage 3 (output) | The prompt or the final reply | Validates; can short-circuit with a refusal. **Pure** (no LLM call). | PII scrubber, jailbreak heuristic, output relevance check. |
| Middleware | Wraps each LLM call inside the loop    | The wire request and response | Transforms wire payloads; provider-local. | Prompt-cache header injection, request signing, custom retry. |
| Hook       | Lifecycle callbacks throughout the run | Lifecycle events              | Observes; can mutate run state via the callback signature. | `on_step`, `on_handoff`, `on_tool_call`, audit emission. |
| Sandbox    | Wraps each tool execution              | Tool arguments and effects    | Isolates the *blast radius* of the call (filesystem, network). | Docker, K8s, hosted bridges (E2B/Modal/...). |

**Rule of thumb:** if you want to **reject** an input — guardrail. If
you want to **modify** the wire payload — middleware. If you want to
**observe** what happened — hook. If you want to **contain** a tool's
side-effects — sandbox.

## Tool kinds — Function vs Hosted vs MCP

| Kind             | Source                              | Defined in your code? | Provider-specific?           |
| ---------------- | ----------------------------------- | --------------------- | ---------------------------- |
| `FunctionTool`   | Wraps a Python callable.            | Yes.                  | No.                          |
| Hosted tool      | Provider runs it (web search, code exec, file search). | No — opt-in by config. | Yes (provider-native). |
| MCP tool         | An external [MCP](https://modelcontextprotocol.io/) server advertises it. | No — discovered at runtime. | No (MCP is the protocol). |

**Rule:** function tools are the default. Hosted tools and MCP tools
extend the surface; they don't replace it.

## Multi-agent patterns — Handoffs vs Swarms vs Graphs

The three composition axes. All three coordinate multiple agents, but
along different shapes.

| Pattern  | Shape                          | Termination                         | When                                            |
| -------- | ------------------------------ | ----------------------------------- | ----------------------------------------------- |
| Handoff  | Directed routing (A → B → ...) | A leaf agent emits a final reply.   | One agent decides "who handles this next".      |
| Swarm    | Cycle (members iterate)        | `swarm_done` tool / `max_turns` / predicate. | Several agents refine an answer together. |
| Graph    | State machine (nodes + edges)  | Reach a terminal node.              | Long workflow with branching + HITL + checkpoints. |

See [Handoffs & Swarms](../architecture/handoffs-and-swarms.md) and
[Graphs](../architecture/graphs.md).

## Memory layers — Episodic vs Semantic vs Sessions vs Context

Four words that all sound like "memory". They are not the same.

| Layer       | Lifetime                           | Shape                              | Used for                                       |
| ----------- | ---------------------------------- | ---------------------------------- | ---------------------------------------------- |
| Context     | One run.                           | Free-form developer object (`RunContext.context`). | Pass per-run state to tools and hooks. |
| Sessions    | Across runs for the same identity. | `Session` (SQLite-backed by default). | "Remember the user's previous turns." |
| Episodic memory | Across runs, agent-controlled. | `MemoryItem` records.            | Facts learned from past conversations.         |
| Semantic memory | Across runs, vector-indexed.   | `VectorStore` records + embeddings. | "What did we discuss about X?" retrieval. |

**Rule:** `context` is per-run developer state; everything else is
across-run persistence. Episodic memory is what the agent has *experienced*;
semantic memory is what the agent can *retrieve* by similarity.

## Cost mechanisms — Estimator vs Ledger vs Router vs Budget

| Mechanism      | When it runs                | What it does                                       |
| -------------- | --------------------------- | -------------------------------------------------- |
| `CostEstimator`| Before an LLM call.         | Predicts token cost.                               |
| `CostLedger`   | After each LLM call.        | Appends a `CostEntry`. Implementations in `budgets/`. |
| `LLMRouter`    | Before an LLM call.         | Picks the model — `CheapestFirstRouter`, `LatencyFirstRouter`, or your own. |
| `BudgetConfig` | Throughout the run.         | Hard ceiling; the Runner short-circuits when exhausted. |

**Rule:** estimator predicts, ledger records, router selects, budget
enforces. Four separable concerns; they compose freely.

## Observability — Tracing vs Evals vs Logging vs Verbose

| Layer     | Purpose                                     | Output                                  |
| --------- | ------------------------------------------- | --------------------------------------- |
| Logging   | Generic Python `logging` per module.        | Logs to stderr or wherever you configure. |
| Verbose   | Human-readable per-step rendering (`[verbose]` extra). | Rich panels in the terminal. |
| Tracing   | OpenInference / OpenTelemetry spans.        | OTel collector → Arize / Phoenix / Langfuse / OTLP. |
| Evals     | Empirical correctness measurement.          | Graders + reports in `evals/`.           |

**Rule:** logging is for developers debugging; verbose is for humans
watching; tracing is for ops; evals are for correctness. They are
**additive**, not alternatives.

## Persistence — Checkpointers vs Sessions vs Memory

These three persist different things and live in different modules.

| Persistence    | What it stores                                  | Module                              |
| -------------- | ----------------------------------------------- | ----------------------------------- |
| Checkpointer   | Graph / swarm in-flight state (resumable runs). | `graphs/checkpointers/`, `swarms/checkpointers/`. |
| Session        | Conversation history for an identity.           | `session/`.                         |
| Memory         | Long-lived facts (episodic + semantic).         | `memory/`.                          |

**Rule:** if a run was interrupted and needs to resume, a checkpointer
restores it. If a *user* returns later and you want continuity, a
session loads their history. If the *agent* wants to recall facts
across users / runs, memory retrieves them.

## Skills vs Tools — Packages vs Primitives

A **skill** is a *bundle* — instructions + tools + governance —
composed onto an Agent. A **tool** is a *primitive*. A skill might add
five tools at once, plus instruction snippets, plus an allow-list of
which other tools the LLM can call.

| Aspect       | Skill                          | Tool                |
| ------------ | ------------------------------ | ------------------- |
| Granularity  | Bundle of related capabilities. | Single callable.   |
| Instructions | Yes — appended to the agent.    | No.                |
| Governance   | Can restrict tool use.          | Itself governed.   |
| Composition  | Multiple skills per agent.      | Multiple tools per agent. |

**Rule:** reach for a skill when you need to add a *capability stack*
(e.g. "research"); reach for a tool when you need to expose a single
function.

## A2A vs MCP — Process-level vs Tool-level

Both extend the ADK across process boundaries, but at different layers.

| Protocol | What's exchanged                         | Boundary               | Direction                            |
| -------- | ---------------------------------------- | ---------------------- | ------------------------------------ |
| MCP      | **Tool calls** advertised by a server.   | ADK → external server. | ADK consumes tools.                  |
| A2A      | **Agent-to-agent invocations**.          | ADK ↔ ADK (or compatible). | ADK delegates to another ADK process. |

**Rule:** MCP is for *tool surfaces*. A2A is for *agent surfaces*.

## Durable execution — Temporal vs in-process

| Mode        | Where state lives                       | Recovery                            | When                              |
| ----------- | --------------------------------------- | ----------------------------------- | --------------------------------- |
| In-process  | Python objects in memory + checkpointer. | Resume from last checkpoint if the checkpointer was wired. | Most workflows.                  |
| Temporal    | Temporal server.                        | Workflow history replay.            | Long-running / multi-day workflows / cross-process resume. |

**Rule:** in-process + checkpointer covers most cases. Temporal is for
workflows that must survive deploys and need replay semantics.

## Type layers — Layer 1 / Layer 2 / Layer 3 (recap)

See [Type layers](../architecture/type-layers.md) for the full treatment.

| Layer | Direction | Owner              | Where it appears                          |
| ----- | --------- | ------------------ | ----------------------------------------- |
| 1     | In        | Framework          | `LLMInputContentItem` and friends.         |
| 2     | Wire      | Provider (local)   | `ChatCompletion*` TypedDicts.              |
| 3     | Out       | Framework          | `RunItem` (history).                       |

**Rule:** developers see Layer 1 and Layer 3. Layer 2 stays inside the
provider module.

## `LLM` ABC vs provider config classes (recap)

See [LLM ABC](../architecture/llm-abc.md).

`LLM` is the abstract base class. Each provider subclasses it
(`LiteLLMModel`, `AnthropicModel`, `OpenAIResponsesModel`,
`OpenAIChatCompletionsModel`, `GeminiModel`) and pairs with a config:
`LiteLLMConfig`, `AnthropicConfig`, `OpenAIResponsesConfig`,
`OpenAIChatCompletionsConfig`, `GeminiConfig`. All configs subclass
`LLMConfig` — provider-agnostic fields stay there.

## `RunItem` variants (recap)

See [Type layers](../architecture/type-layers.md) Layer 3 table for
the full list. The variants you'll most often pattern-match against:

- `UserItem` / `SystemItem` — input messages.
- `MessageOutputItem` — assistant reply.
- `ToolCallItem` / `ToolCallOutputItem` — tool call + result pair.
- `HandoffCallItem` / `HandoffOutputItem` — handoff transition pair.
- `ReasoningItem` — provider-emitted reasoning.
- `CompactionItem` — a summary that replaced earlier turns.
- `MCPListToolsItem` / `MCPApprovalRequestItem` / `MCPApprovalResponseItem` — MCP exchange artefacts.

## Halting / Rice / NFL — the math limits (recap)

See [Foundations](../foundations/three-mathematical-limits.md).

Three theorems shape every design choice:

- **Halting → bound everything** (`max_turns`, budgets, retries).
- **Rice → measure, don't prove** (evals, not formal verification).
- **NFL → specialise, then compose** (handoffs, swarms, graphs,
  skills).

---

## Index of every named concept

For quick lookup. Each link goes to the architecture page or guide
that covers it in depth.

- [Agent](../architecture/runner.md) (config) / [Runner](../architecture/runner.md) (execution)
- [Guardrails](../guides/guardrails.md), Middleware (`llms/llm_middleware.py`), [Hooks](../architecture/runner.md), [Sandbox](../guides/sandbox.md)
- [Function tools](../guides/tools.md), Hosted tools, [MCP tools](../guides/mcp.md)
- [Handoffs](../architecture/handoffs-and-swarms.md), [Swarms](../architecture/handoffs-and-swarms.md), [Graphs](../architecture/graphs.md)
- [Context](../architecture/runner.md), Sessions, [Memory](../guides/memory.md) (episodic + semantic)
- [Embedders](../guides/memory.md), Vector stores
- [Skills](../guides/skills.md)
- Evals, Graders (LLM-as-judge, agent-as-judge, code)
- [Tracing](../guides/tracing.md), OpenInference, OpenTelemetry exporters
- [Cost ledger](../guides/cost.md), Cost estimator, [LLM router](../guides/cost.md), Budgets
- [Checkpointers](../architecture/graphs.md) (in-memory / sqlite / postgres / redis / s3 / tiered)
- [Temporal durable execution](../architecture/governance.md), [Per-tenant task queues](../architecture/governance.md)
- [Audit substrate](../architecture/governance.md), Audit sinks
- [LLM ABC](../architecture/llm-abc.md), Provider configs, [Type layers](../architecture/type-layers.md)
- [RunItem](../architecture/type-layers.md) variants
- [`LLMResponse` parts](../architecture/llm-abc.md) (TextPart / ThinkingPart / ToolCallPart)
- [Layer 1 input variants](../architecture/type-layers.md)
- [A2A](../guides/a2a.md), [MCP](../guides/mcp.md) (client + server)
