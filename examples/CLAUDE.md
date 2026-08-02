# Examples

Example implementations demonstrating TroopAI Agents patterns. Each example
is runnable and focused on a single ADK feature or capability (it may span a
few files when that one feature needs it).

Heavy, multi-complex-agent application templates live in the separate
`troopai-cookbook-python` repository — not here. The discriminator is purpose:
single-feature demo → `examples/`; full interactive app → the cookbook repo.

Top-level directories: `agent_patterns/`, `skills/`, `handoffs/`,
`llm_providers/{anthropic,litellm,openai}/`, `memory/`, `tools/`,
`guardrails/`, `tracing/`. Pattern tables below show one
row per file with the demonstrated pattern.

## Agent Patterns

| Example | Pattern |
|---------|---------|
| `agents_as_tools.py` | Sub-agents as regular tools via `as_tool()`, LLM orchestrates delegation |
| `agents_as_tools_streaming.py` | `on_stream` callback for real-time sub-agent event monitoring |
| `agents_as_tools_conditional.py` | `enabled` callbacks for context-dependent agent tool visibility |
| `deterministic.py` | Sequential pipeline with structured output gating between agents |
| `parallelization.py` | `asyncio.gather()` for concurrent agent execution with judge selection |
| `governed_delegation.py` | `as_tool()` governance: `timeout`, `budget`, `max_result_tokens`, `get_agent_graph()` |
| `forcing_tool_use.py` | `LLMConfig.tool_choice="required"` + `ToolUseBehavior` modes + `reset_tool_choice` |
| `human_in_the_loop.py` | `requires_approval` on tools, `RunState.approve()/reject()` for resumption |
| `human_in_the_loop_custom_rejection.py` | Rejection messages that guide the LLM toward alternatives |
| `human_in_the_loop_stream.py` | Streaming combined with approval checkpoints |
| `nested_human_in_the_loop.py` | Approvals from sub-agents propagate transparently via `as_tool()` |

## Flows

| Example | Pattern |
|---------|---------|
| `basic_flow.py` | All core primitives in one script: parallel `@start`, `@listen(method_ref)`, `@listen(a & b)`, `@router`, `@listen("label")`, streaming, checkpoint round-trip |
| `router_flow.py` | `@router` returning a route label + branched `@listen("label")` dispatch (no LLM) |
| `and_join_flow.py` | AND gate via `method_a & method_b` fluent operator — fires once after BOTH arrive |
| `or_join_flow.py` | OR gate via `method_a | method_b` — fires ONCE on first arrival; gate is consumed |
| `checkpoint_resume.py` | `FlowCheckpoint.to_json()` + `Runner.arun_flow_from_checkpoint(...)` round-trip |
| `async_flow.py` | Async execution via `await Runner.arun_flow(...)` |
| `sync_flow.py` | Sync wrapper `Runner.run_flow(...)` — blocking, event-loop-aware |
| `streamed_flow.py` | Event streaming via `Runner.arun_flow_streamed(...)` with `isinstance`-narrowed event handling |
| `research_with_agents.py` | Production overview — Agents + Tasks + TaskGroup + `error_policy="route_to_error_handler"` + `@listen("__error__")` recovery + streamed observability (requires API key) |
| `flow_diagram.py` | `Flow.to_mermaid()` / `Flow.to_dot()` topology export with optional render via `viz` + `mermaid` extras |
| `flow_diagram_with_agents.py` | Same topology export on an Agent-calling Flow — diagrams emit BEFORE running (requires API key) |

## Skills

| Example | Pattern |
|---------|---------|
| `skills_agent_with_skills.py` | Multi-skill agent (weather + math) with EAGER and LAZY activation, governance, guardrails |
| `skills_customer_support.py` | Customer support agent with order + billing skills, discovery toolset, cross-skill queries |
| `skills_directory.py` | Loading skills from SKILL.md directories (LangChain/CrewAI/ADK compatible), running with agent |
| `skills_discovery.py` | `SkillDiscoveryToolset` — LLM introspects skills, loads resources, runs diagnostics |

## Handoffs

| Example | Pattern |
|---------|---------|
| `llm_orchestrated.py` | LLM decides transfers: bare agent list, Handoff objects, dynamic enablement, typed input |
| `code_orchestrated.py` | Deterministic intent-based routing with `HandoffRoute` and `handoff_route()` — zero LLM routing tokens |
| `compare_handoffs.py` | Side-by-side token usage comparison of LLM vs code-orchestrated handoffs |
| `compare_handoffs_multiturn.py` | All-in-one vs separate-concerns: how specialist cost scales with tool calls |
| `message_filters.py` | `forward_intent`, `remove_tool_calls`, `keep_last_n`, custom filters, `compose` pipeline |
| `temporal_slicing.py` | `on_handoff` with `HandoffInputData`, temporal-aware filters, forwarded decoupling |
| `cost_optimized.py` | `HandoffConfig` strategies: `full`, `last_n`, `intent_only`, `summary` with budget caps |

## Tools

| Example | Pattern |
|---------|---------|
| `jit_context_aware.py` | `JITContextAwareTool` — active context management with notes, history search, budget monitoring |
| `tool_context_management.py` | `ToolContext` budget tracking, token-aware execution |
| `tool_guardrails.py` | Per-tool input/output guardrails with allow/reject/raise behavior |
| `tool_advanced_features.py` | Artifacts, output types, `return_direct`, `prepare` callback, args validation |
| `tool_dependencies.py` | `requires_env` + `requires_packages` declarations, fail-fast at agent construction |
| `tool_rate_limiting.py` | Sliding-window per-tool throttle with `wait` and `error` behaviours |
| `deferred_tool_loading.py` | `defer_loading=True` + `build_tool_search()` — hide rare tools until queried |
| `deferred_tools_hitl.py` | Static and conditional `requires_approval` for HITL deferral |
| `streaming_tools.py` | Streaming with tool guardrails and HITL stream events |

## Document Search (RAG)

| Example | Pattern |
|---------|---------|
| `rag/document_search.py` | `TXTSearchTool` over a local corpus (lazy-indexed via `LiteLLMEmbedder`); agent retrieves passages before answering. Swap the wrapper or pass `DocumentSearchTool` mixed `sources` to auto-dispatch loaders |

## Document Translation

Multi-file example (`translations/`): translate an untrusted document — in any
source language — into several languages with any LiteLLM provider, config-driven
(model, languages, glossary, context, concurrency), shaped like a production
translation job: security guardrails, a transient-failure retry policy, an
exact-match translation memory (resumable, repeat-free), concurrent languages
under a shared token budget, and a machine-readable job report with per-segment
status + USD cost. The trust boundary is the point: the document AND the
user-supplied glossary/context are untrusted, fenced together, never in the
instruction channel — and the job is uninterruptible past the intake gate (a
flagged chunk is withheld while everything else ships).
Run with `python examples/run_examples.py --allow-network --filter translations`
(needs `pymupdf` + `lingua-language-detector` + the configured provider's key); skipped by
default since it hits the network. Pass `--source <path-or-url>` to translate
any PDF; `--demo` to fire the guardrails on non-English inputs; `--eval` for
offline quality scoring.

| Example | Pattern |
|---------|---------|
| `translations/document_translation.py` | Entry point (the only runnable file): code-orchestrated, provider-agnostic translate pipeline; concurrent languages (semaphore) under a shared `TokenBudget` + per-run `LLMUsageLimits`; writes translated PDFs + `job_report.json`. Support modules live in `core/` `security/` `output/` subpackages |
| `translations/core/translator.py` | Trust boundary: system prompt = task only; document + glossary + context fenced together as untrusted data under a per-document nonce. `LLMConfig.retry_policy` backoff; segment-level fault isolation (withhold, never abort) |
| `translations/security/guardrails.py` | Security guardrails (no LLM judge): fail-closed intake scan on glossary/context (regex + semantic); three-tier input detection — multilingual regex, embedding-codebook `SemanticScanner` (catches regex-blind paraphrases, scans raw content only), nonce fence; output identifier-preservation (injected/dropped), empty, wrong-language, and `{{merge field}}` placeholder integrity |
| `translations/core/translation_memory.py` | Exact-match translation memory: crash-safe JSONL segment cache keyed by model/languages/context/glossary; re-runs resume free |
| `translations/output/reporting.py` | Machine-readable job manifest: per-segment status + withheld reasons, per-language tokens and USD cost via `LiteLLM.cost` |
| `translations/security/demonstrations.py` | Fire the defences on two channels: real poisoned PDFs generated at runtime and read back through the real ingestion path (document channel — injection chunk contained, legit chunks ship), and poisoned glossary/context (config channel — rejected at intake); plus templated-document preservation |

## LLM Providers

| Example | Pattern |
|---------|---------|
| `anthropic/anthropic_example.py` | Native `AnthropicLLM` — basic, streaming, tools, extended thinking, structured output (synthetic tool), prompt caching, retry policy |
| `litellm/prompt_caching.py` | `LLMConfig.prompt_caching` across Anthropic, Gemini, OpenAI |
| `litellm/reasoning.py` | `LLMConfig.reasoning` for extended thinking across providers |
| `openai/responses_example.py` | Native `OpenAIResponsesLLM` + hosted `web_search` via `extra_body` |
| `openai/chatcompletions_example.py` | Native `OpenAIChatCompletionsLLM` — streaming + structured output |

## Tracing

| Example | Pattern |
|---------|---------|
| `tracing/custom_span_example.py` | `custom_span(...)` for application spans + minimal in-memory recording tracer |
| `tracing/otel_console.py` | `setup_otel(console=True)` — prints every framework span to stdout |
| `tracing/otel_otlp.py` | Ship to an OTLP collector (Jaeger / Datadog / Honeycomb) |
| `tracing/multi_tracer.py` | `MultiTracer` fan-out to OTel + in-memory recorder simultaneously |

## Observability

| Example | Pattern |
|---------|---------|
| `observability/metrics_and_openinference.py` | `setup_otel(convention=OPENINFERENCE)` + `setup_metrics()` composed in a `MultiTracer`; `RunConfig(tracing_enabled=True, metrics_enabled=True)` |

## Durable Execution

| Example | Pattern |
|---------|---------|
| `temporal/basic_agent.py` | Agent as a Temporal workflow via `TemporalLLM`, `TroopAIWorkflow`, `TroopAITemporalPlugin` |
| `temporal/graph_workflow.py` | Graph HITL interrupt + Temporal signal resume (`workflow.wait_condition`) |
| `restate/basic_agent.py` | Agent as a Restate durable handler via `RestateLLM` (journaled, replay-safe) + HITL durable promise |

## Cost

| Example | Pattern |
|---------|---------|
| `pre_call_estimation.py` | `LLM.estimate_cost` → `CostEstimate` (pure, no API call) |
| `tenant_budget.py` | `TenantBudget` per-run cap; `TenantBudgetExceeded` pre-call kill |
| `smart_routing.py` | `LLMRouter` fallback demo (fixed-order subclass); `CheapestFirstRouter` is the production cheap-first variant; wired via `RunConfig.router` |
| `cost_aware_compaction.py` | `CompactionConfig.cost_aware` tightening under budget pressure |

## Config

| Example | Pattern |
|---------|---------|
| `config/run_config_agent.py` | `load_agent("agent.json")` — build an Agent from a strict JSON config; tool + output-schema referenced from `__main__` by dotted path |
| `config/run_topology.py` | `load_topology("topology.json")` — multi-agent file with an `agents` map + handoffs by name; two-pass wiring resolves the references |
| `config/run_swarm.py` | `load_topology("swarm.json")` — a `swarm` section (members/entry/policy/composed termination) run via `Runner.configure().swarm(...).arun(...)` |
| `config/run_graph.py` | `load_topology("graph.json")` — a `graph` section (nodes/edges/entry/terminals) run via `Runner.arun_graph(...)` |

## Console Output Convention

Examples render run lifecycle to the console through the verbose event
stream — every top-level Runner entry point passes
`RunConfig(verbose=VerboseConfig())`. Classic `logger.*` calls stay but
reach only the rotating `.log` file (the default logging config attaches
no console handler). NEVER add `logging.basicConfig(...)` to an example —
it is dead code under the package's logging config. Exceptions:
`run_examples.py` / `auto_mode.py` (suite tooling owns its own stdout
handler) and `temporal/` + `restate/` (durable runtimes — no verbose).

## Running Examples

All examples are runnable from the project root (`python examples/<pattern>/<file>.py`). Set up the environment from the root `environment.yaml` (conda). API keys must be configured per the provider docs.

Batch runner: `python examples/run_examples.py` discovers every example, classifies the keys/infrastructure each needs, skips those whose prerequisites are absent, and runs the rest with a per-example timeout. `--list` classifies without running (zero cost); `--auto-mode [--filter <topic>]` runs. `auto_mode.py` exposes `is_auto_mode()` / `input_with_fallback()` / `confirm_with_fallback()` so interactive examples run unattended under `TROOPAI_EXAMPLES_INTERACTIVE_MODE=auto`.
