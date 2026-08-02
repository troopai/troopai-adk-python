# Config Module

Declarative agent configuration: build an `Agent` from a JSON document
validated against a strict schema, instead of constructing it in Python.

## Scope

Operator-facing layer for the **data-heavy** surface of an agent — model,
prompt, tools, output schema, generation knobs. Behavior that requires Python
callables or objects (lifecycle hooks, middleware, dynamic prompts, tool
bodies, guardrail functions) is **not** expressible inline; it is reached
through a dotted-path reference (`"my_pkg.module:symbol"`) or rejected with a
guiding error. This is a configuration surface, not a Python replacement.

The `llm` surface accepts a bare model-name string (optionally paired with a
standalone agnostic `llm_config`) or a typed, `provider`-discriminated block
selecting a provider-native LLM with its own `config`. The two configuration
sources are mutually exclusive.

`tools` accepts dotted `FunctionTool` refs and `{type, args}` provider-hosted
tools; `guardrails` accepts dotted refs (input/output phase by list);
`system_prompt` also accepts a
`{dynamic: ref}` callable reference.

## Files

| File | Purpose |
|---|---|
| `loader.py` | `load_agent(path)` — JSON/YAML file → dict → validate → assemble; `read_config_document` (format-aware, YAML behind optional `pyyaml`) |
| `assembler.py` | `build_agent(AgentConfig)` — explicit named-param `Agent(...)` |
| `topology.py` | `load_topology`/`build_topology` → `AgentTopology` (agents + swarm + graph); two-pass handoff cycle wiring |
| `resolver.py` | `resolve_dotted_spec` + typed `resolve_function_tool` / `resolve_output_schema`; `importable_dir` (refs resolve relative to the config file's dir, appended to `sys.path` so a sibling can't shadow an installed module) |
| `providers.py` | `PROVIDER_REGISTRY` + `register_llm_provider` + per-provider factories (`build_llm` reuses); the only config-layer module importing concrete LLMs / runtime configs (lazy, optional-dep safe) |
| `hosted_tools.py` | `HOSTED_TOOL_REGISTRY` + `register_hosted_tool` + `build_hosted_tool`; builds provider-hosted tools from `{type, args}` |
| `guardrails.py` | `build_guardrails`; resolves dotted guardrail refs (input/output phase by list) into runtime `AgentGuardrails` |
| `schema.py` | `dump_agent_config_schema` / `dump_agent_node_config_schema` / `dump_topology_config_schema` + committed JSON Schemas (regen via `python -m troopai.adk.config.schema`) |

Schema models live in `types/config/` (the source of truth); the generated
`agent_config.schema.json`, `agent_node_config.schema.json` (a sub-agent file —
`AgentConfig` + `handoffs`), and `topology_config.schema.json` sit beside them.

## Multi-agent topologies

A topology file declares an `agents` map (each an `AgentConfig` plus optional
`handoffs` by name), and optional `swarm`/`graph` sections. A map entry may
instead be `{config_path}` — a pointer to a standalone agent file, resolved
relative to the topology file and parsed as an `AgentNodeConfig` (so a
file-sourced member may itself declare handoffs); depth is fixed at one (no
nested topology, no include cycle). Loading is two-pass — build every agent as
a stub, then wire handoffs by name — so A↔B handoff cycles resolve without
proxies (`Agent` is mutable; targets aren't validated at construction). Swarm
policy/termination and graph merges/joins resolve via name→object dispatch
tables; edge conditions via a dotted `ref`.

## Key decisions

| Decision | Rationale |
|---|---|
| Pydantic models are the schema; JSON is just the deserializer | One validation path; a YAML adapter could feed the same `model_validate` later. |
| `extra="forbid"` everywhere | A typo fails loudly instead of being silently ignored. |
| Optional `$schema` pointer tolerated via an aliased, ignored field | Editor/CI validation against the published schema; runtime validation stays Pydantic. |
| Reuse real types in the schema (`SystemPrompt`, `StopAtTools`) | Zero drift; the generated schema documents the actual runtime types. |
| One dotted-path resolver, confined to load time | The single sanctioned dynamic-import boundary; never on a hot path. |
| Refs + `config_path` resolve relative to the config file (dir on `sys.path` for the load) | A sibling module / sub-agent file is found regardless of launch CWD; explicit, bounded to load time, no injected tokens. |
| Builtin name → factory registry (`providers.py`, `hosted_tools.py`) | Short names for first-party types; dotted refs for user symbols. Provider factories are the sole import boundary for concrete LLMs, keeping the schema + assembler agnostic. Guardrails carry no builtins — dotted refs only. |
| Code-only keys raise a guiding error | `hooks`/`middleware`/`skills` point the user back to Python (`handoffs` is declarative-by-name on a topology node). |
| Absent optional fields stay at `Agent` defaults | No implicit injection of a value the operator did not choose. |

## Public surface

`load_agent`, `build_agent`, `load_topology`, `build_topology`,
`AgentTopology`, `resolve_dotted_spec`.

See `examples/config/` for a runnable demo.
