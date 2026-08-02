(config/config)=

# Declarative Agent Configuration

The `troopai.adk.config` subsystem lets you define an `Agent` — or an entire
multi-agent topology — as a JSON (or YAML) document and load it with a single
function call. The document is validated against strict Pydantic models before
the `Agent` is assembled; a typo fails loudly instead of being silently
ignored.

This guide covers the full operator-facing surface: the loaders, every field
in the single-agent schema, the topology schema for multi-agent workflows
(handoffs, swarms, graphs), the `$schema` pointer for editor tooling, the
dotted-`ref` escape hatch, the extension registries, and the security trust
boundary.

---

## Quickstart

### Single agent

Put the tool body in an importable module, `weather.py`, next to the config:

```python
from troopai.adk.tools import function_tool


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather report for a city."""
    return f"It is 21°C and sunny in {city}."
```

Reference it from `agent.json` by its dotted path:

```json
{
  "$schema": "../../src/troopai/adk/types/config/agent_config.schema.json",
  "name": "weather_assistant",
  "description": "Answers weather questions using a tool.",
  "system_prompt": "You are a concise weather assistant. Call get_weather for every city.",
  "llm": "claude-haiku-4-5-20251001",
  "tools": ["weather.get_weather"]
}
```

Load it in Python:

```python
import asyncio
import logging
from pathlib import Path

from troopai.adk.config import load_agent
from troopai.adk.run import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    agent = load_agent(Path("agent.json"))
    result = await Runner.arun(agent, "What is the weather in Paris?")
    logger.info(result.final_output)


asyncio.run(main())
```

`load_agent` validates the document, imports the `weather` module, resolves
the `get_weather` symbol, and returns a ready-to-run `Agent`. References
resolve **relative to the config file's directory**, so `weather.py` sitting
next to `agent.json` is found no matter where you launch from.

:::{note}
If a tool lives in the script you run *directly*, its module is `__main__`, so
the reference would read `__main__:get_weather`. That works for a throwaway
script, but it is not portable — load the same `agent.json` from anywhere else
and `__main__` is a different module. Put tools in their own module for any
config you intend to reuse.
:::

### Multi-agent topology

Create `topology.json`:

```json
{
  "$schema": "../../src/troopai/adk/types/config/topology_config.schema.json",
  "agents": {
    "triage": {
      "name": "triage",
      "system_prompt": "Route Spanish requests to the spanish agent; answer English ones yourself.",
      "llm": "claude-haiku-4-5-20251001",
      "handoffs": [
        {"target": "spanish", "description": "Handle requests written in Spanish."}
      ]
    },
    "spanish": {
      "name": "spanish",
      "system_prompt": "Reply only in Spanish, in one short sentence.",
      "llm": "claude-haiku-4-5-20251001"
    }
  },
  "entry": "triage"
}
```

Load it:

```python
from troopai.adk.config import load_topology
from troopai.adk.run import Runner

topology = load_topology(Path("topology.json"))
entry = topology.agents[topology.entry]
result = await Runner.arun(entry, "Hola, ¿me puedes saludar?")
```

### YAML

`.yaml` and `.yml` files are accepted everywhere `.json` is — the loader
dispatches on the file extension. YAML requires the optional `pyyaml`
dependency (`pip install troopai-adk-python[yaml]`). Both formats pass through
the same Pydantic validation and assembler; no behavior differs.

```yaml
name: weather_assistant
system_prompt: You are a concise weather assistant.
llm: claude-haiku-4-5-20251001
tools:
  - weather.get_weather
```

---

## Public API

| Symbol | Purpose |
|---|---|
| `load_agent(path)` | File path → `Agent` |
| `load_topology(path)` | File path → `AgentTopology` |
| `build_agent(config)` | `AgentConfig` dict or model → `Agent` |
| `build_topology(config)` | `TopologyConfig` dict or model → `AgentTopology` |
| `dump_agent(agent)` | `Agent` → config dict (lossy; see [Round-trip dump](#round-trip-dump)) |
| `resolve_dotted_spec(spec)` | Import and return a dotted-path symbol |
| `register_llm_provider(name, factory)` | Extend the provider registry |
| `register_hosted_tool(name, factory)` | Extend the hosted-tool registry |

---

## Single-agent field reference (`AgentConfig`)

Every field except `name` and `system_prompt` is optional. Fields that are
absent use the `Agent` dataclass defaults — no implicit injection.

### `name`

```json
"name": "weather_assistant"
```

Non-empty string. Unique identifier for the agent.

### `description`

```json
"description": "Answers weather questions using a live tool."
```

One-line description. Used as the tool description when the agent is exposed
via `as_tool()` or listed as a handoff target.

### `system_prompt`

Three forms are accepted.

**Plain string:**

```json
"system_prompt": "You are a helpful assistant."
```

**Structured `SystemPrompt`** (reuses the runtime type directly):

```json
"system_prompt": {
  "role": "You are a concise weather assistant.",
  "guidelines": [
    "Call get_weather for any city the user asks about.",
    "Keep answers to a single sentence."
  ],
  "tone": "friendly"
}
```

The structured form accepts `role`, `context`, `knowledge`, `guidelines`,
`tone`, `constraints`, `output_format`, and `examples` — the same fields as
the runtime `SystemPrompt`.

**Dynamic prompt ref** — a dotted path to a `DynamicSystemPrompt` callable:

```json
"system_prompt": {"dynamic": "my_pkg.prompts.build_prompt"}
```

The referenced callable receives `DynamicSystemPromptData` at runtime and
returns a `str` or `SystemPrompt`. Async callables are accepted.

### `llm`

**Bare model-name string** — selects the active LiteLLM backend by model
name:

```json
"llm": "claude-haiku-4-5-20251001"
```

**Typed provider block** — selects a provider-native LLM with discriminated
configuration:

```json
"llm": {
  "provider": "anthropic",
  "model": "claude-sonnet-4-5",
  "api_key": "sk-...",
  "config": {
    "temperature": 0.7,
    "max_output_tokens": 2000,
    "auto_cache_control": true
  }
}
```

The `provider` key is the discriminator. Accepted values:

| `provider` | LLM class | Notes |
|---|---|---|
| `"anthropic"` | `AnthropicLLM` | `config` accepts `thinking`, `service_tier`, `auto_cache_control`, `cache_control_ttl` |
| `"openai-responses"` | `OpenAIResponsesLLM` | `config` accepts `reasoning`, `include`, `store`, `service_tier`, `max_tool_calls`, `background`, and others |
| `"openai-chat"` | `OpenAIChatCompletionsLLM` | `config` accepts `audio`, `web_search_options`, `prediction`, `modalities`, `service_tier`, `verbosity`, and others |
| `"gemini"` | `GeminiLLM` | `config` accepts `thinking_config`, `safety_settings`, `cached_content_name`, `response_modalities`; `vertexai`, `project`, `location` for Vertex |
| `"litellm"` | `LiteLLM` | `config` accepts `reasoning_effort`, `thinking`, `cache_control_injection_points`, `cached_content`, and others |

All provider blocks also accept `base_url` and `max_retries`. Secrets
(`api_key`) fall back to the corresponding environment variable when unset.

:::{important}
A provider block's `config` and a top-level `llm_config` are mutually
exclusive — validation rejects both at once. Use one source.
:::

### `llm_config`

Standalone provider-agnostic configuration. Pairs with the string `llm` form.

```json
"llm": "claude-haiku-4-5-20251001",
"llm_config": {
  "temperature": 0.4,
  "max_output_tokens": 1024,
  "num_retries": 2
}
```

Common fields: `temperature`, `top_k`, `top_p`, `max_output_tokens`,
`frequency_penalty`, `presence_penalty`, `stop_sequences`, `seed`, `timeout`,
`num_retries`, `fallbacks`, `tool_choice`, `tool_execution_mode`,
`reset_tool_choice`, and the nested `retry_policy`.

### `tools`

A list where each item is either a dotted `ref` string or a hosted-tool
block.

**Function-tool ref** — a dotted path to a `@function_tool`-decorated
callable:

```json
"tools": ["weather.get_weather", "my_pkg.tools.lookup"]
```

The path names a module and a symbol in it. Two separators are accepted:
`"package.module.symbol"` (dotted — the form shown throughout this guide) and
`"package.module:symbol"` (colon). They are equivalent for a top-level symbol;
the colon form additionally disambiguates a nested attribute (`module:Outer.inner`),
which is why [`dump_agent`](#round-trip-dump) emits it. The module is imported
at load time (relative to the config file's directory), and the symbol must
resolve to a `FunctionTool` instance.

**Hosted-tool block** — a `{"type": ..., "args": {...}}` block selecting a
provider-native tool:

```json
"tools": [
  {"type": "web_search", "args": {"max_uses": 5}},
  {"type": "code_execution", "args": {}},
  "my_pkg.tools.lookup"
]
```

Accepted `type` values:

| `type` | Hosted class | Supported providers |
|---|---|---|
| `"web_search"` | `WebSearchTool` | Anthropic, OpenAI Responses, Gemini |
| `"code_execution"` | `CodeExecutionTool` | OpenAI Responses, Gemini |
| `"file_search"` | `FileSearchTool` | OpenAI Responses |
| `"image_generation"` | `ImageGenerationTool` | OpenAI Responses |
| `"url_context"` | `URLContextTool` | Gemini |
| `"hosted_mcp"` | `HostedMCPTool` | Anthropic |

`args` are forwarded verbatim to the hosted-tool constructor. See the
[Tools guide](../guides/tools.md) for per-tool attribute details.

### `output_schema`

**Bare class ref:**

```json
"output_schema": "my_pkg.schemas.WeatherReport"
```

**With explicit enforcement:**

```json
"output_schema": {
  "ref": "my_pkg.schemas.WeatherReport",
  "enforcement": "normalized"
}
```

`enforcement` values:

| Value | Behavior |
|---|---|
| `"none"` | Raw schema, no enforcement |
| `"normalized"` | Provider-agnostic defaults applied (default) |
| `"strict"` | Full strict-mode for providers that support it |
| `"compact"` | Minimal schema representation |

### `guardrails`

```json
"guardrails": {
  "input": [
    {"ref": "my_pkg.guards.content_check"},
    {"ref": "my_pkg.guards.topic_filter"}
  ],
  "output": [
    {"ref": "my_pkg.guards.content_policy"}
  ]
}
```

Each entry in `input` and `output` is a dotted `ref` resolving to a
user-authored `AgentInputGuardrail` or `AgentOutputGuardrail` instance
in an importable module:

- `{"ref": "pkg.module.symbol"}` — imports `pkg.module` and returns the
  `symbol` attribute, which must be an `AgentInputGuardrail` (for `input`)
  or `AgentOutputGuardrail` (for `output`) instance.

All guardrails are user-authored. See the
[Guardrails guide](../guides/guardrails.md) for how to write them with
the `@agent_input_guardrail` and `@agent_output_guardrail` decorators.

### `skill_activation`

```json
"skill_activation": "lazy"
```

`"lazy"` (default) or `"eager"`. Controls when skill instructions enter the
system prompt. Ignored when the agent has no skills.

### `tool_use_behavior`

```json
"tool_use_behavior": "run_llm_again"
```

| Value | Behavior |
|---|---|
| `"run_llm_again"` (default) | LLM continues after each tool call |
| `"stop_on_first_tool"` | Run stops and returns the first tool result |
| `{"stop_at_tool_names": ["lookup"]}` | Stop when one of the named tools is called |

### `verbose`

```json
"verbose": {
  "enabled": true,
  "mode": "auto",
  "use_color": true,
  "use_rich": true,
  "show_timestamps": false
}
```

Per-agent verbose output override. All fields are optional and fall back to
their listed defaults.

---

## Topology field reference (`TopologyConfig`)

A topology file declares an `agents` map plus optional multi-agent structures.
The map keys are local names — the handles that `handoffs`, `swarm.members`,
`graph.nodes`, and `entry` resolve against.

### `agents`

A required map of local agent name to `AgentNodeConfig`. Each entry is a full
`AgentConfig` that may additionally carry a `handoffs` list:

```json
"agents": {
  "triage": {
    "name": "triage",
    "system_prompt": "...",
    "llm": "claude-haiku-4-5-20251001",
    "handoffs": [
      {"target": "specialist", "description": "Escalate complex issues."}
    ]
  },
  "specialist": {
    "name": "specialist",
    "system_prompt": "...",
    "llm": "claude-haiku-4-5-20251001"
  }
}
```

`handoffs` entries are either a bare local agent name (`"specialist"`) or a
`HandoffRef` object (`{"target": "specialist", "description": "..."}`). The
loader uses a two-pass wiring strategy so mutual handoffs (A hands off to B,
B hands off to A) resolve without proxies.

#### Sub-agent files (`config_path`)

An `agents` entry may instead point at a standalone agent file, so each agent
lives in its own document:

```json
"agents": {
  "triage":  {"config_path": "triage.json"},
  "spanish": {"config_path": "spanish.json"}
}
```

Each `config_path` is resolved **relative to the topology file** (an absolute
path is used as-is). The referenced file is a normal agent document that may
itself declare `handoffs` by name — so `triage.json` can route to `spanish`
while living in its own file. Point such a file's `$schema` at
`agent_node_config.schema.json` (an `AgentConfig` plus `handoffs`) for editor
validation:

```json
{
  "$schema": "../../src/troopai/adk/types/config/agent_node_config.schema.json",
  "name": "triage",
  "system_prompt": "Route Spanish requests to the spanish agent.",
  "llm": "claude-haiku-4-5-20251001",
  "handoffs": [{"target": "spanish", "description": "Handle Spanish."}]
}
```

Inline and `config_path` entries can be mixed in one map. A `config_path`
target is a single agent file, never a nested topology — resolution depth is
fixed at one, so there is no include cycle to worry about. (Because resolving a
`config_path` needs a base directory, this form works through `load_topology`;
calling `build_topology` on an in-memory model with a `config_path` entry
raises.)

### `entry`

```json
"entry": "triage"
```

Optional. The name of the agent to use as the entry point. Not required for
swarm and graph topologies, which declare their own `entry`.

### `swarm`

An optional swarm built over the agents in the `agents` map.

```json
"swarm": {
  "members": ["author", "reviewer"],
  "entry": "author",
  "policy": {"type": "round_robin"},
  "termination": {
    "type": "or",
    "conditions": [
      {"type": "explicit_done"},
      {"type": "max_turns", "limit": 6}
    ]
  },
  "config": {"max_handoffs": 6}
}
```

| Field | Required | Description |
|---|---|---|
| `members` | Yes | Local agent names participating in the swarm |
| `entry` | Yes | Local name of the first agent to act |
| `policy` | No | `{"type": "llm_handoff"}` (default) or `{"type": "round_robin"}` |
| `termination` | Yes | Stop condition (see below) |
| `config` | No | `{"max_handoffs": N, "max_total_tokens": N}` budget caps |

**Termination conditions:**

| `type` | Fields | Description |
|---|---|---|
| `"max_turns"` | `limit` (int, > 0) | Stop after N turns |
| `"explicit_done"` | — | Stop when an agent signals completion |
| `"handoff_to"` | `target` (string) | Stop when control transfers to the named agent |
| `"or"` | `conditions` (list, ≥ 2) | Stop when ANY child condition fires |
| `"and"` | `conditions` (list, ≥ 2) | Stop when ALL child conditions fire |

`or` and `and` conditions are recursive, so you can compose arbitrarily deep
termination trees.

Run a swarm from a loaded topology:

```python
topology = load_topology(Path("swarm.json"))
result = await Runner.configure().swarm(topology.swarm).arun("Explain caching briefly.")
```

### `graph`

An optional state-machine graph built over the agents in the `agents` map.

```json
"graph": {
  "id": "outline_then_write",
  "nodes": {
    "outliner": {"agent": "outliner"},
    "writer": {"agent": "writer"}
  },
  "edges": [{"from": "outliner", "to": "writer"}],
  "entry": "outliner",
  "terminals": ["writer"]
}
```

| Field | Required | Description |
|---|---|---|
| `id` | Yes | Graph identifier |
| `nodes` | Yes | Map of node id to `{"agent": "<local name>", "merge"?: ..., "join"?: ...}` |
| `edges` | No | List of `{"from": "<node>", "to": "<node>", "when"?: "<ref>"}` |
| `entry` | Yes | Entry node id |
| `terminals` | Yes | List of terminal node ids |

**Node fields:**

| Field | Values | Description |
|---|---|---|
| `agent` | local name | Which agent this node runs |
| `merge` | `"concat_text"`, `"last_wins"`, `"extend_items"`, `"first_wins"` | Fan-in merge strategy for nodes with multiple incoming edges |
| `join` | `"and"` (default), `"or"` | Fan-in join semantics |

**Edge fields:**

| Field | Values | Description |
|---|---|---|
| `from` | node id | Source node |
| `to` | node id | Destination node |
| `when` | dotted ref | Optional predicate `(NodeResult) -> bool`; edge fires only when `True` |

Run a graph from a loaded topology:

```python
topology = load_topology(Path("graph.json"))
result = await Runner.arun_graph(topology.graph, "The impact of caching on cost")
```

### Swarm and graph together

A topology may declare both `swarm` and `graph`. They are independent views
over the same `agents` map; the caller picks which one to run. The
`AgentTopology` returned by `load_topology` carries both on `.swarm` and
`.graph`, and either may be `None`.

---

## The `$schema` pointer

Config files may carry an optional `"$schema"` key pointing at the committed
generated schema:

```{code-block} json
:force:

{
  "$schema": "../../src/troopai/adk/types/config/agent_config.schema.json",
  "name": "my_agent",
  ...
}
```

The generated schemas sit in the source tree:

- `src/troopai/adk/types/config/agent_config.schema.json` — a single agent.
- `src/troopai/adk/types/config/agent_node_config.schema.json` — a sub-agent
  file (an agent plus `handoffs`), pointed at by a topology's `config_path`.
- `src/troopai/adk/types/config/topology_config.schema.json` — a topology.

Editors that support JSON Schema (VS Code, PyCharm, Neovim with LSP) will
autocomplete field names, flag unknown keys, and show inline documentation.
External CI tools can validate files against the schema before load time.

The loader ignores the `$schema` key entirely — runtime validation is always
performed by the Pydantic models, not the JSON Schema file. The schemas are
regenerated with:

```bash
python -m troopai.adk.config.schema
```

Run this command after changing any type in `src/troopai/adk/types/config/`
and commit both the source change and the updated schema files together. A
drift-guard test verifies they stay in sync.

:::{note}
The JSON Schema validates YAML files too, because YAML is a superset of
JSON. Point `$schema` at the same file when authoring YAML configs.
:::

---

## Dotted-ref escape hatch and the registries

### Dotted refs

Any field that accepts a `ref` resolves a Python path at load time. Two
separators are accepted:

```
"my_pkg.module.callable_or_object"   # dotted — used throughout this guide
"my_pkg.module:callable_or_object"   # colon — disambiguates nested attributes
```

The part before the separator is the module path (`importlib.import_module`);
the part after is the attribute. The dotted form takes the final segment as the
attribute, so the colon form is the one to reach a nested attribute
(`module:Outer.inner`) — and the one [`dump_agent`](#round-trip-dump) emits.
A single dynamic-import boundary handles all resolution — `resolve_dotted_spec`
in `config/resolver.py` — with the config file's own directory placed on the
import path for the load, so a sibling module resolves by its bare name.

Fields that accept refs: `tools` (function tools), `output_schema`,
`guardrails.input` and `guardrails.output` (dotted ref form),
`system_prompt` (dynamic form), graph edge `when` predicates.

### Extension registries

First-party providers, hosted tools, and guardrails are selected by short
name through registries. You can extend each registry to add your own:

**Custom LLM provider:**

```python
from troopai.adk.config import register_llm_provider
from troopai.adk.llms import LLM, LLMConfig

def my_provider_factory(block: dict) -> tuple[LLM, LLMConfig | None]:
    return MyCustomLLM(model=block["model"]), None

register_llm_provider("my-provider", my_provider_factory)
```

After registration, `"llm": {"provider": "my-provider", "model": "..."}` is
accepted.

**Custom hosted tool:**

```python
from troopai.adk.config import register_hosted_tool
from troopai.adk.tools.hosted import HostedTool

def my_tool_factory(args: dict) -> HostedTool:
    return MyCustomHostedTool(**args)

register_hosted_tool("my_tool", my_tool_factory)
```

---

## Round-trip dump

`dump_agent(agent)` serializes an agent's static surface back to a config
dict. The result validates as an `AgentConfig`.

```python
from troopai.adk.config import dump_agent
import json

data = dump_agent(my_agent)
print(json.dumps(data, indent=2))
```

The dump is **lossy by design**: code-only fields cannot be recovered.

| Field | Dumped? |
|---|---|
| `name`, `description` | Yes |
| `system_prompt` (string or `SystemPrompt`) | Yes |
| `system_prompt` (dynamic callable) | No — omitted |
| `llm` (model-name string or known provider) | Yes |
| `llm` (custom `LLM` subclass) | No — omitted |
| `llm_config` | Yes |
| `output_schema` | Yes (if it wraps a named class) |
| Hosted tools | Yes |
| Function tools | No — bodies are code |
| Guardrails | No — guardrail functions are code |
| `api_key` | No — never serialized |

---

## Security trust boundary

:::{danger}
Loading a config file imports and then calls the Python modules it
references via dotted paths. This is equivalent to importing those modules
directly — it executes Python code from them.

**Load only config files you trust**, exactly as you would a Python entry
point. A malicious config with a crafted `ref` value can execute arbitrary
code in your process.

The same trust requirement extends to every file a topology pulls in through
`config_path` and to the Python modules in those files' directories: loading a
topology trusts each referenced agent file as much as the topology itself.
:::

This boundary is intentional: it is what gives dotted refs their power —
referencing any Python function, guardrail, or output schema without
restricting the vocabulary to a closed set. Keep trust evaluation at the
config-file level.

---

## See also

- [Agents guide](../guides/agents.md) — the `Agent` dataclass, tools,
  handoffs, guardrails.
- [Tools guide](../guides/tools.md) — function tools, hosted tools, tool
  guardrails.
- [Guardrails guide](../guides/guardrails.md) — input/output guardrail
  decorator syntax, severity, remediation.
- Running examples in `examples/config/`:
  - `run_config_agent.py` — single-agent with a function tool and output schema.
  - `run_topology.py` — multi-agent handoff topology.
  - `run_swarm.py` — swarm with composed termination.
  - `run_graph.py` — state-machine graph.
