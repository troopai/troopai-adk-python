# Declarative agent configuration (JSON)

Build an `Agent` from a JSON file instead of Python, via
`troopai.adk.config.load_agent`.

```bash
python examples/config/run_config_agent.py
```

## What it shows

- `agent.json` — a strict, schema-validated config: `name`, a structured
  `system_prompt`, an `llm` model name, a `tools` list, and a structured
  `output_schema`.
- `weather.py` — an importable sibling module holding the tool body
  (`get_weather`) and the output-schema class (`WeatherReport`).
- `run_config_agent.py` — loads the JSON, which references those symbols by a
  normal dotted path (`weather.get_weather`, `weather.WeatherReport`). Code-only
  behavior stays in Python; the JSON wires it together. References resolve at
  load time relative to the config file's directory, so a sibling module loads
  by its bare name — no `__main__:` prefix.

## The `$schema` pointer

`agent.json` carries an optional `$schema` key pointing at the generated
schema (`src/troopai/adk/types/config/agent_config.schema.json`). It enables
editor autocomplete and external/CI validation; the loader ignores it at
runtime (validation is always the Pydantic models). Regenerate the schema
with:

```bash
python -m troopai.adk.config.schema
```

## Multi-agent topologies

- `topology.json` + `run_topology.py` — an `agents` map whose entries point at
  per-file agents (`triage.json`, `spanish.json`) via `config_path`; handoffs
  are wired by name across the files.
- `swarm.json` + `run_swarm.py` — a `swarm` over the agents (members, entry,
  round-robin policy, composed `or` termination), run via
  `Runner.configure().swarm(...).arun(...)`.
- `graph.json` + `run_graph.py` — a `graph` over the agents (nodes, edges,
  entry, terminals), run via `Runner.arun_graph(...)`.

A topology file may declare both a `swarm` and a `graph` — they are
independent views over the same `agents`; the caller picks which to run.

## Strictness

Validation rejects unknown keys, so a typo fails loudly. Behavior that needs
Python callables/objects (`hooks`, `middleware`, `skills`) is rejected with a
message pointing you back to Python. (`handoffs` are declarative — by name on a
topology node.)

Loading the agent needs no API key. The final live turn calls the model and
requires an LLM API key in the environment.
