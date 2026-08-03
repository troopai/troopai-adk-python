# TroopAI Agent Development Kit (ADK)

**Where language becomes action.**

[![CI](https://github.com/troopai/troopai-adk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/troopai/troopai-adk-python/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/troopai/troopai-adk-python/branch/main/graph/badge.svg)](https://codecov.io/gh/troopai/troopai-adk-python)

A lightweight, provider-agnostic Python framework for building multi-agent workflows with 100+ LLMs via litellm.

> [!NOTE]
> The JavaScript/TypeScript version of this ADK will be released soon. Stay tuned!

## The concept

An agent is a model that stopped talking and started doing: it calls tools,
changes state, and leaves side effects in the world. One agent is useful. A
unit of specialists that can't coordinate is a liability.

A **troop** puts that unit under a single leader with real authority — not a
vibe in a system prompt, but authority the runtime enforces. The leader tasks
each unit, and every unit executes inside per-unit doctrine: bounded retries,
hard token budgets, timeouts, and output that only commits to shared state
after inspection. The leader's judgment operates inside those rails —
command-by-exception by default, direct micromanagement of any unit whenever
you want it. Command and control, for agents.

The ADK also ships the opposite topology on purpose. `Swarm` is decentralized:
peers hand control to each other under a routing policy, nobody in charge —
right for open-ended exploration. A troop is what you reach for when someone
has to be accountable for the objective: production units with budgets,
deadlines, and a leader that answers for them.

## Design tenets

1. **Explicit over magical.** If you can't step through it in a debugger, it
   doesn't belong in the orchestration path.
2. **One obvious way.** Fewer knobs, sharper edges — the ADK has opinions.
3. **Everything is inspectable.** Decisions, tool I/O, and token costs are
   structured traces, not anecdotes.
4. **Benchmarks or it didn't happen.** Claims ship with eval evidence or not
   at all — including this framework's own.

## Status

- **Today (v0.1.0 groundwork):** agents, `Runner` (sync/async/streaming),
  swarms, graphs, flows, task pipelines, tools, handoffs, guardrails, memory
  and sessions, MCP, A2A, sandboxed code execution, durable execution
  (Temporal/Restate), OpenTelemetry tracing, deploy targets, and a strict
  JSON/YAML config layer — all in this repository, MIT licensed.
- **Next:** the troop primitive (leader doctrine + commander + `TroopRunner`).
- **Then:** `troopai-evals-python` (benchmarks vs. other frameworks) and
  `troopai-cookbook-python` (production-grade examples) — build with the ADK,
  prove it with the evals, learn it from the cookbook.

## Installation

### Prerequisites

- Python 3.12+
- [Conda](https://docs.conda.io/) (recommended) or a virtual environment manager

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/troopai/troopai-adk-python.git
cd troopai-adk-python

# 2. Create and activate the conda environment
#    (this also runs `pip install -e '.[dev]'` — full contributor install)
conda env create -f environment.yaml
conda activate troopai-adk-python

# 3. Optional: choose a leaner install if you don't need the dev surface
pip install -e .                # core only (litellm + pydantic + griffe + aiosqlite)
pip install -e '.[anthropic]'   # native Anthropic SDK path
pip install -e '.[otel]'        # OpenTelemetry tracing bridge
pip install -e '.[mcp]'         # Model Context Protocol client
pip install -e '.[viz]'         # Agent graph visualization (graphviz)
pip install -e '.[verbose]'     # Rich-backed panel/line verbose renderer (ANSI fallback without it)
pip install -e '.[all]'         # all of the above
pip install -e '.[dev]'         # everything + test + lint + typecheck (default)
```

The editable install (`-e .`) registers `troopai.adk` as an importable package using the `src/` layout defined in `pyproject.toml`. Changes to source code take effect immediately without reinstalling.

Feature extras are discovered automatically by `pip`; install only what you need. The core install is deliberately lean — `litellm`, `pydantic`, `griffe`, `aiosqlite`, `typing-extensions` — and every optional provider / exporter / UI enhancement is gated behind its own extra (including `rich` via `[verbose]`).

### API Keys

Set the API keys for the LLM providers you want to use:

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

### Verify Installation

```bash
python -c "from troopai.adk import Agent, Runner; print('OK')"
```

## Quick Start

```python
import asyncio
import logging

from troopai.adk import Agent, Runner

logger = logging.getLogger(__name__)

agent = Agent(
    name="Assistant",
    system_prompt="You are a helpful assistant.",
)

result = asyncio.run(Runner.arun(agent, "Hello!"))
logger.info(result.final_output)
```

## Command-Line Interface

The `troopai` console script drives agents from the terminal — scaffold a
project, validate its config without spending a token, then run or chat:

```bash
troopai new my_agent                       # scaffold config + tools + schema
troopai validate my_agent/agent.json       # strict schema check, no tokens
troopai run my_agent/agent.json "hello"    # one-shot run (config or --agent module:var)
troopai chat my_agent/agent.json           # interactive REPL, optional --session-db
troopai serve my_agent/agent.json                    # REST + health over HTTP ([serve] extra)
```

`run` auto-dispatches agents, swarms, graphs, and topologies; every
cost-affecting behavior (sessions, verbose rendering, tracing, env
files) stays off until you pass its flag. See `docs/cli/cli.md` for the
full command reference.

## Deployment

Serve an agent over HTTP, then ship the container to any cloud. The
framework imports no server runtime and no cloud SDK — every piece is
opt-in, and you keep control of the runtime.

```bash
pip install 'troopai-adk-python[serve]'

# Serve locally: REST (POST /run, POST /run_sse) + health (/healthz, /readyz).
troopai serve --agent my_agent.app:agent --host 0.0.0.0 --port 8000

# Generate the deployment artifacts you own (Dockerfile + manifests):
troopai deploy init --target k8s --agent my_agent.app:agent --image my-agent:latest

# Or build and ship to a target via your installed CLIs:
troopai deploy build      --agent my_agent.app:agent --image my-agent:latest --push
troopai deploy cloud-run  --agent my_agent.app:agent --image gcr.io/PROJECT/my-agent --project PROJECT --region REGION
troopai deploy gke        --agent my_agent.app:agent --image IMAGE --project P --region R --cluster C
troopai deploy ecs        --agent my_agent.app:agent --image ACCT.dkr.ecr.R.amazonaws.com/my-agent --region R --execution-role-arn ARN
```

`troopai deploy` targets `docker`, `k8s`, `gke`, `helm`, `cloudrun`,
`ecs`, `app-runner`, and `lambda`. The generated image satisfies the
universal container contract (binds `0.0.0.0:$PORT`, config from env,
non-root, `/healthz` + `/readyz` probes), so the same image runs
everywhere. Because the package is private, the generated
`requirements.txt` must make `troopai-adk-python` installable in your image
(private index, vendored wheel, or VCS URL).

A single replica works out of the box on the default per-pod SQLite
stores. For multi-replica (horizontally-scaled) deployments, back A2A
tasks and REST sessions with Postgres so state is shared across pods —
`troopai serve --task-dsn "$PG_DSN" --session-dsn "$PG_DSN"` (install
`troopai-adk-python[a2a-postgres,session-postgres]`). The AWS deploy commands
also accept `--push` to log in to ECR and build/push the image for you.
See [`docs/deploy/`](docs/deploy/) for the full guide.

## Running Examples

All examples are runnable from the project root:

```bash
python examples/agent_patterns/agents_as_tools.py
python examples/handoffs/llm_orchestrated.py
python examples/tools/tool_guardrails.py
```

## Core Concepts

- [**Agents**](docs/agents/) — Autonomous entities with tools, guardrails, and handoffs
- [**Tools**](docs/tools/) — Function wrappers with schema validation and guardrails
- [**Handoffs**](docs/handoffs/) — Agent-to-agent routing (LLM-orchestrated or code-orchestrated)
- [**Guardrails**](docs/guardrails/) — Pre/post execution validation at agent and tool level
- [**Memory**](docs/memory/) — Persistent knowledge across sessions
- [**Skills**](docs/skills/) — Reusable capability packages (instructions + tools + governance)
- [**Tracing**](docs/tracing/) — OpenTelemetry observability

## Project Structure

```
src/troopai/adk/       # Source code (namespace package)
tests/                 # Unit and integration tests
examples/              # Single-file runnable examples (one concept each)
docs/                  # Usage documentation
configs/               # Logging and other configs
```

## Key Dependencies

**Core**: `litellm` | `pydantic` | `griffe` | `aiosqlite` | `typing-extensions`
**Optional extras**: `anthropic` (`.[anthropic]`) | `mcp` (`.[mcp]`) | `opentelemetry-*` (`.[otel]`) | `graphviz` (`.[viz]`) | `rich` (`.[verbose]`)

## Acknowledgements

This ADK draws on prior art and ongoing work from across the
multi-agent ecosystem:

- [**LangGraph**](https://github.com/langchain-ai/langgraph) —
  state-machine multi-agent orchestration; influence on the
  `graphs/` subsystem.
- [**CrewAI**](https://github.com/crewAIInc/crewAI) — multi-agent
  collaboration patterns.
- [**OpenAI Swarm**](https://github.com/openai/swarm) — swarm cycle
  pattern (reference shape for `swarms/`).
- [**OpenAI Agents SDK**](https://github.com/openai/openai-agents-python) —
  Runner design and handoff mechanism reference.
- [**Anthropic Claude Agent SDK**](https://github.com/anthropics/anthropic-quickstarts) —
  Anthropic-native provider design.
- [**LiteLLM**](https://github.com/BerriAI/litellm) — provider-agnostic
  LLM abstraction over 100+ models.
- [**Model Context Protocol**](https://modelcontextprotocol.io/) —
  tool-integration substrate.
- [**Pydantic**](https://docs.pydantic.dev/) — typed validation.
- [**Temporal**](https://temporal.io/) — durable execution backbone.
- [**OpenTelemetry**](https://opentelemetry.io/) and
  [**OpenInference**](https://github.com/Arize-ai/openinference) —
  observability conventions.
- [**Sphinx**](https://www.sphinx-doc.org/) +
  [**MyST-Parser**](https://github.com/executablebooks/MyST-Parser) —
  documentation pipeline.

Inclusion here records influence, not endorsement.
