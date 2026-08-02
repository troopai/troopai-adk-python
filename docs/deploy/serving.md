(deploy/serving)=

# Serving Layer

The `troopai.adk.serving` module turns a local `Agent` into an ASGI app that
any ASGI runtime can serve. Two surfaces are available: a plain-REST surface
for generic HTTP clients and an A2A JSON-RPC surface for peer agents. Both are
opt-in — the framework never serves a route the developer did not request.

## Installation

```bash
pip install 'troopai-adk-python[serve]'
```

This pulls in Starlette and sse-starlette. The A2A surface additionally
requires the `a2a` extra:

```bash
pip install 'troopai-adk-python[serve,a2a]'
```

## Quick start: `troopai serve`

`troopai serve` is the fastest path from a local agent to a running HTTP
endpoint. It exposes the REST surface and health routes by default:

```bash
# Serve the agent defined at my_pkg.agents:assistant
troopai serve --agent my_pkg.agents:assistant
```

By default this binds `127.0.0.1:8000` and serves:

- `POST /run` — run the agent to completion, returns JSON
- `POST /run_sse` — run the agent and stream events as Server-Sent Events
- `GET /healthz` — liveness probe
- `GET /readyz` — readiness probe

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--agent MODULE:VAR` | — | Dotted reference to the `Agent` object (required) |
| `--host TEXT` | `127.0.0.1` | Bind address. Use `0.0.0.0` inside a container |
| `--port INTEGER` | `8000` | Bind port |
| `--rest/--no-rest` | on | Expose `POST /run` and `POST /run_sse` |
| `--health/--no-health` | on | Expose `GET /healthz` and `GET /readyz` |
| `--a2a/--no-a2a` | off | Enable the A2A surface (requires `--card`) |
| `--card FILE` | — | Developer-authored AgentCard JSON; enables the A2A surface |
| `--max-turns INTEGER` | framework default | Per-request agent-loop turn limit |
| `--task-db FILE` | in-memory | SQLite file for durable A2A task storage with restart recovery (single replica) |
| `--task-dsn DSN` | — | Postgres DSN for a shared A2A task store across replicas; excludes `--task-db` |
| `--session-db FILE` | — | SQLite file for persistent REST sessions (single replica); excludes `--session-dsn` |
| `--session-dsn DSN` | — | Postgres DSN for shared REST sessions across replicas; excludes `--session-db` |
| `--env-file FILE` | — | Load `KEY=VALUE` pairs from a file (never auto-discovered) |

### Enabling the A2A surface

Pass `--card` with a developer-authored AgentCard JSON to publish the A2A
JSON-RPC and discovery surface alongside the REST surface:

```bash
troopai serve --agent my_pkg.agents:assistant --card card.json
```

The card is published at `GET /.well-known/agent-card.json`. The CLI never
synthesises a card; every field is intentional.

:::{warning}
The default A2A task store is in-memory. Tasks are lost on process restart,
which breaks the A2A continuation token resume contract for any task that
outlives the process. Pass `--task-db tasks.sqlite` for a durable store
that recovers non-terminal tasks from prior processes on startup.
:::

### Container invocation

Inside a container, bind all interfaces and read the platform port:

```bash
troopai serve --agent app.agents:assistant --host 0.0.0.0 --port "$PORT"
```

The `troopai deploy` tooling bakes this command into the generated
Dockerfile automatically.

## REST request and response format

Both `POST /run` and `POST /run_sse` accept the same JSON body:

```json
{
  "prompt": "What is the capital of France?",
  "max_turns": 10,
  "session": {
    "user_id": "alice",
    "session_id": "chat-123"
  }
}
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | yes | Non-empty string sent to the agent |
| `max_turns` | no | Positive integer; overrides the server default when present |
| `session` | no | Session block; requires a session factory wired into the app |

### `POST /run` — synchronous response

Returns a JSON object with the run result when the agent finishes:

```
{
  "final_output": "Paris.",
  "items": [...],
  "usage": {
    "input_tokens": 42,
    "output_tokens": 3
  }
}
```

Response fields are Layer-1 and Layer-3 types only — the provider wire format
never crosses this boundary.

### `POST /run_sse` — streaming response

Returns a stream of `text/event-stream` frames. Each frame carries a JSON
payload; the stream ends with an `event: result` frame carrying the final
summary:

```
data: {"type": "text_delta", "text_delta": "Par"}

data: {"type": "text_delta", "text_delta": "is."}

event: result
data: {"final_output": "Paris.", "usage": {...}}
```

## Programmatic API: `build_app`

For production deployments with custom ASGI runtimes or more control over
the lifecycle, use `build_app` directly instead of the CLI:

```python
import uvicorn
from troopai.adk.agents import Agent
from troopai.adk.serving import build_app

agent = Agent(name="assistant", system_prompt="You are a helpful assistant.")

app = build_app(
    agent,
    rest=True,
    health=True,
)

uvicorn.run(app, host="0.0.0.0", port=8080)
```

`build_app` raises `ValueError` if no surface is enabled.

### Signature

```python
def build_app(
    agent: Agent,
    *,
    rest: bool = False,
    a2a_server: A2AServer | None = None,
    health: bool = False,
    max_turns: int | None = None,
    run_config: RunConfig | None = None,
    session_factory: SessionFactory | None = None,
    readiness_probe: ReadinessProbe | None = None,
) -> Starlette: ...
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `agent` | The agent the REST surface runs |
| `rest` | Mount `POST /run` and `POST /run_sse`; off by default |
| `a2a_server` | An `A2AServer` config to mount the A2A JSON-RPC surface; `None` leaves A2A off |
| `health` | Mount `GET /healthz` and `GET /readyz`; off by default |
| `max_turns` | Default per-request agent-loop budget; `None` defers to the framework default |
| `run_config` | Optional `RunConfig` applied to every REST run |
| `session_factory` | Async `(user_id, session_id) -> SessionStore` for requests carrying a `session` block |
| `readiness_probe` | Async predicate for `GET /readyz`; when `None`, readiness always reports ready |

### Adding a session backend

Wire a `SessionFactory` to persist conversation history across requests.

For a single replica, `SQLiteMultiSessions` is sufficient:

```python
from troopai.adk.session import SQLiteMultiSessions
from troopai.adk.serving import build_app

sessions = SQLiteMultiSessions(app_name="assistant", path="sessions.sqlite")

app = build_app(
    agent,
    rest=True,
    health=True,
    session_factory=lambda user_id, session_id: sessions.get_or_create(user_id, session_id),
)
```

For multiple replicas, use `PostgresMultiSessions` so every replica reads and
writes the same conversation state, or pass `--session-dsn` to `troopai serve`
to have the CLI wire it automatically. See [Horizontal scaling](scaling.md) for
the full multi-replica setup.

### Adding a custom readiness probe

Provide an async predicate to signal when the service is ready for traffic:

```python
async def my_readiness() -> bool:
    # Return False until external dependencies are reachable.
    return await db.ping()

app = build_app(agent, health=True, readiness_probe=my_readiness)
```

When the probe returns `False`, `GET /readyz` responds with HTTP 503.

## See also

- [Container contract](container.md) — what every generated image must satisfy
- [Kubernetes and Helm](kubernetes.md) — Kubernetes manifests and Helm charts
- [GCP Cloud Run](gcp.md) — Cloud Run deployment
- [AWS](aws.md) — ECS, App Runner, Lambda
- [Horizontal scaling](scaling.md) — multi-replica and shared backends
- [A2A guide](../a2a/a2a.md) — full A2A protocol documentation
