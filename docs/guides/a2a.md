(guides/a2a)=

# 🤝 A2A

The **Agent-to-Agent (A2A) protocol** is an open standard for autonomous
agents to communicate with each other as peers over HTTP + Server-Sent
Events. In the ADK, A2A is the answer to one specific question: *how does
one ADK process delegate work to a completely separate process?*

```{admonition} When to reach for A2A
:class: tip

A2A is appropriate when the agent you need to call lives in a different
process, a different service, or belongs to a different team or framework.
For agents that share the same Python process, `Agent.as_tool()` or
handoffs are cheaper and simpler.
```

The A2A support is an **optional extra**. Install it before using any
symbol from `troopai.adk.a2a`:

```bash
pip install 'troopai-adk-python[a2a]'
```

This pulls in `a2a-sdk[http-server]` (Starlette + sse-starlette for the
server path) and `httpx` for the client path. When the extra is absent,
every public symbol in `troopai.adk.a2a` is `None`; downstream code can
branch on `A2AAgent is None` to degrade gracefully.

---

## A2A vs handoffs vs MCP

Three extension mechanisms extend the ADK across different boundaries. The
right choice depends entirely on where the other party lives:

| Situation | Mechanism |
| --- | --- |
| Target agent is in **the same Python process** | `Agent.as_tool()` or handoffs — zero network, shared context window |
| Target is a **different process / service / framework** | `A2AAgent` + `A2ARunner` — typed remote-agent peer over HTTP |
| Target is a **stateless tool, API, or database** | MCP — tool semantics, no agent loop on the other side |

The key distinction between A2A and MCP is the shape of what lives at the
other end. MCP standardises *tool surfaces* — the far end is a collection of
callable functions. A2A standardises *agent surfaces* — the far end is a
full agent loop with its own LLM, tools, guardrails, and reasoning. See
{doc}`../concepts/index` (§ A2A vs MCP) for the one-sentence rule.

**Handoffs** are an intra-process mechanism. When a handoff fires, the
`Runner` routes execution from one local `Agent` to another within the same
Python process — no HTTP, no serialisation, shared tool registry and
context window. Once the work crosses a network boundary, handoffs no longer
apply; that is where A2A steps in. See {doc}`handoffs` for the full handoff
guide.

---

## Server side — exposing an ADK agent over A2A

Any local `Agent` can be exposed as an A2A endpoint with three
components: an `A2AServer` config object, a manually-authored `AgentCard`,
and `build_starlette_app` to materialise the ASGI app.

```python
import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from troopai.adk.a2a import A2AServer, build_starlette_app
from troopai.adk.agents import Agent

local_agent = Agent(
    name="research_helper",
    system_prompt="You help users find information.",
    tools=[...],
)

card = AgentCard(
    name="research-helper",
    description="Helps users find information.",
    version="1.0.0",
    supported_interfaces=[
        AgentInterface(
            url="https://my.example.com",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        ),
    ],
    capabilities=AgentCapabilities(streaming=True),
)

server = A2AServer(agent=local_agent, agent_card=card)
app = build_starlette_app(server)

# The ADK does not spawn a process — pick your own ASGI runtime:
uvicorn.run(app, host="0.0.0.0", port=8080)
```

`A2AServer` is a frozen dataclass — pure config with no running state. It
pairs the local `Agent` with its `AgentCard` and optional settings
(`max_turns`, `run_config`, `task_store`, `rpc_url`). The ADK does not own
the ASGI lifecycle; `build_starlette_app` returns a Starlette app that your
chosen runtime (uvicorn, hypercorn, granian) serves.

The resulting app exposes two routes:

- `GET /.well-known/agent-card.json` — your `AgentCard`, the discovery
  contract every A2A client reads before sending a task.
- `POST /` (or `server.rpc_url`) — JSON-RPC dispatcher handling
  `send_message` (blocking and streaming), `get_task`, `cancel_task`,
  `list_tasks`, and push-notification config endpoints. SSE responses are
  emitted by the a2a-sdk for streaming methods.

**The `AgentCard` is manually authored** — every field (`name`,
`description`, `url`, `version`, `capabilities`, `skills`) is intentional.
There is no auto-derivation from the `Agent` object. This matches the
upstream A2A spec's recommendation: the card is the public contract the LLM
on the other side reads, so it must be deliberate.

```{warning}
`A2AServer.task_store` defaults to `None`, which causes `build_starlette_app`
to install an `InMemoryTaskStore`. This works for development but **loses
every task on process restart**, breaking the `A2AContinuationToken` resume
contract for any task that outlives the process. Supply a persistent store
for production:

```python
from a2a.server.tasks import DatabaseTaskStore

server = A2AServer(
    agent=local_agent,
    agent_card=card,
    task_store=DatabaseTaskStore(...),
)
```

`build_starlette_app` logs a `WARNING` when falling back to the in-memory
default so the gap is visible in deployment logs.
```

See `examples/a2a/server_basic.py` for a complete runnable server.

---

## Client side — calling a remote A2A agent

`A2AAgent` is the client-side primitive. It extends `BaseAgent` (the same
base as local `Agent`) and is **pure config** — it carries the remote URL,
optional timeout, and interceptors. Execution flows through `A2ARunner`.

### Basic call

```python
import asyncio
import logging
from troopai.adk.a2a import A2AAgent, A2ARunner

logger = logging.getLogger(__name__)

async def main() -> None:
    async with A2AAgent(
        name="ResearchBot",
        url="https://research.example.com",
    ) as remote:
        result = await A2ARunner.arun(remote, "What is the latest A2A spec version?")
        logger.info("result: %s", result.text)

asyncio.run(main())
```

The `async with` form is preferred so the underlying `httpx.AsyncClient`
closes cleanly. The return is a typed `A2ARunResult` with three fields:
`.text`, `.task_id`, and `.context_id`.

`A2ARunner` accepts **only** `A2AAgent` instances. Passing a local `Agent`,
`Swarm`, or `Graph` raises `TypeError` with a message pointing the caller
back at `Runner`. The split exists because the wire format, lifecycle, and
error model of a remote A2A peer have nothing in common with a local agent
loop.

### As a tool

`A2AAgent.as_tool()` wraps the remote agent as a `FunctionTool` so a local
agent's LLM can invoke it mid-turn alongside its other tools:

```python
from troopai.adk.agents import Agent

remote = A2AAgent(name="Researcher", url="https://research.example.com")
local = Agent(
    name="Coordinator",
    system_prompt="Use the researcher tool when you need fresh information.",
    tools=[remote.as_tool(max_result_tokens=2_000)],
)
```

The wrapping `FunctionTool` flows through the standard middleware, tracing,
and hooks pipeline. The inner call dispatches via `A2ARunner.arun` (not
`Runner.arun`) so the local-vs-remote boundary stays clean.

### As a Graph node

`A2AAgent` can also be placed as a node inside a local `Graph` alongside
local `Agent`, `Swarm`, and callable nodes. The `A2AExecutableAdapter`
handles the dispatch transparently via `Graph.to_executable()`.

See `examples/a2a/client_in_graph.py` for a complete example.

---

## Wire format

The A2A protocol runs over HTTP with JSON-RPC 2.0 as the messaging layer.
Request and response bodies are encoded as JSON. Streaming responses are
delivered as Server-Sent Events (SSE) on the same HTTP connection.

The `a2a-sdk` owns the protobuf wire types (`a2a.types`). The ADK confines
all `a2a.types` imports to three files: `converters.py`, `a2a_client.py`,
and `executor.py`. Framework code elsewhere works exclusively with ADK-owned
types (`A2ARunResult`, `A2AStreamEvent`, `A2AContinuationToken`,
`A2ATaskStatus`, `A2ATaskStateLiteral`) — no protobuf imports leak into
`agents/`, `run/`, `tools/`, or any developer-facing surface.

The primary JSON-RPC methods the ADK uses on the client side are:

- `message/send` — blocking send; returns a completed `Task`.
- `message/stream` — streaming send; emits SSE events until terminal state.
- `tasks/get` — poll a previously-submitted task by `task_id`.
- `tasks/cancel` — request cancellation of an in-flight task.

On the server side the ADK's `DefaultRequestHandler` (from `a2a-sdk`)
handles all of the above plus `tasks/list` and push-notification config
endpoints.

---

## Authentication

Auth is plumbed through `a2a-sdk`'s `ClientCallInterceptor` interface —
the ADK does not introduce a new auth surface:

```python
from a2a.client import AuthInterceptor, InMemoryContextCredentialStore

credentials = InMemoryContextCredentialStore()
credentials.set_credentials(...)
interceptor = AuthInterceptor(credential_service=credentials)

async with A2AAgent(
    name="Secure",
    url="https://secure.example.com",
    interceptors=[interceptor],
) as remote:
    result = await A2ARunner.arun(remote, "authenticated request")
```

Server-side authentication is the responsibility of the `DefaultRequestHandler`
and any middleware you stack on the Starlette app. The ADK adds no auth code
of its own; auth is a deployment concern, not a framework concern.

---

## Streaming and long-running tasks

### Streaming

Pass `stream=True` to receive incremental text deltas as the remote agent
works:

```python
chunks: list[str] = []
async for event in await A2ARunner.arun(remote, "Write a long essay.", stream=True):
    if event["type"] == "text_delta":
        chunks.append(event["text_delta"])
    elif event["type"] == "completed":
        logger.info("final: %s", "".join(chunks))
        break
    elif event["type"] == "failed":
        logger.error("failed: %s", event.get("message", ""))
        break
```

The stream is bounded by `A2AClient.max_stream_chunks` (default 10,000) and
`max_stream_bytes` (default 8 MiB). A runaway remote that streams forever
raises `A2AProtocolError` rather than exhausting memory.

### Background tasks and continuation tokens

For tasks that may exceed an HTTP timeout, submit in the background and poll
later:

```python
async with A2AAgent(name="LongJob", url="https://jobs.example.com") as remote:
    token = await A2ARunner.arun(remote, "Crawl the entire archive.", background=True)
    # token is an A2AContinuationToken — JSON-serialisable, durable
    # across process restarts as long as the remote TaskStore retains it.

    # Later (possibly from a different process):
    status = await A2ARunner.poll_task(remote, token)
    if status.state == "completed":
        logger.info("done: %s", status.result)
    elif status.state in ("failed", "rejected", "cancelled"):
        logger.warning("ended: %s: %s", status.state, status.message)
```

`A2AContinuationToken` is a frozen dataclass with three fields: `task_id`,
`context_id`, and `remote_url`. Persist it via
`dataclasses.asdict() + json.dumps()`, restore via
`A2AContinuationToken(**json.loads(payload))`.

`A2ARunner.poll_task` returns a one-shot status snapshot — it does not
block until terminal state. To wait, call it in a bounded polling loop with
your own timeout and retry budget. To cancel, call
`A2ARunner.cancel_task(remote, token)`.

---

## Common patterns

### Federation across teams

Different teams own different ADK processes. A coordinator process holds only
`A2AAgent` references to the specialist services and delegates sub-tasks to
them via `A2ARunner.arun`. Each specialist service exposes a different
`AgentCard` with its own capability description. The coordinator's LLM reads
the `description` and `skills` fields in each card to decide which service
handles a given sub-task.

### Specialist remote agents exposed as tools

When the coordinator is itself an `Agent`, use `A2AAgent.as_tool()` to
register each remote specialist as a tool. The LLM orchestrates delegation
mid-turn using standard tool calling — the remote boundary is transparent.
This is the lowest-friction pattern for adding a remote specialist to an
existing local agent.

### Tenant-isolated agents on separate infrastructure

Multi-tenant systems sometimes require per-tenant isolation at the infra
level — separate databases, network policies, or compliance boundaries. Each
tenant's agent runs on its own service. The calling process constructs an
`A2AAgent` with the tenant-specific URL (looked up from a routing table) and
dispatches via `A2ARunner.arun`. The ADK's `A2AContinuationToken` carries
`remote_url` so background tasks can be resumed even from a fresh process.

---

## Error handling

All A2A failures surface as typed exceptions under `troopai.adk.a2a`:

| Exception | Cause |
| --- | --- |
| `A2ATransportError` | Network failure (connect, timeout, DNS, TLS) |
| `A2AProtocolError` | Malformed response, version mismatch, auth rejection |
| `A2ATaskError` | Remote task ended in `failed` or `rejected` state |
| `A2ATaskCancelledError` | Remote task was cancelled (subclass of `A2ATaskError`) |

All extend `A2AError`, which extends the root `TroopAIError`.

```{warning}
`A2ATaskError.remote_message` is **untrusted input** sourced from the peer
agent. It may contain prompt-injection bait or escape sequences. Sanitise or
escape before rendering to end-users, downstream LLMs, or log aggregators
that might re-interpret it.
```

```python
from troopai.adk.a2a import A2ATaskError, A2ATransportError

try:
    result = await A2ARunner.arun(remote, "...")
except A2ATransportError:
    # network problem — consider retry
    ...
except A2ATaskError as exc:
    logger.warning(
        "task %s ended in %s: %s",
        exc.task_id,
        exc.state,
        exc.remote_message,
    )
```

---

## Tracing

A2A calls participate automatically in OpenTelemetry tracing via the
existing `function_span` infrastructure. Client-side spans are named
`a2a.<task_id>` and carry an `troopai.a2a.remote_url` attribute;
server-side spans carry `troopai.a2a.agent_name`. These two attributes let
you correlate the client and server sides of a network boundary in your
trace UI.

The same secret-redaction that runs on tool I/O runs on `a2a_data` span
attributes — embedded credentials are masked before leaving the process.

Enable OTel: `pip install 'troopai-adk-python[otel]'`. See {doc}`tracing` for the
full tracing guide.

---

## What A2A does not try to do

The current implementation deliberately leaves several concerns outside its
scope:

- **Shared state between processes.** Each A2A peer owns its own context
  window, tool registry, and guardrails. The protocol exchanges text prompts
  and text responses — no shared memory, no shared session. If you need
  shared state, manage it externally (a database, a cache) and pass
  references in the prompt.
- **Long-lived stateful sessions.** The A2A protocol has a `context_id`
  concept for multi-turn conversations, but there is no long-lived session
  object on the ADK client side. Each `A2ARunner.arun` call is a self-
  contained request; multi-turn state lives on the server's `TaskStore`.
- **W3C trace propagation through `ClientCallInterceptor`.** Client and
  server spans are linked by naming convention (`a2a.<task_id>`), but a
  shared trace ID across the HTTP boundary via W3C `traceparent` propagation
  is not yet implemented.
- **`TASK_STATE_INPUT_REQUIRED` round-trips (human-in-the-loop).** The
  server can return `input_required` state, but the ADK does not yet have a
  built-in mechanism to pause, collect user input, and resume the same
  server-side task. This requires a persistent `TaskStore` and application-
  level polling logic.
- **Push notifications.** The JSON-RPC dispatcher exposes push-notification
  config endpoints, but the ADK's client surface does not yet expose a
  high-level push-notification API.

---

## See also

- {doc}`../concepts/index` — § A2A vs MCP for the one-sentence rule
- {doc}`mcp` — MCP guide (tool-level remote calls)
- {doc}`handoffs` — intra-process agent routing
- {doc}`tracing` — OpenTelemetry integration
- `examples/a2a/` — runnable client and server examples
- A2A specification: <https://a2a-protocol.org/>
