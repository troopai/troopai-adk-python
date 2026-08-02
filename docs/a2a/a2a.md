# Agent-to-Agent (A2A) Protocol

The TroopAI ADK ships first-class support for the
[Agent-to-Agent (A2A) protocol](https://a2a-protocol.org/) — an open
standard for autonomous agents to talk to each other as peers over
HTTP+SSE. This is distinct from MCP: MCP standardises how an agent
talks to its **tools**, A2A standardises how an agent talks to other
**agents**.

The two protocols are complementary, not competing. A single agent
can (and often should) expose itself via both: MCP for structured
tool callers and A2A for peer agents that need multi-turn, stateful
collaboration.

## Installation

A2A support is an optional extra:

```bash
pip install 'troopai-adk-python[a2a]'
```

This pulls in `a2a-sdk[http-server]` (which includes Starlette and
sse-starlette for the server path) plus `httpx` for the client path.
When the extra is missing, every public symbol in
`troopai.adk.a2a` is `None`; downstream code can branch on
`A2AAgent is None` to skip A2A wiring gracefully.

## When to use A2A vs MCP vs `Agent.as_tool()`

| Situation | Use |
|---|---|
| One agent calls another that lives in **the same Python process** | `Agent.as_tool()` — zero network overhead, shares context window |
| One agent calls another that lives in **a different process / vendor / framework** | `A2AAgent` — typed remote-agent peer over HTTP |
| One agent calls a **stateless tool / API / database** | MCP — tool semantics, no agent loop on the other side |

## Quick start: client side

Call a remote A2A-compatible agent from your local agent:

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
        logger.info(result.text)

asyncio.run(main())
```

`A2AAgent` is a `BaseAgent` peer — sibling to `Agent`. It is **pure
config**: it carries the URL, timeout, interceptors, and other
remote-endpoint settings. Execution flows through `A2ARunner`. The
`async with` form is recommended so the underlying
`httpx.AsyncClient` gets closed cleanly. The result is a typed
`A2ARunResult` exposing `.text`, `.task_id`, `.context_id`.

### A2ARunner is for A2AAgent only

`A2ARunner` is the execution sibling of `A2AAgent`, the way `Runner`
is the execution sibling of local `Agent` / `Swarm` / `Graph`. The
two runners are **strictly partitioned**:

* `Runner.arun(agent, ...)` — local `Agent` (and `Runner.arun_swarm`,
  `Runner.arun_graph` for those primitives).
* `A2ARunner.arun(agent, ...)` — remote `A2AAgent`. **Only**
  `A2AAgent` instances. Passing a local `Agent`, `Swarm`, or `Graph`
  raises `TypeError` with a message pointing the caller back at
  `Runner`.

The reason for the split is that the wire format, lifecycle, and
error model of a remote A2A peer have nothing in common with a local
agent loop. Multiplexing them on a single runner would force every
entry point to discriminate at the type level for no benefit. A
type checker confirms the constraint at edit time; the runtime
guard catches the `Any`-typed boundary cases (dynamic dispatch
tables, JSON-loaded configs, untyped third-party callers).

### As an orchestration peer or as a tool — your choice

The dual surface lets you treat the remote agent either way:

**As a peer** (direct call via `A2ARunner.arun`):

```python
remote = A2AAgent(name="Researcher", url="https://research.example.com")
result = await A2ARunner.arun(remote, "Find recent papers on retrieval augmentation.")
```

**As a tool** (LLM invokes it mid-turn alongside your local tools):

```python
from troopai.adk.agents import Agent

remote = A2AAgent(name="Researcher", url="https://research.example.com")
local = Agent(
    name="Coordinator",
    system_prompt="Use the researcher tool when you need fresh information.",
    tools=[remote.as_tool(max_result_tokens=2_000)],
)
```

`as_tool()` mirrors `Agent.as_tool()` exactly. The wrapping
`FunctionTool` flows through the standard middleware / tracing /
hooks plumbing; the inner remote call is dispatched through
`A2ARunner.arun` (not `Runner.arun`) so the local-vs-remote
boundary stays clean.

## Quick start: server side

Expose any local `Agent` as an A2A endpoint:

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

# The ADK does NOT spawn a process — pick your own ASGI runtime:
uvicorn.run(app, host="0.0.0.0", port=8080)
```

The Starlette app exposes:

* `GET /.well-known/agent-card.json` — your AgentCard, the discovery
  contract every A2A client reads first.
* `POST /` — JSON-RPC dispatcher handling `send_message` (blocking
  and streaming), `get_task`, `cancel_task`, etc. SSE responses are
  emitted by the a2a-sdk for streaming methods.

The AgentCard is **manually authored** (Microsoft pattern) — every
field is intentional, no auto-derivation magic.

### Production warning: persistent task store

`A2AServer.task_store` defaults to `None`, in which case
`build_starlette_app` constructs an `InMemoryTaskStore`. This works
for development but **loses every task on process restart** —
breaking the `A2AContinuationToken` resume contract for any task
that outlives the process. Production deployments should pass a
persistent store:

```python
from a2a.server.tasks import DatabaseTaskStore

server = A2AServer(
    agent=local_agent,
    agent_card=card,
    task_store=DatabaseTaskStore(...),
)
```

`build_starlette_app` logs a `WARNING` when falling back to the
in-memory default so the choice is visible in deployment logs.

## Long-running tasks: the typed continuation token

For tasks that exceed an HTTP timeout, submit in the background and
poll later:

```python
from troopai.adk.a2a import A2AAgent, A2ARunner

async with A2AAgent(name="LongJob", url="https://jobs.example.com") as remote:
    token = await A2ARunner.arun(remote, "Crawl the entire archive.", background=True)
    # `token` is an A2AContinuationToken — JSON-serialisable, durable
    # across process restarts.

    # Hours later, possibly from a different process:
    status = await A2ARunner.poll_task(remote, token)
    if status.state == "completed":
        logger.info(status.result)
    elif status.state in ("failed", "rejected", "cancelled"):
        logger.warning("Task ended: %s: %s", status.state, status.message)
```

The `A2AContinuationToken` is a frozen dataclass with three fields:
`task_id`, `context_id`, `remote_url`. Persist it via
`dataclasses.asdict() + json.dumps()`, restore via
`A2AContinuationToken(**json.loads(payload))`.

## Streaming

Receive incremental updates as the remote agent works:

```python
chunks: list[str] = []
async for event in await A2ARunner.arun(remote, "Write a long essay.", stream=True):
    if event["type"] == "text_delta":
        chunks.append(event["text_delta"])
        logger.info("delta: %s", event["text_delta"])
    elif event["type"] == "completed":
        logger.info("Final text: %s", "".join(chunks))
        break
    elif event["type"] == "failed":
        logger.error("Failed: %s: %s", event["state"], event.get("message", ""))
        break
```

The stream is bounded by `A2AClient.max_stream_chunks` (default
10,000) and `max_stream_bytes` (default 8 MiB). A runaway remote
that streams forever raises `A2AProtocolError` rather than exhausting
memory.

## Authentication

Auth is plumbed through `a2a-sdk`'s `ClientCallInterceptor` —
the ADK does not invent a new auth surface:

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

Server-side auth is the responsibility of `a2a-sdk`'s
`DefaultRequestHandler` and any middleware you stack on the Starlette
app. The ADK does not add new auth code.

## Error handling

Every A2A failure surfaces as a typed exception under
`troopai.adk.a2a`:

| Exception | Cause |
|---|---|
| `A2ATransportError` | Network failure (connect, timeout, DNS, TLS) |
| `A2AProtocolError` | Malformed response, version mismatch, auth rejection |
| `A2ATaskError` | Remote task ended in `failed` or `rejected` state |
| `A2ATaskCancelledError` | Remote task was cancelled (subclass of `A2ATaskError`) |

All extend `A2AError`, which extends `TroopAIError` — catch the framework
root for any framework error including A2A.

> **Security note**: `A2ATaskError.remote_message` is **untrusted
> input** sourced from the peer agent. It MAY contain prompt-injection
> bait or escape sequences. Sanitise / escape before rendering to
> end-users — the snippet below routes it through `logger.warning`
> with structured args, but you MUST review your handler before
> exposing the message to a downstream LLM, UI, or log aggregator
> that might re-interpret it.

```python
from troopai.adk.a2a import A2ATaskError, A2ATransportError

try:
    result = await A2ARunner.arun(remote, "...")
except A2ATransportError:
    # network problem — retry?
    ...
except A2ATaskError as exc:
    # remote agent finished but reported failure
    logger.warning(
        "task %s ended in %s: %s",
        exc.task_id,
        exc.state,
        exc.remote_message,
    )
```

## Tracing

A2A calls automatically participate in OpenTelemetry tracing via
the existing `function_span` infrastructure. Both client-side and
server-side spans use the `a2a.<task_id>` naming convention; the
client span's `troopai.a2a.remote_url` attribute and the server
span's `troopai.a2a.agent_name` attribute let you correlate the two
sides of a network boundary.

The same secret-redaction that runs on tool I/O runs on the
`a2a_data` attributes — embedded credentials are masked before
leaving the process.

Install the OTel extra to enable: `pip install 'troopai-adk-python[otel]'`.

## See also

* `examples/a2a/` — runnable client + server examples
* The A2A spec: <https://a2a-protocol.org/>
* `src/troopai/adk/mcp/` — sibling integration for tool-level remote calls
