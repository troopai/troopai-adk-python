# A2A Module

Agent-to-Agent (A2A) protocol — peer client and server-side
exposure of local Agents over HTTP+SSE. Optional extra: install with
`pip install 'troopai-adk-python[a2a]'`.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Soft-import guard mirroring `mcp/__init__.py`. Public re-exports of every framework-typed symbol. |
| `exceptions.py` | `A2AError` hierarchy under `TroopAIError` — `A2ATransportError`, `A2AProtocolError`, `A2ATaskError`, `A2ATaskCancelledError`. |
| `a2a_continuation_token.py` | `A2AContinuationToken` (typed handle for a long-running remote task) + `A2ATaskStatus` (poll snapshot) + `A2ATaskStateLiteral` enum. |
| `converters.py` | Bidirectional conversion between A2A protobuf wire types and framework-typed pieces. The single boundary where `a2a.types` crosses into the framework. |
| `a2a_client.py` | `A2AClient` — framework-typed wrapper over `a2a.client`. Owns the httpx lifecycle, surfaces `A2A*Error` for typed branching, opens a `function_span` per call. |
| `a2a_agent.py` | `A2AAgent(BaseAgent)` — peer-agent class. **Pure config**: no execution methods. Carries URL, timeout, interceptors, optional `a2a.client.ClientConfig`, framework-side streaming bounds, plus `as_tool()` mirror of `Agent.as_tool()`. |
| `a2a_runner.py` | `A2ARunner` — execution entry point for `A2AAgent`. Pure namespace of `@classmethod`s (`arun`, `poll_task`, `cancel_task`). Accepts ONLY `A2AAgent` instances — local primitives raise `TypeError`. |
| `server.py` | `A2AServer` frozen-dataclass config object. |
| `app_factory.py` | `build_starlette_app(server)` — Starlette routes from the config. |
| `executor.py` | `A2AExecutor(AgentExecutor)` — bridges A2A task execution into `Runner.arun()`. |
| `adapters.py` | `A2AExecutableAdapter(Executable)` — wraps `A2AAgent` so it slots into a `Graph` node alongside local Agent / Swarm / callable nodes. |

## Architectural Decisions

| # | Decision | Why |
|---|---|---|
| 1 | `A2AAgent` extends `BaseAgent`, not `Agent` and not `BaseTool` | Sibling of local `Agent` — both peers under `BaseAgent`. SRP: a remote agent is a different kind of orchestration entity than a local one. The `as_tool()` method gives the tool-shaped surface when needed. |
| 2 | `A2ARunner` is the execution entry point — `A2AAgent` is config only | Mirrors `Runner` ↔ `Agent` / `Swarm` / `Graph`. `A2ARunner.arun(agent, ...)` accepts ONLY `A2AAgent` instances; local primitives raise `TypeError` at runtime (with a static type-checker rejection at edit time). `A2ARunner` does not enter `Runner.arun` — the wire format, lifecycle, and error model differ enough that multiplexing the two on one runner buys nothing. |
| 3 | `a2a.types` (protobuf) confined to `converters.py`, `a2a_client.py`, `executor.py` | Same discipline as `llms/litellm/` confines `litellm.types`. Three-layer rule: wire types never appear in `agents/`, `run/`, `tools/`, or any developer-facing surface. |
| 4 | `A2AContinuationToken` is the typed long-running-task primitive | JSON-serialisable frozen dataclass — durable across process restarts. Caller passes `background=True` to `A2ARunner.arun(agent, prompt, ...)`, gets the token, polls or resumes via `A2ARunner.poll_task(agent, token)` / `A2ARunner.arun(agent, prompt, continuation_token=token)`. |
| 5 | Server is a config object + factory, not a running server | `A2AServer` is frozen config; `build_starlette_app(server)` returns a Starlette app the developer's own `uvicorn.run` serves. The ADK does not own the ASGI lifecycle. Matches "Agent = config, Runner = execution" rule. |
| 6 | A2A spans reuse `FunctionSpanData.a2a_data` — no new span kind | Same approach as MCP (decision #6 in `tracing/CLAUDE.md`). The OTel bridge name-switches `tool.<n>` → `a2a.<n>` based on `a2a_data` presence. Zero new entries on the `Tracer` Protocol. |
| 7 | a2a-sdk is an optional extra | The core framework has zero runtime dependency on `a2a`. Soft imports raise `ImportError` with the install command when the extra is missing. |
| 8 | Manual `AgentCard` — no auto-derivation | Microsoft pattern. Every field in the card is intentional. Auto-derivation hides what the LLM-on-the-other-side will see. |

## Pointers

- `docs/a2a/a2a.md` — usage walkthrough, client + server quick-starts, auth, streaming, long-running tasks.
- `examples/a2a/` — runnable client + server examples.

## Status

Shipped:

- Client side — `A2AAgent`, `A2AClient`, `A2AContinuationToken`, `converters`, tracing extension.
- Server side — `A2AServer`, `A2AExecutor`, `build_starlette_app`, integration test, docs, examples.
- Graph-node adapter (`A2AExecutableAdapter`) — `A2AAgent` slots into a `Graph` node alongside local primitives. `to_executable()` auto-dispatches.

Not yet implemented: W3C trace propagation through `ClientCallInterceptor` so client + server spans share a trace ID; HITL `TASK_STATE_INPUT_REQUIRED` round-trip via persistent `TaskStore`.
