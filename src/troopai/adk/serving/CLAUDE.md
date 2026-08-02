# Serving Module

HTTP serving layer — turns a local `Agent` into an ASGI app the
developer's own runtime serves. Sibling to `a2a/`: this package owns the
provider-agnostic surfaces (plain REST + health), and mounts the A2A
JSON-RPC + discovery routes by delegating to `a2a.build_starlette_app`.
Optional extra: `pip install 'troopai-adk-python[server]'` (Starlette +
sse-starlette).

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Soft-import guard mirroring `a2a/`/`mcp/`. Public names are `None` when the `server` extra is absent (branch on `build_app is None`). |
| `app.py` | `build_app(agent, *, rest=, a2a_server=, health=, …)` — composes the enabled surfaces into one `Starlette` app. Every surface is OFF by default. |
| `rest.py` | `rest_routes(agent, …)` — `POST /run` (collect → JSON) and `POST /run_sse` (SSE). Wraps `Runner.arun`. `SessionFactory` seam for per-request sessions. |
| `health.py` | `health_routes(...)` — `GET /healthz` (liveness) + `GET /readyz` (readiness, optional async `ReadinessProbe`). |
| `serializers.py` | `run_result_to_dict` / `streaming_result_to_dict` / `stream_event_to_dict` / `usage_to_dict`. The single Layer-3 → JSON boundary. |

## Architectural decisions

| # | Decision | Why |
|---|---|---|
| 1 | Surfaces are opt-in; `build_app` defaults every one OFF and raises if none is enabled | No-implicit-behavior / cost-conservative invariant — the developer never serves a route they did not request. |
| 2 | Serialize Layer-1 / Layer-3 only, via `ItemHelpers.run_items_to_params()` | Reuses the framework's existing replay path; the provider wire format (Layer 2) never crosses the HTTP boundary. |
| 3 | The framework never imports a server runtime | `build_app` returns the `Starlette` app; the `serve` CLI (and only it) owns `uvicorn`. Mirrors `a2a` decision #5 (config vs. execution). |
| 4 | `SessionFactory = Callable[[str, str], SessionStore]` injected by the caller | The REST layer never hard-codes a session backend; app-name + storage are bound by whoever wires the factory (the `serve` command). |
| 5 | A2A routes are mounted by delegating to `a2a.build_starlette_app` | No duplication of the A2A surface; the `a2a` extra is required only when an `a2a_server` is passed (lazy import in `_a2a_routes`). |
| 6 | `final_output` is coerced defensively (`str()` fallback) | It carries the agent's arbitrary `output_type` — a genuinely dynamic value, not a closed union — so the response always stays valid JSON. |

## Container contract

The generated container binds `0.0.0.0:$PORT` explicitly via the image
CMD (the `serve` default stays `127.0.0.1:8000`). `/healthz` + `/readyz`
back Kubernetes probes, Cloud Run startup checks, and load-balancer
health checks. SSE (`/run_sse`) needs the runtime to keep the connection
open; size drain windows for in-flight agent turns.

## Testing

`tests/unit/serving/` — network-free `ScriptedLLM` agent + Starlette
`TestClient`. `conftest.py` exposes `scripted_agent` / `streaming_agent`;
no port is ever bound.
