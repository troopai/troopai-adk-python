# Sandbox Module

Sandbox agents — Agent + isolated execution environment.

## Files

| Path | Purpose |
|---|---|
| `agent.py` | `SandboxAgent` dataclass (Agent subclass) |
| `config.py` | `SandboxRunConfig` on `RunConfig.sandbox` |
| `capabilities/` | `SandboxCapability` base + concrete capabilities |
| `clients/` | `BaseSandboxClient` ABC + concrete backends |
| `tools/` | Capability-bound `FunctionTool`s |
| `guardrails/` | `SandboxCommandGuardrail` typed verdicts |
| `policy/` | NetworkPolicy + ResourceLimits backend translation |
| `observability/` | `AuditSink` + `sandbox_span`; `observability.py` carries `SandboxObservability`, the run-scoped emission handle wired through `run_command` |
| `snapshot/` | `SnapshotStore` ABC + Local/S3/GCS impls |
| `selector.py` | `SandboxSelector` ABC + `CheapestFirstSelector` — cost-aware backend selection |
| `session/` | Session orchestration helpers + manifest-entry materialization |
| `runner_integration/` | Per-run lifecycle helpers |

## Key Architectural Decisions

- **Agent = config**: `SandboxAgent` extends `Agent` but stays
  config-only; the Runner brackets the loop with a sandbox session.
- **Cost-conservative**: `Capabilities.default()` returns only
  `[CompactionCapability()]`. Shell + Filesystem are opt-in.
- **Layer-1 types live under `troopai.adk.types.sandbox`**; runtime
  / capabilities / clients live under `troopai.adk.sandbox`.
- **`RemoteVMSandboxClient`** is an intermediate base shared by every
  hosted bridge (E2B, Vercel, Modal, Daytona, Cloudflare, Blaxel,
  Runloop). New hosted providers cost ~150 LoC.

See `docs/sandbox/index.md` for usage.
