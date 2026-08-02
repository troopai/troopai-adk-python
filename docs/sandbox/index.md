# Sandbox Agents

Sandbox agents pair an ADK Agent with an isolated execution
environment — filesystem, shell, mounted storage, exposed ports,
snapshots — so the model can manipulate real files and run real
commands inside a controlled boundary.

This index links to the per-topic documentation.

| Topic | Document |
|---|---|
| Types | [types.md](types.md) |
| Capabilities | [capabilities.md](capabilities.md) |
| Clients (backends) | [clients.md](clients.md) |
| Selection (cost-aware) | [selection.md](selection.md) |
| Cost & billing | [cost.md](cost.md) |
| Security policy | [security.md](security.md) |
| Observability | [observability.md](observability.md) |
| Snapshots | [snapshots.md](snapshots.md) |
| Manifest materialization | [manifest-materialization.md](manifest-materialization.md) |
| IaC integration | [iac.md](iac.md) |
| Runner integration | [runner_integration.md](runner_integration.md) |

## Quickstart

```python
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.clients.local import LocalSubprocessSandboxClient
from troopai.adk.sandbox.config import SandboxRunConfig

agent = SandboxAgent(
    name="coder",
    system_prompt="You are a sandboxed coder.",
    capabilities=[ShellCapability()],
)

client = LocalSubprocessSandboxClient()
run_config = RunConfig(sandbox=SandboxRunConfig(client=client))
result = await Runner.arun(agent, "List files in /tmp", run_config=run_config)
```

## Architecture

The Runner detects `isinstance(agent, SandboxAgent)` (or non-None
`RunConfig.sandbox`) and brackets the agent loop with a
`sandbox_run_context` that:

1. Acquires a per-agent `SandboxConcurrencyGuard`.
2. Validates capability dependency requirements.
3. Clones capabilities for per-run isolation.
4. Resolves the session by priority: explicit session → session_state
   resume → client.create with manifest → selector picks from candidates.
5. Binds session + run_as on every cloned capability.
6. Folds the manifest through `process_manifest`.
7. Calls `session.start()` for runner-owned sessions.
8. Yields the lifecycle handle for the rest of the agent loop.
9. On exit, calls `session.aclose()` (for runner-owned) and releases
   the guard.

See [runner_integration.md](runner_integration.md) for detail.

```{toctree}
:hidden:
:maxdepth: 1

types
capabilities
clients
selection
cost
security
observability
snapshots
manifest-materialization
iac
runner_integration
```
