# Sandbox Clients (Backends)

A `BaseSandboxClient[ClientOptionsT]` builds a `BaseSandboxSession`
per run; the session is the live handle the Runner brackets the
agent loop with.

## Built-in backends

| Backend | Class | Extra | Best for |
|---|---|---|---|
| Local subprocess | `LocalSubprocessSandboxClient` | (none) | Dev / CI; **no isolation**, banner WARN logged |
| Docker container | `DockerSandboxClient` | `[sandbox-docker]` | Local container isolation, resource limits, network_mode |
| Kubernetes pod | `K8sPodSandboxClient` | `[sandbox-k8s]` | Cluster-scale isolation, NetworkPolicy CR, restricted PSS |
| Hosted REST bridge | `RemoteVMSandboxClient` + 7 concrete bridges | `[sandbox-remote-vm]` + per-provider | E2B, Vercel, Modal, Daytona, Cloudflare, Blaxel, Runloop |

## Hosted bridges

Each hosted bridge is ~140 LoC and shares the `RemoteVMSandboxClient`
intermediate base for HTTP retry / error mapping / `RemoteVMSandboxSession`
wire-up. Bridges expose REST-only paths by default; users wanting a
provider's official Python SDK can subclass and override `create()` /
`resume()`.

| Provider | Class | Extra | Create endpoint |
|---|---|---|---|
| E2B | `E2bSandboxClient` | `[sandbox-e2b]` | `POST /sandboxes` |
| Vercel | `VercelSandboxClient` | `[sandbox-vercel]` | `POST /v1/sandboxes` |
| Modal | `ModalSandboxClient` | `[sandbox-modal]` | `POST /v1/sandboxes` |
| Daytona | `DaytonaSandboxClient` | `[sandbox-daytona]` | `POST /sandboxes` |
| Cloudflare | `CloudflareSandboxClient` | `[sandbox-cloudflare]` | `POST /sandboxes` |
| Blaxel | `BlaxelSandboxClient` | `[sandbox-blaxel]` | `POST /sandboxes` |
| Runloop | `RunloopSandboxClient` | `[sandbox-runloop]` | `POST /v1/devboxes` |

## Session surface

`BaseSandboxSession` covers:

- **Lifecycle**: `start`, `stop`, `shutdown`, `aclose`, plus
  `async with session: ...` ctx-manager support.
- **Run-a-command primitive**: `async run(*command, timeout, shell,
  user) -> ExecResult`.
- **PTY family**: `pty_start`, `pty_write_stdin`, `pty_terminate_all`
  (Docker drives `exec_run(socket=True, tty=True)`; K8s drives
  `kubernetes.stream(..., tty=True)`).
- **File ops**: `read`, `write`, `ls`, `rm`, `mkdir`, `extract`.
- **Workspace**: `persist_workspace`, `hydrate_workspace`,
  `apply_manifest`, `apply_patch`.
- **Network**: `resolve_exposed_port(port) -> ExposedPortEndpoint`.
- **Capability flags**: `supports_docker_volume_mounts`,
  `supports_pty`.

See `examples/sandbox/docker_shell_agent.py`, `k8s_pod_agent.py`,
`e2b_hosted_agent.py` for end-to-end runs.
