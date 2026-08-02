(guides/sandbox)=

# 🪣 Sandbox

A sandbox wraps every tool execution inside an isolated environment so that
an agent's shell commands, file writes, and network calls cannot affect the
host process or other tenants. This guide explains why sandboxing matters,
what backends are available, how the framework picks one, and how you wire
everything together.

---

## Why sandboxes

When an agent calls a shell tool, the framework has no way to know in
advance what that tool will do. A single command can delete files, open
network connections, install packages, or fork new processes. Without
isolation, a mistake or a prompt injection that reaches the shell tool has
unbounded blast radius — it can touch anything the host process can touch.

A sandbox draws a hard boundary. All execution inside it is subject to:

- **Network policy** — allowlist which hosts and ports are reachable.
- **Resource limits** — CPU cores, memory, wall-clock time, process count, disk.
- **Workspace isolation** — the agent sees only what the manifest declares.
- **Audit trail** — every command is emitted to an `AuditSink` before and
  after it runs.

The sandbox layer is the last line of containment. Guardrails decide whether
a command is *permitted*; the sandbox decides *where* it runs and what it can
reach while it runs. Those are orthogonal concerns — see
{ref}`guides/sandbox` § *Tool permissions vs sandboxing* below.

---

## Two backend families

The framework ships two families of sandbox backend.

**Local backends** run on the same machine as the framework process. The
framework speaks directly to the Docker daemon or the Kubernetes API server.
Network round-trips are sub-millisecond; cold-start for a cached image is
roughly one second. Local backends are ideal for CI pipelines, on-premises
deployments, and any environment where you own the infrastructure.

**Hosted bridges** forward execution to a remote sandbox service over HTTPS.
The framework speaks a thin REST dialect; each vendor translates it to their
own substrate. You pay compute costs per minute but avoid managing
infrastructure entirely. Hosted bridges are the right default for cloud
deployments and for agents that need more resources than a single machine can
provide.

| Property | Local (Docker / K8s) | Hosted bridges |
|---|---|---|
| Cold-start latency | < 2 s (cached image) | 2–10 s |
| Infrastructure ops | You manage the daemon / cluster | Vendor manages |
| Persistent workspace | Yes (volumes) | Provider-dependent |
| Network isolation | Linux namespaces | Provider-dependent |
| Cost model | Infrastructure cost only | Per-minute metered |
| Offline capable | Yes | No |

---

## `SandboxSelector` ABC

When you supply more than one candidate backend, the framework delegates the
choice to a `SandboxSelector`. The selector receives the list of
`SandboxCandidate` objects (each pairing a `BaseSandboxClient` with its
options) and a `SandboxRequirements` struct that describes what the run
needs (network, persistence, custom capabilities).

```python
import abc

class SandboxSelector(abc.ABC):
    @abc.abstractmethod
    def select(
        self,
        candidates: list[SandboxCandidate],
        requirements: SandboxRequirements,
    ) -> SandboxCandidate: ...
```

The built-in `CheapestFirstSelector` filters the list to candidates whose
`capabilities` satisfy `requirements`, then returns the one with the lowest
`cost.usd_per_minute`. Unpriced backends (local Docker, K8s) sort after all
priced ones so hosted bridges are preferred only when you explicitly price
them lower.

```python
from troopai.adk.sandbox.selector import CheapestFirstSelector, SandboxCandidate
from troopai.adk.sandbox.clients.hosted.e2b import E2bSandboxClient, E2bSandboxClientOptions
from troopai.adk.sandbox.clients.docker import DockerSandboxClient, DockerSandboxClientOptions

selector = CheapestFirstSelector()
candidates = [
    SandboxCandidate(
        client=DockerSandboxClient(),
        options=DockerSandboxClientOptions(image="python:3.12-slim"),
    ),
    SandboxCandidate(
        client=E2bSandboxClient(),
        options=E2bSandboxClientOptions(api_key="..."),
    ),
]
```

Wire the selector into `SandboxRunConfig`:

```python
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.types.sandbox.cost import SandboxRequirements

config = SandboxRunConfig(
    selector=selector,
    candidates=candidates,
    requirements=SandboxRequirements(network=True),
)
```

See {doc}`../sandbox/selection` for the full cost-aware selection
documentation.

---

## Local backends

### `DockerSandboxClient`

Spawns one long-lived container per session using `sleep infinity` as PID 1,
then drives workloads with `docker exec`. The container is removed when the
session closes.

```python
from troopai.adk.sandbox.clients.docker import DockerSandboxClient, DockerSandboxClientOptions
from troopai.adk.sandbox.config import SandboxRunConfig

client = DockerSandboxClient()
config = SandboxRunConfig(
    client=client,
    options=DockerSandboxClientOptions(
        image="python:3.12-slim",
        memory_mb=512,
        cpu_count=1.0,
        working_directory="/workspace",
    ),
)
```

Key options:

- `image` — Docker image (required).
- `memory_mb` / `cpu_count` / `pid_limit` — resource caps translated to
  Docker's `mem_limit` / `nano_cpus` / `pids_limit`.
- `network_policy` — `NetworkPolicy` translated to Docker network arguments.
- `environment` — environment variables injected at container start.

Install the optional extra: `pip install 'troopai-adk-python[sandbox-docker]'`.

**When to use**: local development, CI pipelines, single-machine deployments.
`DockerSandboxClient` is the recommended local backend for production
workloads where you control the Docker daemon.

### `K8sPodSandboxClient`

Spawns an ephemeral Pod per session. `NetworkPolicy` objects are translated
to Kubernetes `NetworkPolicy` custom resources; resource limits become
`ResourceQuota` constraints.

```python
from troopai.adk.sandbox.clients.k8s import K8sPodSandboxClient, K8sSandboxClientOptions
from troopai.adk.sandbox.config import SandboxRunConfig

client = K8sPodSandboxClient()
config = SandboxRunConfig(
    client=client,
    options=K8sSandboxClientOptions(
        image="python:3.12-slim",
        namespace="agents",
        service_account="sandbox-runner",
    ),
)
```

Key options:

- `image` — container image (required).
- `namespace` — Kubernetes namespace (default `"default"`).
- `service_account` — `serviceAccountName` for the pod; `None` uses the
  namespace default.

Install the optional extra: `pip install 'troopai-adk-python[sandbox-k8s]'`.

**When to use**: multi-tenant workloads where you need Kubernetes scheduling,
autoscaling, and namespace-scoped RBAC. K8s PodSecurity admission labels are
applied at the namespace level; the pod itself is created with the
`restricted` profile by default.

### `LocalSubprocessSandboxClient`

Runs commands as child processes of the host Python process inside a
temporary working directory.

```{warning}
`LocalSubprocessSandboxClient` provides **no isolation**. Commands run as
the same user and with the same filesystem access as the framework process.
Use only for local development and examples. Production deployments must
use `DockerSandboxClient`, `K8sPodSandboxClient`, or a hosted bridge.
```

```python
from troopai.adk.sandbox.clients.local import (
    LocalSubprocessSandboxClient,
    LocalSandboxClientOptions,
)
from troopai.adk.sandbox.config import SandboxRunConfig

client = LocalSubprocessSandboxClient()
config = SandboxRunConfig(
    client=client,
    options=LocalSandboxClientOptions(working_directory="/tmp/agent-workspace"),
)
```

The client logs a `WARNING` banner on construction so the lack of isolation
is visible in every trace and audit log.

---

## Hosted bridges

Each hosted bridge is a thin REST client that wraps one SaaS sandbox
provider. All seven inherit from `RemoteVMSandboxClient`, which provides
shared HTTP construction, retry logic, and error mapping. Every bridge uses
httpx under the hood; the `[sandbox-hosted]` extra installs it.

The table below summarises the bridges, their `backend_id`, the rate the
`CheapestFirstSelector` uses for ranking, and whether the backend declares
a persistent workspace:

| Bridge class | `backend_id` | Rate (USD/min) | Persistent | Key auth field |
|---|---|---|---|---|
| `E2bSandboxClient` | `e2b` | 0.06 | Yes | `api_key` |
| `ModalSandboxClient` | `modal` | 0.10 | Yes | `api_key` |
| `DaytonaSandboxClient` | `daytona` | 0.08 | Yes | `api_key` |
| `VercelSandboxClient` | `vercel` | 0.12 | No | `api_key` |
| `CloudflareSandboxClient` | `cloudflare` | 0.05 | No | `api_key` |
| `BlaxelSandboxClient` | `blaxel` | 0.09 | Yes | `api_key` |
| `RunloopSandboxClient` | `runloop` | 0.10 | Yes | `api_key` |

Rates shown are the static defaults in the source; override the class-level
`cost` attribute when actual provider pricing differs.

### Configuration per backend

Every options class inherits the shared `RemoteVMSandboxClientOptions` base,
which provides `base_url`, `max_retries`, and `request_timeout`. Each bridge
then adds provider-specific fields:

```python
from troopai.adk.sandbox.clients.hosted.e2b import E2bSandboxClient, E2bSandboxClientOptions

# E2B — template selects the sandbox environment image
options = E2bSandboxClientOptions(
    api_key="e2b_...",
    template_id="python-data-science",
    region="us-east-1",      # optional
)

from troopai.adk.sandbox.clients.hosted.modal import ModalSandboxClient, ModalSandboxClientOptions

# Modal — app_name + environment_name scope the sandbox
options = ModalSandboxClientOptions(
    api_key="ak-...",
    app_name="my-agent-app",
    environment_name="main",
    image="python:3.12-slim",
)

from troopai.adk.sandbox.clients.hosted.cloudflare import (
    CloudflareSandboxClient,
    CloudflareSandboxClientOptions,
)

# Cloudflare — account_id scopes billing
options = CloudflareSandboxClientOptions(
    api_key="cf_...",
    account_id="abc123",
)

from troopai.adk.sandbox.clients.hosted.runloop import RunloopSandboxClient, RunloopSandboxClientOptions

# Runloop — blueprint_id selects the sandbox image
options = RunloopSandboxClientOptions(
    api_key="rl_...",
    blueprint_id="bp_python312",
)
```

The remaining bridges (`DaytonaSandboxClientOptions`, `VercelSandboxClientOptions`,
`BlaxelSandboxClientOptions`) follow the same pattern: `api_key` plus one or
two provider-specific identifiers (`workspace_id`, `project_id` / `team_id`,
`workspace`).

---

## `SnapshotStore`

A `SnapshotStore` persists and restores sandbox workspace state between runs.
The framework calls `BaseSandboxSession.persist_workspace()` at the end of a
run to produce a tar stream, then hands it to the store's `save()` method.
On the next run, `BaseSandboxClient.create()` receives the `SnapshotSpec` and
asks the store for the bytes, then restores the workspace before the agent
loop starts.

```python
class SnapshotStore(abc.ABC):
    async def save(
        self, *, snapshot_id: str, data: IOBase, manifest_hash: str | None = None
    ) -> SnapshotMetadata: ...
    async def load(self, ref: SnapshotRef) -> IOBase: ...
    async def delete(self, ref: SnapshotRef) -> None: ...
    async def list(self, prefix: str | None = None) -> list[SnapshotMetadata]: ...
    async def exists(self, ref: SnapshotRef) -> bool: ...
```

Three concrete implementations ship out of the box:

- **`LocalSnapshotStore`** — writes tar files under a local directory.
  Suitable for single-machine development or CI.
- **`S3SnapshotStore`** — reads and writes snapshots from an S3-compatible
  bucket. Requires the `[sandbox-s3]` extra.
- **`GCSSnapshotStore`** — reads and writes snapshots from Google Cloud
  Storage. Requires the `[sandbox-gcs]` extra.

```{note}
Not all backends support `snapshot_store`. Docker, K8s, and the hosted
bridges raise `UnsupportedSnapshotFeatureError` if you pass a non-`None`
store today. The session-level `persist_workspace` / restore path works
independently of the store; the store is the *persistence* layer for
cross-run restore.
```

Wire the store via `SandboxRunConfig`:

```python
from troopai.adk.sandbox.snapshot import LocalSnapshotStore
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.types.sandbox.snapshot import SnapshotSpec

store = LocalSnapshotStore(base_dir="/var/troopai/snapshots")
config = SandboxRunConfig(
    client=client,
    options=options,
    snapshot_store=store,
    snapshot=SnapshotSpec(snapshot_id="run-42"),
)
```

See {doc}`../sandbox/snapshots` for the full snapshot documentation.

---

## `RunHooks` — sandbox lifecycle callbacks

`RunHooks` exposes four sandbox-specific callbacks that fire around every
sandboxed run. Subclass `RunHooks` and override only the callbacks you need:

```python
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.context import RunContext
from troopai.adk.types.sandbox.usage import SandboxUsage
from troopai.adk.types.sandbox.exec_result import ExecResult

class MySandboxHooks(RunHooks):
    async def on_sandbox_start(self, context, agent, session) -> None:
        """Fires once when the sandbox session is acquired."""
        ...

    async def on_sandbox_stop(self, context, agent, session, usage: SandboxUsage) -> None:
        """Fires once when the session is released. usage carries wall-clock
        duration and, if capture_live_cost=True, the provider-reported cost."""
        ...

    async def on_sandbox_exec_start(self, context, agent, command: str) -> None:
        """Fires before each non-PTY command."""
        ...

    async def on_sandbox_exec_end(self, context, agent, command: str, result: ExecResult) -> None:
        """Fires after each non-PTY command. Non-zero exit codes surface here,
        not as exceptions — the hook is observation only."""
        ...
```

Pass the hooks instance to `Runner`:

```python
from troopai.adk.run.runner import Runner

runner = Runner(agent=agent, hooks=MySandboxHooks())
```

The `on_sandbox_snapshot` callback (also on `RunHooks`) fires after a
successful snapshot save and is the right place to log snapshot metadata or
trigger an external notification.

---

## Tool permissions vs sandboxing

Tool permissions and sandboxing are **orthogonal** — they answer different
questions.

**Tool permissions** (the `allowed_tools` allowlist on an agent or runner)
answer: *"Is this tool allowed to run at all?"* A tool not in the allowlist
is never invoked regardless of the sandbox backend.

**Sandboxing** answers: *"In which environment does an allowed tool execute,
and what can it reach?"* Even if a tool is permitted, the sandbox can
constrain its network access, CPU, memory, and filesystem visibility.

Both checks happen before the tool executes:

```
LLM tool call
    │
    ▼
  Allowed-tools check  ── rejected? → ToolNotAllowedError
    │
    ▼
  Sandbox environment  ── command policy? → SandboxCommandGuardrail
    │
    ▼
  Execution inside sandbox
```

Because the concerns are separate you can, for example, allow all shell
tools on a development agent but confine them to an unnetworked Docker
container, while a production agent has a tighter tool allowlist *and*
a networked hosted bridge for the subset of commands that need it.

See {doc}`../permissions/permissions` for the tool-permission model.

---

## Common patterns

### Sandbox-by-default with per-tool opt-out

Attach a sandbox to every run via `RunConfig.sandbox` and let agents that
do not need isolation opt out by omitting the sandbox field:

```python
from troopai.adk.run.config import RunConfig
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.clients.docker import DockerSandboxClient, DockerSandboxClientOptions

sandboxed_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=DockerSandboxClient(),
        options=DockerSandboxClientOptions(image="python:3.12-slim"),
    )
)

# Unsafe agent that should never touch the host filesystem
await runner.arun(input="...", run_config=sandboxed_config)

# Coordination agent with no shell tools — no sandbox needed
await coordinator_runner.arun(input="...")
```

### Tenant-aware backend selection

In multi-tenant deployments you may want each tenant to run in a dedicated
namespace or project. Build the `SandboxRunConfig` dynamically:

```python
def make_sandbox_config(tenant_id: str) -> SandboxRunConfig:
    return SandboxRunConfig(
        client=K8sPodSandboxClient(),
        options=K8sSandboxClientOptions(
            image="python:3.12-slim",
            namespace=f"tenant-{tenant_id}",
        ),
    )
```

Pair this with the per-tenant task-queue routing documented in
{doc}`../architecture/governance` when you need Temporal-level
tenant isolation alongside sandbox isolation.

### Cost-aware backend selection

Cheap local backends handle the bulk of triage work; expensive hosted
bridges are reserved for heavy lifting. `CheapestFirstSelector` implements
this automatically when you supply a mixed candidate list and a
`SandboxRequirements` that matches only what each job needs:

```python
from troopai.adk.types.sandbox.cost import SandboxRequirements

# Triage run — no network, any backend
triage_config = SandboxRunConfig(
    selector=CheapestFirstSelector(),
    candidates=candidates,           # Docker first, then hosted
    requirements=SandboxRequirements(network=False),
)

# Heavy lifting — network required, pick cheapest qualifying hosted bridge
heavy_config = SandboxRunConfig(
    selector=CheapestFirstSelector(),
    candidates=candidates,
    requirements=SandboxRequirements(network=True, persistent=True),
)
```

`CheapestFirstSelector` will route triage jobs to the unpriced local
Docker backend and heavy jobs to whichever hosted bridge has the lowest
`usd_per_minute` rate while satisfying the requirements.

---

## See also

- {doc}`../architecture/governance` — per-tenant Temporal task queue
  routing and other cross-cutting governance invariants.
- {doc}`../concepts/index` — the Guardrails vs Middleware vs Hooks vs Sandbox
  comparison table.
- {doc}`../architecture/governance` — framework governance and the
  non-negotiable architectural invariants.
- {doc}`../sandbox/selection` — full cost-aware selection documentation.
- {doc}`../sandbox/snapshots` — full snapshot and `SnapshotStore` documentation.
