(deploy/scaling)=

# Horizontal Scaling

A single replica of a served agent works correctly out of the box. Adding
replicas requires a shared backend for any state that must survive across
replicas or process restarts.

## Single-replica default

The generated Kubernetes Deployment sets `replicas: 1`. A single replica:

- Runs the SQLite A2A task store entirely in-process (pass `--task-db` to
  `troopai serve` to make it durable across restarts).
- Uses any `SessionStore` the developer wires into the app (in-memory or
  SQLite-backed via the `session` extra).
- Satisfies the [container contract](container.md): binds `0.0.0.0:$PORT`,
  reads config from env, handles `SIGTERM` with a 45-second drain window,
  exposes `/healthz` and `/readyz`.

For many workloads — especially those where agent turns are short — a single
replica with a durable SQLite store is sufficient.

## Scaling to multiple replicas

When traffic requires more than one replica, two shared backends must be
configured before increasing the replica count. Both are provided by the ADK
and wired via `troopai serve` flags.

### A2A task store — `PostgresTaskStore`

`PostgresTaskStore` (`troopai.adk.a2a.postgres_task_store`) is a durable,
shared A2A executor task store backed by a PostgreSQL `a2a_tasks` table reached
through a `psycopg` async connection pool. Because every replica talks to the
same database, background task state and `A2AContinuationToken` resumption
survive across replicas — which a per-pod SQLite file cannot provide.

Wire it via `--task-dsn`:

```bash
troopai serve \
  --agent my_pkg.agents:assistant \
  --card card.json \
  --task-dsn "postgresql://user:pass@db-host/agents"
```

Install the required extras:

```bash
pip install 'troopai-adk-python[a2a,a2a-postgres]'
```

`PostgresTaskStore` also runs `recover_on_startup`: any task a prior process
left non-terminal is marked FAILED before the server accepts requests, so
clients receive a clean error and can resubmit.

:::{note}
`--task-dsn` wires the ADK's own executor task store (background task
snapshots for restart recovery and shared state across replicas). The a2a-sdk
request-handler's own task store is separate; for full A2A wire-protocol task
lookups across replicas an operator may additionally supply a shared a2a-sdk
store via `build_starlette_app`. REST sessions via `--session-dsn` are fully
shared with no further configuration.
:::

`--task-db` (SQLite file) and `--task-dsn` (Postgres DSN) are mutually
exclusive; choose one.

### Session store — `PostgresMultiSessions`

`PostgresMultiSessions` (`troopai.adk.session.postgres_multi_sessions`) is
the Postgres counterpart of `SQLiteMultiSessions`. It owns a `psycopg` async
connection pool and makes REST session history fully shared across replicas —
any replica can continue any conversation from the last persisted message.

Wire it via `--session-dsn`:

```bash
troopai serve \
  --agent my_pkg.agents:assistant \
  --session-dsn "postgresql://user:pass@db-host/agents"
```

Install the required extra:

```bash
pip install 'troopai-adk-python[session-postgres]'
```

`--session-db` (SQLite file) and `--session-dsn` (Postgres DSN) are mutually
exclusive; choose one.

### Combined multi-replica invocation

For a horizontally-scaled deployment, point both DSNs at the same Postgres
instance and set `replicas > 1`:

```bash
troopai serve \
  --agent my_pkg.agents:assistant \
  --card card.json \
  --task-dsn "$PG_DSN" \
  --session-dsn "$PG_DSN"
```

```bash
pip install 'troopai-adk-python[a2a,a2a-postgres,session-postgres]'
```

## Enabling the HPA

The generated Kubernetes Deployment includes a `HorizontalPodAutoscaler`
manifest set to `minReplicas: 1` and `maxReplicas: 4`. It is disabled (no
metrics target is configured) until the operator enables it.

To activate autoscaling once a shared backend is in place:

1. Configure `--task-dsn` and `--session-dsn` in the Deployment's container
   args as described above.
2. Set a CPU or custom metric target in `deploy/k8s/HPA.yaml`.
3. Apply the updated manifests:

```bash
kubectl apply -k deploy/k8s
```

A typical CPU-based target:

```yaml
# deploy/k8s/HPA.yaml
spec:
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

## Cloud Run and App Runner

Both Cloud Run and App Runner scale horizontally by default, managing replicas
automatically. The same shared-backend requirement applies: any stateful
resource (A2A task store, session store) must use a shared backend
when more than one instance can be active simultaneously. Use `--task-dsn` and
`--session-dsn` as described above.

For Cloud Run, set `--min-instances` to control cold-start behaviour:

```bash
troopai deploy cloud-run \
  --agent my_pkg.agents:assistant \
  --project my-project \
  --region us-central1 \
  --min-instances 1
```

## Drain window sizing

In-flight agent turns can outlast the default `SIGTERM` → kill window of
30 seconds (especially for multi-turn conversations, tool calls, and LLM
providers with high latency). The generated Kubernetes manifests set
`terminationGracePeriodSeconds: 45`. Increase this value for agents with
longer expected turn durations:

```yaml
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 120  # adjust for your agent's turn budget
```

Cloud Run allows configuring the request timeout and drain timeout via the
service configuration.

## See also

- [Container contract](container.md) — the universal image contract
- [Kubernetes and Helm](kubernetes.md) — Kustomize manifests and HPA
- [Serving layer](serving.md) — `troopai serve` flag reference
- [A2A guide](../a2a/a2a.md#production-warning-persistent-task-store) — persistent task store setup
