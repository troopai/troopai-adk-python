(deploy/kubernetes)=

# Kubernetes and Helm

Two targets ship to Kubernetes: `k8s` uses Kustomize and raw kubectl;
`helm` renders a Helm chart and installs it with `helm upgrade --install`.
Both generate the same container artifacts (Dockerfile, `.dockerignore`,
`requirements.txt`) plus the orchestration layer specific to each target.

## Prerequisites

| Target | Required CLIs |
|--------|--------------|
| `k8s` | `kubectl` |
| `gke` | `gcloud`, `docker`, `kubectl` |
| `helm` | `helm` |

The CLIs must be installed by the operator; `troopai deploy` shells out to
them and imports no cloud SDK.

---

## `troopai deploy k8s`

Generate Kustomize manifests and apply them to the current kubeconfig context.

### Step 1 — generate artifacts

```bash
troopai deploy init \
  --target k8s \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --env-key OPENAI_API_KEY
```

This writes into `deploy/k8s/`:

| File | Purpose |
|------|---------|
| `Deployment.yaml` | Deployment with startup / readiness / liveness probes, resource limits, 45s grace period |
| `Service.yaml` | ClusterIP service |
| `HPA.yaml` | HorizontalPodAutoscaler (disabled by default; see [Scaling](scaling.md)) |
| `ConfigMap.yaml` | Non-secret config (e.g. `AGENT_REF`, `PORT`) |
| `Secret.yaml` | Example Secret with your `--env-key` names as keys |
| `kustomization.yaml` | Kustomize overlay wiring the above |

Edit `deploy/k8s/Secret.yaml` to populate the actual secret values, or
replace it with a reference to your cluster's secret management solution
before applying.

### Step 2 — build and push the image

```bash
troopai deploy build \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --push
```

### Step 3 — apply

```bash
troopai deploy k8s \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --no-generate
```

`--no-generate` skips writing new artifacts and applies what is already on
disk. Omit it to regenerate and apply in one step.

Pass `--context` to target a specific kubeconfig context:

```bash
troopai deploy k8s \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --context my-prod-cluster \
  --no-generate
```

### Reference: k8s flags

Shared flags (`--agent`, `--image`, `--app-name`, `--port`, `--extras`,
`--env-key`) apply to all `deploy` subcommands and are documented in the
[flag reference](#shared-flags) below.

| Flag | Default | Description |
|------|---------|-------------|
| `--context TEXT` | current context | kubeconfig context to target |
| `--dir PATH` | `.` | Directory holding or to receive the manifests |
| `--no-generate` | off | Use manifests already on disk; do not regenerate |

### Probes

The generated Deployment maps the health endpoints to Kubernetes probes:

| Probe | Endpoint | Interval | Purpose |
|-------|----------|----------|---------|
| `startupProbe` | `GET /healthz` | every 5s, 30 attempts | Allow slow startup before liveness kicks in |
| `readinessProbe` | `GET /readyz` | every 10s | Gate traffic until the agent is fully ready |
| `livenessProbe` | `GET /healthz` | every 15s | Restart the pod if the process hangs |

### Resource limits

The generated Deployment requests 250m CPU and 512 Mi memory, with limits of
1 CPU and 1 Gi memory. Adjust these for your agent's actual profile before
going to production.

---

## `troopai deploy gke`

GKE reuses the same Kubernetes manifests; the deploy action builds and pushes
the image, fetches cluster credentials via `gcloud`, then applies the
Kustomize set with `kubectl`.

```bash
troopai deploy gke \
  --agent my_pkg.agents:assistant \
  --image gcr.io/my-project/my-agent:latest \
  --project my-gcp-project \
  --region us-central1 \
  --cluster my-cluster
```

This runs in sequence:

1. `docker build` + `docker push` (skip push with `--no-push`)
2. `gcloud container clusters get-credentials <cluster> --region <region> --project <project>`
3. `kubectl apply -k deploy/k8s`

### Reference: gke flags

| Flag | Required | Description |
|------|----------|-------------|
| `--project TEXT` | yes | GCP project id |
| `--region TEXT` | yes | Cluster region or location |
| `--cluster TEXT` | yes | GKE cluster name |
| `--no-push` | — | Build but do not push the image |
| `--dir PATH` | `.` | Build context / manifest directory |
| `--no-generate` | — | Use artifacts already on disk |

---

## `troopai deploy helm`

Render a Helm chart and install or upgrade the release with
`helm upgrade --install`.

### Step 1 — generate the chart

```bash
troopai deploy init \
  --target helm \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --env-key OPENAI_API_KEY
```

This writes a Helm chart under `deploy/helm/<app-name>/`:

| File | Purpose |
|------|---------|
| `Chart.yaml` | Chart metadata |
| `values.yaml` | Default values (image, port, replica count, resource limits) |
| `templates/` | Go templates for Deployment, Service, HPA, ConfigMap, Secret |

### Step 2 — push the image and install

```bash
# Push the image first
troopai deploy build \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --push

# Install or upgrade the release
troopai deploy helm \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --no-generate
```

Specify a namespace; the namespace is created if it does not exist:

```bash
troopai deploy helm \
  --agent my_pkg.agents:assistant \
  --image registry.example.com/my-agent:latest \
  --namespace production \
  --no-generate
```

### Reference: helm flags

| Flag | Default | Description |
|------|---------|-------------|
| `--namespace TEXT` | cluster default | Namespace to install into (created if missing) |
| `--dir PATH` | `.` | Directory holding or to receive the chart |
| `--no-generate` | — | Use the chart already on disk |

---

## Shared flags

All `troopai deploy` subcommands accept these flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--agent MODULE:VAR` | — | `module:var` reference the container serves (required) |
| `--image IMAGE[:TAG]` | `troopai-agent:latest` | Container image name with optional registry and tag |
| `--app-name TEXT` | derived from `--image` | Service / resource name (RFC 1123 label) |
| `--port INTEGER` | `8080` | Container port |
| `--extras TEXT` | `serve,a2a` | `troopai-adk-python` extras installed in the image |
| `--env-key TEXT` | — | Env var name to surface as a Secret reference (repeatable) |

## See also

- [Container contract](container.md) — what the generated image must satisfy
- [Scaling](scaling.md) — enabling the HPA once a shared Postgres backend is configured
- [GCP Cloud Run](gcp.md) — managed serverless alternative
