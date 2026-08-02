(deploy/gcp)=

# GCP Cloud Run

`troopai deploy cloud-run` deploys an agent to Google Cloud Run using
`gcloud run deploy --source`. Cloud Build builds the generated Dockerfile in
the cloud — no local Docker daemon is required. The only prerequisite is the
`gcloud` CLI, authenticated and configured.

## Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login` or a service
  account)
- Cloud Run, Cloud Build, and Secret Manager APIs enabled in the project
- The project has a default region set, or you pass `--region` explicitly

## Deployment

### Step 1 — generate artifacts

```bash
troopai deploy init \
  --target cloudrun \
  --agent my_pkg.agents:assistant \
  --image gcr.io/my-project/my-agent:latest \
  --env-key OPENAI_API_KEY \
  --env-key ANTHROPIC_API_KEY
```

This writes:

| File | Purpose |
|------|---------|
| `Dockerfile` | Image built by Cloud Build |
| `.dockerignore` | Files excluded from the build context |
| `requirements.txt` | Package install seam (see [Container contract](container.md#package-installation)) |
| `deploy/cloudrun/service.yaml` | Knative `Service` reference manifest |

### Step 2 — edit requirements.txt

Cloud Build runs `pip install -r requirements.txt` inside the image. The
generated file installs `troopai-adk-python` from PyPI by name; adjust it
beforehand only if you want a pin, a vendored wheel, or a VCS URL instead
(see [Container contract](container.md#package-installation)).

### Step 3 — create secrets in Secret Manager

Each `--env-key` name maps to a Secret Manager secret of the same name.
Create the secrets before deploying (Cloud Run fails the deploy if a
referenced secret does not exist):

```bash
echo -n "$OPENAI_API_KEY" | \
  gcloud secrets create OPENAI_API_KEY \
    --project my-project \
    --data-file=-

echo -n "$ANTHROPIC_API_KEY" | \
  gcloud secrets create ANTHROPIC_API_KEY \
    --project my-project \
    --data-file=-
```

### Step 4 — deploy

```bash
troopai deploy cloud-run \
  --agent my_pkg.agents:assistant \
  --image gcr.io/my-project/my-agent:latest \
  --project my-project \
  --region us-central1 \
  --env-key OPENAI_API_KEY \
  --env-key ANTHROPIC_API_KEY \
  --no-generate
```

This runs:

```
gcloud run deploy <app-name> \
  --source . \
  --project my-project \
  --region us-central1 \
  --port 8080 \
  --no-allow-unauthenticated \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest \
  --set-secrets ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest
```

`--set-secrets` maps each `--env-key` to a Secret Manager secret of the same
name at the `latest` version.

## Authentication

By default the service requires authentication (`--no-allow-unauthenticated`).
To allow public access without credentials, pass `--allow-unauthenticated`:

```bash
troopai deploy cloud-run \
  --agent my_pkg.agents:assistant \
  --project my-project \
  --region us-central1 \
  --allow-unauthenticated
```

:::{warning}
A publicly accessible endpoint exposes your agent and the LLM calls it
makes to the internet. Require authentication for any agent that accesses
private data or consumes paid LLM tokens.
:::

## Minimum instances

Cloud Run scales to zero by default. Set `--min-instances` to keep warm
instances and avoid cold-start latency:

```bash
troopai deploy cloud-run \
  --agent my_pkg.agents:assistant \
  --project my-project \
  --region us-central1 \
  --min-instances 1
```

## Reference: cloud-run flags

Shared flags (`--agent`, `--image`, `--app-name`, `--port`, `--extras`,
`--env-key`) are documented in the [Kubernetes page](kubernetes.md#shared-flags).

| Flag | Required | Description |
|------|----------|-------------|
| `--project TEXT` | yes | GCP project id |
| `--region TEXT` | yes | Cloud Run region |
| `--allow-unauthenticated` | — | Make the service publicly invokable |
| `--min-instances INTEGER` | `0` | Warm instances to keep (0 scales to zero) |
| `--dir PATH` | `.` | Build context directory (must contain the Dockerfile) |
| `--no-generate` | — | Use artifacts already on disk |

## See also

- [Container contract](container.md) — the image the deploy builds
- [Scaling](scaling.md) — multi-replica and shared backends
- [AWS](aws.md) — ECS, App Runner, Lambda targets
