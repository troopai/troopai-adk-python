# Deploying an agent

`app.py` defines a single `agent`. This walkthrough serves it over HTTP
and ships it to a cloud target. Everything is opt-in — the framework owns
no server runtime and no cloud SDK.

## 1. Install the serving extra

```bash
pip install 'troopai-adk-python[serve]'
```

## 2. Serve locally

```bash
# From this directory. REST + health are on by default.
troopai serve --agent app:agent --host 0.0.0.0 --port 8000

# In another shell:
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/run -H 'content-type: application/json' \
  -d '{"prompt": "hello"}'
```

`POST /run` returns the result as JSON; `POST /run_sse` streams run items
as Server-Sent Events; `GET /healthz` (liveness) and `GET /readyz`
(readiness) back container probes.

## 3. Generate deployment artifacts

```bash
# Container artifacts only:
troopai deploy init --agent app:agent --image my-agent:latest

# Or a full target set (Dockerfile + Kubernetes manifests):
troopai deploy init --target k8s --agent app:agent --image my-agent:latest \
  --env-key OPENAI_API_KEY
```

Edit the generated `requirements.txt` so `troopai-adk-python` is installable in
the image — the package is private, so point pip at a private index, a
vendored wheel, or a VCS URL, then add your agent's own dependencies.

## 4. Build and ship

```bash
# Build (and push) the image:
troopai deploy build --agent app:agent --image my-agent:latest --push

# GCP Cloud Run (Cloud Build — no local docker needed):
troopai deploy cloud-run --agent app:agent \
  --image gcr.io/PROJECT/my-agent --project PROJECT --region REGION

# Kubernetes (any cluster):
troopai deploy k8s --agent app:agent --image my-agent:latest

# AWS ECS Fargate (push the image to ECR first):
troopai deploy ecs --agent app:agent \
  --image ACCOUNT.dkr.ecr.REGION.amazonaws.com/my-agent \
  --region REGION --execution-role-arn arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole
```

Targets: `docker`, `k8s`, `gke`, `helm`, `cloudrun`, `ecs`, `app-runner`,
`lambda`. See [`docs/deploy/`](../../docs/deploy/) for the full guide,
including horizontal scaling (a shared task store + networked session
backend for multi-replica deployments).
