(deploy/aws)=

# AWS

Three AWS targets are available, each suited to a different traffic pattern:

| Target | Best for |
|--------|---------|
| `ecs` | Long-running agents, sustained traffic, fine-grained VPC control |
| `apprunner` | Simple container-to-URL deploys with managed load balancing and TLS |
| `lambda` | Spiky or intermittent traffic where pay-per-invocation economics apply |

All three shell out to the `aws` CLI. No AWS SDK is imported by the framework.

## Prerequisites

- `aws` CLI installed and configured (`aws configure` or an IAM role assumed
  in the environment)
- Docker installed (for building and pushing images to ECR)
- ECR repository created in the target region

## Pushing an image to ECR

All AWS targets use images from ECR. You can push manually or let the deploy
command handle it automatically with `--push`.

### Automatic ECR login and push (recommended)

Pass `--push` to any of the three AWS deploy subcommands. The command logs in
to ECR, builds the image, and pushes it — all before the register/create/update
step:

```bash
troopai deploy ecs \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:1 \
  --region us-east-1 \
  --execution-role-arn arn:aws:iam::123456789:role/ecsTaskExecutionRole \
  --push
```

Under the hood `--push` runs:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com
```

and then builds and pushes `ctx.image` via `docker build` + `docker push`. The
password is passed through stdin — never placed in the process argument list.

### Manual ECR login and push (alternative)

If you prefer to push the image yourself before deploying:

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# Build and push via troopai deploy build
troopai deploy build \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --push
```

Then run the deploy subcommand without `--push`.

---

## `troopai deploy ecs`

Registers a Fargate task definition and optionally forces a new deployment on
an existing ECS service.

### Step 1 — generate artifacts

```bash
troopai deploy init \
  --target ecs \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --env-key OPENAI_API_KEY
```

This writes:

| File | Purpose |
|------|---------|
| `Dockerfile` | Standard container image |
| `.dockerignore` | Build context exclusions |
| `requirements.txt` | Package install seam |
| `deploy/aws-ecs/task-definition.json` | ECS task definition template with `REPLACE_*` placeholders |

Edit `deploy/aws-ecs/task-definition.json` to fill in your cluster ARN,
log group, and any other account-specific values marked with `REPLACE_`.

### Step 2 — register the task definition and roll the service

Use `--push` to have the command log in to ECR, build, and push the image
before registering the task definition:

```bash
troopai deploy ecs \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1 \
  --execution-role-arn arn:aws:iam::123456789:role/ecsTaskExecutionRole \
  --push
```

Or, if you already pushed the image manually, omit `--push`:

```bash
troopai deploy ecs \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1 \
  --execution-role-arn arn:aws:iam::123456789:role/ecsTaskExecutionRole \
  --no-generate
```

This runs:

```
aws ecs register-task-definition \
  --region us-east-1 \
  --cli-input-json <rendered task definition>
```

To also force a new deployment on an existing service, pass `--cluster` and
`--service`:

```bash
troopai deploy ecs \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1 \
  --execution-role-arn arn:aws:iam::123456789:role/ecsTaskExecutionRole \
  --cluster my-cluster \
  --service my-agent-service \
  --push
```

### Reference: ecs flags

| Flag | Required | Description |
|------|----------|-------------|
| `--region TEXT` | yes | AWS region |
| `--execution-role-arn TEXT` | yes | Task execution role ARN (ECR pull + secrets access) |
| `--cluster TEXT` | — | Existing ECS cluster name (required with `--service`) |
| `--service TEXT` | — | Existing ECS service to force a new deployment on |
| `--push` | — | Log in to ECR, build, and push the image before deploying |
| `--dir PATH` | `.` | Directory to write artifacts into |
| `--no-generate` | — | Use artifacts already on disk |

---

## `troopai deploy app-runner`

Creates an App Runner service from an ECR image. App Runner manages the
load balancer, TLS, and scaling automatically.

### Step 1 — generate artifacts

```bash
troopai deploy init \
  --target apprunner \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest
```

This writes:

| File | Purpose |
|------|---------|
| `Dockerfile` | Standard container image |
| `.dockerignore` | Build context exclusions |
| `requirements.txt` | Package install seam |
| `deploy/aws-apprunner/create-service.json` | App Runner create-service input |

### Step 2 — create the service

Pass `--push` to log in to ECR, build, and push the image, then create the
service in one step:

```bash
troopai deploy app-runner \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1 \
  --access-role-arn arn:aws:iam::123456789:role/AppRunnerECRAccessRole \
  --push
```

Or push the image first with `troopai deploy build --push`, then create the
service without `--push`:

```bash
troopai deploy app-runner \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1 \
  --access-role-arn arn:aws:iam::123456789:role/AppRunnerECRAccessRole \
  --no-generate
```

This runs:

```
aws apprunner create-service \
  --region us-east-1 \
  --cli-input-json <rendered create-service document>
```

The access role must grant App Runner permission to pull images from ECR.

### Reference: app-runner flags

| Flag | Required | Description |
|------|----------|-------------|
| `--region TEXT` | yes | AWS region |
| `--access-role-arn TEXT` | yes | IAM role ARN App Runner uses to pull from ECR |
| `--push` | — | Log in to ECR, build, and push the image before deploying |
| `--dir PATH` | `.` | Directory to write artifacts into |
| `--no-generate` | — | Use artifacts already on disk |

---

## `troopai deploy lambda`

Packages the agent as a Lambda function image using the
[Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) and
updates an existing function's image URI.

:::{note}
Lambda is best suited to spiky or intermittent traffic. Sustained multi-turn
agent conversations benefit from ECS or App Runner, which keep the process
running between requests.
:::

### Step 1 — generate artifacts

```bash
troopai deploy init \
  --target lambda \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest
```

This writes a Lambda Web Adapter Dockerfile under `deploy/aws-lambda/`:

```dockerfile
# Lambda Web Adapter: forward HTTP requests to troopai serve
FROM public.ecr.aws/awsguru/aws-lambda-adapter:0.8.1 AS adapter
FROM python:3.12-slim

COPY --from=adapter /lambda-adapter /opt/extensions/lambda-adapter

# ... (same pip install, non-root user, and CMD as the standard image)
```

The Web Adapter intercepts Lambda invocations and forwards them as HTTP
requests to the `troopai serve` process running on port 8080.

### Step 2 — push to ECR and update the function

Pass `--push` to log in to ECR, build, and push the Lambda image, then update
the function in one step:

```bash
troopai deploy lambda \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1 \
  --push
```

Or push first and then update separately:

```bash
# Build and push the Lambda image manually
troopai deploy build \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --push

# Update the function's image
troopai deploy lambda \
  --agent my_pkg.agents:assistant \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1
```

This runs:

```
aws lambda update-function-code \
  --function-name <app-name> \
  --image-uri 123456789.dkr.ecr.us-east-1.amazonaws.com/my-agent:latest \
  --region us-east-1
```

### Reference: lambda flags

| Flag | Required | Description |
|------|----------|-------------|
| `--region TEXT` | yes | AWS region |
| `--function-name TEXT` | — | Target function name; defaults to the app name |
| `--push` | — | Log in to ECR, build, and push the image before deploying |
| `--dir PATH` | `.` | Directory to write artifacts into |
| `--no-generate` | — | Use artifacts already on disk |

---

## Shared flags

Shared flags (`--agent`, `--image`, `--app-name`, `--port`, `--extras`,
`--env-key`) are documented in the [Kubernetes page](kubernetes.md#shared-flags).

## See also

- [Container contract](container.md) — what every generated image must satisfy
- [Scaling](scaling.md) — multi-replica and shared backends
- [GCP Cloud Run](gcp.md) — GCP equivalent of App Runner
