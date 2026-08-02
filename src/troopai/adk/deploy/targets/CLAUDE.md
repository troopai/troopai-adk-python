# Deploy Targets

One adapter per deployment target behind the `DeployTarget` protocol
(`base.py`). `generate(ctx)` is the uniform method `deploy init` calls for
every target; build/ship actions are target-specific methods (their flags
differ per cloud) invoked directly by each `deploy` CLI subcommand.

## Targets

| Key | Class | Ship action | Tools |
|---|---|---|---|
| `docker` | `DockerTarget` | `build()` — docker build/push | docker |
| `k8s` | `K8sTarget` | `apply()` — kubectl apply -k | kubectl |
| `gke` | `GKETarget` | `deploy()` — build/push + get-credentials + apply | gcloud, docker, kubectl |
| `helm` | `HelmTarget` | `install()` — helm upgrade --install | helm |
| `cloudrun` | `CloudRunTarget` | `deploy()` — gcloud run deploy --source | gcloud |
| `ecs` | `ECSTarget` | `deploy()` — register-task-definition (+ update-service) | aws |
| `apprunner` | `AppRunnerTarget` | `deploy()` — apprunner create-service | aws |
| `lambda` | `LambdaTarget` | `deploy()` — lambda update-function-code | aws |

## Decisions

- The protocol's uniform surface is `key` / `required_tools` / `generate`;
  ship methods are bespoke per target (varying flags) and not on the
  protocol. `TARGETS` (in `__init__.py`) maps key → instance for
  `deploy init` discovery.
- Every non-docker target's `generate()` composes `DockerTarget().generate()`
  (or, for Lambda, its own Web Adapter Dockerfile) so the artifact set is
  self-contained and deployable.
- GKE reuses the Kubernetes manifests; only its ship sequence differs.
