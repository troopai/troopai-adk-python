# Deploy Module

Deployment engine: generates infra artifacts and ships the agent by
driving the operator's installed CLIs. Stdlib + click only — **no cloud
SDK, no new runtime dependency**. Drives the `troopai deploy` command
group (`cli/deploy.py`).

## Files

| File | Purpose |
|---|---|
| `context.py` | `DeployContext` — frozen, validated inputs (agent_ref, image, app_name RFC-1123, port, python_version, extras, env_keys). |
| `commands.py` | `CommandRunner` seam: `SubprocessRunner` (shells to host CLIs), `RecordingRunner` (tests), `require_tool` / `run_checked`, `DeployToolMissing` / `DeployCommandFailed`. |
| `templates.py` | Container artifacts: `render_dockerfile` / `render_dockerignore` / `render_requirements`. |
| `k8s_manifests.py` | Kubernetes manifest renderers (Deployment/Service/HPA/ConfigMap/Secret/Kustomization). |
| `helm_chart.py` | Helm chart renderer (Chart.yaml + values.yaml substituted; `templates/` are static Go templates). |
| `cloudrun_manifests.py` | Knative `service.yaml` renderer. |
| `aws_manifests.py` | ECS task definition (JSON), App Runner create-service input (JSON), Lambda Web Adapter Dockerfile. |
| `artifacts.py` | `write_artifacts(files, dest, *, force)` — writes a `{path: content}` map, skipping existing files unless `force`. |
| `targets/` | One adapter per target behind the `DeployTarget` protocol (see `targets/`). |

## Architectural decisions

| # | Decision | Why |
|---|---|---|
| 1 | Active deploy shells out to operator CLIs (docker/gcloud/kubectl/aws/helm) via `CommandRunner`, never a Python cloud SDK | Zero new runtime deps; `RecordingRunner` makes the active path unit-testable with no cloud access; mirrors how an operator deploys by hand. |
| 2 | Templates render with `string.Template.safe_substitute` and lowercase `$placeholders` | Uppercase shell vars (`$PORT`, `$AGENT_REF`) and Go-template `{{ }}` pass through untouched — no escaping. Helm `templates/` files are static (filled by Helm at install, not generation). |
| 3 | The generated image binds `0.0.0.0:$PORT` explicitly in the CMD | Meets the universal container contract; `troopai serve` keeps its secure `127.0.0.1` default for local runs. |
| 4 | `requirements.txt` is the install seam, not a public index | The package is private, so the Dockerfile installs `troopai-adk-python[...]` from whatever the operator's `requirements.txt` points at (private index / vendored wheel / VCS). |
| 5 | Account-specific values (region, role ARNs) are deploy-time args, injected into the JSON inline; generated reference artifacts carry clear `REPLACE_*` placeholders | A registered/created resource is never built from a placeholder; the artifact stays a readable reference. |
| 6 | `DeployTarget.generate()` returns a complete artifact set (container + orchestration) | `deploy init --target X` yields everything needed to ship; build/ship methods are target-specific (their flags differ per cloud) and called directly by each CLI subcommand. |

## Container contract

Every target's image binds `0.0.0.0:$PORT`, takes config from env, runs
non-root, starts fast, and exposes `/healthz` + `/readyz`. The same image
runs on Kubernetes, Cloud Run, ECS, and App Runner; Lambda uses a Web
Adapter variant.

## Testing

`tests/unit/deploy/` — render assertions + YAML/JSON validity; ship
actions run through `RecordingRunner` asserting exact argv. No cloud is
ever contacted. `tests/unit/cli/test_deploy.py` exercises the CLI with the
runner monkeypatched (the `deploy` group shadows the `cli.deploy`
submodule, so tests reach it via `sys.modules`).
