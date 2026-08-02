# Sandbox IaC Integration

`IaCBundle` declares an infrastructure stack that should be applied
before the sandbox session starts and destroyed after it ends. The
runner provisions the stack with the user's chosen tool, extracts
outputs, and maps them to env vars the sandbox can read.

```
@dataclass(frozen=True, kw_only=True)
class IaCBundle:
    provider: Literal["terraform", "pulumi"]
    working_directory: str
    variables: dict[str, str]
    output_env_mapping: dict[str, str]  # IaC output → env var name
    destroy_on_exit: bool = True
    timeout: float = 300.0
```

## Terraform

`apply_iac(bundle)` runs:

1. `terraform init` (in the bundle's `working_directory`).
2. `terraform apply -auto-approve -var key=value ... -json`.
3. Parses outputs JSON and maps to env vars per
   `output_env_mapping`.

Cleanup: `destroy_iac(bundle)` runs `terraform destroy -auto-approve`.

Subprocess invocation goes through `asyncio.create_subprocess_exec`
so the runner never blocks. Errors map to `IaCApplyError` /
`IaCDestroyError` (both under `SandboxConfigurationError`).

## Pulumi

Same shape, with `pulumi up --yes --non-interactive --skip-preview`
on apply and `pulumi destroy --yes --non-interactive` on destroy.
Stack outputs come via `pulumi stack output --json`.

## Wiring

Set `SandboxRunConfig.iac` to an `IaCBundle`; the `sandbox_run_context`
calls `apply_iac` before `client.create(...)` and `destroy_iac` on
exit. The applied env mapping is forwarded into the sandbox's
container/pod via `options.environment`.

See `src/troopai/adk/sandbox/runner_integration/iac_runner.py`.
