# Sandbox Security

Three layers of security policy ship out of the box; all three are
opt-in (cost-conservative default is "off").

## 1. Command allowlist / denylist

`SandboxCommandGuardrail` is a `ToolInputGuardrail` that inspects
parsed tool args carrying a `command` field. Three pattern modes:

- `"exact"` — base command must match an allowlist entry exactly.
- `"prefix"` — the command must start with an allowlist entry.
- `"regex"` — the command must match a regex anchored at start.

`None` for either list disables that side. A trip raises
`SandboxCommandRejected` and surfaces in `RunResult.guardrail_results`.

Verdicts always flow through guardrails — never middleware — so
audit completeness is preserved.

## 2. Network policy

`NetworkPolicy(allow_hosts, allow_ports, deny_default=True,
allow_dns=True)` is declarative and translates per backend:

- Docker: `network_mode="none"` when `deny_default=True` and lists are
  empty; otherwise defers to the backend's default networking. A
  full sidecar / iptables-firewall sweep is a deployment concern.
- K8s: emits a `networking.k8s.io/v1 NetworkPolicy` CR on the pod,
  with explicit egress rules per `allow_ports` + DNS allow when
  `allow_dns=True`. Hostname allowlists require a service-mesh
  sidecar — recorded as an annotation for downstream operators.
- LocalSubprocess: raises `SandboxNetworkPolicyViolation` when
  `deny_default=True` (cannot enforce inside the host's process).

## 3. Resource limits

`SandboxResourceLimits(cpu_cores, memory_mb, disk_mb, exec_timeout,
session_timeout, max_processes, max_egress_bytes)` — every field
defaults to `None`. The backend interprets `None` as "no ADK-
enforced limit; respect backend default."

- Docker: maps to `cpu_period` + `cpu_quota`, `mem_limit`,
  `pids_limit`.
- K8s: maps to container `resources.limits` + `resources.requests`,
  plus `ephemeral-storage` for `disk_mb`.

## 4. PodSecurity (K8s only)

`K8sSandboxClientOptions.pod_security_standard` defaults to
`"restricted"`. Restricted profile adds: `runAsNonRoot=True`,
`runAsUser=1000`, `allowPrivilegeEscalation=False`,
`capabilities.drop=["ALL"]`, `seccompProfile.type="RuntimeDefault"`.

See `src/troopai/adk/sandbox/guardrails/command_guardrail.py` and
`src/troopai/adk/sandbox/policy/` for the implementations.
