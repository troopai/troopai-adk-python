"""Kubernetes manifest renderers (Deployment / Service / HPA / config).

Produces manifests that fix the gaps a naive generator leaves: startup +
readiness + liveness probes on the health endpoints, a HorizontalPodAuto-
scaler, Secret-sourced API keys, resource requests/limits, and a
termination grace period sized for in-flight agent turns. Rendering uses
``safe_substitute`` with lowercase ``$placeholders``; the repeated env and
secret blocks are built in Python (templates cannot loop).
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.deploy.context import DeployContext

# Drain window for in-flight agent turns, which can outlast the 30s default.
_GRACE_SECONDS = 45

_DEPLOYMENT = Template(
    """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $app_name
  labels:
    app: $app_name
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $app_name
  template:
    metadata:
      labels:
        app: $app_name
    spec:
      terminationGracePeriodSeconds: $grace
      containers:
        - name: $app_name
          image: $image
          ports:
            - containerPort: $port
          env:
$env_entries
          envFrom:
            - configMapRef:
                name: $app_name-config
          resources:
            requests:
              cpu: 250m
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi
          startupProbe:
            httpGet:
              path: /healthz
              port: $port
            failureThreshold: 30
            periodSeconds: 5
          readinessProbe:
            httpGet:
              path: /readyz
              port: $port
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: $port
            periodSeconds: 15
"""
)

_SERVICE = Template(
    """\
apiVersion: v1
kind: Service
metadata:
  name: $app_name
  labels:
    app: $app_name
spec:
  type: ClusterIP
  selector:
    app: $app_name
  ports:
    - name: http
      port: 80
      targetPort: $port
"""
)

_HPA = Template(
    """\
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $app_name
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $app_name
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
"""
)

_CONFIGMAP = Template(
    """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: $app_name-config
data:
  # Non-secret runtime config. Add KEY: "value" entries as needed.
  LOG_LEVEL: "INFO"
"""
)

_SECRET = Template(
    """\
# Example Secret — DO NOT commit real values. Create the real one with:
#   kubectl create secret generic $app_name-secrets --from-literal=KEY=VALUE
apiVersion: v1
kind: Secret
metadata:
  name: $app_name-secrets
type: Opaque
stringData:
$secret_data
"""
)

_KUSTOMIZATION = Template(
    """\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
  - hpa.yaml
  - configmap.yaml
"""
)


def _env_entries(ctx: DeployContext) -> str:
    """Build the container ``env:`` list: PORT, AGENT_REF, then Secret refs."""
    lines = [
        "            - name: PORT",
        f'              value: "{ctx.port}"',
        "            - name: AGENT_REF",
        f'              value: "{ctx.agent_ref}"',
    ]
    for key in ctx.env_keys:
        lines.extend(
            [
                f"            - name: {key}",
                "              valueFrom:",
                "                secretKeyRef:",
                f"                  name: {ctx.app_name}-secrets",
                f"                  key: {key}",
            ]
        )
    return "\n".join(lines)


def _secret_data(ctx: DeployContext) -> str:
    """Build the Secret ``stringData`` block (placeholder values)."""
    if len(ctx.env_keys) == 0:
        return '  # Add secret keys here, e.g. OPENAI_API_KEY: "sk-..."\n  EXAMPLE_API_KEY: "REPLACE_ME"'
    return "\n".join(f'  {key}: "REPLACE_ME"' for key in ctx.env_keys)


def render_deployment(ctx: DeployContext) -> str:
    """Render the Deployment manifest for *ctx*."""
    return _DEPLOYMENT.safe_substitute(
        app_name=ctx.app_name,
        image=ctx.image,
        port=ctx.port,
        grace=_GRACE_SECONDS,
        env_entries=_env_entries(ctx),
    )


def render_service(ctx: DeployContext) -> str:
    """Render the ClusterIP Service manifest for *ctx*."""
    return _SERVICE.safe_substitute(app_name=ctx.app_name, port=ctx.port)


def render_hpa(ctx: DeployContext) -> str:
    """Render the HorizontalPodAutoscaler manifest for *ctx*."""
    return _HPA.safe_substitute(app_name=ctx.app_name)


def render_configmap(ctx: DeployContext) -> str:
    """Render the ConfigMap manifest for *ctx*."""
    return _CONFIGMAP.safe_substitute(app_name=ctx.app_name)


def render_secret_example(ctx: DeployContext) -> str:
    """Render the example Secret manifest for *ctx*."""
    return _SECRET.safe_substitute(app_name=ctx.app_name, secret_data=_secret_data(ctx))


def render_kustomization() -> str:
    """Render the Kustomization that ties the manifests together (static)."""
    return _KUSTOMIZATION.template


def render_k8s_manifests(ctx: DeployContext) -> dict[str, str]:
    """Render the full Kubernetes manifest set, keyed by ``deploy/k8s/`` path.

    Args:
        ctx: The deploy context.

    Returns:
        Map of relative path to manifest content.
    """
    return {
        "deploy/k8s/deployment.yaml": render_deployment(ctx),
        "deploy/k8s/service.yaml": render_service(ctx),
        "deploy/k8s/hpa.yaml": render_hpa(ctx),
        "deploy/k8s/configmap.yaml": render_configmap(ctx),
        "deploy/k8s/secret.example.yaml": render_secret_example(ctx),
        "deploy/k8s/kustomization.yaml": render_kustomization(),
    }
