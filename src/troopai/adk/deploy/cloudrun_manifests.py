"""Cloud Run (Knative) service manifest renderer.

A declarative ``service.yaml`` for ``gcloud run services replace``. The
active ``deploy cloud-run`` path uses ``gcloud run deploy --source`` (Cloud
Build) instead, so this manifest is the reference/GitOps artifact.
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.deploy.context import DeployContext

_SERVICE = Template(
    """\
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: $app_name
spec:
  template:
    spec:
      # Concurrency >1 — the agent server is async and handles parallel
      # requests. Raise timeoutSeconds for long multi-turn runs.
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: $image
          ports:
            - name: http1
              containerPort: $port
          env:
            - name: AGENT_REF
              value: "$agent_ref"
          resources:
            limits:
              cpu: "1"
              memory: 512Mi
"""
)


def render_cloudrun_service(ctx: DeployContext) -> str:
    """Render the Knative Service manifest for *ctx*.

    Args:
        ctx: The deploy context.

    Returns:
        The ``service.yaml`` contents.
    """
    return _SERVICE.safe_substitute(app_name=ctx.app_name, image=ctx.image, port=ctx.port, agent_ref=ctx.agent_ref)


def render_cloudrun(ctx: DeployContext) -> dict[str, str]:
    """Render the Cloud Run artifact set, keyed by ``deploy/cloudrun/`` path.

    Args:
        ctx: The deploy context.

    Returns:
        Map of relative path to file content.
    """
    return {"deploy/cloudrun/service.yaml": render_cloudrun_service(ctx)}
