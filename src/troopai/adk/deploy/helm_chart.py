"""Helm chart renderer for a served TroopAI agent.

``Chart.yaml`` and ``values.yaml`` are rendered with ``safe_substitute``
at generation time; the ``templates/`` files are static Go templates that
Helm fills from ``values.yaml`` at install time, so their ``{{ }}`` and
``$`` pass through untouched (they are never run through substitution).
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.deploy.context import DeployContext

_CHART = Template(
    """\
apiVersion: v2
name: $app_name
description: A TroopAI agent served over HTTP (REST + health).
type: application
version: 0.1.0
appVersion: "0.1.0"
"""
)

_VALUES = Template(
    """\
replicaCount: 1

image:
  repository: $image_repo
  tag: "$image_tag"
  pullPolicy: IfNotPresent

agentRef: "$agent_ref"
containerPort: $port

service:
  type: ClusterIP
  port: 80

autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 5
  targetCPUUtilizationPercentage: 70

resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: "1"
    memory: 1Gi

terminationGracePeriodSeconds: 45

# Name of a pre-created Secret holding the agent's API keys. Create it with:
#   kubectl create secret generic $app_name-secrets --from-literal=KEY=VALUE
secretName: "$app_name-secrets"

# Secret keys injected as environment variables (must exist in secretName).
secretEnv: [$secret_env_list]
"""
)

_TPL_HELPERS = """\
{{- define "agent.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "agent.fullname" -}}
{{- printf "%s" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agent.labels" -}}
app.kubernetes.io/name: {{ include "agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
"""

_TPL_DEPLOYMENT = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "agent.fullname" . }}
  labels:
    {{- include "agent.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "agent.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "agent.selectorLabels" . | nindent 8 }}
    spec:
      terminationGracePeriodSeconds: {{ .Values.terminationGracePeriodSeconds }}
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - containerPort: {{ .Values.containerPort }}
          env:
            - name: PORT
              value: "{{ .Values.containerPort }}"
            - name: AGENT_REF
              value: "{{ .Values.agentRef }}"
            {{- range .Values.secretEnv }}
            - name: {{ . }}
              valueFrom:
                secretKeyRef:
                  name: {{ $.Values.secretName }}
                  key: {{ . }}
            {{- end }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          startupProbe:
            httpGet:
              path: /healthz
              port: {{ .Values.containerPort }}
            failureThreshold: 30
            periodSeconds: 5
          readinessProbe:
            httpGet:
              path: /readyz
              port: {{ .Values.containerPort }}
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: {{ .Values.containerPort }}
            periodSeconds: 15
"""

_TPL_SERVICE = """\
apiVersion: v1
kind: Service
metadata:
  name: {{ include "agent.fullname" . }}
  labels:
    {{- include "agent.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  selector:
    {{- include "agent.selectorLabels" . | nindent 4 }}
  ports:
    - name: http
      port: {{ .Values.service.port }}
      targetPort: {{ .Values.containerPort }}
"""

_TPL_HPA = """\
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "agent.fullname" . }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "agent.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
{{- end }}
"""


def split_image_reference(image: str) -> tuple[str, str]:
    """Split a container image reference into ``(repository, tag)``.

    The tag is the segment after the final ``:`` that follows the last
    ``/`` — so a registry port (``registry:5000/team/app``) is never
    mistaken for a tag. Returns ``"latest"`` when the reference carries
    no tag.

    Args:
        image: A container image reference, optionally prefixed with a
            registry (which may itself carry a ``:port``) and suffixed
            with a ``:tag``.

    Returns:
        The ``(repository, tag)`` pair. ``tag`` is ``"latest"`` when the
        reference has none.
    """
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        tag = image[last_colon + 1 :]
        return image[:last_colon], tag if len(tag) > 0 else "latest"
    return image, "latest"


def render_helm_chart(ctx: DeployContext) -> dict[str, str]:
    """Render the Helm chart for *ctx*, keyed by ``deploy/helm/<app>/`` path.

    Args:
        ctx: The deploy context.

    Returns:
        Map of relative path to chart file content.
    """
    repo, image_tag = split_image_reference(ctx.image)
    secret_env_list = ", ".join(f'"{key}"' for key in ctx.env_keys)
    chart = _CHART.safe_substitute(app_name=ctx.app_name)
    values = _VALUES.safe_substitute(
        app_name=ctx.app_name,
        image_repo=repo,
        image_tag=image_tag,
        agent_ref=ctx.agent_ref,
        port=ctx.port,
        secret_env_list=secret_env_list,
    )
    base = f"deploy/helm/{ctx.app_name}"
    return {
        f"{base}/Chart.yaml": chart,
        f"{base}/values.yaml": values,
        f"{base}/templates/_helpers.tpl": _TPL_HELPERS,
        f"{base}/templates/deployment.yaml": _TPL_DEPLOYMENT,
        f"{base}/templates/service.yaml": _TPL_SERVICE,
        f"{base}/templates/hpa.yaml": _TPL_HPA,
    }
