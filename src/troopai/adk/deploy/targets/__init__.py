"""Deploy targets and the registry ``deploy init`` dispatches on.

New targets register here so ``troopai deploy init --target <key>``
discovers them. Each target's ``generate`` returns a complete, deployable
artifact set (container artifacts plus its orchestration manifests).
"""

from __future__ import annotations

from troopai.adk.deploy.targets.apprunner import AppRunnerTarget
from troopai.adk.deploy.targets.aws_lambda import LambdaTarget
from troopai.adk.deploy.targets.base import DeployTarget
from troopai.adk.deploy.targets.cloudrun import CloudRunTarget
from troopai.adk.deploy.targets.docker import DockerTarget
from troopai.adk.deploy.targets.ecs import ECSTarget
from troopai.adk.deploy.targets.gke import GKETarget
from troopai.adk.deploy.targets.helm import HelmTarget
from troopai.adk.deploy.targets.k8s import K8sTarget

TARGETS: dict[str, DeployTarget] = {
    DockerTarget.key: DockerTarget(),
    K8sTarget.key: K8sTarget(),
    GKETarget.key: GKETarget(),
    HelmTarget.key: HelmTarget(),
    CloudRunTarget.key: CloudRunTarget(),
    ECSTarget.key: ECSTarget(),
    AppRunnerTarget.key: AppRunnerTarget(),
    LambdaTarget.key: LambdaTarget(),
}

__all__ = [
    "TARGETS",
    "AppRunnerTarget",
    "CloudRunTarget",
    "DeployTarget",
    "DockerTarget",
    "ECSTarget",
    "GKETarget",
    "HelmTarget",
    "K8sTarget",
    "LambdaTarget",
]
