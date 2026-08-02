"""``troopai deploy`` — generate deployment artifacts and ship the agent.

``deploy init`` writes a Dockerfile, ``.dockerignore``, and a starter
``requirements.txt`` (plus per-target manifests as targets are added).
``deploy build`` builds the image by driving the operator's installed
``docker`` CLI through the :class:`CommandRunner` seam. The framework
imports no cloud SDK — deployment is index/operator-owned.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import click

from troopai.adk.cli.errors import framework_errors
from troopai.adk.deploy.artifacts import write_artifacts
from troopai.adk.deploy.commands import DeployCommandFailed, DeployToolMissing, SubprocessRunner
from troopai.adk.deploy.context import DeployContext
from troopai.adk.deploy.helm_chart import split_image_reference
from troopai.adk.deploy.targets import TARGETS
from troopai.adk.deploy.targets.apprunner import AppRunnerTarget
from troopai.adk.deploy.targets.aws_lambda import LambdaTarget
from troopai.adk.deploy.targets.cloudrun import CloudRunTarget
from troopai.adk.deploy.targets.docker import DockerTarget
from troopai.adk.deploy.targets.ecs import ECSTarget
from troopai.adk.deploy.targets.gke import GKETarget
from troopai.adk.deploy.targets.helm import HelmTarget
from troopai.adk.deploy.targets.k8s import K8sTarget

logger = logging.getLogger(__name__)


def _ship(action: Callable[[], None]) -> None:
    """Run a ship action, mapping deploy failures to clean CLI errors.

    Args:
        action: The zero-arg deploy step to run.

    Raises:
        click.UsageError: If a required external tool is missing.
        click.ClickException: If a deploy command exits non-zero.
    """
    try:
        action()
    except DeployToolMissing as exc:
        raise click.UsageError(str(exc)) from exc
    except DeployCommandFailed as exc:
        raise click.ClickException(str(exc)) from exc


def context_options[F: Callable[..., object]](f: F) -> F:
    """Attach the flags that build a :class:`DeployContext`."""
    f = click.option("--agent", "agent_ref", required=True, help="Agent reference 'module:var' the container serves.")(
        f
    )
    f = click.option("--image", default="troopai-agent:latest", show_default=True, help="Container image name[:tag].")(
        f
    )
    f = click.option(
        "--app-name", default=None, help="Service/resource name (defaults to the image name without tag/registry)."
    )(f)
    f = click.option("--port", default=8080, show_default=True, type=int, help="Container port.")(f)
    f = click.option(
        "--extras", default="serve,a2a", show_default=True, help="troopai-adk-python extras installed in the image."
    )(f)
    f = click.option(
        "--env-key",
        "env_keys",
        multiple=True,
        help="Env var name surfaced as a Secret reference in manifests (repeatable).",
    )(f)
    return f


def _context(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
) -> DeployContext:
    """Build a validated :class:`DeployContext`, mapping bad input to UsageError.

    Args:
        agent_ref: The ``module:var`` reference the container serves.
        image: Container image name with optional tag/registry.
        app_name: Explicit service name, or ``None`` to derive from *image*.
        port: Container port.
        extras: ``troopai-adk-python`` extras to install.
        env_keys: Secret env-var names to surface in manifests.

    Returns:
        The validated context.

    Raises:
        click.UsageError: If validation fails.
    """
    name = app_name if app_name is not None else split_image_reference(image)[0].rsplit("/", 1)[-1]
    try:
        return DeployContext(
            agent_ref=agent_ref, image=image, app_name=name, port=port, extras=extras, env_keys=tuple(env_keys)
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


@click.group(name="deploy")
def deploy() -> None:
    """Generate deployment artifacts and ship the agent to a cloud target."""


@deploy.command(name="init")
@context_options
@click.option(
    "--target",
    type=click.Choice(sorted(TARGETS)),
    default="docker",
    show_default=True,
    help="Artifact set to generate.",
)
@click.option(
    "--dir",
    "dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    help="Directory to write artifacts into.",
)
@click.option("--force", is_flag=True, default=False, help="Overwrite existing files.")
@framework_errors
def init(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    target: str,
    dest: Path,
    force: bool,
) -> None:
    """Generate deployment artifacts for the chosen TARGET."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    written, skipped = write_artifacts(TARGETS[target].generate(ctx), dest, force=force)
    for path in written:
        click.echo(f"created {path}")
    for path in skipped:
        click.echo(f"skipped {path} (exists; pass --force to overwrite)")
    click.echo("")
    click.echo("Next: make requirements.txt install troopai-adk-python, then build the image:")
    click.echo(f"  troopai deploy build --agent {agent_ref} --image {ctx.image}")


@deploy.command(name="build")
@context_options
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Docker build context.",
)
@click.option("--push", is_flag=True, default=False, help="docker push after a successful build.")
@click.option(
    "--no-generate", is_flag=True, default=False, help="Use the Dockerfile already present; do not write one."
)
@framework_errors
def build(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    dest: Path,
    push: bool,
    no_generate: bool,
) -> None:
    """Build the container image with docker (and optionally push it)."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = DockerTarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(lambda: target.build(ctx, runner, context_dir=dest, push=push))
    click.echo(f"built {ctx.image}" + (" and pushed" if push else ""))


@deploy.command(name="k8s")
@context_options
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Directory holding (or to receive) the manifests.",
)
@click.option("--context", "kube_context", default=None, help="kubeconfig context to target.")
@click.option("--no-generate", is_flag=True, default=False, help="Use manifests already present; do not write them.")
@framework_errors
def k8s(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    dest: Path,
    kube_context: str | None,
    no_generate: bool,
) -> None:
    """Render Kubernetes manifests and apply them with kubectl."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = K8sTarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(lambda: target.apply(runner, context_dir=dest, kube_context=kube_context))
    click.echo(f"applied Kubernetes manifests for {ctx.app_name!r}")


@deploy.command(name="gke")
@context_options
@click.option("--project", required=True, help="GCP project id.")
@click.option("--region", required=True, help="Cluster region/location.")
@click.option("--cluster", required=True, help="GKE cluster name.")
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Build context / manifest directory.",
)
@click.option("--no-push", is_flag=True, default=False, help="Build but do not push the image.")
@click.option("--no-generate", is_flag=True, default=False, help="Use artifacts already present; do not write them.")
@framework_errors
def gke(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    project: str,
    region: str,
    cluster: str,
    dest: Path,
    no_push: bool,
    no_generate: bool,
) -> None:
    """Build/push the image and apply manifests to a GKE cluster."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = GKETarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(
        lambda: target.deploy(
            ctx, runner, project=project, region=region, cluster=cluster, context_dir=dest, push=not no_push
        )
    )
    click.echo(f"deployed {ctx.image} to GKE cluster {cluster!r}")


@deploy.command(name="helm")
@context_options
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Directory holding (or to receive) the chart.",
)
@click.option("--namespace", default=None, help="Namespace to install into (created if missing).")
@click.option("--no-generate", is_flag=True, default=False, help="Use the chart already present; do not write it.")
@framework_errors
def helm(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    dest: Path,
    namespace: str | None,
    no_generate: bool,
) -> None:
    """Render a Helm chart and install/upgrade the release."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = HelmTarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(lambda: target.install(ctx, runner, context_dir=dest, namespace=namespace))
    click.echo(f"installed Helm release {ctx.app_name!r}")


@deploy.command(name="cloud-run")
@context_options
@click.option("--project", required=True, help="GCP project id.")
@click.option("--region", required=True, help="Cloud Run region.")
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Build context (must contain the generated Dockerfile).",
)
@click.option(
    "--allow-unauthenticated",
    is_flag=True,
    default=False,
    help="Make the service publicly invokable (default requires authentication).",
)
@click.option(
    "--min-instances",
    type=int,
    default=0,
    show_default=True,
    help="Warm instances to avoid cold starts (0 scales to zero).",
)
@click.option("--no-generate", is_flag=True, default=False, help="Use artifacts already present; do not write them.")
@framework_errors
def cloud_run(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    project: str,
    region: str,
    dest: Path,
    allow_unauthenticated: bool,
    min_instances: int,
    no_generate: bool,
) -> None:
    """Build from source via Cloud Build and deploy to Cloud Run."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = CloudRunTarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(
        lambda: target.deploy(
            ctx,
            runner,
            project=project,
            region=region,
            source_dir=dest,
            allow_unauthenticated=allow_unauthenticated,
            min_instances=min_instances,
        )
    )
    click.echo(f"deployed {ctx.app_name!r} to Cloud Run in {region}")


@deploy.command(name="ecs")
@context_options
@click.option("--region", required=True, help="AWS region.")
@click.option("--execution-role-arn", required=True, help="ECS task execution role ARN (ECR pull + secrets).")
@click.option("--cluster", default=None, help="Existing ECS cluster to roll (with --service).")
@click.option("--service", default=None, help="Existing ECS service to force a new deployment on.")
@click.option("--push", is_flag=True, default=False, help="Log in to ECR, build, and push the image before deploying.")
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Directory to write artifacts into.",
)
@click.option("--no-generate", is_flag=True, default=False, help="Use artifacts already present; do not write them.")
@framework_errors
def ecs(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    region: str,
    execution_role_arn: str,
    cluster: str | None,
    service: str | None,
    push: bool,
    dest: Path,
    no_generate: bool,
) -> None:
    """Register an ECS Fargate task definition (and optionally roll a service)."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = ECSTarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(
        lambda: target.deploy(
            ctx,
            runner,
            region=region,
            execution_role_arn=execution_role_arn,
            cluster=cluster,
            service=service,
            push=push,
            context_dir=dest,
        )
    )
    click.echo(f"registered ECS task definition {ctx.app_name!r}")


@deploy.command(name="app-runner")
@context_options
@click.option("--region", required=True, help="AWS region.")
@click.option("--access-role-arn", required=True, help="IAM role ARN App Runner uses to pull from ECR.")
@click.option("--push", is_flag=True, default=False, help="Log in to ECR, build, and push the image before deploying.")
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Directory to write artifacts into.",
)
@click.option("--no-generate", is_flag=True, default=False, help="Use artifacts already present; do not write them.")
@framework_errors
def app_runner(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    region: str,
    access_role_arn: str,
    push: bool,
    dest: Path,
    no_generate: bool,
) -> None:
    """Create an App Runner service from the ECR image."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = AppRunnerTarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(
        lambda: target.deploy(ctx, runner, region=region, access_role_arn=access_role_arn, push=push, context_dir=dest)
    )
    click.echo(f"created App Runner service {ctx.app_name!r}")


@deploy.command(name="lambda")
@context_options
@click.option("--region", required=True, help="AWS region.")
@click.option("--function-name", default=None, help="Target function name (defaults to the app name).")
@click.option("--push", is_flag=True, default=False, help="Log in to ECR, build, and push the image before deploying.")
@click.option(
    "--dir",
    "dest",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    help="Directory to write artifacts into.",
)
@click.option("--no-generate", is_flag=True, default=False, help="Use artifacts already present; do not write them.")
@framework_errors
def lambda_(
    agent_ref: str,
    image: str,
    app_name: str | None,
    port: int,
    extras: str,
    env_keys: tuple[str, ...],
    region: str,
    function_name: str | None,
    push: bool,
    dest: Path,
    no_generate: bool,
) -> None:
    """Update a Lambda function's image (built from the Web Adapter Dockerfile)."""
    ctx = _context(agent_ref, image, app_name, port, extras, env_keys)
    target = LambdaTarget()
    if not no_generate:
        write_artifacts(target.generate(ctx), dest, force=False)
    runner = SubprocessRunner()
    _ship(lambda: target.deploy(ctx, runner, region=region, function_name=function_name, push=push, context_dir=dest))
    click.echo(f"updated Lambda function image for {ctx.app_name!r}")
