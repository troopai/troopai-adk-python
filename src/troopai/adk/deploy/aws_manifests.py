"""AWS artifact renderers — ECS Fargate, App Runner, and Lambda.

* ECS: a Fargate task definition (JSON) registered via the aws CLI.
* App Runner: an ``aws apprunner create-service`` input document (JSON).
* Lambda: a container image with the AWS Lambda Web Adapter so the same
  ``troopai serve`` HTTP app runs behind Lambda's invoke model.

Account-specific values (region, IAM role ARNs) default to clearly-marked
placeholders for the generated reference artifacts; the deploy commands
inject the real values at ship time.
"""

from __future__ import annotations

import json
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.deploy.context import DeployContext

PLACEHOLDER_REGION = "REPLACE_REGION"
PLACEHOLDER_ROLE_ARN = "REPLACE_WITH_ROLE_ARN"

# Pinned third-party image providing the AWS Lambda Web Adapter binary.
_LWA_IMAGE = "public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4"


def render_ecs_task_definition(
    ctx: DeployContext,
    *,
    region: str = PLACEHOLDER_REGION,
    execution_role_arn: str = PLACEHOLDER_ROLE_ARN,
) -> str:
    """Render a Fargate task definition as a JSON string.

    Args:
        ctx: The deploy context.
        region: AWS region (for the CloudWatch log group).
        execution_role_arn: Task execution role ARN (ECR pull + secrets).

    Returns:
        The task definition as pretty-printed JSON.
    """
    container = {
        "name": ctx.app_name,
        "image": ctx.image,
        "essential": True,
        "portMappings": [{"containerPort": ctx.port, "protocol": "tcp"}],
        "environment": [
            {"name": "PORT", "value": str(ctx.port)},
            {"name": "AGENT_REF", "value": ctx.agent_ref},
        ],
        "secrets": [{"name": key, "valueFrom": key} for key in ctx.env_keys],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": f"/ecs/{ctx.app_name}",
                "awslogs-region": region,
                "awslogs-stream-prefix": "ecs",
            },
        },
        "healthCheck": {
            "command": [
                "CMD-SHELL",
                f"python -c \"import urllib.request as u; u.urlopen('http://localhost:{ctx.port}/healthz')\"",
            ],
            "interval": 30,
            "timeout": 5,
            "retries": 3,
            "startPeriod": 30,
        },
    }
    task_definition = {
        "family": ctx.app_name,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "512",
        "memory": "1024",
        "executionRoleArn": execution_role_arn,
        "containerDefinitions": [container],
    }
    return json.dumps(task_definition, indent=2)


def render_ecs(ctx: DeployContext) -> dict[str, str]:
    """Render the ECS artifact (task definition), keyed by path."""
    return {"deploy/aws-ecs/task-definition.json": render_ecs_task_definition(ctx)}


def render_apprunner_create_service(ctx: DeployContext, *, access_role_arn: str = PLACEHOLDER_ROLE_ARN) -> str:
    """Render the ``aws apprunner create-service`` input as a JSON string.

    Args:
        ctx: The deploy context.
        access_role_arn: IAM role ARN App Runner uses to pull from ECR.

    Returns:
        The create-service input as pretty-printed JSON.
    """
    document = {
        "ServiceName": ctx.app_name,
        "SourceConfiguration": {
            "ImageRepository": {
                "ImageIdentifier": ctx.image,
                "ImageRepositoryType": "ECR",
                "ImageConfiguration": {
                    "Port": str(ctx.port),
                    "RuntimeEnvironmentVariables": {"AGENT_REF": ctx.agent_ref},
                },
            },
            "AuthenticationConfiguration": {"AccessRoleArn": access_role_arn},
            "AutoDeploymentsEnabled": False,
        },
        "InstanceConfiguration": {"Cpu": "1024", "Memory": "2048"},
        "HealthCheckConfiguration": {"Protocol": "HTTP", "Path": "/healthz"},
    }
    return json.dumps(document, indent=2)


def render_apprunner(ctx: DeployContext) -> dict[str, str]:
    """Render the App Runner artifact (create-service input), keyed by path."""
    return {"deploy/aws-apprunner/create-service.json": render_apprunner_create_service(ctx)}


_LAMBDA_DOCKERFILE = Template(
    """\
# Lambda container image: the AWS Lambda Web Adapter forwards each Lambda
# invocation to the troopai serve HTTP app on AWS_LWA_PORT. The same
# image contract as the other targets — only the front door differs.
FROM python:$python_version-slim

COPY --from=$lwa_image /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    AWS_LWA_PORT=$port \\
    PORT=$port \\
    AGENT_REF=$agent_ref

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

# The adapter polls READINESS on $PORT; serve binds 0.0.0.0 and exposes
# /healthz + /readyz.
CMD ["sh", "-c", "troopai serve --agent \\"$AGENT_REF\\" --host 0.0.0.0 --port \\"$PORT\\""]
"""
)


def render_lambda_dockerfile(ctx: DeployContext) -> str:
    """Render the Lambda (Web Adapter) Dockerfile for *ctx*."""
    return _LAMBDA_DOCKERFILE.safe_substitute(
        python_version=ctx.python_version,
        lwa_image=_LWA_IMAGE,
        port=ctx.port,
        agent_ref=ctx.agent_ref,
    )


def render_lambda(ctx: DeployContext) -> dict[str, str]:
    """Render the Lambda artifact (LWA Dockerfile), keyed by path."""
    return {"deploy/aws-lambda/Dockerfile": render_lambda_dockerfile(ctx)}
