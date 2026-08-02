"""Tests for the AWS targets (ECS, App Runner, Lambda)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from troopai.adk.deploy.aws_manifests import (
    render_apprunner_create_service,
    render_ecs_task_definition,
    render_lambda_dockerfile,
)
from troopai.adk.deploy.commands import CommandResult, RecordingRunner
from troopai.adk.deploy.context import DeployContext
from troopai.adk.deploy.targets.apprunner import AppRunnerTarget
from troopai.adk.deploy.targets.aws_lambda import LambdaTarget
from troopai.adk.deploy.targets.ecs import ECSTarget


def _ctx(env_keys: tuple[str, ...] = ()) -> DeployContext:
    return DeployContext(
        agent_ref="app:agent",
        image="acct.dkr.ecr.r.amazonaws.com/my-agent:1",
        app_name="my-agent",
        env_keys=env_keys,
    )


def test_ecs_task_definition_valid_json_with_port_and_secrets() -> None:
    doc = json.loads(
        render_ecs_task_definition(
            _ctx(env_keys=("OPENAI_API_KEY",)), region="us-east-1", execution_role_arn="arn:role"
        )
    )
    assert doc["requiresCompatibilities"] == ["FARGATE"]
    assert doc["executionRoleArn"] == "arn:role"
    container = doc["containerDefinitions"][0]
    assert container["portMappings"][0]["containerPort"] == 8080
    assert {secret["name"] for secret in container["secrets"]} == {"OPENAI_API_KEY"}
    assert container["logConfiguration"]["options"]["awslogs-region"] == "us-east-1"


def test_ecs_deploy_registers_and_rolls_service() -> None:
    runner = RecordingRunner()
    ECSTarget().deploy(_ctx(), runner, region="us-east-1", execution_role_arn="arn:role", cluster="c", service="s")
    assert runner.calls[0][:3] == ["aws", "ecs", "register-task-definition"]
    assert runner.calls[1][:3] == ["aws", "ecs", "update-service"]


def test_ecs_deploy_without_service_only_registers() -> None:
    runner = RecordingRunner()
    ECSTarget().deploy(_ctx(), runner, region="r", execution_role_arn="arn")
    assert len(runner.calls) == 1


def test_ecs_deploy_with_push_logs_in_builds_pushes_then_registers() -> None:
    runner = RecordingRunner(results=[CommandResult(returncode=0, stdout="pw", stderr="")])
    ECSTarget().deploy(_ctx(), runner, region="us-east-1", execution_role_arn="arn", push=True, context_dir=Path("/wk"))
    assert [call[0] for call in runner.calls[:4]] == ["aws", "docker", "docker", "docker"]
    assert runner.calls[0][:3] == ["aws", "ecr", "get-login-password"]
    assert "register-task-definition" in runner.calls[-1]


def test_ecs_push_without_context_dir_raises() -> None:
    runner = RecordingRunner()
    with pytest.raises(ValueError):
        ECSTarget().deploy(_ctx(), runner, region="r", execution_role_arn="arn", push=True)


def test_apprunner_create_service_json() -> None:
    doc = json.loads(render_apprunner_create_service(_ctx(), access_role_arn="arn:apprunner"))
    assert doc["ServiceName"] == "my-agent"
    image = doc["SourceConfiguration"]["ImageRepository"]
    assert image["ImageIdentifier"] == "acct.dkr.ecr.r.amazonaws.com/my-agent:1"
    assert image["ImageConfiguration"]["Port"] == "8080"
    assert doc["SourceConfiguration"]["AuthenticationConfiguration"]["AccessRoleArn"] == "arn:apprunner"
    assert doc["HealthCheckConfiguration"]["Path"] == "/healthz"


def test_apprunner_deploy_runs_create_service() -> None:
    runner = RecordingRunner()
    AppRunnerTarget().deploy(_ctx(), runner, region="us-east-1", access_role_arn="arn")
    assert runner.calls[0][:3] == ["aws", "apprunner", "create-service"]


def test_lambda_dockerfile_uses_web_adapter() -> None:
    text = render_lambda_dockerfile(_ctx())
    assert "lambda-adapter" in text
    assert "AWS_LWA_PORT" in text
    assert "--host 0.0.0.0" in text


def test_lambda_generate_uses_lambda_dockerfile() -> None:
    files = LambdaTarget().generate(_ctx())
    assert "deploy/aws-lambda/Dockerfile" in files
    assert "requirements.txt" in files


def test_lambda_deploy_updates_function_code() -> None:
    runner = RecordingRunner()
    LambdaTarget().deploy(_ctx(), runner, region="us-east-1", function_name="my-fn")
    call = runner.calls[0]
    assert call[:3] == ["aws", "lambda", "update-function-code"]
    assert "my-fn" in call
    assert "--image-uri" in call
