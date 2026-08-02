"""Tests for ``troopai.adk.types.sandbox.iac``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.iac import IaCBundle


class TestIaCBundleConstruction:
    def test_terraform_minimal(self) -> None:
        b = IaCBundle(provider="terraform", working_directory="/opt/iac/myinfra")
        assert b.provider == "terraform"
        assert b.working_directory == "/opt/iac/myinfra"
        assert b.variables == {}
        assert b.output_env_mapping == {}
        assert b.destroy_on_exit is True
        assert b.timeout == 300.0

    def test_pulumi_minimal(self) -> None:
        b = IaCBundle(provider="pulumi", working_directory="/opt/iac/myproj")
        assert b.provider == "pulumi"


class TestIaCBundleValidation:
    def test_empty_working_directory_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            IaCBundle(provider="terraform", working_directory="")

    def test_relative_working_directory_rejected(self) -> None:
        with pytest.raises(ValueError, match="absolute path"):
            IaCBundle(provider="terraform", working_directory="myinfra")

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            IaCBundle(provider="terraform", working_directory="/opt/iac", timeout=0.0)

    def test_empty_output_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="keys must be non-empty"):
            IaCBundle(
                provider="terraform",
                working_directory="/opt/iac",
                output_env_mapping={"": "FOO"},
            )

    def test_empty_env_var_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            IaCBundle(
                provider="terraform",
                working_directory="/opt/iac",
                output_env_mapping={"db_endpoint": ""},
            )


class TestIaCBundleFullConfig:
    def test_full_terraform_bundle(self) -> None:
        b = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac/prod",
            variables={"region": "us-east-1", "instance_type": "t3.medium"},
            output_env_mapping={
                "db_endpoint": "DATABASE_URL",
                "s3_bucket": "STORAGE_BUCKET",
            },
            destroy_on_exit=False,
            timeout=600.0,
        )
        assert len(b.variables) == 2
        assert b.output_env_mapping["db_endpoint"] == "DATABASE_URL"
        assert b.destroy_on_exit is False
        assert b.timeout == 600.0
