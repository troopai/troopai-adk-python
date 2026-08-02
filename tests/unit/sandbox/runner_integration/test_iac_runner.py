"""Tests for the IaC runner helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import NoReturn
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.exceptions.exceptions import SandboxConfigurationError
from troopai.adk.sandbox.runner_integration.iac_runner import apply_iac, destroy_iac
from troopai.adk.types.sandbox.iac import IaCBundle

_PATCH = "troopai.adk.sandbox.runner_integration.iac_runner.asyncio.create_subprocess_exec"
_WAIT_FOR = "troopai.adk.sandbox.runner_integration.iac_runner.asyncio.wait_for"


class TestIaCBundle:
    def test_terraform_bundle(self) -> None:
        b = IaCBundle(provider="terraform", working_directory="/opt/iac")
        assert b.provider == "terraform"

    def test_pulumi_bundle(self) -> None:
        b = IaCBundle(provider="pulumi", working_directory="/opt/iac")
        assert b.provider == "pulumi"


class TestApplyIacIsAsync:
    def test_apply_iac_is_coroutine_function(self) -> None:
        import inspect

        assert inspect.iscoroutinefunction(apply_iac)

    def test_destroy_iac_is_coroutine_function(self) -> None:
        import inspect

        assert inspect.iscoroutinefunction(destroy_iac)


def _proc(*, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    """A fake subprocess.Process with awaitable communicate/wait."""
    p = MagicMock()
    p.communicate = AsyncMock(return_value=(stdout, stderr))
    p.wait = AsyncMock(return_value=returncode)
    p.returncode = returncode
    p.kill = MagicMock()
    return p


_REAL_WAIT_FOR = asyncio.wait_for


class _WaitForExpiry:
    """Model ``asyncio.wait_for`` expiry only for the Nth invocation.

    The runner calls ``wait_for`` more than once per apply (apply stage,
    then the output-fetch stage). To pin a timeout on a SPECIFIC stage,
    expire only on the chosen call index and otherwise delegate to the
    genuine ``asyncio.wait_for`` captured before patching. The inner
    coroutine is closed before raising so the AsyncMock coroutine does
    not emit a spurious "never awaited" warning.
    """

    def __init__(self, *, expire_on_call: int) -> None:
        self._expire_on_call = expire_on_call
        self._calls = 0

    async def __call__(
        self,
        awaitable: Coroutine[object, object, object],
        *,
        timeout: object = None,
    ) -> object:
        self._calls += 1
        if self._calls == self._expire_on_call:
            awaitable.close()
            raise TimeoutError
        return await _REAL_WAIT_FOR(awaitable, timeout=timeout)  # type: ignore[arg-type]


async def _expire_wait_for(awaitable: Coroutine[object, object, object], *, timeout: object = None) -> NoReturn:
    """Model ``asyncio.wait_for`` always expiring (first invocation)."""
    del timeout
    awaitable.close()
    raise TimeoutError


class TestTerraformChdirArgFormat:
    """``-chdir`` MUST be the ``-chdir=PATH`` single-arg form.

    Terraform's global ``-chdir`` option is only recognised when it
    carries its value via ``=`` (e.g. ``terraform -chdir=/p apply``).
    A space-separated pair (``-chdir`` then ``/p`` as the next argv
    entry) is misparsed — the path becomes the subcommand and the run
    fails — so every terraform invocation must use the joined form.
    """

    async def test_apply_uses_chdir_equals_form(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        with patch(_PATCH, AsyncMock(side_effect=[_proc(), _proc(stdout=b"{}")])) as m:
            await apply_iac(bundle)
        apply_args = m.call_args_list[0].args
        output_args = m.call_args_list[1].args
        assert "-chdir=/opt/iac" in apply_args
        assert "-chdir=/opt/iac" in output_args
        # The broken two-arg form must NOT appear.
        assert "-chdir" not in apply_args
        assert "-chdir" not in output_args

    async def test_destroy_uses_chdir_equals_form(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        with patch(_PATCH, AsyncMock(side_effect=[_proc()])) as m:
            await destroy_iac(bundle)
        args = m.call_args_list[0].args
        assert "-chdir=/opt/iac" in args
        assert "-chdir" not in args


class TestOutputFetchTimeout:
    """The output-fetch stage must enforce the timeout: kill + map error."""

    async def test_terraform_output_timeout_kills_and_raises(self) -> None:
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            output_env_mapping={"db_url": "DATABASE_URL"},
            timeout=3.0,
        )
        apply_p = _proc(returncode=0)
        out_p = _proc()
        # Expire on the SECOND wait_for (the `output -json` stage); the
        # first (apply) succeeds normally.
        with (
            patch(_PATCH, AsyncMock(side_effect=[apply_p, out_p])),
            patch(_WAIT_FOR, _WaitForExpiry(expire_on_call=2)),
            pytest.raises(SandboxConfigurationError, match="terraform output -json timed out after 3.0s"),
        ):
            await apply_iac(bundle)
        out_p.kill.assert_called_once()
        apply_p.kill.assert_not_called()

    async def test_pulumi_output_timeout_kills_and_raises(self) -> None:
        bundle = IaCBundle(
            provider="pulumi",
            working_directory="/opt/iac",
            output_env_mapping={"endpoint": "ENDPOINT"},
            timeout=4.0,
        )
        apply_p = _proc(returncode=0)
        out_p = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[apply_p, out_p])),
            patch(_WAIT_FOR, _WaitForExpiry(expire_on_call=2)),
            pytest.raises(SandboxConfigurationError, match="pulumi stack output --json timed out after 4.0s"),
        ):
            await apply_iac(bundle)
        out_p.kill.assert_called_once()
        apply_p.kill.assert_not_called()


class TestDestroyTimeout:
    """destroy must kill the CLI process on timeout (documented contract)."""

    async def test_terraform_destroy_timeout_kills_and_raises(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac", timeout=2.0)
        proc = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[proc])),
            patch(_WAIT_FOR, _expire_wait_for),
            pytest.raises(SandboxConfigurationError, match="terraform destroy timed out after 2.0s"),
        ):
            await destroy_iac(bundle)
        proc.kill.assert_called_once()

    async def test_pulumi_destroy_timeout_kills_and_raises(self) -> None:
        bundle = IaCBundle(provider="pulumi", working_directory="/opt/iac", timeout=1.0)
        proc = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[proc])),
            patch(_WAIT_FOR, _expire_wait_for),
            pytest.raises(SandboxConfigurationError, match="pulumi destroy timed out after 1.0s"),
        ):
            await destroy_iac(bundle)
        proc.kill.assert_called_once()
