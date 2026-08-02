"""Subprocess-level tests for the IaC runner (apply_iac / destroy_iac).

The existing smoke test only checks `IaCBundle` construction and that
the helpers are coroutine functions. This module mocks
``asyncio.create_subprocess_exec`` to exercise the real control flow:
provider dispatch, zero / non-zero exit, timeout, malformed output
JSON, the output → env-var mapping, and destroy.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Coroutine
from typing import NoReturn
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from troopai.adk.exceptions.exceptions import SandboxConfigurationError
from troopai.adk.sandbox.runner_integration.iac_runner import _terminate_and_reap, apply_iac, destroy_iac
from troopai.adk.types.sandbox.iac import IaCBundle

_PATCH = "troopai.adk.sandbox.runner_integration.iac_runner.asyncio.create_subprocess_exec"
_WAIT_FOR = "troopai.adk.sandbox.runner_integration.iac_runner.asyncio.wait_for"


async def _expire_wait_for(awaitable: Coroutine[object, object, object], *, timeout: object = None) -> NoReturn:
    """Model ``asyncio.wait_for`` detecting wall-clock expiry.

    Patching at the ``wait_for`` boundary (rather than making
    ``communicate`` raise) means the timeout test fails loud if the
    SUT ever drops the ``wait_for`` wrapper — i.e. it pins that the
    timeout is actually ENFORCED, not just that the handler exists.
    The inner coroutine is closed first so the AsyncMock coroutine
    does not emit a spurious "never awaited" warning.
    """
    del timeout
    awaitable.close()
    raise TimeoutError


async def _cancel_wait_for(awaitable: Coroutine[object, object, object], *, timeout: object = None) -> NoReturn:
    del timeout
    awaitable.close()
    raise asyncio.CancelledError


class _ExpireWaitForOnCall:
    """A ``wait_for`` stub that expires on a chosen call index.

    Lets a test drive a timeout on the SECOND subprocess (the ``output``
    query) after the first (``apply`` / ``up``) succeeded.
    """

    def __init__(self, expire_at: int) -> None:
        self._expire_at = expire_at
        self._n = 0

    async def __call__(
        self,
        awaitable: Coroutine[object, object, object],
        *,
        timeout: object = None,
    ) -> object:
        del timeout
        i = self._n
        self._n += 1
        if i == self._expire_at:
            awaitable.close()
            raise TimeoutError
        return await awaitable


def _proc(*, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"", pid: int | None = None) -> MagicMock:
    """A fake subprocess.Process with awaitable communicate/wait."""
    p = MagicMock()
    p.communicate = AsyncMock(return_value=(stdout, stderr))
    p.wait = AsyncMock(return_value=returncode)
    p.returncode = returncode
    p.kill = MagicMock()
    if pid is not None:
        p.pid = pid
    return p


class TestTerraformApply:
    async def test_zero_exit_unwraps_value_dict(self) -> None:
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            variables={"region": "us-east-1"},
            output_env_mapping={"db_url": "DATABASE_URL"},
            timeout=5.0,
        )
        apply_p = _proc(returncode=0)
        out_p = _proc(stdout=b'{"db_url": {"value": "postgres://h/db"}}')
        with patch(_PATCH, AsyncMock(side_effect=[apply_p, out_p])) as m:
            env = await apply_iac(bundle)
        assert env == {"DATABASE_URL": "postgres://h/db"}
        # apply invocation carries the -var pair + -chdir + -json.
        apply_args = m.call_args_list[0].args
        assert "apply" in apply_args
        assert "region=us-east-1" in apply_args

    async def test_plain_scalar_output_stringified(self) -> None:
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            output_env_mapping={"port": "PORT"},
        )
        with patch(_PATCH, AsyncMock(side_effect=[_proc(), _proc(stdout=b'{"port": 8080}')])):
            env = await apply_iac(bundle)
        assert env == {"PORT": "8080"}

    async def test_nonzero_exit_raises_configuration_error(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        failed = _proc(returncode=1, stderr=b"backend init failed")
        with (
            patch(_PATCH, AsyncMock(side_effect=[failed])),
            pytest.raises(SandboxConfigurationError, match=r"terraform apply failed \(exit=1\)"),
        ):
            await apply_iac(bundle)

    async def test_timeout_raises_and_kills(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac", timeout=1.0)
        # communicate would succeed; the timeout originates from
        # asyncio.wait_for itself — so this fails loud if the SUT ever
        # drops the wait_for wrapper (timeout no longer ENFORCED).
        proc = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[proc])),
            patch(_WAIT_FOR, _expire_wait_for),
            pytest.raises(SandboxConfigurationError, match="terraform apply timed out after 1.0s"),
        ):
            await apply_iac(bundle)
        proc.kill.assert_called_once()

    async def test_malformed_output_json_raises(self) -> None:
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            output_env_mapping={"db_url": "DATABASE_URL"},
        )
        # Unparseable `output -json` must raise, not silently map to {} and
        # drop every output_env_mapping var after the infra was applied.
        with (
            patch(_PATCH, AsyncMock(side_effect=[_proc(), _proc(stdout=b"not-json")])),
            pytest.raises(SandboxConfigurationError, match="unparseable JSON"),
        ):
            await apply_iac(bundle)

    async def test_output_nonzero_exit_raises(self) -> None:
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            output_env_mapping={"db_url": "DATABASE_URL"},
        )
        # apply succeeds (exit 0) but `output -json` fails (locked state):
        # the env vars must NOT silently vanish.
        with (
            patch(
                _PATCH,
                AsyncMock(side_effect=[_proc(returncode=0), _proc(returncode=1, stderr=b"state locked")]),
            ),
            pytest.raises(SandboxConfigurationError, match=r"terraform output -json failed \(exit=1\)"),
        ):
            await apply_iac(bundle)


class TestPulumiApply:
    async def test_zero_exit_maps_outputs(self) -> None:
        bundle = IaCBundle(
            provider="pulumi",
            working_directory="/opt/iac",
            output_env_mapping={"endpoint": "ENDPOINT"},
        )
        with patch(_PATCH, AsyncMock(side_effect=[_proc(), _proc(stdout=b'{"endpoint": "https://x"}')])):
            env = await apply_iac(bundle)
        assert env == {"ENDPOINT": "https://x"}

    async def test_nonzero_exit_raises(self) -> None:
        bundle = IaCBundle(provider="pulumi", working_directory="/opt/iac")
        with (
            patch(_PATCH, AsyncMock(side_effect=[_proc(returncode=2, stderr=b"pulumi boom")])),
            pytest.raises(SandboxConfigurationError, match=r"pulumi up failed \(exit=2\)"),
        ):
            await apply_iac(bundle)

    async def test_output_nonzero_exit_raises(self) -> None:
        bundle = IaCBundle(
            provider="pulumi",
            working_directory="/opt/iac",
            output_env_mapping={"endpoint": "ENDPOINT"},
        )
        # `pulumi up` succeeds but `stack output --json` fails (no stack
        # selected): raise rather than drop the env mapping silently.
        with (
            patch(
                _PATCH,
                AsyncMock(side_effect=[_proc(returncode=0), _proc(returncode=255, stderr=b"no stack selected")]),
            ),
            pytest.raises(SandboxConfigurationError, match=r"pulumi stack output --json failed \(exit=255\)"),
        ):
            await apply_iac(bundle)

    async def test_timeout_raises_and_kills(self) -> None:
        bundle = IaCBundle(provider="pulumi", working_directory="/opt/iac", timeout=2.0)
        # Timeout originates from asyncio.wait_for (production path),
        # not from communicate raising — pins that wait_for ENFORCES it.
        proc = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[proc])),
            patch(_WAIT_FOR, _expire_wait_for),
            pytest.raises(SandboxConfigurationError, match="pulumi up timed out after 2.0s"),
        ):
            await apply_iac(bundle)
        proc.kill.assert_called_once()


class TestProviderDispatch:
    async def test_unknown_provider_raises_configuration_error(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        # IaCBundle is a frozen dataclass; bypass the Literal to drive
        # the runtime else-branch (test-scaffolding only).
        object.__setattr__(bundle, "provider", "ansible")
        with pytest.raises(SandboxConfigurationError, match="must be 'terraform' or 'pulumi'"):
            await apply_iac(bundle)


class TestDestroy:
    async def test_terraform_destroy_invokes_cli(self) -> None:
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            variables={"k": "v"},
        )
        with patch(_PATCH, AsyncMock(side_effect=[_proc()])) as m:
            await destroy_iac(bundle)  # -> None by signature; behavior asserted via the mock
        args = m.call_args_list[0].args
        assert "destroy" in args
        assert "-auto-approve" in args
        assert "k=v" in args
        assert m.call_args_list[0].kwargs["start_new_session"] is True

    async def test_pulumi_destroy_invokes_cli(self) -> None:
        bundle = IaCBundle(provider="pulumi", working_directory="/opt/iac")
        with patch(_PATCH, AsyncMock(side_effect=[_proc()])) as m:
            await destroy_iac(bundle)  # -> None by signature; behavior asserted via the mock
        args = m.call_args_list[0].args
        assert "destroy" in args
        assert "--yes" in args
        assert m.call_args_list[0].kwargs["start_new_session"] is True

    async def test_terraform_destroy_nonzero_exit_raises(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac")
        with (
            patch(_PATCH, AsyncMock(side_effect=[_proc(returncode=1, stderr=b"destroy failed")])),
            pytest.raises(SandboxConfigurationError, match=r"terraform destroy failed \(exit=1\)"),
        ):
            await destroy_iac(bundle)

    async def test_pulumi_destroy_nonzero_exit_raises(self) -> None:
        bundle = IaCBundle(provider="pulumi", working_directory="/opt/iac")
        with (
            patch(_PATCH, AsyncMock(side_effect=[_proc(returncode=255, stderr=b"pulumi destroy failed")])),
            pytest.raises(SandboxConfigurationError, match=r"pulumi destroy failed \(exit=255\)"),
        ):
            await destroy_iac(bundle)

    async def test_destroy_cancellation_terminates_process_group_and_reaps(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac", timeout=1.0)
        proc = _proc(pid=1234)
        with (
            patch(_PATCH, AsyncMock(side_effect=[proc])),
            patch(_WAIT_FOR, _cancel_wait_for),
            patch("troopai.adk.sandbox.runner_integration.iac_runner.os.getpgid", return_value=1234) as getpgid,
            patch("troopai.adk.sandbox.runner_integration.iac_runner.os.killpg") as killpg,
            pytest.raises(asyncio.CancelledError),
        ):
            await destroy_iac(bundle)

        getpgid.assert_called_once_with(1234)
        killpg.assert_called_once()
        proc.wait.assert_awaited_once()


class TestTimeoutReapsChild:
    """Regression: a timed-out IaC subprocess must be reaped via ``await
    proc.wait()`` after ``kill()``; otherwise the killed child lingers as a
    zombie until GC. The apply / up / output query paths all needed the reap;
    only the destroy paths already had it.
    """

    async def test_terraform_apply_timeout_reaps(self) -> None:
        bundle = IaCBundle(provider="terraform", working_directory="/opt/iac", timeout=1.0)
        proc = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[proc])),
            patch(_WAIT_FOR, _expire_wait_for),
            pytest.raises(SandboxConfigurationError, match="terraform apply timed out"),
        ):
            await apply_iac(bundle)
        proc.kill.assert_called_once()
        proc.wait.assert_awaited_once()

    async def test_terraform_output_timeout_reaps(self) -> None:
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            output_env_mapping={"db_url": "DATABASE_URL"},
            timeout=1.0,
        )
        apply_p = _proc(returncode=0)
        out_p = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[apply_p, out_p])),
            patch(_WAIT_FOR, _ExpireWaitForOnCall(expire_at=1)),
            pytest.raises(SandboxConfigurationError, match="terraform output -json timed out"),
        ):
            await apply_iac(bundle)
        out_p.kill.assert_called_once()
        out_p.wait.assert_awaited_once()

    async def test_unresponsive_process_group_gets_sigkill_after_term(self) -> None:
        proc = _proc(pid=1234)
        wait_calls = 0

        async def wait_side_effect() -> int:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                await asyncio.Event().wait()
            return 0

        proc.wait = AsyncMock(side_effect=wait_side_effect)
        with (
            patch("troopai.adk.sandbox.runner_integration.iac_runner.os.getpgid", return_value=1234),
            patch("troopai.adk.sandbox.runner_integration.iac_runner.os.killpg") as killpg,
        ):
            await _terminate_and_reap(proc, grace_seconds=0.001)

        assert killpg.call_args_list == [
            call(1234, signal.SIGTERM),
            call(1234, signal.SIGKILL),
        ]
        assert proc.wait.await_count == 2

    async def test_pulumi_up_timeout_reaps(self) -> None:
        bundle = IaCBundle(provider="pulumi", working_directory="/opt/iac", timeout=2.0)
        proc = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[proc])),
            patch(_WAIT_FOR, _expire_wait_for),
            pytest.raises(SandboxConfigurationError, match="pulumi up timed out"),
        ):
            await apply_iac(bundle)
        proc.kill.assert_called_once()
        proc.wait.assert_awaited_once()

    async def test_pulumi_output_timeout_reaps(self) -> None:
        bundle = IaCBundle(
            provider="pulumi",
            working_directory="/opt/iac",
            output_env_mapping={"endpoint": "ENDPOINT"},
            timeout=2.0,
        )
        up_p = _proc(returncode=0)
        out_p = _proc()
        with (
            patch(_PATCH, AsyncMock(side_effect=[up_p, out_p])),
            patch(_WAIT_FOR, _ExpireWaitForOnCall(expire_at=1)),
            pytest.raises(SandboxConfigurationError, match="pulumi stack output --json timed out"),
        ):
            await apply_iac(bundle)
        out_p.kill.assert_called_once()
        out_p.wait.assert_awaited_once()


class TestOutputMapping:
    async def test_absent_output_skipped_present_unwrapped_and_stringified(self) -> None:
        # Exercises _map_outputs via apply: an unmapped output is
        # dropped; a {"value": ...} dict is unwrapped; a bare scalar
        # is str()-ified; an output absent from the mapping is ignored.
        bundle = IaCBundle(
            provider="terraform",
            working_directory="/opt/iac",
            output_env_mapping={"present": "PRESENT", "missing": "MISSING", "wrapped": "WRAPPED"},
        )
        outputs = b'{"present": 42, "wrapped": {"value": "deep"}, "unmapped": "ignored"}'
        with patch(_PATCH, AsyncMock(side_effect=[_proc(), _proc(stdout=outputs)])):
            env = await apply_iac(bundle)
        assert env == {"PRESENT": "42", "WRAPPED": "deep"}
        assert "MISSING" not in env  # absent output → no env entry
