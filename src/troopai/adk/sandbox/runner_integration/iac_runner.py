"""IaC runners — apply / destroy Terraform or Pulumi bundles.

Subprocess-backed. Output → env-var mapping flows from
``IaCBundle.output_env_mapping`` into the sandbox environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from typing import TYPE_CHECKING, Any, Protocol

from troopai.adk.exceptions.exceptions import SandboxConfigurationError

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.iac import IaCBundle

logger = logging.getLogger(__name__)

__all__ = ["apply_iac", "destroy_iac"]

_PROCESS_REAP_GRACE_SECONDS = 5.0


class _ProcessLike(Protocol):
    """Small subprocess protocol used by IaC cleanup helpers."""

    @property
    def pid(self) -> int | None: ...

    def kill(self) -> None: ...

    async def wait(self) -> Any: ...


def _terminate_process_group(proc: _ProcessLike, sig: signal.Signals) -> None:
    """Signal a subprocess and, when possible, its process group."""
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(os.getpgid(pid), sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            logger.warning("Failed to signal IaC process group for pid=%s; falling back to child kill.", pid)
    kill = proc.kill
    kill()


async def _terminate_and_reap(
    proc: _ProcessLike,
    *,
    grace_seconds: float = _PROCESS_REAP_GRACE_SECONDS,
) -> None:
    """Terminate an IaC subprocess group and reap the immediate child."""
    _terminate_process_group(proc, signal.SIGTERM)
    wait = proc.wait
    try:
        async with asyncio.timeout(grace_seconds):
            await wait()
            return
    except TimeoutError:
        logger.warning("IaC process group did not exit after %.1fs; sending SIGKILL.", grace_seconds)
    _terminate_process_group(proc, signal.SIGKILL)
    await wait()


async def apply_iac(bundle: IaCBundle) -> dict[str, str]:
    """Apply ``bundle`` and return the output → env-var mapping.

    Routes ``"terraform"`` and ``"pulumi"`` providers to their CLI.
    Errors map to ``SandboxConfigurationError`` so the runner can
    surface them through the standard error hierarchy.
    """
    if bundle.provider == "terraform":
        return await _terraform_apply(bundle)
    if bundle.provider == "pulumi":
        return await _pulumi_apply(bundle)
    raise SandboxConfigurationError(
        f"IaCBundle.provider must be 'terraform' or 'pulumi', got {bundle.provider!r}",
    )


async def destroy_iac(bundle: IaCBundle) -> None:
    """Destroy ``bundle`` infrastructure. Best-effort.

    ``IaCBundle.provider`` is a ``Literal["terraform", "pulumi"]`` so
    the discriminator is exhaustive.
    """
    if bundle.provider == "terraform":
        await _terraform_destroy(bundle)
        return
    await _pulumi_destroy(bundle)


async def _terraform_apply(bundle: IaCBundle) -> dict[str, str]:
    args = ["terraform", f"-chdir={bundle.working_directory}", "apply", "-auto-approve", "-json"]
    for k, v in bundle.variables.items():
        args.extend(["-var", f"{k}={v}"])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=bundle.timeout,
        )
    except TimeoutError as exc:
        await _terminate_and_reap(proc)
        raise SandboxConfigurationError(
            f"terraform apply timed out after {bundle.timeout}s",
        ) from exc
    except asyncio.CancelledError:
        await _terminate_and_reap(proc)
        raise
    if proc.returncode != 0:
        raise SandboxConfigurationError(
            f"terraform apply failed (exit={proc.returncode}): {stderr.decode(errors='replace')[:500]}",
        )
    # terraform output -json provides the outputs we need.
    out_proc = await asyncio.create_subprocess_exec(
        "terraform",
        f"-chdir={bundle.working_directory}",
        "output",
        "-json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out_stdout, out_stderr = await asyncio.wait_for(out_proc.communicate(), timeout=bundle.timeout)
    except TimeoutError as exc:
        await _terminate_and_reap(out_proc)
        raise SandboxConfigurationError(
            f"terraform output -json timed out after {bundle.timeout}s",
        ) from exc
    except asyncio.CancelledError:
        await _terminate_and_reap(out_proc)
        raise
    # A non-zero exit (locked state, uninitialised workspace) would otherwise
    # fall through JSONDecodeError to an empty dict, silently dropping EVERY
    # output_env_mapping var from the sandbox after the infra was applied.
    if out_proc.returncode != 0:
        raise SandboxConfigurationError(
            f"terraform output -json failed (exit={out_proc.returncode}): {out_stderr.decode(errors='replace')[:500]}",
        )
    try:
        outputs = json.loads(out_stdout.decode())
    except json.JSONDecodeError as exc:
        raise SandboxConfigurationError(
            f"terraform output -json returned unparseable JSON: {out_stdout.decode(errors='replace')[:200]}",
        ) from exc
    return _map_outputs(outputs, bundle)


async def _terraform_destroy(bundle: IaCBundle) -> None:
    args = ["terraform", f"-chdir={bundle.working_directory}", "destroy", "-auto-approve"]
    for k, v in bundle.variables.items():
        args.extend(["-var", f"{k}={v}"])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=bundle.timeout)
    except TimeoutError as exc:
        logger.warning("terraform destroy timed out; terminating IaC process group.")
        await _terminate_and_reap(proc)
        raise SandboxConfigurationError(
            f"terraform destroy timed out after {bundle.timeout}s",
        ) from exc
    except asyncio.CancelledError:
        logger.warning("terraform destroy cancelled; terminating IaC process group.")
        await _terminate_and_reap(proc)
        raise
    if proc.returncode != 0:
        logger.warning("terraform destroy failed with exit=%s", proc.returncode)
        raise SandboxConfigurationError(
            f"terraform destroy failed (exit={proc.returncode}): {stderr.decode(errors='replace')[:500]}",
        )


async def _pulumi_apply(bundle: IaCBundle) -> dict[str, str]:
    proc = await asyncio.create_subprocess_exec(
        "pulumi",
        "up",
        "--yes",
        "--non-interactive",
        "--skip-preview",
        cwd=bundle.working_directory,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=bundle.timeout)
    except TimeoutError as exc:
        await _terminate_and_reap(proc)
        raise SandboxConfigurationError(
            f"pulumi up timed out after {bundle.timeout}s",
        ) from exc
    except asyncio.CancelledError:
        await _terminate_and_reap(proc)
        raise
    if proc.returncode != 0:
        raise SandboxConfigurationError(
            f"pulumi up failed (exit={proc.returncode}): {stderr.decode(errors='replace')[:500]}",
        )
    out_proc = await asyncio.create_subprocess_exec(
        "pulumi",
        "stack",
        "output",
        "--json",
        cwd=bundle.working_directory,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out_stdout, out_stderr = await asyncio.wait_for(out_proc.communicate(), timeout=bundle.timeout)
    except TimeoutError as exc:
        await _terminate_and_reap(out_proc)
        raise SandboxConfigurationError(
            f"pulumi stack output --json timed out after {bundle.timeout}s",
        ) from exc
    except asyncio.CancelledError:
        await _terminate_and_reap(out_proc)
        raise
    # A non-zero exit (no stack selected, locked state) would otherwise fall
    # through JSONDecodeError to an empty dict, silently dropping EVERY
    # output_env_mapping var from the sandbox after the infra was applied.
    if out_proc.returncode != 0:
        raise SandboxConfigurationError(
            f"pulumi stack output --json failed (exit={out_proc.returncode}): "
            f"{out_stderr.decode(errors='replace')[:500]}",
        )
    try:
        outputs = json.loads(out_stdout.decode())
    except json.JSONDecodeError as exc:
        raise SandboxConfigurationError(
            f"pulumi stack output --json returned unparseable JSON: {out_stdout.decode(errors='replace')[:200]}",
        ) from exc
    return _map_outputs(outputs, bundle)


async def _pulumi_destroy(bundle: IaCBundle) -> None:
    proc = await asyncio.create_subprocess_exec(
        "pulumi",
        "destroy",
        "--yes",
        "--non-interactive",
        cwd=bundle.working_directory,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=bundle.timeout)
    except TimeoutError as exc:
        logger.warning("pulumi destroy timed out; terminating IaC process group.")
        await _terminate_and_reap(proc)
        raise SandboxConfigurationError(
            f"pulumi destroy timed out after {bundle.timeout}s",
        ) from exc
    except asyncio.CancelledError:
        logger.warning("pulumi destroy cancelled; terminating IaC process group.")
        await _terminate_and_reap(proc)
        raise
    if proc.returncode != 0:
        logger.warning("pulumi destroy failed with exit=%s", proc.returncode)
        raise SandboxConfigurationError(
            f"pulumi destroy failed (exit={proc.returncode}): {stderr.decode(errors='replace')[:500]}",
        )


def _map_outputs(outputs: dict[str, object], bundle: IaCBundle) -> dict[str, str]:
    env: dict[str, str] = {}
    for output_name, env_var in bundle.output_env_mapping.items():
        value = outputs.get(output_name)
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if value is not None:
            env[env_var] = str(value)
    return env
