"""Materialize a declared workspace manifest into a sandbox.

A ``RunConfig.sandbox`` ``Manifest`` is a declarative contract: it
says "before the agent runs, the workspace MUST contain these files
and directories." This example builds a manifest with three entry
kinds —

* ``File``      — inline bytes written verbatim,
* ``Dir``       — a directory with nested children,
* ``LocalFile`` — a file copied from a host path into the sandbox,

— and drives it through ``sandbox_run_context``, the SAME per-run
lifecycle bracket the Runner opens around every ``SandboxAgent``
loop. Materialization fires INSIDE that bracket, after the backend
session starts and BEFORE the agent loop, so by the time control
reaches the ``async with`` body every declared path already exists
on disk. We prove that by reading the bytes back.

A ``LocalFile`` whose ``src`` lives OUTSIDE the ADK working
directory is refused unless the manifest carries an explicit
``SandboxPathGrant`` for that location — the host-path
exfiltration defense. This example shows the intended workflow:
a least-privilege, read-only, audited grant scoped to exactly the
host directory the copy needs.

The local ``LocalSubprocessSandboxClient`` backend needs no network
and no provider account, so this runs end-to-end with no setup.

No external API key required.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import tempfile
from pathlib import Path

from troopai.adk.sandbox.clients.local.subprocess_client import (
    LocalSandboxClientOptions,
    LocalSubprocessSandboxClient,
)
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.runner_integration.lifecycle import sandbox_run_context
from troopai.adk.types.sandbox.entries import Dir, File, LocalFile
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.workspace_paths import SandboxPathGrant

logger = logging.getLogger(__name__)


def _write_host_source(host_root: Path) -> Path:
    """Write a real file on the HOST for the ``LocalFile`` entry to copy."""
    src = host_root / "host_seed.txt"
    src.write_bytes(b"copied-from-host\n")
    return src


def _build_manifest(host_src: Path, host_grant_dir: Path) -> Manifest:
    """Declare a workspace: an inline file, a nested dir, a host copy.

    ``host_grant_dir`` is granted read-only so the ``LocalFile`` copy
    of ``host_src`` (which lives outside the sandbox workspace) is
    permitted — the least-privilege host-path grant.
    """
    return Manifest(
        entries={
            "README.txt": File(content=b"materialized by the run lifecycle\n"),
            "pkg": Dir(
                children={
                    "__init__.py": File(content=b""),
                    "main.py": File(content=b"def run():\n    return 42\n"),
                }
            ),
            "seed.txt": LocalFile(src=host_src),
        },
        extra_path_grants=(
            SandboxPathGrant(
                path=str(host_grant_dir),
                read_only=True,
                description="host seed directory for the LocalFile copy",
            ),
        ),
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as host_tmp, tempfile.TemporaryDirectory() as work_tmp:
        host_dir = Path(host_tmp)
        host_src = _write_host_source(host_dir)
        workdir = Path(work_tmp)
        manifest = _build_manifest(host_src, host_dir)
        logger.info("Granted read-only host path for the LocalFile copy: %s", host_dir)

        config = SandboxRunConfig(
            client=LocalSubprocessSandboxClient(warn_banner=False),
            manifest=manifest,
            options=LocalSandboxClientOptions(working_directory=str(workdir)),
        )

        logger.info("=== Opening the run-lifecycle bracket (materialization fires inside) ===")
        async with sandbox_run_context(
            config=config,
            capabilities=[],
            run_as=None,
            concurrency_guard=None,
        ) as handle:
            # We are now at the "agent loop" phase. Every declared
            # entry already exists — apply_manifest ran between
            # session.start() and this point.
            session_id = handle.session.session_id
            logger.info("Live session id: %s", session_id if session_id is not None else "<unset>")

            readme = (workdir / "README.txt").read_bytes()
            init_py = (workdir / "pkg" / "__init__.py").read_bytes()
            main_py = (workdir / "pkg" / "main.py").read_bytes()
            seed = (workdir / "seed.txt").read_bytes()

            logger.info("README.txt        -> %r", readme)
            logger.info("pkg/__init__.py   -> %r (empty file)", init_py)
            logger.info("pkg/main.py       -> %r", main_py)
            logger.info("seed.txt (LocalFile copy) -> %r", seed)

            assert readme == b"materialized by the run lifecycle\n"
            assert init_py == b""
            assert main_py == b"def run():\n    return 42\n"
            assert seed == b"copied-from-host\n"

            # The Dir entry is materialized traversable (read implies
            # execute on directories — standard POSIX), so the
            # materializer could create its children.
            pkg_mode = (workdir / "pkg").stat().st_mode & 0o777
            logger.info("pkg/ mode -> %s (traversable directory)", oct(pkg_mode))
            assert pkg_mode == 0o755

        logger.info("Manifest materialization complete — workspace was ready before the agent loop.")


if __name__ == "__main__":
    asyncio.run(main())
