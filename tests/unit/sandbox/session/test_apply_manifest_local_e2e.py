"""End-to-end: LocalSubprocessSandboxClient.apply_manifest materializes.

Proves the manifest wiring (`create(manifest=)` → session retains it
→ `apply_manifest` drives `materialize_manifest`) against a REAL
local workspace on disk — hermetic (no external deps; `tmp_path`
workdir). The Docker/K8s `apply_manifest` bodies are byte-identical
to the local one; their real materialization is integration-tier
(needs container/cluster infra) and out of unit scope.
"""

from __future__ import annotations

from pathlib import Path

from troopai.adk.sandbox.clients.local.subprocess_client import (
    LocalSandboxClientOptions,
    LocalSubprocessSandboxClient,
)
from troopai.adk.types.sandbox.entries import Dir, File
from troopai.adk.types.sandbox.manifest import Manifest


class TestLocalApplyManifestE2E:
    async def test_manifest_materializes_to_real_workspace(self, tmp_path: Path) -> None:
        manifest = Manifest(
            entries={
                "a.txt": File(content=b"hello-manifest"),
                "pkg": Dir(children={"mod.py": File(content=b"print(1)\n")}),
            },
        )
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(
            manifest=manifest,
            options=LocalSandboxClientOptions(working_directory=str(tmp_path)),
        )
        await session.start()
        try:
            result = await session.apply_manifest()
        finally:
            await session.aclose()
        assert (tmp_path / "a.txt").read_bytes() == b"hello-manifest"
        assert (tmp_path / "pkg").is_dir()
        assert (tmp_path / "pkg" / "mod.py").read_bytes() == b"print(1)\n"
        # Explicit dir-traverse canary: a Dir's default 0o644 is
        # augmented to 0o755 (+x where read); reverting that fix makes
        # the pkg/mod.py write above fail with EACCES.
        assert (tmp_path / "pkg").stat().st_mode & 0o777 == 0o755
        materialized = {f.path for f in result.files}
        assert materialized == {"a.txt", "pkg", "pkg/mod.py"}

    async def test_no_manifest_returns_empty_not_error(self, tmp_path: Path) -> None:
        # A None manifest is "nothing declared", NOT a bug — empty
        # result, no crash (the guard before materialize_manifest).
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(
            options=LocalSandboxClientOptions(working_directory=str(tmp_path)),
        )
        await session.start()
        try:
            result = await session.apply_manifest()
        finally:
            await session.aclose()
        assert result.files == []
        assert result.skipped_mounts == []

    async def test_session_retains_manifest(self, tmp_path: Path) -> None:
        manifest = Manifest(entries={"x.txt": File(content=b"x")})
        client = LocalSubprocessSandboxClient(warn_banner=False)
        session = await client.create(
            manifest=manifest,
            options=LocalSandboxClientOptions(working_directory=str(tmp_path)),
        )
        try:
            assert session.get_manifest() is manifest
        finally:
            await session.aclose()
