"""LocalSandboxSession.apply_patch delivers the diff to ``patch`` via a file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from troopai.adk.sandbox.clients.local.subprocess_client import LocalSandboxSession
from troopai.adk.types.sandbox.exec_result import ExecResult


async def test_apply_patch_delivers_diff_via_file(tmp_path: Path) -> None:
    """The diff must reach ``patch`` through a real ``-i <file>``.

    Regression: ``apply_patch`` called ``run("patch","-p1","-i","-")``, but
    ``run`` does not pipe stdin, so the diff was never delivered — the patch
    silently applied nothing (or hung). It now writes the diff to a temp file
    and feeds ``patch -i <file>``; this captures the ``-i`` path and confirms
    its contents are the diff.
    """
    session = LocalSandboxSession(
        working_directory=tmp_path,
        cleanup_on_shutdown=False,
        default_env={},
    )

    captured: dict[str, str] = {}

    async def _fake_run(*command: Any, **kwargs: Any) -> ExecResult:
        del kwargs
        patch_file = str(command[-1])
        captured["i_arg"] = patch_file
        captured["content"] = Path(patch_file).read_text(encoding="utf-8")
        return ExecResult(stdout=b"", stderr=b"", exit_code=0, duration_ms=1)

    session.run = _fake_run  # type: ignore[method-assign]

    diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"
    out = await session.apply_patch(diff)

    # The diff reached patch through a real file, not the literal "-".
    assert captured["i_arg"] != "-"
    assert captured["content"] == diff
    assert "OK" in out
    # The temp file is cleaned up after the call.
    assert not Path(captured["i_arg"]).exists()
