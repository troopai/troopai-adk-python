"""Tests for ``troopai.adk.sandbox.session.runtime_helpers``.

These execute the embedded POSIX-shell scripts directly with the
host ``sh`` so the actual workspace-escape / fingerprint logic is
verified — not merely the argv construction.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from troopai.adk.exceptions.exceptions import SandboxStartFailed
from troopai.adk.sandbox.session import (
    RESOLVE_WORKSPACE_PATH_HELPER,
    WORKSPACE_FINGERPRINT_HELPER,
    RuntimeHelperScript,
    install_runtime_helpers,
)
from troopai.adk.types.sandbox.exec_result import ExecResult


def _write_script(tmp_path: Path, helper: RuntimeHelperScript) -> Path:
    script = tmp_path / helper.name
    script.write_text(helper.content, encoding="utf-8")
    script.chmod(0o755)
    return script


class TestRuntimeHelperScript:
    def test_content_addressed_install_path(self) -> None:
        helper = RuntimeHelperScript.from_content(name="demo", content="#!/bin/sh\ntrue\n")
        assert helper.install_path.name.startswith("demo-")
        # 12-hex-char sha256 prefix.
        suffix = helper.install_path.name.removeprefix("demo-")
        assert len(suffix) == 12
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_distinct_content_distinct_path(self) -> None:
        a = RuntimeHelperScript.from_content(name="x", content="a")
        b = RuntimeHelperScript.from_content(name="x", content="b")
        assert a.install_path != b.install_path

    def test_install_path_under_troopai_root(self) -> None:
        helper = RuntimeHelperScript.from_content(name="x", content="a")
        assert str(helper.install_path).startswith("/tmp/troopai-adk-python/bin/")

    def test_present_command_shape(self) -> None:
        helper = RuntimeHelperScript.from_content(name="x", content="a")
        cmd = helper.present_command()
        assert cmd[0] == "test"
        assert cmd[1] == "-x"
        assert cmd[2] == str(helper.install_path)

    def test_install_command_is_idempotent_sh(self, tmp_path: Path) -> None:
        # Construct the helper with a tmp-rooted install_path so the
        # content-addressed temp file ("<install_path>.tmp.$$") and the
        # destination share a directory (the production invariant —
        # ``from_content`` would root both under /tmp/troopai-adk-python/bin).
        install_path = tmp_path / "bin" / "echo-helper-deadbeef0000"
        helper = RuntimeHelperScript(
            name="echo-helper",
            content="#!/bin/sh\necho installed\n",
            install_path=install_path,
        )
        argv = helper.install_command()
        # argv = ("sh","-c",<script>,"sh",<install_path>)
        run = subprocess.run(list(argv), capture_output=True, text=True, check=False)
        assert run.returncode == 0, run.stderr
        assert install_path.is_file()
        first_mtime = install_path.stat().st_mtime_ns
        # Re-running with identical content must be a no-op (cmp -s short-circuit).
        run2 = subprocess.run(list(argv), capture_output=True, text=True, check=False)
        assert run2.returncode == 0, run2.stderr
        assert install_path.stat().st_mtime_ns == first_mtime


class TestResolveWorkspacePathScript:
    def test_path_inside_root_is_accepted(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        (root / "sub").mkdir(parents=True)
        candidate = root / "sub"
        run = subprocess.run(
            [str(script), str(root), str(candidate), "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == str(candidate.resolve())

    def test_path_escaping_root_is_rejected(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        run = subprocess.run(
            [str(script), str(root), str(outside), "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 111
        assert "workspace escape" in run.stderr

    def test_symlink_escape_is_rejected(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "secret"
        outside.mkdir()
        link = root / "escape"
        link.symlink_to(outside)
        run = subprocess.run(
            [str(script), str(root), str(link), "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 111
        assert "workspace escape" in run.stderr

    def test_extra_grant_allows_outside_path(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        root.mkdir()
        grant_root = tmp_path / "granted"
        grant_root.mkdir()
        candidate = grant_root / "data"
        candidate.mkdir()
        run = subprocess.run(
            [str(script), str(root), str(candidate), "0", str(grant_root), "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == str(candidate.resolve())

    def test_read_only_grant_deny_wins_over_workspace_root(self, tmp_path: Path) -> None:
        # Regression for the deny-wins fix: a candidate under BOTH the
        # writable workspace root AND a read-only grant covering a
        # workspace subpath must be REJECTED for a write — upstream
        # silently allowed it because the unconditional workspace-root
        # check ran before the read-only grant was consulted.
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        ro_subdir = root / "readonly"
        target = ro_subdir / "file"
        target.mkdir(parents=True)
        run = subprocess.run(
            [str(script), str(root), str(target), "1", str(ro_subdir), "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 114, run.stdout + run.stderr
        assert "read-only extra path grant" in run.stderr

    def test_read_only_grant_subpath_read_still_allowed(self, tmp_path: Path) -> None:
        # Same overlap, but a READ (for_write=0) is still permitted.
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        ro_subdir = root / "readonly"
        target = ro_subdir / "file"
        target.mkdir(parents=True)
        run = subprocess.run(
            [str(script), str(root), str(target), "0", str(ro_subdir), "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == str(target.resolve())

    def test_read_only_grant_rejects_write(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        root.mkdir()
        grant_root = tmp_path / "ro"
        grant_root.mkdir()
        candidate = grant_root / "f"
        candidate.mkdir()
        run = subprocess.run(
            [str(script), str(root), str(candidate), "1", str(grant_root), "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 114
        assert "read-only extra path grant" in run.stderr

    def test_invalid_for_write_flag_rejected(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        root.mkdir()
        run = subprocess.run(
            [str(script), str(root), str(root), "9"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 64
        assert "for_write must be 0 or 1" in run.stderr

    def test_odd_grant_pairs_rejected(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, RESOLVE_WORKSPACE_PATH_HELPER)
        root = tmp_path / "workspace"
        root.mkdir()
        run = subprocess.run(
            [str(script), str(root), str(root), "0", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 64
        assert "root/read_only pairs" in run.stderr


class TestWorkspaceFingerprintScript:
    def test_fingerprint_is_deterministic(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.txt").write_text("hello")
        out1 = tmp_path / "fp1.json"
        out2 = tmp_path / "fp2.json"
        r1 = subprocess.run(
            [str(script), str(root), "tag1", str(out1), "manifest-digest-abc"],
            capture_output=True,
            text=True,
            check=False,
        )
        r2 = subprocess.run(
            [str(script), str(root), "tag1", str(out2), "manifest-digest-abc"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == 0, r2.stderr
        fp1 = json.loads(out1.read_text())
        fp2 = json.loads(out2.read_text())
        assert fp1["fingerprint"] == fp2["fingerprint"]
        assert fp1["tag"] == "tag1"

    def test_fingerprint_changes_with_content(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.txt").write_text("v1")
        out = tmp_path / "fp.json"
        subprocess.run(
            [str(script), str(root), "t", str(out), "d"],
            capture_output=True,
            text=True,
            check=False,
        )
        first = json.loads(out.read_text())["fingerprint"]
        (root / "a.txt").write_text("v2-different")
        subprocess.run(
            [str(script), str(root), "t", str(out), "d"],
            capture_output=True,
            text=True,
            check=False,
        )
        second = json.loads(out.read_text())["fingerprint"]
        assert first != second

    def test_manifest_digest_affects_fingerprint(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.txt").write_text("same")
        out = tmp_path / "fp.json"
        subprocess.run(
            [str(script), str(root), "t", str(out), "digest-A"],
            capture_output=True,
            text=True,
            check=False,
        )
        a = json.loads(out.read_text())["fingerprint"]
        subprocess.run(
            [str(script), str(root), "t", str(out), "digest-B"],
            capture_output=True,
            text=True,
            check=False,
        )
        b = json.loads(out.read_text())["fingerprint"]
        assert a != b

    def test_missing_root_rejected(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        run = subprocess.run(
            [str(script), str(tmp_path / "nope"), "t", str(tmp_path / "o.json"), "d"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 66
        assert "workspace root not found" in run.stderr

    def test_failed_tar_refuses_to_emit_fingerprint(self, tmp_path: Path) -> None:
        # Regression for the missing-pipefail silent corruption: a tar
        # that fails on an unreadable subtree must NOT silently hash
        # empty/partial input into a valid-looking fingerprint. The
        # script must exit non-zero AND leave NO output file (so the
        # restore-skip decision can never act on a wrong fingerprint).
        import os

        if os.geteuid() == 0:
            pytest.skip("running as root bypasses the chmod-000 unreadable-subtree guard")
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        root = tmp_path / "ws"
        unreadable = root / "secret"
        unreadable.mkdir(parents=True)
        (unreadable / "data").write_text("classified")
        unreadable.chmod(0o000)
        out = tmp_path / "fp.json"
        try:
            run = subprocess.run(
                [str(script), str(root), "t", str(out), "digest"],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            unreadable.chmod(0o755)  # restore so pytest tmp cleanup works
        assert run.returncode != 0, run.stdout
        assert "workspace archive failed" in run.stderr
        assert not out.exists(), "no fingerprint file may be written on tar failure"

    def test_path_with_single_quote_now_supported(self, tmp_path: Path) -> None:
        # The set-- argv rewrite removed the eval, so a workspace path
        # containing a single quote is no longer rejected.
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        root = tmp_path / "it's a dir"
        root.mkdir()
        (root / "f.txt").write_text("ok")
        out = tmp_path / "fp.json"
        run = subprocess.run(
            [str(script), str(root), "t", str(out), "d"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        assert json.loads(out.read_text())["fingerprint"]

    def test_too_few_args_rejected(self, tmp_path: Path) -> None:
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        run = subprocess.run(
            [str(script), str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 64
        assert "usage:" in run.stderr

    @pytest.mark.parametrize("bad_rel", ["..", "../escape", "/abs", "a/../b"])
    def test_traversal_exclude_relpath_rejected(self, tmp_path: Path, bad_rel: str) -> None:
        script = _write_script(tmp_path, WORKSPACE_FINGERPRINT_HELPER)
        root = tmp_path / "ws"
        root.mkdir()
        run = subprocess.run(
            [str(script), str(root), "t", str(tmp_path / "o.json"), "d", bad_rel],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 65
        assert "concrete relative path" in run.stderr


class _FakeRunSession:
    """Minimal session double recording argv + scripted ExecResults.

    The installer issues two argv shapes per helper; this double
    distinguishes them by ``command[0]``: ``"test"`` is the
    ``present_command`` probe, ``"sh"`` is the ``install_command``.
    A helper whose content-addressed path is in ``present_paths``
    reports present (probe exit 0); otherwise absent (exit 1).
    """

    def __init__(
        self,
        *,
        present_paths: set[str] | None = None,
        install_exit: int = 0,
        install_stderr: bytes = b"",
    ) -> None:
        self._present_paths = present_paths if present_paths is not None else set()
        self._install_exit = install_exit
        self._install_stderr = install_stderr
        self.calls: list[tuple[tuple[str, ...], object]] = []

    async def run(
        self,
        *command: str,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: object | None = None,
    ) -> ExecResult:
        del timeout, user
        self.calls.append((command, shell))
        if command[0] == "test":
            present = command[2] in self._present_paths
            return ExecResult(stdout=b"", stderr=b"", exit_code=0 if present else 1)
        return ExecResult(
            stdout=b"",
            stderr=self._install_stderr,
            exit_code=self._install_exit,
        )

    def installs(self) -> list[tuple[str, ...]]:
        return [c for c, _ in self.calls if c[0] == "sh"]

    def probes(self) -> list[tuple[str, ...]]:
        return [c for c, _ in self.calls if c[0] == "test"]


class TestInstallRuntimeHelpers:
    async def test_absent_helpers_are_installed(self) -> None:
        session = _FakeRunSession()  # nothing present → all absent
        await install_runtime_helpers(session, backend_id="docker")
        installed = {c[4] for c in session.installs()}
        assert str(RESOLVE_WORKSPACE_PATH_HELPER.install_path) in installed
        assert str(WORKSPACE_FINGERPRINT_HELPER.install_path) in installed
        # argv-mode (shell=False) is load-bearing: the multi-line
        # install script must NOT be space-joined by a shell wrapper.
        assert all(shell is False for _, shell in session.calls)

    async def test_present_helper_skips_install(self) -> None:
        present = {
            str(RESOLVE_WORKSPACE_PATH_HELPER.install_path),
            str(WORKSPACE_FINGERPRINT_HELPER.install_path),
        }
        session = _FakeRunSession(present_paths=present)
        await install_runtime_helpers(session, backend_id="docker")
        assert session.installs() == []
        assert len(session.probes()) == 2

    async def test_install_failure_raises_start_failed(self) -> None:
        session = _FakeRunSession(install_exit=2, install_stderr=b"disk full")
        with pytest.raises(SandboxStartFailed) as excinfo:
            await install_runtime_helpers(session, backend_id="k8s_pod")
        exc = excinfo.value
        assert exc.backend_id == "k8s_pod"
        assert "resolve-workspace-path" in exc.reason
        assert "disk full" in exc.reason
        assert "exit 2" in exc.reason
        # Fails fast on the first helper — does not attempt the second.
        assert len(session.installs()) == 1

    async def test_custom_helper_subset_only(self) -> None:
        session = _FakeRunSession()
        await install_runtime_helpers(
            session,
            backend_id="docker",
            helpers=[RESOLVE_WORKSPACE_PATH_HELPER],
        )
        installed = {c[4] for c in session.installs()}
        assert installed == {str(RESOLVE_WORKSPACE_PATH_HELPER.install_path)}

    async def test_mixed_present_and_absent(self) -> None:
        session = _FakeRunSession(
            present_paths={str(RESOLVE_WORKSPACE_PATH_HELPER.install_path)},
        )
        await install_runtime_helpers(session, backend_id="docker")
        installed = session.installs()
        assert len(installed) == 1
        assert installed[0][4] == str(WORKSPACE_FINGERPRINT_HELPER.install_path)
