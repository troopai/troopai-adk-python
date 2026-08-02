#!/usr/bin/env python3
"""G-15 artifact/content check — ASM-2026-08-01-troopai-adk-v3 (R-004, R-001, R-007).

Normative threshold (not editable here; changes require an ASM successor):
built artifacts and generated output carry no mis-renames; the changelog
page is non-empty. Three parts, each enabled by its input argument:

1. Wheel content (`--wheel PATH`): unzip the built wheel — every member
   must live under `troopai/**` or `troopai_adk_python-*.dist-info/**`;
   anything else is a failure.
2. Generated-output sweep (`--cli PATH`): run `troopai new` and
   `troopai deploy init` into a temp dir and G-06-sweep the generated files
   (zero hits). CLI invocation defaults mirror the source CLI contract:
   `troopai new g15probe --dir <tmp>` and
   `troopai deploy init --agent g15probe:agent --dir <tmp>`
   (override with --new-args / --deploy-args).
3. Changelog non-empty (`--changelog-html PATH`): the rendered changelog
   page exists, has visible text, and contains at least one changelog-entry
   heading (default pattern `Unreleased` or a semver heading) — the
   silent-empty `{include}` branch must fail here (G-04 catches the
   unmatched-marker CRITICAL; this catches the rendered-empty case).

Sibling import: this script reuses `ci_residual_sweep.py` (same directory,
ships together) for the generated-file sweep.

Exit: 0 when every enabled part passes, else 1. At least one part required.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_residual_sweep  # noqa: E402 — sibling script, ships in the same directory

_PACKAGE_RE = re.compile(r"^troopai(/|$)")
_DIST_INFO_RE = re.compile(r"^troopai_adk_python-[^/]+\.dist-info/")

DEFAULT_REQUIRE_PATTERN = r"Unreleased|[0-9]+\.[0-9]+\.[0-9]+"
DEFAULT_NEW_ARGS = ("new", "g15probe")
DEFAULT_DEPLOY_ARGS = ("deploy", "init", "--agent", "g15probe:agent")

_TAG_RE = re.compile(r"<[^>]+>")


def _cat(*parts: str) -> str:
    """Concatenate — source-brand literals are never written literally in
    this shipped file, keeping scripts/ sweep-clean by construction (G-06)."""
    return "".join(parts)


def check_wheel(wheel_path: Path) -> list[str]:
    """Wheel members outside troopai/** + troopai_adk_python-*.dist-info/."""
    failures: list[str] = []
    with zipfile.ZipFile(wheel_path) as zf:
        names = sorted(zf.namelist())
    if not names:
        return ["wheel-empty"]
    for name in names:
        if _PACKAGE_RE.match(name) or _DIST_INFO_RE.match(name):
            continue
        failures.append(f"unexpected-member:{name}")
    if not any(_PACKAGE_RE.match(name) for name in names):
        failures.append("no-troopai-package-members")
    if not any(_DIST_INFO_RE.match(name) for name in names):
        failures.append("no-dist-info-members")
    return failures


def run_cli(cli_argv: list[str], cwd: Path) -> tuple[int, str]:
    """Run one CLI command in cwd; returns (exit code, captured output)."""
    proc = subprocess.run(
        cli_argv,
        check=False,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def sweep_generated(root: Path) -> list[tuple[str, int, str]]:
    """G-06-sweep every generated file under root (no skips except symlinks)."""
    hits: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, term in ci_residual_sweep.sweep_file(path):
            hits.append((rel, lineno, term))
    hits.sort()
    return hits


def visible_text(html: str) -> str:
    return _TAG_RE.sub(" ", html)


def check_changelog(changelog_html: Path, require_pattern: str) -> list[str]:
    """Rendered changelog page: exists, visible text, entry heading present."""
    failures: list[str] = []
    if not changelog_html.is_file():
        return [f"missing:{changelog_html}"]
    try:
        text = changelog_html.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return [f"undecodable:{changelog_html}"]
    body = visible_text(text)
    if not body.split():
        failures.append("empty-visible-text")
    if not re.search(require_pattern, body):
        failures.append(f"no-changelog-entry-heading:/{require_pattern}/")
    return failures


def _self_test() -> int:
    checks = []

    def make_wheel(path: Path, members: dict[str, str]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)

    def check_wheel_clean() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "troopai_adk_python-0.1.0-py3-none-any.whl"
            make_wheel(
                wheel,
                {
                    "troopai/__init__.py": "",
                    "troopai/adk/__init__.py": "__all__ = []\n",
                    "troopai_adk_python-0.1.0.dist-info/METADATA": "Name: troopai-adk-python\n",
                    "troopai_adk_python-0.1.0.dist-info/RECORD": "",
                },
            )
            assert check_wheel(wheel) == [], check_wheel(wheel)

    def check_wheel_rejects_strays() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "x.whl"
            old_pkg = _cat("langact", "ai") + "/adk/__init__.py"
            make_wheel(
                wheel,
                {
                    "troopai/adk/__init__.py": "",
                    "troopai_adk_python-0.1.0.dist-info/METADATA": "",
                    old_pkg: "",
                    "README.md": "stray top-level file\n",
                },
            )
            failures = check_wheel(wheel)
            assert f"unexpected-member:{old_pkg}" in failures, failures
            assert "unexpected-member:README.md" in failures, failures

    def check_wheel_rejects_missing_parts() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "x.whl"
            make_wheel(wheel, {"random/file.py": ""})
            failures = check_wheel(wheel)
            assert "no-troopai-package-members" in failures and "no-dist-info-members" in failures, failures

    def check_sweep_generated() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brand = _cat("langact", "ai")
            (root / "agent.json").write_text('{"name": "g15probe"}\n', encoding="utf-8")
            assert sweep_generated(root) == []
            (root / "Dockerfile").write_text(f"RUN pip install {brand}-adk\n", encoding="utf-8")
            hits = sweep_generated(root)
            assert hits == [("Dockerfile", 1, brand)], hits

    def check_changelog_verdicts() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "changelog.html"
            page.write_text("<html><body><h1>Changelog</h1><h2>[Unreleased]</h2><p>x</p></body></html>",
                            encoding="utf-8")
            assert check_changelog(page, DEFAULT_REQUIRE_PATTERN) == []
            page.write_text("<html><body><h1>Changelog</h1></body></html>", encoding="utf-8")
            failures = check_changelog(page, DEFAULT_REQUIRE_PATTERN)
            assert any("no-changelog-entry-heading" in f for f in failures), failures
            page.write_text("<html><body>   </body></html>", encoding="utf-8")
            failures = check_changelog(page, DEFAULT_REQUIRE_PATTERN)
            assert "empty-visible-text" in failures, failures
            failures = check_changelog(Path(tmp) / "missing.html", DEFAULT_REQUIRE_PATTERN)
            assert any(f.startswith("missing:") for f in failures), failures

    def check_run_cli_stub() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            code, out = run_cli(
                [sys.executable, "-c", "from pathlib import Path; Path('out.txt').write_text('made')"], work
            )
            assert code == 0 and (work / "out.txt").read_text() == "made", (code, out)

    checks = (
        ("wheel: only troopai/** + dist-info passes", check_wheel_clean),
        ("wheel: stray/old-name members rejected", check_wheel_rejects_strays),
        ("wheel: missing package/dist-info rejected", check_wheel_rejects_missing_parts),
        ("generated-output sweep (G-06 terms)", check_sweep_generated),
        ("changelog non-empty verdicts", check_changelog_verdicts),
        ("CLI runner stub", check_run_cli_stub),
    )
    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — any failure is a RED; report and continue
            failures += 1
            print(f"FAIL - {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok - {name}")
    print(f"self-test: {len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wheel", help="built wheel path (enables wheel content check)")
    parser.add_argument("--cli", help="troopai CLI executable (enables generated-output sweep)")
    parser.add_argument(
        "--new-args",
        nargs="+",
        default=list(DEFAULT_NEW_ARGS),
        help="args after the CLI for the scaffold command (default: new g15probe; --dir is appended)",
    )
    parser.add_argument(
        "--deploy-args",
        nargs="+",
        default=list(DEFAULT_DEPLOY_ARGS),
        help="args for the deploy init command (default: deploy init --agent g15probe:agent; --dir appended)",
    )
    parser.add_argument("--work-dir", help="dir for generated output (default: fresh temp dir)")
    parser.add_argument("--changelog-html", help="rendered changelog.html (enables changelog check)")
    parser.add_argument(
        "--require-pattern",
        default=DEFAULT_REQUIRE_PATTERN,
        help="regex the changelog visible text must match (default: %(default)s)",
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic-fixture checks and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    if not (args.wheel or args.cli or args.changelog_html):
        print("error: at least one of --wheel, --cli, --changelog-html is required", file=sys.stderr)
        return 1

    failed = False
    if args.wheel:
        wheel_failures = check_wheel(Path(args.wheel))
        for failure in wheel_failures:
            print(f"FAIL-WHEEL {failure}")
        print(f"G-15 wheel-content: {'PASS' if not wheel_failures else 'FAIL'} wheel={args.wheel}")
        failed |= bool(wheel_failures)

    if args.cli:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(args.work_dir) if args.work_dir else Path(tmp)
            work.mkdir(parents=True, exist_ok=True)
            cli_failed = False
            for label, extra in (("new", args.new_args), ("deploy-init", args.deploy_args)):
                dest = work / label
                dest.mkdir(exist_ok=True)
                code, output = run_cli([args.cli, *extra, "--dir", str(dest)], work)
                if code != 0:
                    print(f"FAIL-CLI {label} exit={code}: {output.strip()[:200]}")
                    cli_failed = True
            hits = sweep_generated(work)
            for rel, lineno, term in hits:
                print(f"FAIL-GENERATED {rel}:{lineno}: {term}")
            print(
                f"G-15 generated-sweep: {'PASS' if not (cli_failed or hits) else 'FAIL'} "
                f"cli-failed={cli_failed} hits={len(hits)}"
            )
            failed |= cli_failed or bool(hits)

    if args.changelog_html:
        changelog_failures = check_changelog(Path(args.changelog_html), args.require_pattern)
        for failure in changelog_failures:
            print(f"FAIL-CHANGELOG {failure}")
        print(
            f"G-15 changelog-non-empty: {'PASS' if not changelog_failures else 'FAIL'} "
            f"path={args.changelog_html}"
        )
        failed |= bool(changelog_failures)

    print(f"RESULT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
