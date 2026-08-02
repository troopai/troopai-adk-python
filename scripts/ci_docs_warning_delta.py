#!/usr/bin/env python3
"""G-04 docs warning delta + dropped-page sweep — ASM-2026-08-01-troopai-adk-v3 (R-007).

Normative threshold (not editable here; changes require an ASM successor):
the target docs build must produce **zero warnings vs. the pre-conversion
source `-W` baseline (delta = 0)**, plus D-04-scrub cleanliness: **zero
references to dropped pages**. `-W` stays BLOCKING (never continue-on-error).

Mechanism (plan F-4):
1. Run `sphinx-build -W -b html docs <out>` (or read a pre-captured build
   log via --warnings-file, so callers control the docs venv). Warnings are
   normalized to a stable form and compared as a SET against
   `scripts/docs-warning-baseline.txt`. Any NEW warning (present now, absent
   from baseline) fails the gate. Baseline warnings that disappeared are
   reported as informational only. Note: with pre-existing warnings in the
   baseline, sphinx's own exit code is expected non-zero under `-W`; the
   gate's signal is the warning-SET comparison, which is what blocks.
2. Dropped-page reference sweep: grep docs/ for references to the
   D-04-dropped page/asset (`translating`, `wordmark`) — zero hits. The
   kept dormant i18n plumbing (`locale_dirs`, `gettext_compact`,
   `SPHINX_LANG`, sphinx-intl) does not match these patterns.

Warning normalization (both sides, shared with
.augments/tools/gates/docs_warning_baseline.py):
    /abs/path/docs/deploy/container.md:75: WARNING: msg   ->  deploy/container.md: WARNING: msg
    docs/index.md:: WARNING: msg                          ->  index.md: WARNING: msg
    WARNING: msg (no location)                            ->  WARNING: msg
Line numbers are stripped so moved content compares equal; the comparison
is a set comparison (duplicates collapse). Severity classes matched:
WARNING, ERROR, CRITICAL.

Baseline file format: one normalized warning per line, sorted; lines
starting with `#` are comments (the T-003 capture writes a `# count: N`
header). Missing/empty baseline = empty set.

Exit: 0 when new-warnings == 0 and dropped-refs == 0, else 1.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# <path>:<lineno>: SEVERITY: message  /  <path>:: SEVERITY: message  /  SEVERITY: message
_WARNING_RE = re.compile(
    r"^(?:(?P<path>.*?\.(?:md|rst|txt|py|yaml|yml|json))(?::\d*)?: )?(?P<severity>WARNING|ERROR|CRITICAL): (?P<msg>.*)$"
)

# D-04 dropped page/asset references. Literal, case-sensitive (file names).
DROPPED_REF_PATTERNS: tuple[str, ...] = ("translating", "wordmark")
_DROPPED_RE = re.compile("|".join(re.escape(p) for p in DROPPED_REF_PATTERNS))

DOCS_TEXT_SUFFIXES = (".md", ".rst", ".txt", ".py", ".yaml", ".yml", ".toml", ".css")


def normalize_warnings(text: str) -> set[str]:
    """Normalize sphinx/docutils output into the comparable warning set."""
    warnings: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        match = _WARNING_RE.match(line)
        if not match:
            continue
        severity, msg = match.group("severity"), match.group("msg").strip()
        path = match.group("path")
        if path:
            posix = path.replace("\\", "/")
            if "/docs/" in posix:
                posix = posix.rsplit("/docs/", 1)[1]
            elif posix.startswith("docs/"):
                posix = posix[len("docs/"):]
            warnings.add(f"{posix}: {severity}: {msg}")
        else:
            warnings.add(f"{severity}: {msg}")
    return warnings


def parse_baseline(text: str) -> set[str]:
    """Parse a baseline file: sorted normalized warnings, `#` comments."""
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def new_warnings(current: set[str], baseline: set[str]) -> set[str]:
    return current - baseline


def find_dropped_refs(docs: Path) -> list[tuple[str, int, str]]:
    """References to D-04-dropped pages/assets under docs/: (rel, lineno, line)."""
    hits: list[tuple[str, int, str]] = []
    for path in sorted(docs.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(docs).as_posix()
        if rel.split("/", 1)[0] == "_build" or path.suffix.lower() not in DOCS_TEXT_SUFFIXES:
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _DROPPED_RE.search(line):
                hits.append((rel, lineno, line.strip()))
    hits.sort()
    return hits


def run_sphinx(sphinx_build: str, docs: Path, out: Path) -> tuple[int, str]:
    """`sphinx-build -W -b html docs out`; returns (exit code, captured output)."""
    proc = subprocess.run(
        [sphinx_build, "-W", "-b", "html", str(docs), str(out)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def _self_test() -> int:
    import tempfile

    def check_normalize_absolute_paths_and_linenos() -> None:
        text = (
            "/home/x/repo/docs/deploy/container.md:75: WARNING: unknown document: foo\n"
            "/home/x/repo/docs/index.md:: WARNING: toctree glob failed\n"
            "docs/swarms/index.md:12: ERROR: broken\n"
            "WARNING: no location here\n"
            "building [html]: all files\n"
        )
        got = normalize_warnings(text)
        assert got == {
            "deploy/container.md: WARNING: unknown document: foo",
            "index.md: WARNING: toctree glob failed",
            "swarms/index.md: ERROR: broken",
            "WARNING: no location here",
        }, got

    def check_normalize_dedupes_and_ignores_info() -> None:
        text = "docs/a.md:1: WARNING: dup\ndocs/a.md:9: WARNING: dup\ndocs/a.md:1: INFO: not a warning\n"
        got = normalize_warnings(text)
        assert got == {"a.md: WARNING: dup"}, got

    def check_new_warning_delta() -> None:
        baseline = parse_baseline("# count: 1\na.md: WARNING: old\n")
        current = {"a.md: WARNING: old", "b.md: WARNING: NEW"}
        assert new_warnings(current, baseline) == {"b.md: WARNING: NEW"}, new_warnings(current, baseline)
        assert new_warnings(baseline, baseline) == set()

    def check_parse_baseline_skips_comments() -> None:
        got = parse_baseline("# header\n# count: 2\n\nx.md: WARNING: a\ny.md: WARNING: b\n")
        assert got == {"x.md: WARNING: a", "y.md: WARNING: b"}, got

    def check_dropped_refs() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "maintenance").mkdir()
            (docs / "maintenance" / "index.md").write_text(
                "See {doc}`translating` for i18n.\n", encoding="utf-8"
            )
            (docs / "conf.py").write_text("locale_dirs = ['locale/']\n", encoding="utf-8")  # kept plumbing
            (docs / "index.md").write_text("clean\n", encoding="utf-8")
            (docs / "logo.md").write_text("see wordmark.svg asset\n", encoding="utf-8")
            hits = find_dropped_refs(docs)
            assert [h[0] for h in hits] == ["logo.md", "maintenance/index.md"], hits

    def check_dropped_refs_clean() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "index.md").write_text("All clean, locale_dirs plumbing kept in conf.py.\n", encoding="utf-8")
            build = docs / "_build"
            build.mkdir()
            (build / "x.md").write_text("translating\n", encoding="utf-8")  # build output ignored
            assert find_dropped_refs(docs) == [], find_dropped_refs(docs)

    checks = (
        ("normalize: abs paths, line numbers, severities", check_normalize_absolute_paths_and_linenos),
        ("normalize: dedupe + ignore non-warnings", check_normalize_dedupes_and_ignores_info),
        ("delta: new warnings detected, equal sets pass", check_new_warning_delta),
        ("baseline parser skips comments/blank", check_parse_baseline_skips_comments),
        ("dropped-page reference sweep finds translating/wordmark", check_dropped_refs),
        ("dropped-page sweep clean (kept plumbing, _build ignored)", check_dropped_refs_clean),
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
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--docs", default=str(repo_root / "docs"), help="docs directory (default: <repo>/docs)")
    parser.add_argument(
        "--baseline",
        default=str(repo_root / "scripts" / "docs-warning-baseline.txt"),
        help="normalized-warning baseline file (default: scripts/docs-warning-baseline.txt)",
    )
    parser.add_argument(
        "--warnings-file",
        help="pre-captured sphinx build output (skip running sphinx; callers control the docs venv)",
    )
    parser.add_argument("--sphinx-build", default="sphinx-build", help="sphinx-build executable (default: PATH)")
    parser.add_argument("--out", help="html output dir (default: <docs>/_build)")
    parser.add_argument("--self-test", action="store_true", help="run synthetic-fixture checks and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    docs = Path(args.docs)
    if not docs.is_dir():
        print(f"error: --docs {docs} is not a directory", file=sys.stderr)
        return 1

    sphinx_exit: int | None = None
    if args.warnings_file:
        build_output = Path(args.warnings_file).read_text(encoding="utf-8", errors="replace")
    else:
        out = Path(args.out) if args.out else docs / "_build"
        sphinx_exit, build_output = run_sphinx(args.sphinx_build, docs, out)

    current = normalize_warnings(build_output)
    baseline_path = Path(args.baseline)
    baseline = parse_baseline(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else set()
    new = sorted(new_warnings(current, baseline))
    stale = sorted(baseline - current)
    for warning in new:
        print(f"NEW-WARNING {warning}")
    dropped = find_dropped_refs(docs)
    for rel, lineno, line in dropped:
        print(f"DROPPED-REF {rel}:{lineno}: {line[:100]}")

    exit_note = f" sphinx-exit={sphinx_exit}" if sphinx_exit is not None else ""
    print(
        f"G-04 warning-delta: new={len(new)} baseline={len(baseline)} current={len(current)} "
        f"stale={len(stale)}{exit_note}"
    )
    print(f"G-04 dropped-page-sweep: hits={len(dropped)}")
    failed = bool(new) or bool(dropped)
    print(f"RESULT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
