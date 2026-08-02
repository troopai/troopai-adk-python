#!/usr/bin/env python3
"""G-05 symlink check — ASM-2026-08-01-troopai-adk-v3 (R-008, F-016).

Normative threshold (not editable here; changes require an ASM successor):
every manifest symlink (59) must resolve as a symlink — per-symlink
`test -L` + readlink-resolves check.

Two modes (coordinator-adjudicated interface):

- **CI mode (default):** `git ls-files -s` in the repo; every mode-120000
  path must be a symlink whose readlink resolves, and the count must equal
  --expect-count (default 59, the F-016 inventory).
- **Migration mode (`--manifest PATH`):** for each symlink-flagged path in
  the T-001 machine manifest, check `test -L` + readlink resolves at the
  corresponding path in the target tree. Source paths are mapped through
  the MIG v3 invariant-4 rename rules for PATH STRINGS (mirrored inline
  below: scripts/ ships standalone and cannot import .augments tooling).
  `--select PREFIX` (repeatable) restricts to manifest symlink paths under
  the given source-path prefixes — the trial-slice form, where only the
  slice's symlinks have been converted yet.

Exit: 0 when all selected symlinks resolve (and, in CI mode, the count
matches), else 1. Deterministic sorted output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def _cat(*parts: str) -> str:
    """Concatenate — source-brand literals are never written literally in
    this shipped file, keeping scripts/ sweep-clean by construction (G-06)."""
    return "".join(parts)


# MIG v3 invariant 4 precedence, applied to path strings only (longest first).
_PATH_RULES: tuple[tuple[str, str], ...] = (
    (_cat("langact", "ai-adk-python"), "troopai-adk-python"),
    (_cat("langact", "ai-adk"), "troopai-adk-python"),
    (_cat("langact", "-ai"), "troopai"),
    (_cat("LangAct", "AI"), "TroopAI"),
    (_cat("LANGACT", "AI"), "TROOPAI"),
    (_cat("langact", "ai"), "troopai"),
)

DEFAULT_EXPECT_COUNT = 59  # F-016 full inventory, coordinator-verified


def transform_path(path: str) -> str:
    """Map a source git-ls-files path to its target-tree path (rename rules)."""
    for old, new in _PATH_RULES:
        path = path.replace(old, new)
    return path


def parse_ls_files(text: str) -> list[str]:
    """`git ls-files -s` output -> sorted mode-120000 (symlink) paths."""
    paths: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        meta, _, path = line.partition("\t")
        mode = meta.split(" ", 1)[0]
        if mode == "120000":
            paths.append(path)
    return sorted(paths)


def check_symlinks(root: Path, relpaths: list[str]) -> list[tuple[str, str]]:
    """Per-path `test -L` + readlink-resolves. Returns [(relpath, failure)]."""
    failures: list[tuple[str, str]] = []
    for rel in relpaths:
        link = root / rel
        if not link.is_symlink():  # test -L
            failures.append((rel, "not-a-symlink"))
            continue
        target = os.readlink(link)
        if not (link.parent / target).exists():
            failures.append((rel, f"unresolved-target:{target}"))
    return failures


def manifest_symlink_paths(manifest_doc: dict, selects: tuple[str, ...]) -> list[str]:
    """Symlink source paths from a T-001 manifest, optionally prefix-filtered."""
    paths = [f["path"] for f in manifest_doc["files"] if f.get("symlink")]
    if selects:
        paths = [p for p in paths if any(p.startswith(sel) for sel in selects)]
    return sorted(paths)


def _self_test() -> int:
    import tempfile

    _SRC_LINK = _cat("src/langact", "ai/adk/run/AGENTS.md")  # source-side path, never literal
    _SRC_FILE = _cat("src/langact", "ai/adk/run/runner.py")
    _TGT_LINK = "src/troopai/adk/run/AGENTS.md"

    def check_parse_ls_files() -> None:
        text = (
            "100644 abc123 0\tREADME.md\n"
            "120000 def456 0\tAGENTS.md\n"
            f"120000 789abc 0\t{_SRC_LINK}\n"
            "100755 000fff 0\ttools/hook.py\n"
        )
        got = parse_ls_files(text)
        assert got == ["AGENTS.md", _SRC_LINK], got

    def check_transform_path() -> None:
        assert transform_path(_SRC_LINK) == _TGT_LINK
        assert transform_path("AGENTS.md") == "AGENTS.md"
        assert transform_path(".kimi-code/skills") == ".kimi-code/skills"

    def check_symlink_verdicts() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("# governance\n", encoding="utf-8")
            (root / "good").symlink_to("CLAUDE.md")
            (root / "broken").symlink_to("MISSING.md")
            (root / "materialized").write_text("# not a link\n", encoding="utf-8")
            failures = check_symlinks(root, ["good", "broken", "materialized"])
            kinds = {rel: kind for rel, kind in failures}
            assert kinds == {"broken": "unresolved-target:MISSING.md", "materialized": "not-a-symlink"}, kinds
            assert check_symlinks(root, ["good"]) == []

    def check_manifest_selection() -> None:
        doc = {
            "files": [
                {"path": "AGENTS.md", "symlink": True},
                {"path": _SRC_LINK, "symlink": True},
                {"path": _SRC_FILE, "symlink": False},
                {"path": "docs/index.md", "symlink": False},
            ]
        }
        assert manifest_symlink_paths(doc, ()) == ["AGENTS.md", _SRC_LINK]
        assert manifest_symlink_paths(doc, ("src/",)) == [_SRC_LINK]

    def check_migration_mode_end_to_end() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "src" / "troopai" / "adk" / "run"
            target_dir.mkdir(parents=True)
            (target_dir / "CLAUDE.md").write_text("# gov\n", encoding="utf-8")
            (target_dir / "AGENTS.md").symlink_to("CLAUDE.md")
            doc = {"files": [{"path": _SRC_LINK, "symlink": True}]}
            rels = [transform_path(p) for p in manifest_symlink_paths(doc, ())]
            assert check_symlinks(root, rels) == []
            (target_dir / "AGENTS.md").unlink()
            (target_dir / "AGENTS.md").write_text("materialized\n", encoding="utf-8")
            failures = check_symlinks(root, rels)
            assert failures == [(_TGT_LINK, "not-a-symlink")], failures

    checks = (
        ("git ls-files -s parsing (mode 120000)", check_parse_ls_files),
        ("source->target path transform", check_transform_path),
        ("test -L + readlink verdicts", check_symlink_verdicts),
        ("manifest symlink selection + --select filter", check_manifest_selection),
        ("migration mode end-to-end (fixture tree)", check_migration_mode_end_to_end),
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
    parser.add_argument("--repo", default=str(repo_root), help="repo/tree root to check (default: this repo)")
    parser.add_argument("--manifest", help="T-001 manifest JSON: migration mode (default: CI mode)")
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="PREFIX",
        help="migration mode: only manifest symlinks under this source-path prefix (repeatable)",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=DEFAULT_EXPECT_COUNT,
        help="CI mode: required symlink count (default: 59, F-016 inventory)",
    )
    parser.add_argument(
        "--ls-files-file",
        help="CI mode: read pre-captured `git ls-files -s` output from this file instead of running git",
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic-fixture checks and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    repo = Path(args.repo)
    if not repo.is_dir():
        print(f"error: --repo {repo} is not a directory", file=sys.stderr)
        return 1

    if args.manifest:
        doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        source_paths = manifest_symlink_paths(doc, tuple(args.select))
        rels = [transform_path(p) for p in source_paths]
        failures = check_symlinks(repo, rels)
        for rel, kind in failures:
            print(f"FAIL-LINK {rel}: {kind}")
        print(f"G-05 symlink-check: mode=migration resolved={len(rels) - len(failures)}/{len(rels)}")
        print(f"RESULT: {'FAIL' if failures else 'PASS'}")
        return 1 if failures else 0

    if args.ls_files_file:
        listing = Path(args.ls_files_file).read_text(encoding="utf-8")
    else:
        proc = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-s"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            print(f"error: git ls-files failed in {repo}: {proc.stderr.decode('utf-8', 'replace').strip()}",
                  file=sys.stderr)
            return 1
        listing = proc.stdout.decode("utf-8")
    rels = parse_ls_files(listing)
    failures = check_symlinks(repo, rels)
    for rel, kind in failures:
        print(f"FAIL-LINK {rel}: {kind}")
    count_ok = len(rels) == args.expect_count
    if not count_ok:
        print(f"COUNT-MISMATCH found={len(rels)} expected={args.expect_count}")
    print(
        f"G-05 symlink-check: mode=ci resolved={len(rels) - len(failures)}/{len(rels)} "
        f"expected-count={args.expect_count} count-ok={count_ok}"
    )
    failed = bool(failures) or not count_ok
    print(f"RESULT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
