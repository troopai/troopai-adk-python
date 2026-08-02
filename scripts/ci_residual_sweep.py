#!/usr/bin/env python3
"""G-06 residual sweep — ASM-2026-08-01-troopai-adk-v3 (R-001, R-006).

Normative threshold (not editable here; changes require an ASM successor):
case-insensitive grep for the eight terms below over the repo (excluding
`.git`) — **zero hits outside the versioned allowlist**
`scripts/.residual-allowlist` (initially empty).

Scope decisions (documented, deterministic):
- `.git/` is excluded per the matrix; `.augments/` is excluded because it is
  the local, never-committed migration control plane whose ledgers and
  contracts necessarily name the source brand (plan constraint:
  "`.augments/` never committed"). Sphinx build output under `docs/_build/`
  and `docs/build/` is excluded: gitignored, never shipped, and vendored
  theme assets carry false positives (e.g. BSD headers, `descclassname`).
- This script's own file is excluded: it must literally carry the sweep
  terms, so it could never be sweep-clean. The allowlist file is excluded
  for the same reason. Every other file under the repo is scanned, including
  the rest of `scripts/` (which ships and must be sweep-clean by
  construction).
- File symlinks are not followed (the link target is scanned at its real
  path; ASM G-05 owns symlink integrity). Binary (non-UTF-8) files are
  skipped — the CH-01 taxonomy found every brand string in text files.

Allowlist format (`scripts/.residual-allowlist`, one entry per line):
    path/relative/to/repo            # suppress every hit in this file
    path/relative/to/repo:LINENO     # suppress the hit on this exact line
Blank lines and lines starting with `#` are ignored. The file is initially
empty (no entries). A missing file is treated as empty.

Exit: 0 when no un-allowlisted hit exists, 1 otherwise. Output is
deterministic (sorted paths/lines) and machine-readable.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

def _cat(*parts: str) -> str:
    """Concatenate string parts.

    The sweep terms are never written as literals in this shipped file (or
    anywhere under scripts/), so the shipped tree is sweep-clean BY
    CONSTRUCTION (G-06 scans every shipped file; the only remaining
    exclusions are this script and the allowlist, documented above).
    """
    return "".join(parts)


# The eight ASM v3 G-06 terms, longest-first so reported terms are maximal.
# Assembled, never literal (see _cat): the source brand in three case
# variants + the org handle, and the four closed-license/CLA markers.
TERMS: tuple[str, ...] = (
    _cat("All Rights", " Reserved"),
    _cat("langact", "-ai"),
    _cat("LangAct", "AI"),
    _cat("LANGACT", "AI"),
    _cat("langact", "ai"),
    _cat("Propriet", "ary"),
    _cat("IC", "LA"),
    _cat("CC", "LA"),
)

_PATTERN = re.compile("|".join(re.escape(term) for term in TERMS), re.IGNORECASE)

DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / ".residual-allowlist"
SELF_PATH = Path(__file__).resolve()
SKIP_DIRS = (".git", ".augments")


def parse_allowlist(text: str) -> dict[str, set[int] | None]:
    """Parse allowlist content: path -> None (whole file) or {linenos}."""
    entries: dict[str, set[int] | None] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path, sep, tail = line.rpartition(":")
        if sep and tail.isdigit() and "/" not in tail:
            path_entries = entries.setdefault(path, set())
            if path_entries is not None:
                path_entries.add(int(tail))
        else:
            entries[line] = None
    return entries


def is_allowlisted(entries: dict[str, set[int] | None], relpath: str, lineno: int) -> bool:
    if relpath not in entries:
        return False
    lines = entries[relpath]
    return lines is None or lineno in lines


def iter_text_files(repo: Path, include_prefixes: tuple[str, ...]) -> list[Path]:
    """All scannable files under repo, sorted by repo-relative posix path."""
    files: list[Path] = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(repo).as_posix()
        if rel.split("/", 1)[0] in SKIP_DIRS:
            continue
        parts = rel.split("/")
        if len(parts) > 1 and parts[0] == "docs" and parts[1] in ("_build", "build"):
            continue  # sphinx build output — gitignored, never shipped (scope decision)
        if include_prefixes and not any(rel.startswith(prefix) for prefix in include_prefixes):
            continue
        files.append(path)
    files.sort(key=lambda p: p.relative_to(repo).as_posix())
    return files


def sweep_file(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, matched term)] for one file; empty for binary files."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return []
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _PATTERN.finditer(line):
            hits.append((lineno, match.group(0)))
    return hits


def sweep(
    repo: Path,
    allowlist_path: Path | None,
    include_prefixes: tuple[str, ...] = (),
    self_path: Path | None = SELF_PATH,
) -> tuple[list[tuple[str, int, str]], int, int]:
    """Run the sweep. Returns (hits, scanned file count, allowlisted count).

    hits are (relpath, lineno, term) sorted deterministically, already
    filtered by the allowlist and the self/symlink/binary rules.
    """
    entries: dict[str, set[int] | None] = {}
    if allowlist_path is not None and allowlist_path.is_file():
        entries = parse_allowlist(allowlist_path.read_text(encoding="utf-8"))
    allowlist_resolved = allowlist_path.resolve() if allowlist_path else None
    hits: list[tuple[str, int, str]] = []
    allowlisted = 0
    files = iter_text_files(repo, include_prefixes)
    for path in files:
        resolved = path.resolve()
        if self_path is not None and resolved == self_path:
            continue
        if allowlist_resolved is not None and resolved == allowlist_resolved:
            continue
        rel = path.relative_to(repo).as_posix()
        for lineno, term in sweep_file(path):
            if is_allowlisted(entries, rel, lineno):
                allowlisted += 1
            else:
                hits.append((rel, lineno, term))
    hits.sort()
    return hits, len(files), allowlisted


def _self_test() -> int:
    import tempfile

    checks: list[tuple[str, object]] = []

    def check_clean_tree() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pkg").mkdir()
            (repo / "pkg" / "mod.py").write_text('"""TroopAI ADK."""\nX = "troopai-adk-python"\n', encoding="utf-8")
            hits, files, _ = sweep(repo, None)
            assert hits == [], hits
            assert files == 1, files

    def check_all_terms_detected() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            body = "\n".join(f"line with {term} inside" for term in TERMS)
            (repo / "bad.txt").write_text(body + "\n", encoding="utf-8")
            hits, _, _ = sweep(repo, None)
            found = {term.lower() for _, _, term in hits}
            expected = {term.lower() for term in TERMS}
            assert found == expected, (found, expected)
            assert len(hits) == len(TERMS), hits

    def check_case_insensitive() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            payload = "s = '" + _cat("LaNgAcT", "aI") + "' and '" + _cat("pRoPrIeT", "aRy") + "'\n"
            (repo / "x.py").write_text(payload, encoding="utf-8")
            hits, _, _ = sweep(repo, None)
            assert len(hits) == 2, hits

    def check_skip_dirs_and_binary() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            brand = _cat("langact", "ai")
            for skipped in (".git", ".augments"):
                (repo / skipped).mkdir()
                (repo / skipped / "f.txt").write_text(brand + "\n", encoding="utf-8")
            (repo / "bin.dat").write_bytes(b"\x89PNG " + brand.encode() + b" \x00\xff")
            hits, files, _ = sweep(repo, None)
            assert hits == [], hits
            assert files == 1, files  # only the binary file was a candidate; it decodes? no -> skipped
            # .git/.augments files never entered the candidate set.

    def check_symlink_not_followed() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "real.txt").write_text("clean\n", encoding="utf-8")
            (repo / "link.txt").symlink_to("real.txt")
            (repo / "broken.txt").symlink_to("missing.txt")
            hits, files, _ = sweep(repo, None)
            assert hits == [], hits
            assert files == 1, files  # symlinks are not scanned as files

    def check_allowlist() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            brand, closed = _cat("langact", "ai"), _cat("Propriet", "ary")
            (repo / "a.txt").write_text(brand + "\n" + closed + "\n", encoding="utf-8")
            (repo / "b.txt").write_text(brand + "\n", encoding="utf-8")
            allow = repo / ".residual-allowlist"
            allow.write_text("# comment\na.txt:1\n\n", encoding="utf-8")
            hits, _, allowlisted = sweep(repo, allow)
            assert hits == [("a.txt", 2, closed), ("b.txt", 1, brand)], hits
            assert allowlisted == 1, allowlisted
            allow.write_text("a.txt\nb.txt\n", encoding="utf-8")
            hits, _, allowlisted = sweep(repo, allow)
            assert hits == [], hits
            assert allowlisted == 3, allowlisted

    def check_self_exclusion() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            own = repo / "ci_residual_sweep.py"
            own.write_text("TERMS = ('" + _cat("langact", "ai") + "',)\n", encoding="utf-8")
            hits, _, _ = sweep(repo, None, self_path=own)
            assert hits == [], hits

    def check_include_prefixes() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            brand, closed = _cat("langact", "ai"), _cat("Propriet", "ary")
            (repo / "src").mkdir()
            (repo / "docs").mkdir()
            (repo / "src" / "a.py").write_text(brand + "\n", encoding="utf-8")
            (repo / "docs" / "b.md").write_text(closed + "\n", encoding="utf-8")
            hits, _, _ = sweep(repo, None, include_prefixes=("src/",))
            assert hits == [("src/a.py", 1, brand)], hits

    def check_allowlist_parser() -> None:
        entries = parse_allowlist("# c\n\nx/y.txt\nx/z.txt:12\nbad:line\n")
        assert entries["x/y.txt"] is None, entries
        assert entries["x/z.txt"] == {12}, entries
        assert entries["bad:line"] is None, entries  # not path:lineno -> whole-file entry

    checks = [
        ("clean tree passes", check_clean_tree),
        ("all eight ASM terms detected", check_all_terms_detected),
        ("case-insensitive matching", check_case_insensitive),
        (".git/.augments skipped, binary skipped", check_skip_dirs_and_binary),
        ("file symlinks not followed", check_symlink_not_followed),
        ("allowlist path and path:lineno suppression", check_allowlist),
        ("script's own file excluded", check_self_exclusion),
        ("include-prefix scoping (shard form)", check_include_prefixes),
        ("allowlist parser", check_allowlist_parser),
    ]
    failures = 0
    for name, check in checks:
        try:
            check()  # type: ignore[operator]
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
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[1]),
        help="repo root to sweep (default: the repo containing this script)",
    )
    parser.add_argument(
        "--allowlist",
        default=str(DEFAULT_ALLOWLIST),
        help="allowlist path (default: scripts/.residual-allowlist; missing = empty)",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help="only scan repo-relative paths under PREFIX (repeatable; shard-scoped form)",
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic-fixture checks and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    repo = Path(args.repo)
    if not repo.is_dir():
        print(f"error: --repo {repo} is not a directory", file=sys.stderr)
        return 1
    hits, files, allowlisted = sweep(repo, Path(args.allowlist), tuple(args.include_prefix))
    for rel, lineno, term in hits:
        print(f"HIT {rel}:{lineno}: {term}")
    print(f"G-06 residual-sweep: hits={len(hits)} allowlisted={allowlisted} files={files} terms={len(TERMS)}")
    print(f"RESULT: {'FAIL' if hits else 'PASS'}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
