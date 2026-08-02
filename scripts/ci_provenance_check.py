#!/usr/bin/env python3
"""G-09 provenance check — ASM-2026-08-01-troopai-adk-v3 (R-013, provenance part).

Normative threshold (not editable here; changes require an ASM successor):
MIT provenance holds — no known-vuln deps/secrets/closed-license residue
in license posture. The bandit/pip-audit/CodeQL/secret-scanning parts of
G-09 run in CI; this script owns the provenance part:

1. `LICENSE` is the MIT license (SPDX text match: normalized full-text
   comparison against the canonical MIT text, copyright line wildcarded).
2. `pyproject.toml` license field is MIT (PEP 639 string form `license =
   "MIT"` or legacy `{text = "MIT"}`) AND carries the classifier
   `License :: OSI Approved :: MIT License`.
3. No `Private :: Do Not Upload` classifier anywhere in pyproject.toml.
4. Closed-license banner scan: zero hits for the two G-06 license-banner
   terms (the "propriet-ary" word and the "all rights-reserved" phrase —
   written split here so this shipped file stays sweep-clean by
   construction) in the first --header-lines lines of any text file in the
   repo (excluding `.git`, `.augments`, file symlinks, and this script
   itself — it must build those term patterns).

Exit: 0 when all four checks pass, else 1. Deterministic sorted output.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# Canonical MIT license text (SPDX MIT), copyright line excluded — the
# matcher wildcards the copyright line before comparing.
_MIT_CANONICAL = """\
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

MIT_CLASSIFIER = "License :: OSI Approved :: MIT License"
PRIVATE_CLASSIFIER = "Private :: Do Not Upload"

def _cat(*parts: str) -> str:
    """Concatenate — banner-term literals are never written literally in
    this shipped file, keeping scripts/ sweep-clean by construction (G-06)."""
    return "".join(parts)


_BANNER_RE = re.compile(_cat("propriet", "ary") + "|" + _cat("all rights", " reserved"), re.IGNORECASE)

SELF_PATH = Path(__file__).resolve()
SKIP_DIRS = (".git", ".augments")


def _normalize_license_text(text: str) -> str:
    """Lowercased, whitespace-collapsed text with copyright lines removed."""
    lines = [line for line in text.splitlines() if not line.strip().lower().startswith("copyright")]
    return " ".join(" ".join(lines).split())


def license_is_mit(license_path: Path) -> bool:
    """SPDX text match: normalized LICENSE == normalized canonical MIT."""
    try:
        text = license_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return _normalize_license_text(text) == _normalize_license_text(_MIT_CANONICAL)


def check_pyproject(pyproject_path: Path) -> list[str]:
    """License field + classifiers. Returns failure descriptions (empty = pass)."""
    failures: list[str] = []
    try:
        raw = pyproject_path.read_bytes()
        doc = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return [f"pyproject-unreadable: {exc}"]
    if PRIVATE_CLASSIFIER.encode() in raw:
        failures.append(f"private-classifier: {PRIVATE_CLASSIFIER!r} present")
    project = doc.get("project", {})
    license_field = project.get("license")
    license_ok = False
    if isinstance(license_field, str):  # PEP 639
        license_ok = license_field.strip() == "MIT"
    elif isinstance(license_field, dict):  # legacy table form
        license_ok = str(license_field.get("text", "")).strip() == "MIT"
    if not license_ok:
        failures.append(f"license-field: not MIT (got {license_field!r})")
    classifiers = project.get("classifiers", [])
    if MIT_CLASSIFIER not in classifiers:
        failures.append(f"license-classifier: {MIT_CLASSIFIER!r} missing")
    return failures


_ALLOWLIST_PATH = Path("scripts/.residual-allowlist")


def _load_allowlist(repo: Path) -> tuple[set[str], set[tuple[str, int]]]:
    """Shared G-06/G-09 allowlist (`scripts/.residual-allowlist`): bare paths and path:lineno entries."""
    file_entries: set[str] = set()
    line_entries: set[tuple[str, int]] = set()
    allow = repo / _ALLOWLIST_PATH
    if not allow.is_file():
        return file_entries, line_entries
    for raw in allow.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entry = line.split()[0]
        path_part, sep, lineno = entry.rpartition(":")
        if sep and lineno.isdigit():
            line_entries.add((path_part, int(lineno)))
        else:
            file_entries.add(entry)
    return file_entries, line_entries


def banner_scan(repo: Path, header_lines: int, self_path: Path | None = SELF_PATH) -> list[tuple[str, int, str]]:
    """Closed-license banner hits (relpath, lineno, line) in file headers."""
    file_entries, line_entries = _load_allowlist(repo)
    hits: list[tuple[str, int, str]] = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(repo).as_posix()
        if rel.split("/", 1)[0] in SKIP_DIRS:
            continue
        parts = rel.split("/")
        if len(parts) > 1 and parts[0] == "docs" and parts[1] in ("_build", "build"):
            continue  # sphinx build output — gitignored, never shipped
        if rel == _ALLOWLIST_PATH.as_posix():
            continue  # the allowlist's own reasons may name the terms
        if self_path is not None and path.resolve() == self_path:
            continue
        if rel in file_entries:
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines()[:header_lines], start=1):
            if (rel, lineno) in line_entries:
                continue
            if _BANNER_RE.search(line):
                hits.append((rel, lineno, line.strip()))
    hits.sort()
    return hits


def _self_test() -> int:
    import tempfile

    mit_body = _MIT_CANONICAL.replace("MIT License\n", "MIT License\n\nCopyright (c) 2026 Example\n", 1)

    def check_mit_license_match() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "LICENSE"
            path.write_text(mit_body, encoding="utf-8")
            assert license_is_mit(path)
            # Wrapped differently (line re-flow) still matches.
            path.write_text(mit_body.replace("\n", " ").replace("  ", "\n"), encoding="utf-8")
            assert license_is_mit(path)

    def check_non_mit_license_rejected() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "LICENSE"
            closed = _cat("Propriet", "ary")
            path.write_text(f"Example — {closed} License\n\n" + _cat("All rights", " reserved") + ".\n",
                            encoding="utf-8")
            assert not license_is_mit(path)
            path.write_text(mit_body.replace("free of charge", "for a fee"), encoding="utf-8")
            assert not license_is_mit(path)

    def check_pyproject_pep639() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pyproject.toml"
            path.write_text(
                '[project]\nname = "x"\nlicense = "MIT"\nclassifiers = ["License :: OSI Approved :: MIT License"]\n',
                encoding="utf-8",
            )
            assert check_pyproject(path) == [], check_pyproject(path)

    def check_pyproject_legacy_table() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pyproject.toml"
            path.write_text(
                '[project]\nname = "x"\nlicense = { text = "MIT" }\n'
                'classifiers = ["License :: OSI Approved :: MIT License"]\n',
                encoding="utf-8",
            )
            assert check_pyproject(path) == [], check_pyproject(path)

    def check_pyproject_failures() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pyproject.toml"
            closed = _cat("Propriet", "ary")
            path.write_text(
                f'[project]\nname = "x"\nlicense = {{ text = "{closed}" }}\n'
                'classifiers = ["Private :: Do Not Upload"]\n',
                encoding="utf-8",
            )
            failures = check_pyproject(path)
            assert any("private-classifier" in f for f in failures), failures
            assert any("license-field" in f for f in failures), failures
            assert any("license-classifier" in f for f in failures), failures

    def check_banner_scan_header_window() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            header_hit = repo / "bad.py"
            header_hit.write_text('"""' + _cat("Propriet", "ary") + ' code."""\nx = 1\n', encoding="utf-8")
            deep = repo / "deep.py"
            body = "\n".join(["# ok"] * 30 + ["# " + _cat("All Rights", " Reserved")])
            deep.write_text(body + "\n", encoding="utf-8")
            hits = banner_scan(repo, 10)
            assert [h[0] for h in hits] == ["bad.py"], hits
            hits = banner_scan(repo, 40)
            assert [h[0] for h in hits] == ["bad.py", "deep.py"], hits

    def check_banner_scan_skips() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            closed = _cat("Propriet", "ary")
            (repo / ".git").mkdir()
            (repo / ".git" / "f").write_text(closed + "\n", encoding="utf-8")
            (repo / ".augments").mkdir()
            (repo / ".augments" / "f").write_text(closed + "\n", encoding="utf-8")
            (repo / "bin.dat").write_bytes(b"\xff\xfe " + closed.lower().encode() + b" \x00")
            (repo / "self.py").write_text("terms = ['" + closed + "']\n", encoding="utf-8")
            hits = banner_scan(repo, 10, self_path=repo / "self.py")
            assert hits == [], hits

    checks = (
        ("MIT SPDX text match (incl. re-flowed)", check_mit_license_match),
        ("non-MIT licenses rejected", check_non_mit_license_rejected),
        ("pyproject PEP 639 license string", check_pyproject_pep639),
        ("pyproject legacy license table", check_pyproject_legacy_table),
        ("pyproject Private/closed-license/missing-classifier failures", check_pyproject_failures),
        ("banner scan header window", check_banner_scan_header_window),
        ("banner scan skips .git/.augments/binary/self", check_banner_scan_skips),
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
    parser.add_argument("--repo", default=str(repo_root), help="repo root (default: this repo)")
    parser.add_argument("--license", dest="license_path", help="LICENSE path (default: <repo>/LICENSE)")
    parser.add_argument("--pyproject", help="pyproject.toml path (default: <repo>/pyproject.toml)")
    parser.add_argument("--header-lines", type=int, default=10, help="banner scan window (default: 10 lines)")
    parser.add_argument("--self-test", action="store_true", help="run synthetic-fixture checks and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    repo = Path(args.repo)
    license_path = Path(args.license_path) if args.license_path else repo / "LICENSE"
    pyproject_path = Path(args.pyproject) if args.pyproject else repo / "pyproject.toml"

    failed = False
    mit_ok = license_is_mit(license_path)
    print(f"G-09 provenance license-mit: {'PASS' if mit_ok else 'FAIL'} path={license_path}")
    failed |= not mit_ok

    pyproject_failures = check_pyproject(pyproject_path) if pyproject_path.is_file() else ["pyproject-missing"]
    for failure in pyproject_failures:
        print(f"FAIL-PYPROJECT {failure}")
    print(f"G-09 provenance pyproject: {'PASS' if not pyproject_failures else 'FAIL'} path={pyproject_path}")
    failed |= bool(pyproject_failures)

    banner_hits = banner_scan(repo, args.header_lines)
    for rel, lineno, line in banner_hits:
        print(f"FAIL-BANNER {rel}:{lineno}: {line[:100]}")
    print(f"G-09 provenance banner-scan: {'PASS' if not banner_hits else 'FAIL'} hits={len(banner_hits)}")
    failed |= bool(banner_hits)

    print(f"RESULT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
