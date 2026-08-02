#!/usr/bin/env python3
"""G-04 icon lint — ASM-2026-08-01-troopai-adk-v3 (R-007, icon part).

Normative threshold (not editable here; changes require an ASM successor):
the docs landing page and every section index page must be icon-free —
zero `{octicon}` shortcodes and zero Unicode emoji/pictographic glyphs.

Scope: `docs/index.md` plus every `docs/**/index.md` (section indexes at any
depth). Other docs pages are outside this gate's scope per the matrix
("landing/section indexes icon-free").

Detected patterns:
- MyST/octicon shortcode: the literal role marker ``{octicon}`` (e.g.
  ``{octicon}`star` ``).
- Unicode pictographic ranges: U+1F300–U+1FAFF (pictographs, emoticons,
  transport, supplemental, extended-A), U+2600–U+27BF (misc symbols +
  dingbats), U+2B00–U+2BFF (misc symbols and arrows), U+FE0F (emoji
  variation selector). Plain arrows (U+2190–U+21FF) and box-drawing text
  are NOT flagged.

Exit: 0 on zero hits, 1 otherwise. Output is deterministic (sorted).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OCTICON_RE = re.compile(r"\{octicon\}")
EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿⬀-⯿️]")  # ☀-➿ = U+2600-27BF, ⬀-⯿ = U+2B00-2BFF, ️ = U+FE0F


def index_pages(docs: Path) -> list[Path]:
    """docs/index.md + section index pages (docs/**/index.md), sorted."""
    return sorted(docs.rglob("index.md"), key=lambda p: p.relative_to(docs).as_posix())


def lint_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(lineno, kind, excerpt)] icon hits for one page."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return []
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if OCTICON_RE.search(line):
            hits.append((lineno, "octicon", line.strip()))
        elif EMOJI_RE.search(line):
            hits.append((lineno, "emoji", line.strip()))
    return hits


def lint(docs: Path) -> list[tuple[str, int, str, str]]:
    """All icon hits across index pages: (relpath, lineno, kind, excerpt)."""
    hits: list[tuple[str, int, str, str]] = []
    for page in index_pages(docs):
        rel = page.relative_to(docs).as_posix()
        for lineno, kind, excerpt in lint_file(page):
            hits.append((rel, lineno, kind, excerpt))
    hits.sort()
    return hits


def _self_test() -> int:
    import tempfile

    def check_clean_pages_pass() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "swarms").mkdir()
            (docs / "index.md").write_text("# TroopAI ADK\n\nSee -> the guide and x -> y arrows.\n", encoding="utf-8")
            (docs / "swarms" / "index.md").write_text("# Swarms\n\nPlain text, no icons.\n", encoding="utf-8")
            assert lint(docs) == [], lint(docs)

    def check_octicon_detected() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "index.md").write_text('# Home\n\n{octicon}`star;1em;sd-text-info` Star\n', encoding="utf-8")
            hits = lint(docs)
            assert len(hits) == 1 and hits[0][2] == "octicon" and hits[0][1] == 3, hits

    def check_emoji_ranges_detected() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            page = docs / "index.md"
            page.write_text("# H\n\nlaunch \U0001f680 done \u2705 star \u2b50 heart \u2764\ufe0f\n", encoding="utf-8")
            hits = lint(docs)
            assert len(hits) == 1 and hits[0][2] == "emoji", hits

    def check_only_index_pages_scanned() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "guide.md").write_text("emoji \U0001f680 here\n", encoding="utf-8")  # not an index page
            (docs / "index.md").write_text("clean\n", encoding="utf-8")
            sub = docs / "section"
            sub.mkdir()
            (sub / "index.md").write_text("clean too\n", encoding="utf-8")
            assert lint(docs) == [], lint(docs)

    def check_nested_section_index_scanned() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            deep = docs / "a" / "b"
            deep.mkdir(parents=True)
            (deep / "index.md").write_text("{octicon}`x`\n", encoding="utf-8")
            hits = lint(docs)
            assert [h[0] for h in hits] == ["a/b/index.md"], hits

    def check_arrows_and_ascii_not_flagged() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "index.md").write_text(
                "a -> b, c --> d, x >= y, 100% coverage (tm) (c) (r)\n", encoding="utf-8"
            )
            assert lint(docs) == [], lint(docs)

    checks = (
        ("clean landing + section pages pass", check_clean_pages_pass),
        ("{octicon} shortcode detected", check_octicon_detected),
        ("unicode emoji/pictographic ranges detected", check_emoji_ranges_detected),
        ("non-index pages are out of scope", check_only_index_pages_scanned),
        ("nested section index pages scanned", check_nested_section_index_scanned),
        ("arrows/ascii not flagged", check_arrows_and_ascii_not_flagged),
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
    parser.add_argument(
        "--docs",
        default=str(Path(__file__).resolve().parents[1] / "docs"),
        help="docs directory (default: <repo>/docs)",
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic-fixture checks and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    docs = Path(args.docs)
    if not docs.is_dir():
        print(f"error: --docs {docs} is not a directory", file=sys.stderr)
        return 1
    hits = lint(docs)
    for rel, lineno, kind, excerpt in hits:
        print(f"HIT {rel}:{lineno}: {kind}: {excerpt[:100]}")
    pages = len(index_pages(docs))
    print(f"G-04 icon-lint: hits={len(hits)} index-pages={pages}")
    print(f"RESULT: {'FAIL' if hits else 'PASS'}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
