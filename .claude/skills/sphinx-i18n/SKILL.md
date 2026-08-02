---
name: sphinx-i18n
description: Keep the Sphinx docs' French + German gettext translations in sync — extract strings, update the .po catalogs, translate changed entries, compile .mo, and verify the FR/DE builds render translated (no English fallback). Use after editing docs/ source or src docstrings (autodoc), or when translations drift.
allowed-tools: Bash(make *) Bash(sphinx-intl *) Bash(sphinx-build *) Bash(git diff *) Bash(grep *)
---

# Sphinx i18n (FR / DE) Sync

English is the source of truth. French and German ship as gettext `.po`
translations under `docs/locale/{fr,de}/LC_MESSAGES/`, compiled to `.mo` at
build time. Sphinx renders HTML from the `.mo` binaries, not the `.po`
source — so every `.po` edit needs a recompile. The human contributor guide
is `docs/maintenance/translating.md`; this is the operational procedure. Run
from the repo root; the toolchain is the `[docs]` extra
(`pip install -e '.[docs]'` brings `sphinx-intl`).

## 1. Refresh the catalogs after English changes

```bash
make -C docs gettext       # extract msgids (incl. autodoc'd docstrings)
make -C docs i18n-update   # sphinx-intl update -l fr -l de
```

New source strings appear with an empty `msgstr`; substantially-changed ones
are marked `#, fuzzy`. Both fall back to English until translated.

## 2. Translate the changed entries (FR + DE)

Edit `docs/locale/{fr,de}/LC_MESSAGES/**.po`. Translate **`msgstr` only** —
never touch `msgid`. After translating a fuzzy entry, remove its `#, fuzzy`
marker. Preserve markup verbatim:

- Code blocks and code identifiers in prose (`max_turns` stays `max_turns`).
- MyST/rST roles, directive names (`tip` / `warning` / `note`), and
  `{sub-ref}` substitutions.
- Mermaid diagram labels stay English for now.

See `docs/maintenance/translating.md` for the full style guide.

## 3. Batch discipline (avoid lost writes)

Translate **at most 3–5 `.po` files per pass**, write whole files, and
confirm each write landed before the next batch:

```bash
git diff --stat -- docs/locale
```

Large batches in one pass can freeze or silently drop writes. Whole-site
re-translation is a **workflow** (many parallel agents), not a single pass.

## 4. Compile + build

```bash
make -C docs i18n-build     # sphinx-intl build → .mo  (REQUIRED after any .po edit)
make -C docs html-all       # en + fr + de   (or html-fr / html-de)
```

## 5. Verify no English fallback

Confirm a translated string actually renders in the target HTML — not the
English source:

```bash
grep -Rl "<a known translated phrase>" docs/build/html/fr/ | head
```

If a French / German page shows English, the `.mo` is stale: re-run
`make -C docs i18n-build`. An empty or still-`#, fuzzy` `msgstr` also falls
back — translate it.
