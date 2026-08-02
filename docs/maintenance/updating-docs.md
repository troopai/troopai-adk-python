(maintenance/updating-docs)=

# 📝 Updating the Docs

When you ship code, work through this checklist before declaring
the change done.

## Did you add a new public API symbol?

Add or extend the matching guide under `docs/guides/<topic>.md`. Keep
prose short; show one minimal example.

## Did you change a major architectural decision?

Update or extend the relevant page under `docs/architecture/` to reflect the
new design. If the change is wide-reaching, update `docs/architecture/overview.md`
as well.

## Did you change persisted-format / tolerance behaviour?

Add an entry under `CHANGELOG.md` `[Unreleased]` → `Changed` (or
`Fixed` if it's a bug fix that user code may need to be aware of).

## Did you add a new module under `src/troopai/adk/`?

Add a `docs/guides/<module>.md` (or expand a relevant
`docs/architecture/<page>.md`).

## Did you add user-visible strings to existing docs?

Nothing to do today — the site ships English-only. The i18n plumbing
(`locale_dirs`, `gettext_compact`, `SPHINX_LANG`, `sphinx-intl`) is kept
in place but dormant until a future language is activated. When that
happens, the process will be to refresh the PO catalogs from the current
source and commit them:

```bash
cd docs
make i18n-update    # future target, arrives with the first locale
git add docs/locale/
```

Translators will then fill the new `msgstr` entries in a follow-up.

## Did you change `pyproject.toml` deps?

Update the install instructions in `CONTRIBUTING.md` if a contributor
will need to re-install.

## Before claiming docs are done

Run:

```bash
cd docs
make linkcheck      # no broken links
make html           # English builds clean
```

The hygiene gate (`ruff` / `mypy` / `pyright` / IDE diagnostics)
will not have anything to say about pure doc changes, but run it
anyway for habit.

## Did you touch a public class signature?

The reference autodoc pages (`docs/reference/api/*.md`) pick up
docstrings automatically — make sure the docstring is current.
