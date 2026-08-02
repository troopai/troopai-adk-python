### Summary

<!-- What does this change do, and why? Link the spec/issue if there is one. -->

### Issue

<!-- e.g. "Closes #123". Use "N/A" for a standalone change. -->

### Test plan

<!-- How was this verified? Commands run, examples exercised, cases covered. -->

### Hygiene gate

<!-- The five-check gate from CONTRIBUTING.md — all must be clean. -->

- [ ] `ruff check src tests examples`
- [ ] `ruff format --check src tests examples`
- [ ] `mypy -p troopai.adk`
- [ ] `pyright src/troopai/adk/`
- [ ] IDE diagnostics clean on touched files

### Checklist

- [ ] Tests added/updated for the change
- [ ] Docs updated
- [ ] `CHANGELOG.md` `[Unreleased]` entry added (user-visible effect, not implementation detail)
- [ ] No version bump in this PR — the maintainer cuts releases (see `RELEASING.md`)
- [ ] No version language in shipped code (`src/`, `tests/`, `docs/`, `examples/`, `README.md`)
