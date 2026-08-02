# Releasing

How the TroopAI ADK is versioned and how a release is cut. This is
maintainer guidance, written for humans and AI agents alike — for landing a
change, see [`CONTRIBUTING.md`](CONTRIBUTING.md). Contributors never touch
versions.

> This file is the one place version strings belong outside `pyproject.toml`
> and `CHANGELOG.md`: it is release-management metadata, not shipped code, so
> the project's no-version-language convention does not apply here.

## What a release is

The ADK ships as a **public** package on PyPI (`troopai-adk-python`). A
release is a git tag `vX.Y.Z` on `main` plus a GitHub Release whose notes
come from `CHANGELOG.md`. The release workflow builds the wheel and sdist,
verifies them (twine check, clean-venv install, import + CLI smoke), and
publishes to PyPI via **OIDC trusted publishing** — no long-lived API token.
Publication requires the `pypi` GitHub Environment's required-reviewer
approval (the owner): it is the point of no return.

## Choosing the version

Versions follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
SemVer is a promise about a defined surface — so first, what the promise
covers.

### The versioned surface (the "public API")

A release makes its SemVer promise about all of:

- **The importable API** — every name exported from a package `__all__` under
  `troopai.adk` and documented in `docs/`: `Agent`, `Runner`, the `LLM` ABC,
  tools, guardrails, handoffs, swarms / graphs / flows, and their public
  attributes and signatures.
- **The CLI** — the `troopai` commands, their arguments, and documented output.
- **Declarative config** — the JSON / YAML agent and topology schema accepted
  by `load_agent` / `load_topology`.
- **Persisted formats** — anything the ADK writes and later reads back: session
  stores, checkpoints, `FlowCheckpoint` / `RunState` JSON.
- **The install surface** — the `pip` extras (`[anthropic]`, `[otel]`, …),
  `requires-python`, and dependency floors.
- **The exception contract** — which public `exceptions/` type a documented
  path raises.
- **The observable surface** — span names, span attributes, and metric names
  emitted by the tracing / metrics layer (OpenInference / OTel conventions),
  which downstream dashboards and alerts depend on.
- **Cost and default behaviour** — that defaults stay cost-conservative and
  that no tokens, prompts, tool calls, or retries occur without explicit
  opt-in.

Explicitly **not** covered, and free to change in any release:
underscore-prefixed internals and anything absent from a package `__all__`;
provider wire types inside `llms/<provider>/`; `examples/`; docs prose;
internal authoring / tooling files; and any subsystem documented as
**experimental** (see below).

### The tiers

- **Major** — the surface breaks. A public symbol is removed or renamed; a
  public signature or return type changes incompatibly; a default changes such
  that an existing caller's cost or behaviour changes; the framework starts
  emitting tokens / calls / retries a caller did not opt into; data written by
  an earlier release can no longer be read; an extra is removed or renamed, or a
  supported Python version is dropped; a documented path raises a different,
  incompatible exception; or an established span / metric / attribute name is
  renamed or removed.
- **Minor** — the surface grows, backwards-compatibly: a new provider, tool,
  subsystem, optional parameter (with a cost-conservative default), extra,
  exception subtype (still inheriting the documented base), or telemetry
  span / attribute; a new persisted field older readers ignore and newer readers
  default. Existing callers keep working unchanged.
- **Patch** — the surface is unchanged; existing behaviour is fixed or improved
  at equal-or-lower cost: bug fixes, performance, docs, internal refactors, and
  dependency-floor bumps with no API, behaviour, or runtime-support effect.

### Persisted formats

Persisted formats evolve with no version field: new fields carry safe defaults
and loaders tolerate unknown / missing keys — that is **minor** (new capability)
or **patch** (internal). It is **major** only when an artifact written by an
earlier release can no longer be loaded; a hard break renames the loader rather
than rejecting old data.

### Dependencies & the install surface

Adding an extra is **minor**. Removing or renaming an extra, or raising
`requires-python` so a previously-supported interpreter is dropped, is **major**
— it breaks existing installs. A dependency-floor bump is a **patch** when it
has no API or behaviour effect, but **major** when it changes resolved behaviour
or drops a supported runtime.

### Deprecation

Deprecations are announced in `CHANGELOG.md` (the **Deprecated** subsection) and
the release notes — never as version language in shipped code. A deprecated
symbol keeps working through later **minor** releases and is removed only in a
**major**.

### Security

A security fix is classified by its surface impact like any other change — and
shipped promptly, always recorded under the CHANGELOG **Security** subsection.
If the fix must tighten a default or reject previously-accepted input, that is
still a break and bumps accordingly; we do not disguise a necessary break as a
patch to avoid the bump.

### Experimental subsystems

A young subsystem may be documented as **experimental**. While so marked it sits
*outside* these guarantees and may change in any release. Promotion to stable is
a CHANGELOG-noted event, after which the normal tiers apply.

### Bug fixes that change behaviour

A fix that corrects clearly-wrong behaviour is a **patch**, even if someone
relied on the bug. A fix that changes *documented* behaviour is at least
**minor**.

### The tie-breaker

On a boundary, ask: does a caller gain something *new to reach for* (**minor**),
or does something they already reach for *now work better at no new cost*
(**patch**)? If an existing caller must change their code, their config, their
stored data, their dependency setup, their dashboards, or their token budget to
get the result they had before — it is a **major**.

### Pre-1.0 (the 0.y.z series)

While the project is Alpha, the surface may break in a **minor** release: treat
**minor** as the break tier and **patch** as everything else. Breaking changes
still carry a `BREAKING CHANGE:` footer (or `!`) in the commit and a clear
CHANGELOG note, so the tooling and this policy never disagree.

A release versions the cumulative diff on `main` since the last tag — the
highest tier reached by any change in it decides.

## Who bumps, and when

Nobody bumps in a contribution PR. The version is decided once per release,
by the maintainer, through the release workflow. A PR that edits the
`version` in `pyproject.toml` / `src/troopai/adk/__init__.py`, or adds a
dated `CHANGELOG.md` heading, will be asked to drop it — two parallel PRs
cannot both own the next number.

Contributors *do* add `CHANGELOG.md` entries under `[Unreleased]`. The
release folds those under the new dated heading.

## How a release is cut

The flow is workflow-driven. [Commitizen](https://commitizen-tools.github.io/commitizen/)
performs the atomic version bump — it keeps `pyproject.toml:version` and
`src/troopai/adk/__init__.py:__version__` in lockstep and updates the
CHANGELOG — and two GitHub Actions workflows orchestrate the rest.

1. Decide the tier (above) for everything on `main` since the last tag.
2. Run the **"Release — open PR"** workflow (Actions → that workflow → *Run
   workflow*) and choose the increment (`patch` / `minor` / `major`). It runs
   `cz bump --files-only --increment <tier>`, pushes a `release/vX.Y.Z`
   branch, and opens a PR against `main`.
3. Review the PR: confirm the version, the CHANGELOG heading, and that CI is
   green. Edit the CHANGELOG narrative if needed, then merge it as a normal
   merge commit.
4. Merging fires the **"Release — tag & publish artifacts"** workflow, which
   verifies the merged `pyproject.toml` version matches the branch suffix,
   tags `vX.Y.Z` on `main`, builds the wheel + sdist, and creates the GitHub
   Release with those assets attached.

The wheel and sdist are published to PyPI via OIDC trusted publishing after
the owner's `pypi` Environment approval; the GitHub Release references the
published version.

## Local dry run

Preview the next version and CHANGELOG without changing anything:

```bash
cz bump --dry-run --increment <tier>
```

Build and inspect the artifacts a release would attach:

```bash
python -m build
python -m twine check dist/*
```

## If you are an AI agent asked to release

Follow this file exactly. If the tier is genuinely ambiguous, ask the
maintainer rather than inventing a number. Never push a tag by hand — the tag
is created by the "Release — tag & publish artifacts" workflow after the
release PR merges.
