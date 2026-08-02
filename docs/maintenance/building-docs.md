(maintenance/building-docs)=

# 🏗️ Building the Docs

The documentation site is **MyST Markdown** rendered by **Sphinx**. Every
page under `docs/` is a `.md` file; Sphinx turns the tree into a static
HTML site whose entry point is `index.html`.

## Setup

The toolchain ships as the `[docs]` optional-dependency group (Sphinx,
`myst-parser`, the `sphinx-book-theme`, `sphinx-design`, the Mermaid and
copy-button extensions, and `sphinx-intl` for translations):

```bash
conda activate troopai-adk-python
pip install -e '.[docs]'
```

## Build the English site

From the `docs/` directory:

```bash
cd docs
make html
```

The rendered site lands in `build/html/en/`. Open
`docs/build/html/en/index.html` in any browser.

`index.html` is generated from `docs/index.md` (the root document). Sphinx
walks the `toctree` directives outward from there, so a page only appears
in the site if it is reachable through a `toctree` — an orphaned `.md`
file builds to HTML but is unreachable and triggers a warning.

### Without `make`

`make` only wraps `sphinx-build`. If `make` is not installed, call the
builder directly — it produces the same `index.html`:

```bash
cd docs
sphinx-build -b html . build/html/en
```

## Live preview while editing

```bash
cd docs
make livehtml
```

This runs `sphinx-autobuild`: it serves the site locally, watches both
`docs/` and `../src` (so docstring edits picked up by autodoc rebuild
too), and refreshes the browser on save.

## Build all three languages

The site ships English source plus French and German translations:

```bash
cd docs
make html-all
```

This builds `build/html/en/`, `build/html/fr/`, `build/html/de/` and
writes a top-level `build/html/index.html` that redirects to the English
site.

## Check links

```bash
cd docs
make linkcheck
```

Reports broken external links into `build/linkcheck/`. Bare `*.md`
filenames in prose and the private-repo `blob/` links are intentionally
excluded in `conf.py`.

## Clean

```bash
cd docs
make clean        # removes build/
```

## Make targets at a glance

| Target | Does |
|---|---|
| `make html` | English site → `build/html/en/` |
| `make html-fr` / `make html-de` | One translated site → `build/html/<lang>/` |
| `make html-all` | All three languages + redirect `index.html` |
| `make livehtml` | Live-reload preview server (watches `docs/` + `../src`) |
| `make linkcheck` | External-link checker |
| `make clean` | Delete `build/` |

## How it's wired

`conf.py` holds the whole configuration: the enabled Sphinx extensions
(MyST, autodoc + Napoleon for docstrings, intersphinx, Mermaid), the MyST
syntax extensions (`colon_fence`, `dollarmath`, `deflist`, …), the
`sphinx-book-theme` options, and the i18n settings (`locale_dirs`, and the
`SPHINX_LANG` environment variable the `make` targets set per language).

## Related

- {doc}`updating-docs` — what to touch in `docs/` when you ship code.
