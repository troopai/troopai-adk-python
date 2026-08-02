"""Sphinx configuration for the TroopAI ADK documentation site."""

from __future__ import annotations

import os

project = "TroopAI ADK"
author = "TroopAI ADK contributors"
copyright = "2026, TroopAI ADK contributors"

# -- General configuration ----------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

# -- MyST configuration -------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3

myst_substitutions = {
    "project_name": "TroopAI ADK",
    "min_python": "3.12",
}

# -- Internationalisation ----------------------------------------------

locale_dirs = ["locale/"]
gettext_compact = False
language = os.environ.get("SPHINX_LANG", "en")

# -- Mermaid -----------------------------------------------------------

mermaid_version = "10.9.0"
# Theme picked up from sphinx-book-theme's data-theme toggle and re-rendered
# on change by docs/_static/js/mermaid-theme.js (loaded via html_js_files).
mermaid_init_js = ""

# -- Theme + HTML ------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/mermaid-theme.js", "js/sidebar-collapse.js"]
html_theme_options = {
    "repository_url": "https://github.com/troopai/troopai-adk-python",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": False,
    "show_navbar_depth": 2,
    "max_navbar_depth": 5,
    "collapse_navbar": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/troopai/troopai-adk-python",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_end": ["navbar-icon-links"],
}

# -- autodoc + intersphinx --------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
typehints_fully_qualified = False
always_document_param_types = True

# Render ``Attributes:`` docstring sections as ``:ivar:`` fields rather than
# standalone ``py:attribute`` objects, so they don't collide with autodoc
# ``:members:`` (which documents the same dataclass fields from their inline
# docstrings) and produce "duplicate object description" warnings.
napoleon_use_ivar = True

# PEP 695 ``type X = ...`` aliases whose right-hand side references names that
# are only imported under ``TYPE_CHECKING`` (to avoid import cycles at runtime)
# cannot be evaluated by sphinx-autodoc-typehints on Sphinx 9: accessing
# ``TypeAliasType.__value__`` triggers lazy evaluation in the module's runtime
# namespace, where those guarded names are absent, causing a hard NameError.
# Mapping each such alias to its fully-qualified dotted name here causes
# sphinx-autodoc-typehints to substitute a ``TypeAliasForwardRef`` in the
# ``get_type_hints`` local namespace, which shadows the real object in the
# module globals so ``__value__`` is never evaluated.
#
# ``ApprovalPolicy`` (``types/tools/approval_policy.py``):
#   ``type ApprovalPolicy = bool | Callable[[ToolContext[Any]], ...]``
#   ``ToolContext`` is imported only under ``TYPE_CHECKING`` in that module.
autodoc_type_aliases = {
    "ApprovalPolicy": "troopai.adk.types.tools.ApprovalPolicy",
}

# ``RunContext`` / ``RunResult`` are TYPE_CHECKING-only forward references in
# string-form annotations (``Agent`` is decoupled from the Runner layer by
# design), so autodoc-typehints cannot resolve them at build time. Silence only
# that one category rather than masking genuine cross-reference problems.
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    # The i18n plumbing (`locale_dirs`, `gettext_compact`, `SPHINX_LANG`,
    # sphinx-intl) is kept dormant for future languages; no catalogs ship
    # today. Should a locale activate with largely untranslated catalogs,
    # the autodoc docstrings' English cross-references (e.g. :class:`Flow`)
    # would not be mirrored in the empty/fuzzy msgstrs. That is a
    # translation-completeness signal, not a build defect — the English
    # (with working refs) is rendered as fallback.
    "i18n.inconsistent_references",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# -- Output -----------------------------------------------------------

exclude_patterns = [
    "build",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Empty placeholder stubs — excluded until they carry content, so they
    # neither render as blank, titleless pages nor warn as toctree orphans.
    "providers/providers.md",
    "rag/rag.md",
]
templates_path = ["_templates"]

# -- Linkcheck ---------------------------------------------------------

# Patterns for URLs that linkcheck should not attempt to verify.
# Reasons:
#   *.md host patterns — the myst linkify extension auto-linkifies bare
#     filenames like "SKILL.md" and "NOTES.md" into http:// URLs; those
#     are not real hyperlinks — they're prose references to convention files.
#   github.com/.../blob/main — the repository is private during development;
#     the blob links in changelog.rst / contributing.rst are intentional and
#     correct for a future public release.
linkcheck_ignore = [
    r"http://.*\.md$",  # bare *.md filenames linkified by myst linkify
    r"https://github\.com/troopai/troopai-adk-python/blob/.*",
]
