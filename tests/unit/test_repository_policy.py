"""Repository-level CI, release, and packaging policy tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_release_workflow_tags_exact_merge_commit() -> None:
    workflow = _read(".github/workflows/release-tag.yml")

    assert "ref: ${{ github.event.pull_request.merge_commit_sha }}" in workflow
    assert "RELEASE_COMMIT_SHA: ${{ github.event.pull_request.merge_commit_sha }}" in workflow
    assert 'git tag -a "$tag" -m "Release $tag" "$RELEASE_COMMIT_SHA"' in workflow
    assert "ref: main" not in workflow


def test_release_workflow_install_tests_built_wheel() -> None:
    workflow = _read(".github/workflows/release-tag.yml")

    assert "python -m build" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "python -m venv /tmp/troopai-wheel-smoke" in workflow
    assert "python -m pip install dist/*.whl" in workflow
    assert "troopai.adk.__version__" in workflow
    assert "troopai/adk/py.typed" in workflow
    assert "from troopai.adk import Agent, Runner, function_tool" in workflow
    assert "bin/troopai --help" in workflow


def test_release_workflow_existing_tag_is_verified_and_release_is_completed() -> None:
    workflow = _read(".github/workflows/release-tag.yml")

    assert 'git rev-parse "$tag^{commit}"' in workflow
    assert 'existing_sha="$(git rev-parse "$tag^{commit}")"' in workflow
    assert 'if [ "$existing_sha" != "$RELEASE_COMMIT_SHA" ]; then' in workflow
    assert 'gh release view "$tag"' in workflow
    assert 'gh release upload "$tag" dist/* --clobber' in workflow
    assert "Tag $tag already exists; skipping." not in workflow


def test_postgres_unit_tests_are_marker_selected() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    markers = "\n".join(pyproject["tool"]["pytest"]["ini_options"]["markers"])
    makefile = _read("Makefile")
    ci = _read(".github/workflows/ci.yml")
    integration = _read(".github/workflows/integration.yml")

    assert "postgres:" in markers
    assert '-m "not integration and not postgres"' in makefile
    assert '-m "not integration and not postgres"' in ci
    assert "-m postgres" in integration
    assert "tests/unit/graphs/test_postgres_checkpointer.py" not in makefile
    assert "tests/unit/graphs/test_postgres_checkpointer.py" not in ci


def test_supported_python_metadata_matches_ci_matrix() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    ci = _read(".github/workflows/ci.yml")

    assert pyproject["project"]["requires-python"] == ">=3.12,<3.14"
    assert 'python-version: ["3.12", "3.13"]' in ci


def test_temporary_vulnerability_suppressions_are_owned_and_bounded() -> None:
    security = _read(".github/workflows/security.yml")
    precommit = _read(".pre-commit-config.yaml")
    environment = _read("environment.yaml")

    for cve in ("CVE-2026-40217", "CVE-2026-28684"):
        assert f"--ignore-vuln {cve}" in security
        assert f"--ignore-vuln={cve}" in precommit
        assert f"{cve} owner:" in security
        assert f"{cve} removal:" in security
        assert f"{cve} owner:" in precommit
        assert f"{cve} removal:" in precommit

    assert "requires-python>=3.12 (no upper cap)" not in environment
    assert "we cap the project's supported Python to <3.14" not in precommit
