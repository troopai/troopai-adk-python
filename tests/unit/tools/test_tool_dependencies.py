"""Unit tests for FunctionTool.validate_dependencies()."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from troopai.adk.tools.function_tool import FunctionTool

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _make_tool(**overrides: Any) -> FunctionTool:
    defaults: dict[str, Any] = {"name": "t", "schema": MINIMAL_SCHEMA}
    defaults.update(overrides)
    return FunctionTool(**defaults)


class TestRequiresEnv:
    def test_empty_tuple_default(self) -> None:
        tool = _make_tool()
        assert tool.requires_env == ()
        assert tool.validate_dependencies() == []

    def test_var_set_passes(self) -> None:
        tool = _make_tool(requires_env=("TROOPAI_DEPS_TEST",))
        with patch.dict("os.environ", {"TROOPAI_DEPS_TEST": "value"}, clear=False):
            assert tool.validate_dependencies() == []

    def test_var_unset_fails(self) -> None:
        tool = _make_tool(requires_env=("TROOPAI_DEPS_TEST_UNSET",))
        with patch.dict("os.environ", {}, clear=True):
            assert tool.validate_dependencies() == ["env:TROOPAI_DEPS_TEST_UNSET"]

    def test_var_empty_string_fails(self) -> None:
        tool = _make_tool(requires_env=("TROOPAI_DEPS_TEST_EMPTY",))
        with patch.dict("os.environ", {"TROOPAI_DEPS_TEST_EMPTY": ""}, clear=False):
            assert tool.validate_dependencies() == ["env:TROOPAI_DEPS_TEST_EMPTY"]

    def test_multiple_vars_all_missing(self) -> None:
        tool = _make_tool(requires_env=("VAR_A", "VAR_B"))
        with patch.dict("os.environ", {}, clear=True):
            assert sorted(tool.validate_dependencies()) == ["env:VAR_A", "env:VAR_B"]

    def test_multiple_vars_partial(self) -> None:
        tool = _make_tool(requires_env=("VAR_A", "VAR_B"))
        with patch.dict("os.environ", {"VAR_A": "x"}, clear=True):
            assert tool.validate_dependencies() == ["env:VAR_B"]


class TestRequiresPackages:
    def test_empty_tuple_default(self) -> None:
        tool = _make_tool()
        assert tool.requires_packages == ()
        assert tool.validate_dependencies() == []

    def test_installed_package_no_spec(self) -> None:
        # pydantic is a hard dependency of the project — guaranteed installed.
        tool = _make_tool(requires_packages=("pydantic",))
        assert tool.validate_dependencies() == []

    def test_installed_package_satisfies_spec(self) -> None:
        tool = _make_tool(requires_packages=("pydantic>=2",))
        assert tool.validate_dependencies() == []

    def test_installed_package_violates_spec(self) -> None:
        # pydantic 2.x is installed; pin to 1.x to force violation.
        tool = _make_tool(requires_packages=("pydantic<2",))
        assert tool.validate_dependencies() == ["package:pydantic<2"]

    def test_missing_package(self) -> None:
        tool = _make_tool(requires_packages=("troopai-deps-test-no-such-package-67890",))
        assert tool.validate_dependencies() == ["package:troopai-deps-test-no-such-package-67890"]

    def test_invalid_spec_treated_as_missing(self) -> None:
        tool = _make_tool(requires_packages=("not a valid spec ?!",))
        assert tool.validate_dependencies() == ["package:not a valid spec ?!"]

    def test_url_spec_rejected(self) -> None:
        # URL-based requirements (PEP 508 ``pkg @ git+https://...``)
        # cannot be verified by importlib.metadata — only pip can do
        # provenance. The validator must surface them as unsatisfied.
        spec = "pydantic @ git+https://github.com/pydantic/pydantic.git@deadbeef"
        tool = _make_tool(requires_packages=(spec,))
        assert tool.validate_dependencies() == [f"package:{spec}"]

    def test_name_canonicalisation_underscore_to_hyphen(self) -> None:
        # Distributions are stored under their PEP 503-normalised name
        # (lowercase, hyphens). A spec written with underscores must
        # still match. ``troopai-adk-python`` is this project itself; check via
        # a known stdlib-installed dist that has neither hyphens nor
        # underscores in its canonical form by using a project dep.
        # ``aiosqlite`` is a hard dep — guaranteed installed.
        tool = _make_tool(requires_packages=("aiosqlite",))
        assert tool.validate_dependencies() == []
        # Underscore form must also resolve via canonicalize_name.
        tool_underscored = _make_tool(requires_packages=("typing_extensions",))
        assert tool_underscored.validate_dependencies() == []


class TestCombined:
    def test_env_and_package_both_missing(self) -> None:
        tool = _make_tool(
            requires_env=("UNSET_VAR_X",),
            requires_packages=("troopai-deps-no-such-pkg-99999",),
        )
        with patch.dict("os.environ", {}, clear=True):
            missing = tool.validate_dependencies()
        assert "env:UNSET_VAR_X" in missing
        assert "package:troopai-deps-no-such-pkg-99999" in missing
        assert len(missing) == 2

    def test_all_present(self) -> None:
        tool = _make_tool(
            requires_env=("TROOPAI_PRESENT_VAR",),
            requires_packages=("pydantic",),
        )
        with patch.dict("os.environ", {"TROOPAI_PRESENT_VAR": "x"}, clear=False):
            assert tool.validate_dependencies() == []
