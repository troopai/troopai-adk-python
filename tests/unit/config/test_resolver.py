"""Tests for the dotted-path reference resolver.

The resolver turns a string reference in a config file (e.g.
``"my_pkg.tools:search"``) into the live Python object it names. It is the
single escape hatch that lets a declarative config reach code-only symbols
(tool functions, output-schema classes, dynamic prompts, guardrails).
"""

from __future__ import annotations

import json

import pytest

from troopai.adk.config.resolver import resolve_dotted_spec
from troopai.adk.exceptions import ConfigResolutionError


class TestResolveDottedSpec:
    def test_colon_form_resolves_attribute(self) -> None:
        # "module.path:attr" — the explicit, unambiguous form.
        assert resolve_dotted_spec("json:loads") is json.loads

    def test_dot_form_resolves_attribute(self) -> None:
        # "module.path.attr" — the dotted form (split on the final dot).
        assert resolve_dotted_spec("json.loads") is json.loads

    def test_nested_module_colon_form(self) -> None:
        from os import path as os_path

        assert resolve_dotted_spec("os.path:join") is os_path.join

    def test_unknown_module_raises_resolution_error(self) -> None:
        with pytest.raises(ConfigResolutionError):
            resolve_dotted_spec("troopai_no_such_module_xyz:thing")

    def test_unknown_attribute_raises_resolution_error(self) -> None:
        with pytest.raises(ConfigResolutionError):
            resolve_dotted_spec("json:no_such_attribute_xyz")

    def test_missing_separator_raises_resolution_error(self) -> None:
        # A bare token with no module/attribute separator is unresolvable.
        with pytest.raises(ConfigResolutionError):
            resolve_dotted_spec("justaword")

    @pytest.mark.parametrize("spec", [":loads", "json:", ":"])
    def test_empty_segment_colon_form_raises(self, spec: str) -> None:
        with pytest.raises(ConfigResolutionError):
            resolve_dotted_spec(spec)

    def test_relative_module_path_raises(self) -> None:
        # A leading-dot (relative import) must fail loudly, not raise TypeError.
        with pytest.raises(ConfigResolutionError):
            resolve_dotted_spec(".foo:bar")

    def test_module_that_raises_on_import_is_wrapped(self) -> None:
        # A module that errors during import surfaces as ConfigResolutionError.
        with pytest.raises(ConfigResolutionError):
            resolve_dotted_spec("tests.unit.config.sample_broken:anything")

    def test_resolution_error_names_the_spec(self) -> None:
        # The error must quote the offending spec so the misconfigured
        # config file is easy to locate.
        with pytest.raises(ConfigResolutionError) as exc_info:
            resolve_dotted_spec("troopai_no_such_module_xyz:thing")
        assert "troopai_no_such_module_xyz:thing" in str(exc_info.value)
