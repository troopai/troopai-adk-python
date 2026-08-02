"""Tests for loading an Agent from a JSON config file.

The loader reads bytes, parses JSON, validates against the schema, and
assembles the Agent. Parse and validation failures surface as
``ConfigParseError`` (a ``UserError``); a missing file surfaces as the
natural ``FileNotFoundError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.config import load_agent
from troopai.adk.exceptions import ConfigParseError
from troopai.adk.llms.anthropic.anthropic_model import AnthropicLLM
from troopai.adk.tools import FunctionTool


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(payload))
    return path


def test_loads_valid_agent(tmp_path: Path) -> None:
    path = _write(tmp_path, {"name": "triage", "system_prompt": "You triage."})
    agent = load_agent(path)
    assert isinstance(agent, Agent)
    assert agent.name == "triage"


def test_invalid_json_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text("{ not valid json ")
    with pytest.raises(ConfigParseError):
        load_agent(path)


def test_non_mapping_root_raises_parse_error(tmp_path: Path) -> None:
    path = _write(tmp_path, ["not", "a", "mapping"])
    with pytest.raises(ConfigParseError):
        load_agent(path)


def test_validation_failure_raises_parse_error(tmp_path: Path) -> None:
    path = _write(tmp_path, {"system_prompt": "missing name"})
    with pytest.raises(ConfigParseError):
        load_agent(path)


def test_code_only_key_parse_error_mentions_python(tmp_path: Path) -> None:
    path = _write(tmp_path, {"name": "a", "system_prompt": "p", "hooks": "x"})
    with pytest.raises(ConfigParseError) as exc_info:
        load_agent(path)
    assert "Python" in str(exc_info.value)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_agent(tmp_path / "does_not_exist.json")


def test_loads_agent_with_provider_block(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"name": "a", "system_prompt": "p", "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5"}},
    )
    agent = load_agent(path)
    assert isinstance(agent.llm, AnthropicLLM)
    assert agent.llm.model == "claude-sonnet-4-5"


def test_both_llm_sources_raises_parse_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "name": "a",
            "system_prompt": "p",
            "llm": {"provider": "anthropic", "model": "m", "config": {"temperature": 0.5}},
            "llm_config": {"temperature": 0.7},
        },
    )
    with pytest.raises(ConfigParseError, match="one source"):
        load_agent(path)


def test_loads_agent_with_hosted_tool_guardrails_dynamic_prompt(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "name": "support",
            "system_prompt": {"dynamic": "tests.unit.config.sample_symbols:build_prompt"},
            "tools": [{"type": "web_search", "args": {"max_uses": 2}}],
            "guardrails": {"input": [{"ref": "tests.unit.config.sample_symbols:my_input_guard"}], "output": []},
        },
    )
    agent = load_agent(path)
    assert callable(agent.system_prompt)
    assert len(agent.tools) == 1
    assert len(agent.guardrails.input) == 1


def test_resolves_ref_relative_to_config_dir(tmp_path: Path) -> None:
    """A dotted ref resolves a *sibling* module by its bare name.

    ``weather_tool`` is not importable from the test's working directory; it
    resolves only because the loader puts the config file's own directory on
    ``sys.path`` for the duration of the load. The path is restored afterward.
    """
    import sys

    (tmp_path / "weather_tool.py").write_text(
        "from troopai.adk.tools import function_tool\n\n\n"
        "@function_tool\n"
        "def get_weather(city: str) -> str:\n"
        '    return f"sunny in {city}"\n',
        encoding="utf-8",
    )
    path = _write(tmp_path, {"name": "w", "system_prompt": "p", "tools": ["weather_tool.get_weather"]})

    before = list(sys.path)
    agent = load_agent(path)

    assert len(agent.tools) == 1
    tool = agent.tools[0]
    assert isinstance(tool, FunctionTool)
    assert tool.name == "get_weather"
    # The config directory was added only for the load; sys.path is restored.
    assert sys.path == before
    assert str(tmp_path.resolve()) not in sys.path


def test_importable_dir_appends_not_prepends(tmp_path: Path) -> None:
    """The config directory is appended (lowest precedence), never prepended.

    Appending is what stops a sibling file named like a real module from
    shadowing the installed/stdlib one during the load.
    """
    import sys

    from troopai.adk.config.resolver import importable_dir

    entry = str(tmp_path.resolve())
    sentinel = "/__import_front_sentinel__"
    sys.path.insert(0, sentinel)
    try:
        with importable_dir(tmp_path):
            assert sys.path[-1] == entry  # appended at the end
            assert sys.path[0] == sentinel  # the front is untouched
        assert entry not in sys.path  # removed on exit
    finally:
        sys.path.remove(sentinel)


def test_sys_path_restored_after_build_error(tmp_path: Path) -> None:
    """A build failure still removes the config directory from ``sys.path``."""
    import sys

    from troopai.adk.exceptions import ConfigResolutionError

    (tmp_path / "errmod.py").write_text(
        "from troopai.adk.tools import function_tool\n\n\n@function_tool\ndef good() -> str:\n    return 'ok'\n",
        encoding="utf-8",
    )
    # 'errmod.missing' resolves to a module that lacks the attribute → build error.
    path = _write(tmp_path, {"name": "x", "system_prompt": "p", "tools": ["errmod.missing"]})

    before = list(sys.path)
    with pytest.raises(ConfigResolutionError):
        load_agent(path)
    assert sys.path == before
    assert str(tmp_path.resolve()) not in sys.path


def test_importable_dir_leaves_preexisting_entry(tmp_path: Path) -> None:
    """A directory already on ``sys.path`` is left in place after the load."""
    import sys

    (tmp_path / "keepmod.py").write_text(
        "from troopai.adk.tools import function_tool\n\n\n@function_tool\ndef keep() -> str:\n    return 'ok'\n",
        encoding="utf-8",
    )
    entry = str(tmp_path.resolve())
    sys.path.append(entry)
    try:
        path = _write(tmp_path, {"name": "x", "system_prompt": "p", "tools": ["keepmod.keep"]})
        load_agent(path)
        assert entry in sys.path  # not removed — the loader did not add it
    finally:
        sys.path.remove(entry)


def _write_text(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_yaml_agent(tmp_path: Path) -> None:
    path = _write_text(tmp_path, "agent.yaml", "name: yamler\nsystem_prompt: From YAML.\n")
    agent = load_agent(path)
    assert agent.name == "yamler"
    assert agent.system_prompt == "From YAML."


def test_loads_yml_agent(tmp_path: Path) -> None:
    path = _write_text(tmp_path, "agent.yml", "name: y2\nsystem_prompt: p\n")
    assert load_agent(path).name == "y2"


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    path = _write_text(tmp_path, "agent.txt", "name: x")
    with pytest.raises(ConfigParseError, match="extension"):
        load_agent(path)


def test_yaml_non_mapping_root_rejected(tmp_path: Path) -> None:
    path = _write_text(tmp_path, "agent.yaml", "- just\n- a\n- list\n")
    with pytest.raises(ConfigParseError):
        load_agent(path)


def test_yaml_syntax_error_raises_parse_error(tmp_path: Path) -> None:
    path = _write_text(tmp_path, "agent.yaml", "key: [unclosed bracket\n")
    with pytest.raises(ConfigParseError, match="Invalid YAML"):
        load_agent(path)


def test_yaml_missing_pyyaml_raises_parse_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    path = _write_text(tmp_path, "agent.yaml", "name: x\nsystem_prompt: p\n")
    # Setting the module to None makes ``import yaml`` raise ImportError,
    # simulating an environment without the optional pyyaml dependency.
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ConfigParseError, match="pyyaml"):
        load_agent(path)
