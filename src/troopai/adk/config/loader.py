"""Load an ``Agent`` from a JSON or YAML configuration file.

Reads the file as UTF-8 text, parses it by extension (JSON via the stdlib;
YAML behind the optional ``pyyaml`` gate), validates against the strict
schema, and assembles the agent. The Pydantic models are the validation
authority; JSON and YAML both funnel through the same validation path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from troopai.adk.agents.agent import Agent
from troopai.adk.config.assembler import build_agent
from troopai.adk.config.resolver import importable_dir
from troopai.adk.exceptions import ConfigParseError
from troopai.adk.types.config.agent_config import AgentConfig

logger = logging.getLogger(__name__)


def read_config_document(path: Path) -> dict[str, Any]:
    """Read a JSON or YAML config file and return its root mapping.

    Dispatches on the file extension: ``.json`` via the stdlib; ``.yaml`` /
    ``.yml`` via ``yaml.safe_load`` behind an optional-``pyyaml`` gate. Both
    formats funnel through the same Pydantic validation downstream — JSON
    stays canonical and schema-published.

    Args:
        path: Path to the ``.json`` / ``.yaml`` / ``.yml`` config file.

    Returns:
        The parsed root mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigParseError: For a non-UTF-8 file, invalid JSON/YAML, an
            unsupported extension, a missing ``pyyaml`` for a YAML file, or a
            non-mapping root.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigParseError(f"Config {str(path)!r} is not valid UTF-8: {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data: Any = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigParseError(f"Invalid JSON in config {str(path)!r}: {exc}") from exc
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigParseError(
                f"Loading YAML config {str(path)!r} requires pyyaml. Install with: pip install pyyaml"
            ) from exc
        try:
            data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ConfigParseError(f"Invalid YAML in config {str(path)!r}: {exc}") from exc
    else:
        raise ConfigParseError(f"Unsupported config extension {suffix!r} for {str(path)!r}; use .json, .yaml, or .yml.")

    if not isinstance(data, dict):
        raise ConfigParseError(f"Config {str(path)!r} must have a mapping at the root, got {type(data).__name__}.")
    return data


def load_agent(path: str | Path, *, document: dict[str, Any] | None = None) -> Agent:
    """Load an ``Agent`` from a JSON or YAML config file.

    Args:
        path: Path to the ``.json`` / ``.yaml`` / ``.yml`` configuration file.
        document: Pre-parsed root mapping of ``path``. When provided, the
            file is not re-read — callers that already parsed it (e.g. to
            detect the config kind) avoid a second disk read and the window
            where the file changes between the two reads.

    Returns:
        The constructed :class:`~troopai.adk.agents.agent.Agent`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigParseError: If the file is not valid UTF-8/JSON/YAML, has an
            unsupported extension, requires ``pyyaml`` (a YAML file) that is
            not installed, the root is not a mapping, or the document fails
            schema validation.
        ConfigResolutionError: If assembling the agent fails — a tool /
            output-schema reference cannot be resolved, or a provider block
            names an unregistered provider, a missing optional dependency, or
            a field with no matching runtime config field.
    """
    path = Path(path)
    data = read_config_document(path) if document is None else document

    try:
        config = AgentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigParseError(f"Invalid agent config in {str(path)!r}: {exc}") from exc

    logger.info("Loaded agent %r from %s", config.name, str(path))
    # Resolve dotted references (tools, output schema, …) with the config
    # file's own directory importable, so a sibling module loads by bare name.
    with importable_dir(path.resolve().parent):
        return build_agent(config)
