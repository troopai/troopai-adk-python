"""Project templates for ``troopai new``.

Templates are plain string constants rendered with
:class:`string.Template` (``$name`` placeholders) — transparent to
audit, no template-engine dependency, and no package-data files to
ship. The published JSON Schemas are written next to the generated
config so editors validate it offline via its ``$schema`` pointer.
"""

from __future__ import annotations

import json
from string import Template

AGENT_CONFIG_TEMPLATE = Template(
    """\
{
  "$$schema": "./agent_config.schema.json",
  "name": "$name",
  "description": "Starter agent scaffolded by the TroopAI CLI.",
  "system_prompt": "You are $name, a concise and helpful assistant. Use the current_time tool when the user asks about dates or times.",
  "llm": "$model",
  "tools": ["${name}_tools.current_time"]
}
"""
)

AGENT_TOOLS_TEMPLATE = Template(
    '''\
"""Tools for the $name agent.

Referenced from agent.json by dotted name — references resolve relative
to the config file's directory, so this module is found regardless of
where you launch from.
"""

from datetime import UTC, datetime

from troopai.adk.tools import function_tool


@function_tool
def current_time() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()
'''
)

TOPOLOGY_CONFIG_TEMPLATE = Template(
    """\
{
  "$$schema": "./topology_config.schema.json",
  "agents": {
    "triage": {
      "name": "triage",
      "description": "Routes each request to the right specialist.",
      "system_prompt": "Classify the user's request. Hand off to the expert for anything substantive; answer trivial questions yourself in one sentence.",
      "llm": "$model",
      "handoffs": ["expert"]
    },
    "expert": {
      "name": "expert",
      "description": "Answers in depth.",
      "system_prompt": "Answer the user's request thoroughly and precisely.",
      "llm": "$model"
    }
  },
  "entry": "triage"
}
"""
)

AGENT_README_TEMPLATE = Template(
    """\
# $name

A starter TroopAI ADK agent project.

```bash
troopai validate agent.json            # schema-check (no tokens spent)
troopai validate --resolve agent.json  # also import the tool references
troopai run agent.json "What time is it?"
troopai chat agent.json
```

Set provider credentials first — copy `.env.example` to `.env`, fill it
in, and pass `--env-file .env` to run/chat (the CLI never loads env
files implicitly).

Tool references in `agent.json` (like `${name}_tools.current_time`)
resolve relative to the config file's directory.
"""
)

TOPOLOGY_README_TEMPLATE = Template(
    """\
# $name

A starter TroopAI ADK multi-agent topology (triage → expert handoff).

```bash
troopai validate topology.json             # schema-check (no tokens spent)
troopai run topology.json "Explain BSP supersteps"
```

Set provider credentials first — copy `.env.example` to `.env`, fill it
in, and pass `--env-file .env` to run (the CLI never loads env files
implicitly).
"""
)

ENV_EXAMPLE = """\
# Copy to .env, fill in what your provider needs, then pass it explicitly:
#   troopai run agent.json --env-file .env "hello"
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=
# GEMINI_API_KEY=
"""


def render_project(kind: str, name: str) -> dict[str, str]:
    """Render the file set for a new project.

    Args:
        kind: ``"agent"`` or ``"topology"``.
        name: The validated project (and agent) name.

    Returns:
        Map of file name (relative to the project directory) to content.
    """
    from troopai.adk.config.schema import dump_agent_config_schema, dump_topology_config_schema
    from troopai.adk.run.config import DEFAULT_MODEL

    values = {"name": name, "model": DEFAULT_MODEL}
    if kind == "topology":
        return {
            "topology.json": TOPOLOGY_CONFIG_TEMPLATE.substitute(values),
            "topology_config.schema.json": json.dumps(dump_topology_config_schema(), indent=2) + "\n",
            ".env.example": ENV_EXAMPLE,
            "README.md": TOPOLOGY_README_TEMPLATE.substitute(values),
        }
    return {
        "agent.json": AGENT_CONFIG_TEMPLATE.substitute(values),
        f"{name}_tools.py": AGENT_TOOLS_TEMPLATE.substitute(values),
        "agent_config.schema.json": json.dumps(dump_agent_config_schema(), indent=2) + "\n",
        ".env.example": ENV_EXAMPLE,
        "README.md": AGENT_README_TEMPLATE.substitute(values),
    }
