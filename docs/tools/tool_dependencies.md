# Tool Runtime Dependencies — `requires_env` & `requires_packages`

Declare what a tool needs to run. The ADK validates the declarations
at agent construction and refuses to start if anything is missing.

## When to use

| Situation | Use |
|-----------|-----|
| Tool reads an env var (API key, service URL, region) | `requires_env=(...)` |
| Tool imports a package not in the framework's hard deps | `requires_packages=(...)` |
| You want missing config to fail at boot, not mid-turn | Both |

A misconfigured agent fails *before* the first LLM call. No tokens are
spent on a request that was doomed by a missing `SLACK_TOKEN`.

## Quick example

```python
from troopai.adk.agents import Agent
from troopai.adk.tools import function_tool


@function_tool(
    name="slack_notify",
    description="Post a message to Slack",
    requires_env=("SLACK_TOKEN", "SLACK_CHANNEL"),
    requires_packages=("slack-sdk>=3.0",),
)
def slack_notify(message: str) -> str:
    import os
    from slack_sdk import WebClient

    client = WebClient(token=os.environ["SLACK_TOKEN"])
    client.chat_postMessage(channel=os.environ["SLACK_CHANNEL"], text=message)
    return "posted"


agent = Agent(
    name="Notifier",
    system_prompt="Send notifications when asked.",
    tools=[slack_notify],
)
# If SLACK_TOKEN is unset OR slack-sdk is missing/too-old, Agent(...)
# raises ToolDependencyError listing every offender.
```

## Behaviour

`Agent.__post_init__` walks every `FunctionTool` in `tools` and every
`FunctionTool` in attached skills, calling `tool.validate_dependencies()`
on each. Missing requirements are aggregated into a single
`ToolDependencyError`:

```
troopai.adk.exceptions.ToolDependencyError:
Agent 'Notifier' has tools with unsatisfied dependencies:
  - slack_notify: env:SLACK_TOKEN, package:slack-sdk>=3.0
```

The exception attribute `missing` is a `dict[tool_name, list[str]]` so
ops tooling can inspect failures programmatically:

```python
from troopai.adk.exceptions import ToolDependencyError

try:
    agent = Agent(name="...", system_prompt="...", tools=[...])
except ToolDependencyError as e:
    for tool_name, items in e.missing.items():
        log.error("tool %s: %s", tool_name, ", ".join(items))
```

## Requirement formats

### `requires_env`

A `tuple[str, ...]` of environment variable names. A variable is
considered *unsatisfied* when it is unset or set to an empty string.

```python
requires_env=("API_KEY",)               # one var
requires_env=("API_KEY", "API_REGION")  # multiple
requires_env=()                         # default — no requirement
```

### `requires_packages`

A `tuple[str, ...]` of PEP 508 requirement strings. Each entry can
specify a version constraint:

```python
requires_packages=("requests",)             # any version installed
requires_packages=("requests>=2.30",)       # version floor
requires_packages=("requests>=2.30,<3",)    # range
requires_packages=("slack-sdk>=3.0",)
```

The validator parses each spec via `packaging.requirements.Requirement`,
looks up the installed distribution via `importlib.metadata.version`,
and matches the version against the specifier. An invalid spec is
treated as missing — fix the spec.

## Per-tool API

The validator is also callable directly on a `FunctionTool`:

```python
tool = slack_notify  # the @function_tool result above
unsatisfied = tool.validate_dependencies()
# unsatisfied == ["env:SLACK_TOKEN", "package:slack-sdk>=3.0"]
# (or [] when everything is healthy)
```

Useful for custom validation flows (e.g. validating a tool registry at
service startup before any agents are constructed).

## Why fail fast at construction?

Validating at `Agent(...)` rather than at the first `tool` invocation
matters because:

- Misconfiguration is a deploy-time concern, not a runtime concern.
  The error should surface in the same window as your other startup
  checks.
- The LLM never sees the failure. No tokens spent, no model confusion
  about a tool that "exists" but always errors.
- The error message lists every offender — a single fix-list instead
  of N round-trips to discover them one at a time.

## What this does NOT do

- It does not check **transitive** package requirements. If
  `slack-sdk` is installed but its own dependencies are broken, that
  surfaces at import time, not here.
- It does not check **runtime credentials** (e.g. that `SLACK_TOKEN` is
  *valid*, only that it is non-empty). The first API call still
  validates the credential.
- It does not run on `BuiltinTool` subclasses or hosted-tool dataclasses
  (they have no `requires_env` / `requires_packages` fields). Use
  `FunctionTool` for any tool that needs dependency declaration.

## See also

- `tests/unit/tools/test_tool_dependencies.py` — `validate_dependencies()`
  unit tests
- `tests/unit/agents/test_agent_dependency_validation.py` — agent-level
  integration tests
- `examples/tools/tool_dependencies.py` — runnable example
