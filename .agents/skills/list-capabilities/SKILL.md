---
name: list-capabilities
description: List the current runtime subagents and MCP servers so delegation and live-tool choices use what Codex actually exposes now.
---

# List Capabilities

Goal: produce the current set of things available for delegation or direct
tool use in this session, drawn from the live runtime rather than
machine-specific install paths.

Do not read `~/.claude/plugins/cache` or any `plugins/marketplaces`
directory. Those paths and layouts differ by machine and version. Runtime
tool metadata is the portable source.

## Subagents

List every Codex subagent type available in this runtime, grouped by origin:

- Built-in agents exposed by Codex.
- Project agents defined in `.codex/agents/*.toml`.
- Any plugin or system agents exposed by the current session.

Use the current runtime tools and project files to identify what is actually
available. Do not invent agents from prose.

## MCP Servers

Separately, list every MCP server exposed this session. MCP tools are direct
tools against live systems, not agents to dispatch.

Group by server name and summarize the available tool purpose in one line.

## Selection Rule

For audit, review, security, testing, quality, documentation, or codebase
search, prefer the matching specialist subagent when one is available. Use MCP
when the task needs a live external system such as docs, databases, browser,
payments, or workspace data.
