---
description: List the current, machine-independent roster of dispatchable subagents and available MCP servers, so the right specialist (or MCP tool) is chosen instead of a possibly-stale hardcoded name.
---

Goal: produce the **current** set of things you can delegate to right now —
**subagents** and **MCP servers** — drawn from your live runtime, not from
machine-specific install paths.

**Never** read `~/.claude/plugins/cache` or any `.../plugins/marketplaces/`
directory: those paths and layouts differ per machine and per version
(another developer won't have your `$HOME`), and the cache is an internal
detail. The runtime tool registry is the portable, authoritative source.

## 1. Subagents

List every `subagent_type` the **Agent tool** exposes in your current
system prompt — that live list is exactly what you can dispatch. Present a
compact table (name · one-line purpose · origin), grouped by origin:

- **built-in** — bare names from the harness (e.g. `Explore`, `Plan`,
  `general-purpose`).
- **plugin** — namespaced `plugin:agent` (e.g. `feature-dev:code-reviewer`).
- **project** — defined in this repo. Identify these precisely with a
  repo-relative listing (portable — no `$HOME`):

  ```bash
  ls .claude/agents/*.md 2>/dev/null
  ```

## 2. MCP servers

Separately, list every MCP server exposed this session. Its tools are named
`mcp__<server>__<tool>` (some may be deferred). Group by `<server>`, one
line each summarizing what the server's tools do (fold in any MCP server
instructions in your context). These are **distinct from subagents** — MCP
is tools you call directly against a live external system, not agents you
dispatch.

## Selection rule

For audit / review / security / testing / quality / codebase-search,
dispatch the matching **specialist** subagent; reserve generic
`Explore`/`general-purpose` for trivial lookups. Reach for an **MCP** tool
when you need a live external system (docs, database, browser, payments…).

Report only what your runtime exposes now. Do not invent agents or servers,
and do not enumerate plugin/marketplace directories from disk.
