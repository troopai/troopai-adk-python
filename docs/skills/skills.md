# Skills

Skills are reusable capability packages that bundle instructions, tools, resources, and governance into composable units. They occupy the middle ground between a FunctionTool (single function) and an Agent (full autonomous entity).

## Core Concepts

### What is a Skill?

A Skill combines:
- **Instructions** (markdown) — domain expertise injected into the agent's system prompt
- **Tools** (FunctionTool/BuiltinTool) — capabilities the skill provides
- **Resources** (named files/content) — reference material for discovery tools
- **Governance** (timeout, budget, retries) — applied as defaults to all skill tools
- **Guardrails** (input/output) — applied to all skill tools

### Skill Activation

Skills support two activation strategies:

- **LAZY** (default): Instructions injected only when a skill's tool is first called. Tool descriptions are always visible (so the LLM can discover them), but instructions load on demand. Cost-conservative default — agents pay nothing per turn for skills the LLM never invokes.
- **EAGER**: Instructions injected at run start. All skill instructions are always in the system prompt. Opt in for agents with few, always-relevant skills.

### Skill Sources

Skills can be loaded from multiple sources:

- **Inline** (code-defined): `Skill(name=..., description=..., ...)`
- **Directory**: `Skill.from_directory("./skills/code-review/")` — loads `SKILL.md`
- **Remote URL**: `await Skill.from_url("https://example.com/SKILL.md")`

### SKILL.md Format

Compatible with LangChain/CrewAI/Google ADK:

```yaml
---
name: code-review
description: Expert Python code review with security focus
version: 1.0.0
author: TroopAI
tags: python, security, review
license: MIT
---

## Instructions

When reviewing code:
1. Check for security vulnerabilities first
2. Then check for performance issues
3. Finally suggest style improvements
```

### Governance

`SkillGovernance` applies defaults to all tools within a skill:

- `timeout` — per-tool execution timeout in seconds
- `max_result_tokens` — max tokens for tool results
- `max_retries` — LLM retry budget

Tool's own governance values take precedence over skill-level defaults.

### Guardrails

Skill-level guardrails are prepended to each tool's guardrail list, so skill guardrails run first:

```python
skill = Skill(
    name="secure-ops",
    description="Security-sensitive operations",
    tools=[transfer_tool, delete_tool],
    input_guardrails=[amount_check_guardrail],
)
```

### Discovery Toolset

`SkillDiscoveryToolset` generates FunctionTools for LLM-driven skill management:

- `list_skills` — Returns names + descriptions of all skills
- `load_skill` — Returns full instructions for a named skill
- `load_skill_resource` — Reads a resource file from a skill
- `run_skill_script` — Executes a script (opt-in, requires approval)

Discovery tools are opt-in — added explicitly to `Agent.tools`.

### SkillSet

`SkillSet` is a named collection for organizing and reusing skills:

```python
devtools = SkillSet(
    name="developer-tools",
    skills=[code_review, testing, debugging],
)
# Reuse across agents
agent.skills = devtools.skills
```

## Lifecycle Hooks

- `on_skill_activated(context, agent, skill_name)` — fired when a skill is activated in LAZY mode

## Design Decisions

- **No hidden behavior**: Skill instructions are opt-in via `Agent(skills=[...])`. Nothing is auto-injected.
- **Agent = config**: Skills are declared on the Agent (configuration), activated by the Runner (execution).
- **Zero-overhead LAZY**: Unlike Google ADK's `load_skill` tool call, our LAZY mode injects instructions after the first tool call — no extra LLM call needed.
- **Governance as defaults**: Skill governance provides defaults; tool's own values always win.
