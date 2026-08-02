# Skills Module

Reusable capability packages that bundle instructions, tools, resources, and governance into composable units.

## Key Files

- `skill.py` — `Skill`, `SkillMetadata`, `SkillGovernance` dataclasses
- `activation.py` — `SkillActivation` enum (EAGER, LAZY)
- `skill_set.py` — `SkillSet` named collection container
- `skill_rendering.py` — System prompt rendering for skill instructions
- `discovery.py` — `SkillDiscoveryToolset` (LLM-driven discovery tools)
- `sources/base.py` — `SkillSource` ABC
- `sources/directory.py` — `DirectorySkillSource` (SKILL.md parser)
- `sources/remote.py` — `RemoteSkillSource` (URL loader)

## Skill vs Tool vs Agent

| Dimension | FunctionTool | Skill | Agent |
|-----------|-------------|-------|-------|
| Instructions | None (description only) | Rich markdown | Full system prompt |
| Tools | Single function | 0-N tools bundled | 0-N tools |
| Guardrails | Per-tool only | Skill-level + per-tool | Agent-level |
| Governance | Per-tool | Skill-level defaults | Full RunConfig |
| Execution | Stateless function | Instruction-enriched | Full agent loop |

## Activation Strategies

- **LAZY** (default): Instructions injected only when a skill's tool is first called. Zero-overhead discovery since tool descriptions are already in the tool list. Cost-conservative default — every turn pays nothing for unused skills.
- **EAGER**: Instructions injected into system prompt at run start. Best for agents with few, always-relevant skills — opt in explicitly.

## Governance

`SkillGovernance(timeout, max_result_tokens, max_retries)` applied as defaults to all skill tools. Tool's own values take precedence.

## Guardrails

Skill-level `input_guardrails` and `output_guardrails` are prepended to each tool's guardrails (skill guardrails run first).

## Sources

- **Inline**: `Skill(name=..., description=..., ...)` — code-defined
- **Directory**: `Skill.from_directory(path)` — loads SKILL.md (compatible with LangChain/CrewAI/ADK format)
- **Remote**: `await Skill.from_url(url)` — loads SKILL.md from URL

## Discovery

`SkillDiscoveryToolset(skills=[...])` generates FunctionTools for LLM-driven discovery:
- `list_skills` — Names + descriptions of available skills
- `load_skill` — Full instructions for a named skill
- `load_skill_resource` — Access skill resources
- `run_skill_script` — Execute scripts (opt-in, requires approval)

Discovery tools are opt-in — added to `Agent.tools`, not auto-injected.

## Runner Integration

- `resolve_system_prompt()` in `run/loop.py` appends EAGER skill instructions
- `build_tools()` in `run/llm_calls.py` merges skill tools with governance/guardrails
- LAZY activation tracked via `skill_tool_map` in both loop functions
- `on_skill_activated` hook fired on LAZY activation

See `docs/skills/skills.md` for usage. See `examples/skills/` for examples.
