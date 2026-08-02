# TroopAI Agents ADK

Lightweight, provider-agnostic Python framework for multi-agent workflows
with 100+ LLMs via litellm.

> Terminology: this codebase is an **ADK** (Agent Development Kit), not an
> SDK. Use "ADK" in commits, docstrings, comments. Reserve "SDK" for third
> parties (OpenAI Agents SDK, Anthropic SDK).

## Governance

Architectural invariants live in `.claude/rules/architecture.md` (always
loaded). Detailed style/module rules are path-scoped siblings in
`.claude/rules/` and load only when you edit matching files (`*.py`,
`tests/`, `examples/`, `llms/`, etc.). Read `architecture.md` for the
non-negotiables; trust the path-scoped rules to surface when relevant.

Codex compatibility: every `CLAUDE.md` is symlinked as `AGENTS.md`, so edits
to either filename affect both agents. Codex does not natively interpret
Claude's path-scoped rule front matter; when working as Codex, read
`.claude/rules/architecture.md` plus any `.claude/rules/*.md` whose `paths`
match the files you will edit. Do not copy these instruction rules into
`.codex/rules/`, which is for command permission policy.

## Architecture Overview

```
src/troopai/adk/
├── agents/      # Agent, guardrails, handoffs
├── prompts/     # SystemPrompt, tone, dynamic prompts
├── run/         # Runner, config, context, streaming
├── tools/       # Tool system + guardrails
├── types/       # Source of truth for framework types
├── llms/        # LLM abstraction (LLM ABC, LiteLLM, Anthropic, OpenAI, Gemini)
├── handoffs/    # Agent handoff mechanisms
├── swarms/      # Iterative multi-agent collaboration (cycles)
├── graphs/      # State-machine multi-agent orchestration
├── context/     # Compaction, editing, token counting
├── session/     # Persistence (SQLite)
├── memory/      # Memory tools
├── mcp/         # Model Context Protocol
├── a2a/         # Agent-to-Agent
├── tracing/     # OpenTelemetry
├── hooks/       # Lifecycle callbacks
├── schemas/     # AgentOutputSchema
├── config/      # Declarative JSON agent config (load_agent, strict schema)
└── exceptions/  # Exception hierarchy
```

`Input → Input guardrails → Agent loop (LLM → tools → handoffs) → Output
guardrails → Final result`

Each module under `src/troopai/adk/` (and `examples/`, `cookbook/`,
`tests/`) carries its own `CLAUDE.md` with module-specific decisions.

## .claude/ layout

- `rules/` — architectural invariants (`architecture.md`, always loaded)
  plus path-scoped style/module rules that load on matching edits.
- `skills/` — `code-hygiene-gate` and the `add-*` procedures
  (`add-llm-provider`, `add-hosted-tool`, `add-run-item`).
- `agents/` — project subagents. Run `/list-capabilities` for the current,
  authoritative roster (don't rely on names hardcoded in prose).
- `commands/` — slash commands (e.g. `/list-capabilities`).
- `settings.json` — env, enabled plugins, permissions, auto-memory.

## Codex layout

- `AGENTS.md` — symlinks to matching `CLAUDE.md` files.
- `.agents/skills/` — symlinks to `.claude/skills/` plus Codex-native
  wrappers for command-style prompts.
- `.codex/agents/` — Codex custom-agent wrappers that delegate to
  `.claude/agents/*.md` as the source of truth.
- `.codex/hooks.json` and `.codex/hooks/` — Codex hook configuration and
  symlinks to compatible Claude hook scripts.

## Kimi Code layout

- `.kimi-code/skills/` — symlink to `.claude/skills/` (Kimi Code scans it
  as a project-level skill directory; the Claude `SKILL.md` format is
  compatible — `name` + `description` required, extra front matter like
  `allowed-tools` ignored).
- `.kimi-code/agents/` — Kimi-native agent files, hand-maintained from
  `.claude/agents/*.md` (do NOT symlink: Claude front matter such as
  `model:` / `color:` / `skills:` is ignored by Kimi and must not appear;
  keep the two rosters in sync when either changes).
- `.kimi-code/config.toml` — committed project-scoped config (Kimi K3,
  thinking effort `max`, and the `[[hooks]]` wiring below). Kimi Code reads
  only the user-level `~/.kimi-code/config.toml`; activate the project file
  with `KIMI_CODE_HOME="$PWD/.kimi-code" kimi`.
- `.kimi-code/local.toml` — machine-local workspace settings (gitignored).
- `tools/kimi_hooks.py` — hook entrypoint called by the project config's
  `[[hooks]]` (`Stop` / `SubagentStop`); reuses
  `.claude/hooks/docs_sync_reminder.py` detection, reports as plain stdout
  text per Kimi's hook contract.

## Cost Optimization

Default values affecting token cost MUST be cost-conservative (see
`architecture.md`). Levers: `max_result_tokens`, `max_retries`,
JSON-minified tool results, `prompt_caching`, `CacheStrategy.STABLE`,
context compaction + editing, `HandoffConfig.budget` / `collapse`.

## Quick Start

```bash
conda env create -f environment.yaml && conda activate troopai-adk-python
# Python 3.12+
```

Deps: `litellm`, `pydantic`, `mcp`, `temporalio`.

Hygiene gate (must be clean before work is "done"): run the
`code-hygiene-gate` skill — `ruff check`, `ruff format --check`, `mypy`,
`pyright`, and IDE diagnostics. Fix at source, not via suppression.
