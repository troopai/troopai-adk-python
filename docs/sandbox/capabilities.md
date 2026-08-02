# Sandbox Capabilities

Capabilities are composable extensions that turn a bare sandbox
session into a useful agent surface. Each capability is a Pydantic
`BaseModel` subclass; a `SandboxAgent` carries a `capabilities` list
that the Runner clones, binds to the active session, and queries for
tools + instruction fragments per turn.

## Available capabilities

| Capability | Purpose |
|---|---|
| `CompactionCapability` | LLM-driven context summarisation when the conversation crosses a context-window threshold. Defaults to static (240k tokens) or dynamic (0.9 of model's context window) depending on the agent's model. |
| `ShellCapability` | Adds `RunCommandTool` + optional `WriteStdinTool` (when the backend reports `supports_pty()`). Configurator callback for advanced per-tool customisation. |
| `FilesystemCapability` | Adds `ViewImageTool` (base64-decoded image read) + `SandboxApplyPatchTool` (unified-diff patch application with workspace-escape protection). |
| `SkillsCapability` | Materialises declarative skills into the workspace at `.agents/`. Supports inline skills, eager local-dir / git-repo sources, and lazy `LocalDirLazySkillSource` / `GitRepoLazySkillSource` with the `LoadSkillTool` for progressive disclosure. |
| `MemoryCapability` | Workspace-persisted memory under `memories/`. Reads `memory_summary.md` into the system prompt on each run, appends `sessions/<rollout-id>.jsonl` per turn, exposes `read_raw_memories()` / `write_consolidated_memory()` for the Phase-2 LLM-driven consolidation pipeline. |

## Default capability list

`Capabilities.default()` returns `[CompactionCapability()]` only —
**not** `[Filesystem, Shell, Compaction]` like OpenAI's SDK. Shell
exec is the highest-blast-radius capability; the developer opts in
explicitly.

## Authoring a custom capability

1. Subclass `SandboxCapability` with a `Literal[...]` discriminator
   `type` field.
2. Override `required_capability_types()` to declare hard
   dependencies on other capability types (returns a `set[str]`).
3. Override `tools()` to surface FunctionTools (validates session +
   user binding first).
4. Override `process_manifest()` to add workspace entries.
5. Override `async instructions(manifest)` to contribute a system-
   prompt fragment.
6. Override `clone()` only when your capability holds non-trivially-
   copyable state (asyncio.Lock / Event / Semaphore — fresh instances
   per clone).

See `src/troopai/adk/sandbox/capabilities/` for full implementations
and `examples/sandbox/memory_capability.py` for a runnable scenario.
