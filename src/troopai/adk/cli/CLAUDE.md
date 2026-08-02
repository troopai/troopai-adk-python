# CLI Module

The `troopai` console script (`[project.scripts]` → `troopai.adk.cli:main`,
also `python -m troopai.adk.cli`). Click-based; one module per command.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Root click group `main`; registers every command |
| `__main__.py` | `python -m troopai.adk.cli` entry |
| `options.py` | Shared option decorators (`target_options`, `run_options`, `session_options`, `output_option`) |
| `errors.py` | `framework_errors` — maps ConfigParseError/ConfigResolutionError/FileNotFoundError → `click.UsageError` (exit 2, message verbatim, no traceback) |
| `loading.py` | Target seam: `resolve_target` (config path or `module:var`), `primary_executable` (topology dispatch: graph > swarm > entry), `reconcile_positionals`, `load_env_file`, `detect_config_kind` |
| `run.py` | `troopai run` + shared `build_run_config`, `open_session`, `echo_stream`, `target_app_name` |
| `chat.py` | `troopai chat` — CLI-owned REPL (NOT a `run_demo_loop` wrapper: the demo loop has no `session=`) |
| `validate.py` | `troopai validate` (`--resolve` assembles refs) |
| `schema.py` | `troopai schema agent\|node\|topology` |
| `sessions.py` | `troopai sessions list\|show\|delete` |
| `new.py` + `templates.py` | `troopai new` scaffolding (string.Template, `$name`; schema files written beside the config) |
| `serve.py` | `troopai serve` — REST + health by default via `serving.build_app`, A2A added with `--card`, run under uvicorn; behind the `serve` extra |

## Key decisions

| Decision | Rationale |
|---|---|
| click is a core dep | Zero transitive deps on POSIX; the entry point must work on a base install (first-touch UX: `pip install troopai-adk-python && troopai new`). The upstream mistake to avoid was making the *web/server stack* core, not click. |
| `click.echo` = product output; `logging` = diagnostics | A CLI's stdout is its contract (pipeable); the no-`print()` rule targets debug printing, and `click.echo` is neither. Never add `print()`. |
| Heavy framework imports live inside command bodies | Fast `--help`; the parent `troopai.adk` import is already paid, but command modules must not add more at import time. |
| Dynamic import only via `resolve_dotted_spec` | The config resolver is the framework's single sanctioned dotted-reference boundary; the CLI never calls importlib itself. |
| Flags that don't apply to a target kind raise `UsageError` | Never silently ignore (`--stream`/`--max-turns` on swarms, `--session-db` on graphs). |
| Cost-conservative flag defaults | Sessions, verbose, tracing, env-file are all off until flagged; `--model`/`--max-turns` default to `None` and are only forwarded when set, so the CLI never restates a framework default. |
| `reconcile_positionals` | `run [CONFIG] [PROMPT]` + `--agent` makes a lone positional ambiguous; with `--agent` it re-binds to the prompt. |
| uvicorn imported only inside `serve` | The framework deliberately doesn't own the ASGI runtime; the `serve` extra (`troopai-adk-python[server,a2a]` + uvicorn) gates it with a guiding message. `serve` routes through `serving.build_app`, never importing the app stack at module load. |
| AgentCard optional — required only for `--a2a` | `serve` exposes REST + health with no card; passing `--card` (or `--a2a`) publishes the A2A surface and parses the card via `google.protobuf.json_format.ParseDict` (camelCase, strict). The CLI never synthesizes a card. |
| Durable A2A task store via `--task-db` | Opens a `SQLiteTaskStore` and runs `recover_on_startup` (prior-process non-terminal tasks → FAILED) before binding; in-memory otherwise. |
| Session `app_name` = `target_app_name()` | Agent name / swarm entry name / graph id — deterministic, documented in `sessions --app-name` help. |

## Testing

`tests/unit/cli/` — CliRunner end-to-end, network-free via
`conftest.py`'s `stub_agent_dir` fixture (a `ScriptedLLM` agent + swarm
exposed as an importable module in `tmp_path`, loaded through the real
`--agent` path). `serve` tests monkeypatch `uvicorn.run` — no port is
ever bound.
