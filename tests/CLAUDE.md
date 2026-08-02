# Tests

pytest-only test suite for TroopAI ADK. See
`.claude/rules/testing.md` for the framework rule and
per-pattern guidance (agents, guardrails, tools, runner, LLM mocking).

## Test Structure

```
tests/
├── conftest.py         # Shared fixtures (auto-discovered by pytest)
├── unit/               # Unit tests (one directory per ADK module)
│   ├── agents/         # Agent class tests
│   ├── tools/          # Tool system tests
│   ├── run/            # Runner tests
│   ├── guardrails/     # Guardrail tests
│   ├── memory/         # Memory store / MemoryTool tests
│   ├── graphs/         # Graph primitive tests
│   ├── swarms/         # Swarm orchestration tests
│   ├── handoffs/       # Handoff tests
│   ├── llms/           # LLM abstraction tests
│   ├── verbose/        # Verbose renderer tests
│   └── evals/          # Eval suite tests
├── integration/        # Cross-module integration tests
│   ├── graphs/         # End-to-end graph run tests
│   └── ...
└── fixtures/           # Shared fixtures (imported via conftest.py)
```

## Running Tests

```bash
pytest                                        # All tests
pytest tests/unit/agents/                     # Specific directory
pytest tests/unit/agents/test_agent.py        # Specific file
pytest tests/unit/agents/test_agent.py::test_arun_returns_final_output  # Specific test
pytest -v                                     # Verbose
pytest -x                                     # Stop on first failure
pytest -m "not slow"                          # Skip slow tests
pytest -m integration                         # Only integration tests
pytest -n auto                                # Parallel (pytest-xdist)
pytest --cov=troopai.adk --cov-report=html     # With coverage
```

## Configuration

Pytest config and the test-stack pin list live in `pyproject.toml` —
`[tool.pytest.ini_options]` and `[project.optional-dependencies].test`.
That file is the single source of truth; do not restate its values
here.

## Markers (declared in pyproject)

| Marker | Purpose |
|---|---|
| `slow` | Deselect with `-m "not slow"` |
| `integration` | Multi-module integration tests |
| `e2e` | Full-pipeline smoke tests |
| `allow_call_model_methods` | Tests that call real LLM implementations |
| `serial` | Tests that must run serially |

## Environment Variables

```bash
export OPENAI_API_KEY="test-key"
export ANTHROPIC_API_KEY="test-key"
export TROOPAI_TEST_MODE="true"
```

## Coverage

Configured in `[tool.coverage]` in `pyproject.toml`. HTML reports land
in `.coverage/html/`. `fail_under = 50` is the permissive starting
floor — raise as the suite matures.

## See Also

- `.claude/rules/testing.md` — framework rule (pytest only,
  `unittest.mock` clarification) and per-category patterns for agents,
  guardrails, tools, runner, and LLM mocking.
