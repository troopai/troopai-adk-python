---
paths:
  - "tests/**/*.py"
---

# Testing — CRITICAL · pytest Only

- ALL tests are pytest. NEVER `unittest.TestCase` subclasses or
  `unittest.main()`. `unittest.mock` (`patch`, `AsyncMock`, `MagicMock`) is
  fine; `pytest-mock`'s `mocker` is equally acceptable.
- Bare `def test_*` + fixtures. Async: `async def test_*` with NO decorator
  (`asyncio_mode = "auto"`); add `@pytest.mark.asyncio` only when another
  marker needs it for ordering.
- Parametrize via `@pytest.mark.parametrize`, NOT `parameterized.expand`.
- Class grouping: bare `class TestFoo:` — never `unittest.TestCase`.
- Config in `[tool.pytest.ini_options]`; test deps in
  `[project.optional-dependencies].test`. NEVER duplicate.

| Category | Pattern | Location |
|---|---|---|
| Agent | Build agent in fixture; run via `Runner.arun()`; assert `result.final_output` | `tests/unit/{agents,run}/` |
| Guardrail | Assert `Input/OutputGuardrailTripwireTriggered` | `tests/unit/guardrails/` |
| Tool | Instantiate `FunctionTool`; call `execute()`; assert result | `tests/unit/tools/` |
| LLM mock | Patch `litellm.acompletion` with `AsyncMock` | `tests/unit/llms/` |

## Self-Check

1. Bare `class TestX:` / `def test_*` — not `unittest.TestCase`?
2. Async test has no decorator unless ordering needs it?
3. `@pytest.mark.parametrize`, not `parameterized`?
4. Agent test runs via `Runner.arun()` asserting `result.final_output`?
