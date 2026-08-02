# Anthropic Provider Examples

Native `anthropic` SDK examples using `AnthropicLLM` + `AnthropicConfig`.

## Files

- `anthropic_example.py` — Seven snippets in one file:
  1. **basic** — non-streaming Messages call + usage logging.
  2. **streaming** — `stream=True` with token-by-token rendering.
  3. **tools** — function-tool tool-use loop (multi-turn).
  4. **thinking** — `AnthropicConfig.thinking` extended reasoning.
  5. **structured_output** — synthetic-tool path for `output_schema`.
  6. **caching** — `AnthropicConfig.auto_cache_control` (real
     `cache_creation` then `cache_read` on the second turn).
  7. **retry_policy** — `LLMConfig.retry_policy` with exponential
     backoff.

## Running

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/llm_providers/anthropic/anthropic_example.py
```

Each snippet runs independently inside `main()`; failures are caught
and logged so one broken section does not block the others.
