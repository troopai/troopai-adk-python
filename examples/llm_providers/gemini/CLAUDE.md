# Gemini Provider Examples

Native `google-genai` SDK examples using `GeminiLLM` + `GeminiConfig`.

## Files

- `gemini_example.py` — Six snippets in one file:
  1. **basic** — non-streaming generate_content + usage logging.
  2. **streaming** — `stream=True` with token-by-token rendering.
  3. **tools** — function-tool tool-use loop.
  4. **thinking** — `ThinkingConfig` extended reasoning.
  5. **structured_output** — native `response_schema` for `output_schema`.
  6. **hosted_search** — typed `WebSearchTool` translating to
     Gemini's `google_search` hosted tool.

## Running

```bash
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
python examples/llm_providers/gemini/gemini_example.py
```

For Vertex AI mode:

```bash
export GOOGLE_CLOUD_PROJECT=...
export GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login
# Then in code:
#   GeminiLLM(model=..., vertexai=True)
```
