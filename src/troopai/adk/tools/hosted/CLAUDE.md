# Hosted Tools

Cross-provider hosted-tool classes for capabilities the LLM provider
executes server-side (web search, code execution, file search, image
generation, URL context). The framework forwards typed config to each
provider's wire format via the matching converter's `isinstance`
dispatch.

## Files

- `hosted_tool.py` — `HostedTool` abstract dataclass base. Defines
  `SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]]` for converter
  error messages.
- `exceptions.py` — `UnsupportedHostedToolError` raised by a
  provider's converter when the variant is not supported on that
  provider.
- `web_search_tool.py` — `WebSearchTool` (Anthropic + OpenAI Responses + Gemini)
- `code_execution_tool.py` — `CodeExecutionTool` (OpenAI Responses + Gemini)
- `file_search_tool.py` — `FileSearchTool` (OpenAI Responses only)
- `image_generation_tool.py` — `ImageGenerationTool` (OpenAI Responses only)
- `url_context_tool.py` — `URLContextTool` (Gemini only)

## Authoring Contract

Every concrete subclass MUST:

1. Be a `@dataclass(kw_only=True)`.
2. Declare `SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]]` listing
   every provider that ships the capability natively.
3. Document a "Provider matrix" section in the class docstring.
4. Tag every per-provider attribute with `**<Provider> only.**` or
   `**<Provider> + <Provider>.**` when it applies to a strict subset
   of the supported providers.
5. Keep the class name concept-only (`WebSearchTool`, never
   `AnthropicWebSearchTool`).

## Converter Contract

Every provider's `convert_tools` method MUST handle every
`HostedTool` subclass — translate the variants it supports,
raise `UnsupportedHostedToolError(tool, provider,
supported_providers=tool.SUPPORTED_PROVIDERS)` on the rest. Silent
drops are forbidden. Per-provider attribute filtering (silently
ignoring knobs the active provider doesn't honour) is acceptable
when paired with a `logger.debug` line.
