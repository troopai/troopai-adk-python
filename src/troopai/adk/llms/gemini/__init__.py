"""Google Gemini native LLM implementation.

Calls the Gemini API directly via the ``google-genai`` SDK
— no litellm indirection. Supports both the public Gemini Developer
API (api_key auth) and Vertex AI (project + location + credentials)
in a single :class:`GeminiLLM` class via the SDK's built-in dispatch.

Provider-hosted capabilities (Google Search grounding, code execution,
URL context) are wired via the typed framework hosted-tool classes
(``WebSearchTool``, ``CodeExecutionTool``, ``URLContextTool``); each
provider's converter recognises them via ``isinstance``. Esoteric or
beta shapes can still be passed via ``LLMConfig.extra_body`` if
needed.

Usage::

    from troopai.adk.llms.gemini import GeminiLLM, GeminiConfig
    from troopai.adk.tools import function_tool


    @function_tool(name="lookup", description="Look up a record")
    def lookup(record_id: str) -> str:
        return f"Record {record_id}"


    agent = Agent(
        llm=GeminiLLM(model="gemini-2.5-flash"),
        tools=[lookup],
        llm_config=GeminiConfig(
            thinking_config={"thinking_budget": 2048, "include_thoughts": True},
        ),
    )
"""

from troopai.adk.llms.gemini.gemini_config import GeminiConfig
from troopai.adk.llms.gemini.gemini_model import GeminiLLM

__all__ = ["GeminiConfig", "GeminiLLM"]
