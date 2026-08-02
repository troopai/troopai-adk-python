"""Native OpenAI LLM implementations.

Two first-class ``LLM`` subclasses call the ``openai`` SDK directly —
no litellm indirection.

- :class:`OpenAIResponsesLLM` — backed by ``client.responses.create()``.
- :class:`OpenAIChatCompletionsLLM` — backed by ``client.chat.completions.create()``.

Both classes consume provider-agnostic Layer 1 inputs
(``list[LLMInputContentItem]``) and return framework-owned
``LLMResponse`` values. Provider-native capabilities (web search,
file search, computer use, image generation, code interpreter, hosted
MCP servers, etc.) are NOT wrapped in framework tool classes — pass
the raw provider JSON through ``LLMConfig.extra_body`` /
``LLMConfig.extra_args``.

Configuration for each class is carried by a dedicated
``@dataclass`` subclass of :class:`LLMConfig`:

- :class:`OpenAIResponsesConfig`
- :class:`OpenAIChatCompletionsConfig`

The configs type OpenAI-specific fields verbatim against
``openai.types.*`` — never via framework-owned re-definitions.

Two voice models live here as well, implementing the framework-owned
speech ABCs (``voice/``) against the same SDK:

- :class:`OpenAISTTModel` — buffered transcription plus a realtime
  transcription websocket session.
- :class:`OpenAITTSModel` — streaming PCM speech synthesis.
"""

from __future__ import annotations

from troopai.adk.llms.openai.openai_chatcompletions_config import OpenAIChatCompletionsConfig
from troopai.adk.llms.openai.openai_chatcompletions_model import OpenAIChatCompletionsLLM
from troopai.adk.llms.openai.openai_responses_config import OpenAIResponsesConfig
from troopai.adk.llms.openai.openai_responses_model import OpenAIResponsesLLM
from troopai.adk.llms.openai.openai_stt import OpenAISTTModel
from troopai.adk.llms.openai.openai_tts import OpenAITTSModel

__all__ = [
    "OpenAIChatCompletionsConfig",
    "OpenAIChatCompletionsLLM",
    "OpenAIResponsesConfig",
    "OpenAIResponsesLLM",
    "OpenAISTTModel",
    "OpenAITTSModel",
]
