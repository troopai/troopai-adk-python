# Types Usage

## Import Examples

### Layer 1: Provider-Agnostic Items

**Input types** (TypedDict, sent TO the LLM):

```python
from troopai.adk.types.input import LLMInputContentItem, LLMInputText, LLMInputMessage
```

**Output types** (BaseModel, received FROM the LLM):

```python
from troopai.adk.types.output import (
    LLMOutputContentItem, LLMOutputMessage, LLMOutputFunctionToolCall,
    FunctionToolCallResult,
)
```

### Layer 2: Chat Completions Wire Types

```python
from troopai.adk.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolCall,
    ChatCompletionToolChoiceOptionParam,
)
```

### Layer 3: Items

```python
from troopai.adk.types.items import MessageOutputItem, ToolCallItem, RunItem, ItemHelpers
```

### Tool Types

```python
from troopai.adk.types.tools.tool_types import (
    FunctionToolDefinition, BuiltinToolDefinition, ToolDefinition,
)
```

### Response Types

```python
from troopai.adk.types.responses import LLMResponse, LLMStreamEvent

response.content       # str | None
response.tool_calls    # list[ChatCompletionToolCall]
response.usage         # LLMUsage | None
response.output        # Validated structured output | None
response.refusal       # str | None
```

### Token Types

```python
from troopai.adk.types.tokens import InputTokensDetails, OutputTokensDetails

details = InputTokensDetails(
    cached_tokens=1000,
    cache_creation_input_tokens=500,
)
```

### Prompt Caching Types

```python
from troopai.adk.types.caching import (
    AnthropicPromptCaching, GeminiPromptCaching, OpenAIPromptCaching,
    CacheInjectionPoint, PromptCaching,
)
```

### Common Types

**Reasoning:**

```python
from troopai.adk.types.common import Reasoning, ThinkingBlock, RedactedThinkingBlock
```

### Tool Configuration Types

```python
from troopai.adk.types.tools import ToolChoice, ToolExecutionMode
```
