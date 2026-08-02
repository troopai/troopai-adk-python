# Context Management Usage

## Configuration

```python
from troopai.adk.context import CacheStrategy, ContextManagementConfig, CompactionConfig
from troopai.adk.run import RunConfig

config = RunConfig(
    context_management=ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            trigger_tokens=100_000,
        ),
        cache_strategy=CacheStrategy.STABLE,  # Preserve prompt cache
    )
)
result = await Runner.arun(agent, "Hello!", run_config=config)
```
