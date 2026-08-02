# Session Usage

## Session Protocol

```python
class Session(ABC):
    id: str
    app_name: str
    user_id: str
    state: State
    settings: Optional[SessionSettings]

    async def get(self, limit=None) -> list[LLMInputContentItem]: ...
    async def add(self, inputs: list[LLMInputContentItem]) -> None: ...
    async def pop_last(self) -> Optional[LLMInputContentItem]: ...
    async def clear(self) -> None: ...
    async def save_state(self) -> None: ...
    async def close(self) -> None: ...
```

## SQLiteMultiSessions (Manager)

```python
from troopai.adk.session import SQLiteMultiSessions, SessionSettings

# Create a manager (one per app)
sessions = SQLiteMultiSessions(path="sessions.db", app_name="myapp")

# Create a session
session = await sessions.create("conv-001", user_id="user-1")

# Get existing (returns None if missing)
session = await sessions.get("conv-001", user_id="user-1")

# Get or create (most common)
session = await sessions.get_or_create("conv-001", user_id="user-1")

# Use with Runner
result = await Runner.arun(agent, "Hello!", session=session)

# Collection management
all_sessions = await sessions.list(user_id="user-1")
await sessions.delete("conv-old", user_id="user-1")
count = await sessions.count()
await sessions.close()
```

## Session State

```python
# Read/write state (dict-like)
session.state["preference"] = "dark_mode"
theme = session.state.get("preference")

# App-scoped state (shared across sessions)
session.state["app:config"] = {"version": 2}

# Temp state (in-memory only, not persisted)
session.state["temp:scratch"] = "working..."

# Persist state changes
await session.save_state()
```

## Default Settings

```python
# Manager-level defaults apply to all sessions
sessions = SQLiteMultiSessions(
    path="sessions.db",
    settings=SessionSettings(limit=50),
)

# Per-session override
session = await sessions.create(
    "conv-001",
    settings=SessionSettings(limit=100),
)
```
