# Session Module

Persistence layer for conversation history with multi-tenant scoping and session state.

## Key Files

- `session.py` — `Session` ABC with `get`, `add`, `pop_last`, `clear`, `save_state`, `close`
- `session_settings.py` — `SessionSettings` Pydantic model
- `state.py` — `State` class: dict-like with delta tracking, `temp:` prefix, `app:` prefix
- `sqlite_session.py` — `SQLiteSession`: bound session implementation (shares manager's DB)
- `sqlite_multi_sessions.py` — `SQLiteMultiSessions`: manager (create/get/list/delete/count)

## Architecture

**Manager/Session separation** (inspired by Google ADK):
- `SQLiteMultiSessions(path, app_name)` — manages the collection
- `SQLiteSession` — implements Session ABC, shares manager's DB connection
- `State` — dict-like session state with delta tracking and temp prefix

**Multi-tenant scoping**:
- `app_name` is on the manager (one manager = one app)
- `user_id` is per-operation (create, get, list, delete)
- Composite PK: `(app_name, user_id, session_id)`

**State tiers**:
- Session-scoped (no prefix) — isolated to one session
- App-scoped (`app:` prefix) — shared across all sessions for the app
- Temp (`temp:` prefix) — in-memory only, not persisted

## Runner Integration

Pass `session=` to `Runner.run()` or `Runner.arun()`. Runner calls `session.get()`, `session.add()`, reads `session.settings` and `session.id`.

See `docs/session/session.md` for usage. See `examples/` for examples.
