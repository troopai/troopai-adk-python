# Audit Module

Append-only tool-call audit logging with pluggable, privacy-preserving sinks.

## Files

| File | Purpose |
|---|---|
| `event.py` | `AuditEvent` dataclass + `hash_payload` (sha256 of canonical JSON; never stores raw payloads) |
| `sink.py` | `AuditSink` Protocol (`@runtime_checkable`) + `InMemoryAuditSink` + `JsonlFileAuditSink` |
| `sinks/s3.py` | `S3AuditSink` — one object per event; extra-gated (boto3) |
| `sinks/postgres.py` | `PostgresAuditSink` — INSERT into append-only `audit_events`; extra-gated (psycopg) |

## Architecture Decisions

| Decision | What | Why |
|----------|------|-----|
| **Hashes, not raw payloads** | `args_hash` / `result_hash` are sha256 hex | Audit must not become a PII sink; hashes give tamper-evidence + correlation without storing sensitive data |
| **Inline emit, not hooks** | Emitted from the tool executor, not `on_tool_start`/`on_tool_end` | Lifecycle hooks never fire for denied calls; a governance audit must capture denials |
| **Best-effort by default** | Sink failure logs a warning and continues; `audit_strict` re-raises | A logging backend outage should not take down production runs; compliance deployments opt into fail-closed |
| **Protocol mirrors cost ledger** | `AuditSink` is `@runtime_checkable`, async `record` | Same pluggable-backend pattern as the cost ledger; custom sinks (e.g. Kafka) drop in without inheritance |
| **Extras-gated backends** | S3/Postgres raise `ImportError` with a pip-install hint if their SDK is absent | No optional dep forced on consumers who don't need it |
| **Shared local-tool scope** | Emits cover `FunctionTool` calls and framework-executed built-ins on main + HITL paths | Executable built-ins adapt to `FunctionTool`, so local execution has one governance and audit path |

## Pointers

- Emit + gate logic: `run/governance.py` (`emit_audit`); enforcement fields on `run/config.py` (`audit_sink`, `audit_strict`)
- Usage guide: `docs/governance/multi-tenant.md`
- Examples: `examples/governance/audit_logging.py`
- Extras: `pyproject.toml` (`audit-s3`, `audit-postgres`)
