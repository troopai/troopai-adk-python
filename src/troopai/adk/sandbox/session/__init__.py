"""Sandbox session orchestration helpers.

This package groups infrastructure that supports
``BaseSandboxSession`` without changing the public ABC:

* Audit events (``events.py``) — typed start/finish records every
  backend operation emits.
* Write-payload coercion (``workspace_payloads.py``) — normalize
  arbitrary ``IOBase`` streams into a bounded binary read interface.
* Tar exclusion helpers (``tar_workspace.py``) — build ``--exclude``
  arg lists for shell-driven snapshot capture.
* PTY constants + LRU helpers (``pty_types.py``).
* JSON-line serialization for events (``utils.py``).
* Operation-name + error-code identifiers (``op_codes.py``).
"""

from __future__ import annotations

from troopai.adk.sandbox.session.archive_extraction import (
    ArchiveResourceLimitError,
    ArchiveStreamIntegrityError,
    SandboxArchiveLimits,
    UnsafeTarMemberError,
    UnsafeZipMemberError,
    safe_zip_member_rel_path,
    validate_tar_archive_for_extraction,
    validate_zipfile,
    zipfile_compatible_stream,
)
from troopai.adk.sandbox.session.concurrency import gather_in_order
from troopai.adk.sandbox.session.dependencies import (
    Dependencies,
    DependenciesBindingError,
    DependenciesError,
    DependenciesMissingDependencyError,
    FactoryFn,
)
from troopai.adk.sandbox.session.events import (
    EventPayloadPolicy,
    EventPhase,
    SandboxSessionEvent,
    SandboxSessionEventBase,
    SandboxSessionFinishEvent,
    SandboxSessionStartEvent,
    validate_sandbox_session_event,
)
from troopai.adk.sandbox.session.manager import Instrumentation
from troopai.adk.sandbox.session.op_codes import ErrorCode, OpName
from troopai.adk.sandbox.session.pty_types import (
    PTY_EMPTY_YIELD_TIME_MS_MIN,
    PTY_PROCESS_ID_MAX_EXCLUSIVE,
    PTY_PROCESS_ID_MIN,
    PTY_PROCESSES_MAX,
    PTY_PROCESSES_PROTECTED_RECENT,
    PTY_PROCESSES_WARNING,
    PTY_YIELD_TIME_MS_MAX,
    PTY_YIELD_TIME_MS_MIN,
    PtyExecUpdate,
    allocate_pty_process_id,
    clamp_pty_yield_time_ms,
    process_id_to_prune_from_meta,
    resolve_pty_write_yield_time_ms,
    truncate_text_by_tokens,
)
from troopai.adk.sandbox.session.runtime_helpers import (
    RESOLVE_WORKSPACE_PATH_HELPER,
    WORKSPACE_FINGERPRINT_HELPER,
    RuntimeHelperScript,
    install_runtime_helpers,
)
from troopai.adk.sandbox.session.sinks import (
    NEVER_DOWNGRADE_EXC,
    CallbackSink,
    ChainedSink,
    DeliveryMode,
    EventSink,
    HttpPostSink,
    JsonlOutboxSink,
    OnErrorPolicy,
)
from troopai.adk.sandbox.session.tar_workspace import shell_tar_exclude_args
from troopai.adk.sandbox.session.utils import event_to_json_line, safe_decode_with_max_chars
from troopai.adk.sandbox.session.workspace_payloads import (
    WritePayload,
    coerce_write_payload,
)

__all__ = [
    "NEVER_DOWNGRADE_EXC",
    "PTY_EMPTY_YIELD_TIME_MS_MIN",
    "PTY_PROCESSES_MAX",
    "PTY_PROCESSES_PROTECTED_RECENT",
    "PTY_PROCESSES_WARNING",
    "PTY_PROCESS_ID_MAX_EXCLUSIVE",
    "PTY_PROCESS_ID_MIN",
    "PTY_YIELD_TIME_MS_MAX",
    "PTY_YIELD_TIME_MS_MIN",
    "RESOLVE_WORKSPACE_PATH_HELPER",
    "WORKSPACE_FINGERPRINT_HELPER",
    "ArchiveResourceLimitError",
    "ArchiveStreamIntegrityError",
    "CallbackSink",
    "ChainedSink",
    "DeliveryMode",
    "Dependencies",
    "DependenciesBindingError",
    "DependenciesError",
    "DependenciesMissingDependencyError",
    "ErrorCode",
    "EventPayloadPolicy",
    "EventPhase",
    "EventSink",
    "FactoryFn",
    "HttpPostSink",
    "Instrumentation",
    "JsonlOutboxSink",
    "OnErrorPolicy",
    "OpName",
    "PtyExecUpdate",
    "RuntimeHelperScript",
    "SandboxArchiveLimits",
    "SandboxSessionEvent",
    "SandboxSessionEventBase",
    "SandboxSessionFinishEvent",
    "SandboxSessionStartEvent",
    "UnsafeTarMemberError",
    "UnsafeZipMemberError",
    "WritePayload",
    "allocate_pty_process_id",
    "clamp_pty_yield_time_ms",
    "coerce_write_payload",
    "event_to_json_line",
    "gather_in_order",
    "install_runtime_helpers",
    "process_id_to_prune_from_meta",
    "resolve_pty_write_yield_time_ms",
    "safe_decode_with_max_chars",
    "safe_zip_member_rel_path",
    "shell_tar_exclude_args",
    "truncate_text_by_tokens",
    "validate_sandbox_session_event",
    "validate_tar_archive_for_extraction",
    "validate_zipfile",
    "zipfile_compatible_stream",
]
