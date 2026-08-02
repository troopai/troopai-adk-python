"""Sandbox utility layer.

Foundational helpers used across the sandbox subsystem:
checksum streaming, dict deep-merge, GitHub clone, iterator→IO
adapter, ls-la parsing, async retry, safe tar extraction, and
token-budget text truncation.

All helpers are framework-internal — they MUST NOT import any
provider SDK and MUST NOT depend on a live sandbox session.
"""

from __future__ import annotations

from troopai.adk.sandbox.utils.checksums import sha256_file, sha256_io
from troopai.adk.sandbox.utils.deep_merge import deep_merge
from troopai.adk.sandbox.utils.github import clone_repo, ensure_git_available
from troopai.adk.sandbox.utils.iterator_io import IteratorIO
from troopai.adk.sandbox.utils.parse_utils import EntryKind, LsLaEntry, parse_ls_la
from troopai.adk.sandbox.utils.retry import (
    DEFAULT_TRANSIENT_RETRY_BACKOFF,
    DEFAULT_TRANSIENT_RETRY_INTERVAL_S,
    DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPT,
    TRANSIENT_HTTP_STATUS_CODES,
    BackoffStrategy,
    exception_chain_contains_type,
    exception_chain_has_status_code,
    iter_exception_chain,
    retry_async,
)
from troopai.adk.sandbox.utils.tar_utils import (
    UnsafeTarMemberError,
    safe_extract_tarfile,
    safe_tar_member_rel_path,
    should_skip_tar_member,
    strip_tar_member_prefix,
    validate_tar_bytes,
    validate_tarfile,
)
from troopai.adk.sandbox.utils.token_truncation import (
    APPROX_BYTES_PER_TOKEN,
    TruncationMode,
    TruncationPolicy,
    approx_bytes_for_tokens,
    approx_token_count,
    approx_tokens_from_byte_count,
    assemble_truncated_output,
    format_truncation_marker,
    formatted_truncate_text,
    formatted_truncate_text_with_token_count,
    removed_units_for_source,
    split_budget,
    split_string,
    truncate_text,
    truncate_with_byte_estimate,
    truncate_with_token_budget,
)

__all__ = [
    "APPROX_BYTES_PER_TOKEN",
    "DEFAULT_TRANSIENT_RETRY_BACKOFF",
    "DEFAULT_TRANSIENT_RETRY_INTERVAL_S",
    "DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPT",
    "TRANSIENT_HTTP_STATUS_CODES",
    "BackoffStrategy",
    "EntryKind",
    "IteratorIO",
    "LsLaEntry",
    "TruncationMode",
    "TruncationPolicy",
    "UnsafeTarMemberError",
    "approx_bytes_for_tokens",
    "approx_token_count",
    "approx_tokens_from_byte_count",
    "assemble_truncated_output",
    "clone_repo",
    "deep_merge",
    "ensure_git_available",
    "exception_chain_contains_type",
    "exception_chain_has_status_code",
    "format_truncation_marker",
    "formatted_truncate_text",
    "formatted_truncate_text_with_token_count",
    "iter_exception_chain",
    "parse_ls_la",
    "removed_units_for_source",
    "retry_async",
    "safe_extract_tarfile",
    "safe_tar_member_rel_path",
    "sha256_file",
    "sha256_io",
    "should_skip_tar_member",
    "split_budget",
    "split_string",
    "strip_tar_member_prefix",
    "truncate_text",
    "truncate_with_byte_estimate",
    "truncate_with_token_budget",
    "validate_tar_bytes",
    "validate_tarfile",
]
