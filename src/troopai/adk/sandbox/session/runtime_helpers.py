"""Self-installing POSIX-shell helper scripts for sandbox sessions.

Some workspace operations cannot be done safely from the host
side because they depend on the *sandbox's* filesystem view —
symlink resolution, workspace-escape detection, content
fingerprinting. ``RuntimeHelperScript`` packages a small,
content-addressed shell script that the session installs once
into the sandbox (idempotently — the install command no-ops when
an identical script is already present) and then invokes by
absolute path.

Two helpers ship here:

* ``RESOLVE_WORKSPACE_PATH_HELPER`` — resolve a candidate path
  against the workspace root + extra path grants, following
  symlinks with a depth cap, and reject any path that escapes the
  permitted roots (the in-sandbox enforcement of
  ``WorkspacePathPolicy``).
* ``WORKSPACE_FINGERPRINT_HELPER`` — deterministically hash the
  workspace contents + manifest digest into a single fingerprint,
  used to decide whether a snapshot restore can be skipped.

The install path is content-addressed (``<name>-<sha256[:12]>``)
so a new script version installs to a new path and never races a
running invocation of the old one.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath
from typing import TYPE_CHECKING, Final

from troopai.adk.exceptions.exceptions import SandboxStartFailed

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession

logger = logging.getLogger(__name__)

__all__ = [
    "RESOLVE_WORKSPACE_PATH_HELPER",
    "WORKSPACE_FINGERPRINT_HELPER",
    "RuntimeHelperScript",
    "install_runtime_helpers",
]

_HELPER_INSTALL_ROOT: Final[PurePosixPath] = PurePosixPath("/tmp/troopai-adk-python/bin")
_INSTALL_MARKER: Final[str] = "INSTALL_RUNTIME_HELPER"

_RESOLVE_WORKSPACE_PATH_SCRIPT: Final[str] = """
#!/bin/sh
# RESOLVE_WORKSPACE_REALPATH
set -eu

root="$1"
candidate="$2"
for_write="$3"
shift 3
max_symlink_depth=64

case "$for_write" in
    0|1) ;;
    *)
        printf 'for_write must be 0 or 1: %s\\n' "$for_write" >&2
        exit 64
        ;;
esac

if [ $(( $# % 2 )) -ne 0 ]; then
    printf 'extra path grants must be root/read_only pairs\\n' >&2
    exit 64
fi

resolve_path() {
    path="$1"
    depth="${2:-0}"
    seen="${3:-}"
    if [ "$path" = "/" ]; then
        printf '/\\n'
        return 0
    fi

    if [ "$depth" -ge "$max_symlink_depth" ]; then
        printf 'symlink resolution depth exceeded: %s\\n' "$path" >&2
        exit 112
    fi

    if [ -d "$path" ]; then
        (
            cd "$path"
            pwd -P
        )
        return 0
    fi

    parent=${path%/*}
    base=${path##*/}
    if [ -z "$parent" ] || [ "$parent" = "$path" ]; then
        parent="/"
    fi

    resolved_parent=$(resolve_path "$parent" "$depth" "$seen")
    candidate_path="$resolved_parent/$base"
    if [ -L "$candidate_path" ]; then
        case ":$seen:" in
            *":$candidate_path:"*)
                printf 'symlink resolution depth exceeded: %s\\n' "$candidate_path" >&2
                exit 112
                ;;
        esac
        target=$(readlink "$candidate_path")
        next_depth=$((depth + 1))
        next_seen="${seen}:$candidate_path"
        case "$target" in
            /*) resolve_path "$target" "$next_depth" "$next_seen" ;;
            *) resolve_path "$resolved_parent/$target" "$next_depth" "$next_seen" ;;
        esac
        return 0
    fi

    printf '%s\\n' "$candidate_path"
}

# SECURITY INVARIANT: every resolve_path call here is a BARE
# assignment `var=$(resolve_path ...)`. dash / bash / busybox all
# propagate an inner `exit 111/112/113/114` out of a bare-assignment
# command substitution, so a workspace-escape / depth-exceeded
# rejection terminates the whole script. Do NOT refactor any of
# these into `if var=$(...)`, `var=$(...) && ...`, `|| ...`, `!`,
# or a non-terminal pipeline position — POSIX leaves set -e + cmd
# substitution partly unspecified and those suppressed forms WILL
# silently swallow the escape rejection (empirically confirmed).
resolved_candidate=$(resolve_path "$candidate" 0)
best_grant_root=""
best_grant_original=""
best_grant_read_only="0"
best_grant_len=0

# Predicate (returns 0/1) rather than exiting, so the caller can
# sequence it AFTER the read-only-grant deny check. This is a
# deliberate divergence from the upstream design, where an
# unconditional workspace-root allow ran first and silently
# bypassed a read-only grant covering a workspace subpath. Here a
# read-only grant DENIES a write even when the path is also under
# the writable workspace root — deny wins over allow.
candidate_under_root() {
    allowed_root="$1"
    resolved_root=$(resolve_path "$allowed_root" 0)
    case "$resolved_candidate" in
        "$resolved_root"|"$resolved_root"/*)
            return 0
            ;;
    esac
    return 1
}

reject_root_grant() {
    allowed_root="$1"
    resolved_root=$(resolve_path "$allowed_root" 0)
    if [ "$resolved_root" = "/" ]; then
        printf 'extra path grant must not resolve to filesystem root: %s\\n' "$allowed_root" >&2
        exit 113
    fi
}

consider_extra_grant() {
    allowed_root="$1"
    read_only="$2"
    case "$read_only" in
        0|1) ;;
        *)
            printf 'extra path grant read_only must be 0 or 1: %s\\n' "$read_only" >&2
            exit 64
            ;;
    esac

    reject_root_grant "$allowed_root"
    resolved_root=$(resolve_path "$allowed_root" 0)
    case "$resolved_candidate" in
        "$resolved_root"|"$resolved_root"/*)
            root_len=${#resolved_root}
            if [ "$root_len" -gt "$best_grant_len" ]; then
                best_grant_root="$resolved_root"
                best_grant_original="$allowed_root"
                best_grant_read_only="$read_only"
                best_grant_len="$root_len"
            fi
            ;;
    esac
}

while [ "$#" -gt 0 ]; do
    consider_extra_grant "$1" "$2"
    shift 2
done

# Deny wins: a read-only grant that covers the candidate rejects a
# write FIRST — even if the candidate is also under the writable
# workspace root. (Upstream allowed the workspace-root path
# unconditionally before consulting the grant, silently bypassing
# a read-only restriction on a workspace subpath.)
if [ -n "$best_grant_root" ] && [ "$for_write" = "1" ] && [ "$best_grant_read_only" = "1" ]; then
    printf 'read-only extra path grant: %s\\nresolved path: %s\\n' \
        "$best_grant_original" "$resolved_candidate" >&2
    exit 114
fi

if candidate_under_root "$root"; then
    printf '%s\\n' "$resolved_candidate"
    exit 0
fi

if [ -n "$best_grant_root" ]; then
    printf '%s\\n' "$resolved_candidate"
    exit 0
fi

printf 'workspace escape: %s\\n' "$resolved_candidate" >&2
exit 111
""".strip()

_WORKSPACE_FINGERPRINT_SCRIPT: Final[str] = """
#!/bin/sh
# WORKSPACE_FINGERPRINT
set -eu

if [ "$#" -lt 4 ]; then
    printf '%s\\n' \
        "usage: $0 <workspace-root> <tag> <output-path>" \
        " <manifest-digest> [exclude-relpath ...]" >&2
    exit 64
fi

workspace_root=$1
fingerprint_tag=$2
output_path=$3
manifest_digest=$4
shift 4

if [ ! -d "$workspace_root" ]; then
    printf 'workspace root not found: %s\\n' "$workspace_root" >&2
    exit 66
fi

hash_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 | awk '{print $1}'
        return
    fi
    if command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 | awk '{print $NF}'
        return
    fi
    printf 'workspace fingerprint helper requires sha256sum, shasum, or openssl\\n' >&2
    exit 127
}

# Validate every exclude relpath (reject traversal / absolute) BEFORE it
# reaches tar. With the set-- argv approach below there is NO shell
# re-parse, so single quotes in paths are now safe and no longer
# rejected — but a path-traversal exclude is still a real escape risk.
for rel in "$@"; do
    case "$rel" in
        ""|"."|"/"|*"/.."|*"/../"*|".."|../*|*/../*|/*)
            printf 'exclude relpath must be a concrete relative path: %s\\n' "$rel" >&2
            exit 65
            ;;
    esac
done

# Build the tar argv directly in "$@" — no `sh -lc "$cmd"` eval, so no
# path/relpath value can be word-split or interpreted by a shell. The
# loop shifts each original relpath off the head and appends the two
# --exclude forms to the tail; after $orig_count iterations only the
# --exclude pairs remain, then we append the workspace trailer.
orig_count=$#
while [ "$orig_count" -gt 0 ]; do
    rel=$1
    shift
    set -- "$@" "--exclude=$rel" "--exclude=./$rel"
    orig_count=$((orig_count - 1))
done
set -- "$@" -C "$workspace_root" -cf - .

tmp_output="$output_path.tmp.$$"
tar_status_file="$output_path.tarstatus.$$"
cleanup() {
    rm -f -- "$tmp_output" "$tar_status_file"
}
trap cleanup EXIT INT TERM

# POSIX sh has no `pipefail` and `set -e` only inspects the LAST stage
# of a pipeline, so a failed/partial `tar` would otherwise be silently
# hashed into a structurally-valid (empty-input) fingerprint and make
# the restore-skip decision skip a required restore. Capture tar's
# exit status explicitly via a status file and refuse to emit a
# fingerprint when the archive did not complete cleanly.
workspace_fingerprint=$(
    { tar "$@"; printf '%s' "$?" > "$tar_status_file"; } | hash_stdin
)
tar_status=$(cat "$tar_status_file" 2>/dev/null || printf '1')
if [ "$tar_status" -ne 0 ]; then
    printf 'workspace archive failed (tar exit %s); refusing to emit a fingerprint\\n' \
        "$tar_status" >&2
    exit 74
fi

fingerprint=$(
    printf '%s\\n%s\\n' "$workspace_fingerprint" "$manifest_digest" | hash_stdin
)

payload=$(printf '{"fingerprint":"%s","tag":"%s"}\\n' "$fingerprint" "$fingerprint_tag")
mkdir -p -- "$(dirname -- "$output_path")"
printf '%s' "$payload" > "$tmp_output"
mv -f -- "$tmp_output" "$output_path"
trap - EXIT INT TERM
rm -f -- "$tar_status_file"
printf '%s' "$payload"
""".strip()


@dataclass(frozen=True)
class RuntimeHelperScript:
    """A content-addressed, idempotently-installable sandbox shell helper.

    Attributes:
        name: Human-readable helper name; also the install-path stem.
        content: The full POSIX-shell script body.
        install_path: Absolute in-sandbox path the script installs
            to. Content-addressed (``<name>-<sha256[:12]>``) so a
            changed script lands at a fresh path and never races a
            running invocation of the prior version.
        install_marker: A grep-able marker line prepended to the
            install command so an operator can identify the
            installer in shell history / audit logs.
    """

    name: str
    """Human-readable helper name; also the install-path stem."""

    content: str
    """The full POSIX-shell script body."""

    install_path: PurePath
    """Absolute in-sandbox path the script installs to (content-addressed)."""

    install_marker: str = _INSTALL_MARKER
    """Grep-able marker line prepended to the install command."""

    @classmethod
    def from_content(cls, *, name: str, content: str) -> RuntimeHelperScript:
        """Build a helper whose install path is content-addressed by sha256."""
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        install_path = _HELPER_INSTALL_ROOT / f"{name}-{digest}"
        return cls(name=name, content=content, install_path=install_path)

    def install_command(self) -> tuple[str, ...]:
        """Return the argv that idempotently installs this script in-sandbox.

        The command writes the script to a temp file, makes it
        executable, and atomically renames it into ``install_path``.
        If an identical executable already exists at the destination
        (``cmp -s``), it is a no-op. A ``trap`` cleans up the temp
        file on any early exit.
        """
        tmp_template = f"{self.install_path}.tmp.$$"
        heredoc = f"TROOPAI_AGENTS_HELPER_{self.install_path.name.upper().replace('-', '_')}"
        return (
            "sh",
            "-c",
            f"""
# {self.install_marker}
set -eu

dest="$1"
tmp="{tmp_template}"

mkdir -p -- "$(dirname -- "$dest")"

cleanup() {{
    rm -f -- "$tmp"
}}
trap cleanup EXIT INT TERM

cat > "$tmp" <<'{heredoc}'
{self.content}
{heredoc}
chmod 0555 "$tmp"
if [ -d "$dest" ]; then
    rm -rf -- "$dest"
fi
if [ -x "$dest" ] && command -v cmp >/dev/null 2>&1 && cmp -s "$dest" "$tmp"; then
    rm -f -- "$tmp"
    trap - EXIT INT TERM
    exit 0
fi
# No `rm -f -- "$dest"` before the mv: the `[ -d "$dest" ]` branch
# above already removes a directory at the destination, and
# `mv -f` performs an atomic rename(2) that replaces an existing
# regular file in a single operation. A preceding `rm` would open
# a window where $dest does not exist and a concurrent reader hits
# ENOENT.
mv -f -- "$tmp" "$dest"
trap - EXIT INT TERM
""".strip(),
            "sh",
            str(self.install_path),
        )

    def present_command(self) -> tuple[str, ...]:
        """Return the argv that tests whether the helper is already installed."""
        return ("test", "-x", str(self.install_path))


RESOLVE_WORKSPACE_PATH_HELPER: Final[RuntimeHelperScript] = RuntimeHelperScript.from_content(
    name="resolve-workspace-path",
    content=_RESOLVE_WORKSPACE_PATH_SCRIPT,
)

WORKSPACE_FINGERPRINT_HELPER: Final[RuntimeHelperScript] = RuntimeHelperScript.from_content(
    name="workspace-fingerprint",
    content=_WORKSPACE_FINGERPRINT_SCRIPT,
)

_DEFAULT_RUNTIME_HELPERS: Final[tuple[RuntimeHelperScript, ...]] = (
    RESOLVE_WORKSPACE_PATH_HELPER,
    WORKSPACE_FINGERPRINT_HELPER,
)
"""The helper set every real backend session installs at start."""


async def install_runtime_helpers(
    session: BaseSandboxSession,
    *,
    backend_id: str,
    helpers: Sequence[RuntimeHelperScript] = _DEFAULT_RUNTIME_HELPERS,
) -> None:
    """Idempotently install ``helpers`` into a started sandbox session.

    Called from each real backend's ``start()`` once the working
    directory exists and ``run`` is usable. For each helper:

    * ``present_command()`` (``test -x <content-addressed-path>``) is
      run first. Because ``install_path`` is content-addressed
      (``<name>-<sha256[:12]>``), an executable at that exact path is
      necessarily byte-identical to ``content`` — so a zero exit is a
      sound "already installed" signal and the heavier write is
      skipped (the warm-start fast path).
    * Otherwise ``install_command()`` writes + atomically renames the
      script in. A non-zero install exit raises ``SandboxStartFailed``
      attributed to ``backend_id`` — these helpers enforce
      workspace-escape rejection and gate snapshot-restore-skip, so a
      failed install MUST surface as a start failure, never a latent
      silent breakage discovered later by a dependent capability.

    Args:
        session: A started session whose ``run`` executes argv.
        backend_id: Backend identity stamped on a raised
            ``SandboxStartFailed`` (``"docker"`` / ``"k8s_pod"``).
        helpers: Helper scripts to install; defaults to the two
            workspace helpers every backend needs.

    Raises:
        SandboxStartFailed: A helper's install command exited
            non-zero (stderr included in the reason).
    """
    for helper in helpers:
        present = await session.run(*helper.present_command(), shell=False)
        if present.exit_code == 0:
            logger.debug(
                "runtime helper %r already present at %s — skipping install",
                helper.name,
                helper.install_path,
            )
            continue
        logger.info(
            "installing runtime helper %r at %s",
            helper.name,
            helper.install_path,
        )
        result = await session.run(*helper.install_command(), shell=False)
        if result.exit_code != 0:
            stderr_text = result.stderr.decode("utf-8", "replace").strip()
            raise SandboxStartFailed(
                backend_id,
                f"runtime helper {helper.name!r} install failed (exit {result.exit_code}): {stderr_text}",
            )
