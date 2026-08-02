# Manifest Materialization

A `Manifest` is a declarative workspace contract: it says which files
and directories MUST exist before the agent runs. Materialization is
the step that turns that contract into real on-disk state.
`materialize_manifest` is the single entry point a backend calls from
its `apply_manifest`; it walks `Manifest.iter_entries()` and dispatches
each entry to a concrete materializer. This is distinct from manifest
*processing* (`process_manifest_through_capabilities` in the run
lifecycle), where capabilities may transform the manifest first —
processing decides *what* the manifest is, materialization writes it.

Entries are Layer-1 behaviorless data (they carry no `apply()` method),
so dispatch is external and keyed on the concrete entry type via
`isinstance` over the closed entry union.

## Entry types

`materialize_entry(session, key, entry, *, base_dir, grants)` dispatches
one entry and returns a `MaterializedFile`:

| Entry | Discriminator | What the materializer does |
|---|---|---|
| `File` | `file` | Writes the inline `content` bytes verbatim at `key` |
| `Dir` | `dir` | Creates the directory (children flatten into their own entries) |
| `LocalFile` | `local_file` | Copies a host file into the workspace at `key` |
| `LocalDir` | `local_dir` | Recursively copies a host directory tree |
| `GitRepo` | `git_repo` | Clones a repository (the clone runs inside the sandbox image) |

A `Mount` is NOT file-materialized — the backend attaches it natively
at create time; the materializer records its key in
`MaterializationResult.skipped_mounts`. An entry whose concrete type
has no dispatch arm raises `UnsupportedManifestEntryError` — a
registered entry type without a materializer is a framework gap that
surfaces loudly, never a silently missing workspace file.

## The apply_manifest path

`BaseSandboxSession.apply_manifest(*, only_ephemeral=False)` is the
abstract surface. The local, Docker, and Kubernetes backends implement
it by delegating to `materialize_manifest` (they hold the manifest
threaded in at `create`); `RemoteVMSandboxSession` instead POSTs the
manifest to the hosted provider, which materializes it remotely.

The run lifecycle invokes `apply_manifest` once, only when the runner
owns a freshly-created session — after the backend session starts and
before the agent loop — so the declared workspace exists when the
agent's first turn runs. It is skipped in two cases: a caller-injected
session (the caller supplied `config.session` and owns its workspace),
and a resumed session (`config.session_state`; the existing workspace
is preserved and the manifest is fresh-session-only). A materialization
failure propagates loudly through the lifecycle teardown; it is NOT
wrapped as a session-start failure (a bad workspace is distinct from a
backend that would not start).

## Concurrency and ordering

Entries materialize in declaration order, fanned out up to
`max_entry_concurrency` (default `DEFAULT_MAX_ENTRY_CONCURRENCY = 4`)
through the order-preserving `gather_in_order` helper. The bound is
cost-conservative: a large manifest cannot saturate the event loop or a
backend exec channel. Raise it explicitly via
`materialize_manifest(max_entry_concurrency=...)`; a value below `1`
raises `ValueError`.

The current batch is flushed whenever the next entry's destination
overlaps an already-queued destination, or a `Mount` is reached. This
guarantees an ancestor `mkdir` cannot race a descendant write.

## Host-path resolution

`LocalFile` / `LocalDir` sources resolve against `base_dir`. The
default is the host process working directory — the `Path.cwd()` of
the process running the ADK, sampled at call time, NOT the sandbox's
own internal working directory; pass `base_dir` explicitly to pin
resolution when the caller may `chdir` between manifest construction
and materialization. A source
outside that root requires a `SandboxPathGrant` on the manifest
(`extra_path_grants`) — the host-path exfiltration defense.

`resolve_host_source` rejects a symlink on every path component from
the allowed root down to the source; this pre-open check is
platform-independent and always runs. On POSIX, `O_NOFOLLOW` adds
hardening of the check-to-open window at the final `open`; on
non-POSIX the pre-open check is the sole symlink defense.

## Resumption (only_ephemeral)

`only_ephemeral=True` re-materializes only entries marked `ephemeral`,
assuming durable ancestors already exist on the resumed workspace: a
skipped non-ephemeral `Dir` is not re-created, so a child entry still
materializes but relies on the backend's `write` auto-creating parents
(the in-tree backends do). `ephemeral` is strictly per-entry — marking
a `Dir` ephemeral does not propagate to its children, because
`Manifest.iter_entries` flattens each child into its own entry filtered
on its own flag.

The call returns `MaterializationResult(files, skipped_mounts)`: the
`MaterializedFile`s written and the keys of mounts deferred to native
attach.

See `examples/sandbox/manifest_materialization.py` for a runnable
end-to-end demonstration on the local backend. See
`src/troopai/adk/sandbox/session/materialization/` for the
implementation modules.
