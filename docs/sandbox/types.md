# Sandbox Types (Layer 1)

All Layer-1 sandbox types live in `troopai.adk.types.sandbox`. They
have **no provider SDK imports** and serialize to JSON without
conversion hops.

## Manifest

`Manifest` describes the workspace a session should materialize.

| Field | Purpose |
|---|---|
| `root` | Workspace root path inside the sandbox (default `/workspace`) |
| `entries` | `dict[str, BaseEntry]` keyed by workspace-relative path |
| `environment` | Env vars injected at session start |
| `users` / `groups` | Multi-user workspaces (rare; default empty) |
| `extra_path_grants` | `SandboxPathGrant` permissions outside the manifest |
| `remote_mount_command_allowlist` | Mount-tool argv allowlist |

`BaseEntry` subclasses: `File`, `Dir`, `LocalFile`, `LocalDir`,
`GitRepo`, and the `Mount` family (`S3Mount`, `GCSMount`, `R2Mount`,
`AzureBlobMount`, `BoxMount`, `S3FilesMount`).

## Mounts

`Mount.mount_path` is workspace-relative; absolute paths and `..`
escapes raise at validation time. `Mount.read_only` defaults to True.
`Mount.mount_strategy` is a discriminated union:

- `InContainerMountStrategy(pattern=...)` — backend runs a mount tool
  inside the container. Patterns: `RcloneMountPattern`,
  `MountpointMountPattern`, `FuseMountPattern`, `S3FilesMountPattern`.
- `DockerVolumeMountStrategy(driver, driver_options)` — Docker volume
  driver attaches the storage before container start.

`troopai.adk.sandbox.policy.mounts` translates these to each backend's
wire format (Docker volumes, K8s CSI volumes, hosted-bridge create-
body fields).

## Exec result + ports

- `ExecResult(stdout, stderr, exit_code, duration_ms)` — output of
  `session.run(...)`. Non-zero exits are surfaced (not raised).
- `ExposedPortEndpoint(host, port, tls, query)` — output of
  `session.resolve_exposed_port(port)`. Helpers: `url_for(scheme)`.
- `PtyHandle(session_id, command, backend_payload)` — opaque PTY
  reference; `backend_payload` MUST NOT be introspected outside the
  backend.

## Snapshot + IaC

- `SnapshotRef(snapshot_id, store_uri)` — store-scoped address.
- `SnapshotMetadata(ref, created_at_iso, size_bytes, manifest_hash)` —
  what `SnapshotStore.list()` / `save()` return.
- `IaCBundle` — see `docs/sandbox/iac.md`.

## Span data + usage

- `SandboxSpanData(backend_id, command, exit_code, duration_ms,
  manifest_hash, resource_usage, snapshot_id)` — feeds the tracing
  layer alongside the existing function/generation span types.
- `SandboxSingleExecUsage` + `SandboxUsage(__add__)` — mirrors the
  `LLMUsage` accumulator semantics for cross-run aggregation.

## Permissions

- `RunAsUser`, `User`, `Group`, `Permissions(perm_str)`,
  `FileMode(IntEnum)` — POSIX-style permission modelling for
  workspace files and exec sessions.
- `SandboxPathGrant(path, read_only, description)` +
  `WorkspacePathPolicy` — explicit grants outside the manifest.

See `src/troopai/adk/types/sandbox/` for source.
