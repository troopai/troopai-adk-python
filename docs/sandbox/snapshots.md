# Sandbox Snapshots

A `SnapshotStore` reads and writes serialized workspace snapshots —
typically the tar streams produced by
`BaseSandboxSession.persist_workspace()`.

## Store implementations

| Store | Extra | Persistence |
|---|---|---|
| `LocalSnapshotStore` | (none) | Filesystem under `base_path`; atomic temp-file writes |
| `S3SnapshotStore` | `[sandbox-s3]` | S3 object + sibling `.json` metadata; SSE defaults to AES256; pass `server_side_encryption="aws:kms"` + `kms_key_id` for CMEK |
| `GCSSnapshotStore` | `[sandbox-gcs]` | GCS blob + sibling `.json` metadata; CMEK via `kms_key_name` |

## SnapshotStore surface

```
async save(*, snapshot_id, data, manifest_hash) -> SnapshotMetadata
async load(ref) -> IOBase
async delete(ref) -> None
async list(prefix=None) -> list[SnapshotMetadata]
async exists(ref) -> bool
```

`SnapshotMetadata` carries the `SnapshotRef` (id + `store_uri`),
created-at ISO timestamp, size in bytes, and optional manifest hash.

## Wiring into a run

`SandboxRunConfig.snapshot` is accepted by the per-run lifecycle
for interface conformance, but current backends only log a warning
and discard it. `SandboxRunConfig.snapshot_store` is rejected with
`UnsupportedSnapshotFeatureError`; silently dropping a configured
store would misrepresent durability.

The `SnapshotStore` ABC and its Local / S3 / GCS implementations
are usable directly. Automatic session-start restore and
session-stop persistence are not wired into the backends.

## Storage layout (cloud stores)

```
{prefix}{snapshot_id}.tar    # workspace payload
{prefix}{snapshot_id}.json   # metadata sidecar (created_at, size, manifest_hash)
```

Cloud-store implementations fall back to inferring `created_at` /
`size_bytes` from the blob's own metadata when the sidecar is
missing (e.g. on snapshots written before sidecar metadata was
available).

See `examples/sandbox/memory_capability.py` for a two-run flow that
persists workspace state across agent invocations.
