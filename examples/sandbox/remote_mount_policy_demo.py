"""Render the LLM-visible safety paragraph for remote-mount manifests.

When a manifest declares cloud object-storage mounts (S3, GCS, R2,
Azure Blob, Box), the framework injects a paragraph into the
system prompt that:

* marks mounted paths as untrusted data,
* enumerates each mount path + access mode,
* lists the command allowlist (``ls``, ``cat``, ``head``, ...),
* documents the cp-edit-cp workflow for shell-driven edits.

This script shows the rendered text without spinning up an agent.

No external API key required.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging

from troopai.adk.sandbox.remote_mount_policy import (
    build_remote_mount_policy_instructions,
    get_remote_mounts,
)
from troopai.adk.types.sandbox.entries import File
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.mounts import (
    InContainerMountStrategy,
    RcloneMountPattern,
    S3Mount,
)

logger = logging.getLogger(__name__)


def main() -> None:
    s3 = S3Mount(
        bucket="my-data-bucket",
        mount_path="data",
        read_only=True,
        mount_strategy=InContainerMountStrategy(
            pattern=RcloneMountPattern(remote_name="my-data-bucket"),
        ),
    )
    output_mount = S3Mount(
        bucket="my-output-bucket",
        mount_path="output",
        read_only=False,
        mount_strategy=InContainerMountStrategy(
            pattern=RcloneMountPattern(remote_name="my-output-bucket"),
        ),
    )
    manifest = Manifest(
        root="/workspace",
        entries={
            "data": s3,
            "output": output_mount,
            "src/app.py": File(content=b"# regular workspace file"),
        },
    )

    logger.info("=== Discovered remote mounts ===")
    for path, read_only in get_remote_mounts(manifest):
        mode = "ro" if read_only else "rw"
        logger.info("  %s (%s)", path.as_posix(), mode)

    logger.info("=== Rendered policy paragraph (injected into system prompt) ===")
    policy = build_remote_mount_policy_instructions(manifest)
    logger.info("\n%s", policy)
    assert policy is not None
    assert "untrusted data" in policy
    assert "data" in policy
    assert "output" in policy


if __name__ == "__main__":
    main()
