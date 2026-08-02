"""Docker-daemon integration test (skipped when no daemon reachable).

This test exercises DockerSandboxClient + DockerSandboxSession end to
end against a real Docker daemon. It's marked as ``integration`` so
unit-only runs skip it; pytest collects + skips automatically when
the daemon is unreachable.
"""

from __future__ import annotations

from io import BytesIO

import pytest


def _docker_available() -> bool:
    try:
        import docker
    except ImportError:
        return False
    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable"),
]


@pytest.mark.asyncio
async def test_full_lifecycle_against_real_daemon() -> None:
    from troopai.adk.sandbox.clients.docker import (
        DockerSandboxClient,
        DockerSandboxClientOptions,
    )

    client = DockerSandboxClient()
    options = DockerSandboxClientOptions(image="alpine:3.20")
    session = await client.create(options=options)
    try:
        await session.start()
        # Run a trivial echo.
        result = await session.run("echo", "hello-world", shell=True)
        assert result.exit_code == 0
        assert b"hello-world" in result.stdout
        # Write a file, read it back.
        await session.write("greeting.txt", BytesIO(b"hi there"))
        stream = await session.read("greeting.txt")
        assert stream.read() == b"hi there"
        # ls.
        entries = await session.ls(".")
        names = {e.name for e in entries}
        assert "greeting.txt" in names
    finally:
        await session.aclose()
