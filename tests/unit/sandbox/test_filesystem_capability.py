"""Tests for FilesystemCapability + view_image/apply_patch tools (P14 + P28)."""

from __future__ import annotations

import base64
import json
from io import BytesIO

import pytest

from troopai.adk.sandbox.capabilities.filesystem import FilesystemCapability
from troopai.adk.sandbox.clients.local import (
    LocalSandboxClientOptions,
    LocalSubprocessSandboxClient,
)
from troopai.adk.sandbox.tools.view_image_tool import (
    ViewImageArgs,
    _infer_mime,
    make_view_image_tool,
)


@pytest.fixture
async def local_session() -> object:
    client = LocalSubprocessSandboxClient(warn_banner=False)
    session = await client.create(options=LocalSandboxClientOptions())
    await session.start()
    yield session
    await session.aclose()


class TestViewImageArgs:
    def test_minimal(self) -> None:
        args = ViewImageArgs(path="img.png")
        assert args.path == "img.png"
        assert args.mime_type is None


class TestInferMime:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("foo.png", "image/png"),
            ("foo.jpg", "image/jpeg"),
            ("foo.jpeg", "image/jpeg"),
            ("foo.gif", "image/gif"),
            ("foo.webp", "image/webp"),
            ("foo.bin", "application/octet-stream"),
        ],
    )
    def test_basic(self, name: str, expected: str) -> None:
        assert _infer_mime(name) == expected


class TestViewImageTool:
    @pytest.mark.asyncio
    async def test_round_trip(self, local_session: object) -> None:
        # Write a small "image" file.
        payload = b"\x89PNG\r\n\x1a\n_fake_image_"
        await local_session.write("img.png", BytesIO(payload))  # type: ignore[attr-defined]
        tool = make_view_image_tool(session=local_session)  # type: ignore[arg-type]
        assert tool.on_invoke is not None
        result = await tool.on_invoke(None, json.dumps({"path": "img.png"}))  # type: ignore[arg-type]
        assert result["mime_type"] == "image/png"
        assert result["size_bytes"] == len(payload)
        assert base64.b64decode(result["data_base64"]) == payload


class TestFilesystemCapability:
    def test_unbound_returns_no_tools(self) -> None:
        cap = FilesystemCapability()
        assert cap.tools() == []

    @pytest.mark.asyncio
    async def test_bound_returns_both_tools(self, local_session: object) -> None:
        cap = FilesystemCapability()
        cap.bind(local_session)
        tools = cap.tools()
        names = {t.name for t in tools}
        assert names == {"view_image", "apply_patch"}
