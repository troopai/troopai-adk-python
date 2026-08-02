"""Every concrete backend rejects a configured ``snapshot_store``.

No backend implements snapshot-store persistence; ``create()`` must
raise ``UnsupportedSnapshotFeatureError`` (not silently discard the
store — a configured-but-ignored persistence store is a
data-durability lie). The raise is ``create()``'s first statement,
before any options/network use, so construction can be minimal and
``options`` is never touched.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from troopai.adk.exceptions.exceptions import UnsupportedSnapshotFeatureError

# (kind, hosted_module, client_cls_name, expected_backend_id)
_BACKENDS: list[tuple[str, str | None, str | None, str]] = [
    ("hosted", "e2b", "E2bSandboxClient", "e2b"),
    ("hosted", "vercel", "VercelSandboxClient", "vercel"),
    ("hosted", "modal", "ModalSandboxClient", "modal"),
    ("hosted", "daytona", "DaytonaSandboxClient", "daytona"),
    ("hosted", "cloudflare", "CloudflareSandboxClient", "cloudflare"),
    ("hosted", "blaxel", "BlaxelSandboxClient", "blaxel"),
    ("hosted", "runloop", "RunloopSandboxClient", "runloop"),
    ("docker", None, None, "docker"),
    ("k8s", None, None, "k8s_pod"),
    ("local", None, None, "unix_local"),
]


def _make_client(kind: str, hosted_module: str | None, client_cls_name: str | None) -> Any:
    if kind == "hosted":
        # The parametrize guarantees hosted rows carry both names;
        # an explicit raise (not bare assert) also narrows str | None
        # → str for importlib / getattr.
        if hosted_module is None or client_cls_name is None:
            raise TypeError("hosted backend rows must carry module + client class names")
        module = importlib.import_module(f"troopai.adk.sandbox.clients.hosted.{hosted_module}")
        return getattr(module, client_cls_name)()
    if kind == "docker":
        from troopai.adk.sandbox.clients.docker.docker_client import DockerSandboxClient

        return DockerSandboxClient(docker_client=MagicMock())
    if kind == "k8s":
        from troopai.adk.sandbox.clients.k8s.k8s_client import K8sPodSandboxClient

        return K8sPodSandboxClient(core_v1=MagicMock())
    from troopai.adk.sandbox.clients.local.subprocess_client import LocalSubprocessSandboxClient

    return LocalSubprocessSandboxClient(warn_banner=False)


@pytest.mark.parametrize(
    "kind,hosted_module,client_cls_name,expected_backend_id",
    _BACKENDS,
    ids=[b[3] for b in _BACKENDS],
)
async def test_create_rejects_snapshot_store(
    kind: str,
    hosted_module: str | None,
    client_cls_name: str | None,
    expected_backend_id: str,
) -> None:
    client = _make_client(kind, hosted_module, client_cls_name)
    with pytest.raises(UnsupportedSnapshotFeatureError) as excinfo:
        await client.create(snapshot_store=object(), options=object())
    exc = excinfo.value
    assert exc.feature == "snapshot_store"
    assert exc.backend_id == expected_backend_id
    # No backend implements it yet, so the message states that.
    assert exc.supported_backends == ()
    assert "no backend implements it yet" in str(exc)


def test_message_names_supported_backends_when_present() -> None:
    exc = UnsupportedSnapshotFeatureError(
        "snapshot_store",
        "k8s_pod",
        supported_backends=("e2b", "modal"),
    )
    assert exc.feature == "snapshot_store"
    assert exc.backend_id == "k8s_pod"
    assert exc.supported_backends == ("e2b", "modal")
    rendered = str(exc)
    assert "Supported backends: e2b, modal" in rendered
