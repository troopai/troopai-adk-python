"""Tests for ``troopai.adk.mcp.auth``, ``troopai.adk.mcp.otel`` and
``troopai.adk.mcp.notifications``.

These three modules are small and tightly related (header injection,
trace-context injection, and push-driven cache invalidation), so the
tests live together to keep the file count from growing for tiny
units.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.mcp.auth import HeaderProvider, active_header_provider
from troopai.adk.mcp.notifications import _extract_notification_method, make_message_handler
from troopai.adk.mcp.otel import build_mcp_meta
from troopai.adk.mcp.run_hooks_bridge import (
    active_run_context,
    active_run_hooks,
    fire_on_mcp_connect,
    fire_on_mcp_error,
)

# ------------------------------------------------------------------------ auth


async def test_active_header_provider_isolated_across_concurrent_tasks() -> None:
    """ContextVar isolation — concurrent tasks see only their own provider."""
    seen: dict[str, HeaderProvider | None] = {}

    async def task(label: str, provider: HeaderProvider | None) -> None:
        if provider is not None:
            token = active_header_provider.set(provider)
            try:
                await asyncio.sleep(0)
                seen[label] = active_header_provider.get()
            finally:
                active_header_provider.reset(token)
        else:
            await asyncio.sleep(0)
            seen[label] = active_header_provider.get()

    def p_a() -> dict[str, str]:
        return {"X-Token": "a"}

    def p_b() -> dict[str, str]:
        return {"X-Token": "b"}

    await asyncio.gather(
        task("a", p_a),
        task("b", p_b),
        task("none", None),
    )

    assert seen["a"] is p_a
    assert seen["b"] is p_b
    assert seen["none"] is None


def test_active_header_provider_default_is_none() -> None:
    assert active_header_provider.get() is None


# ------------------------------------------------------------------------ otel


def test_build_mcp_meta_returns_dict_when_no_otel_active() -> None:
    """No active span → returns the ``extra`` dict unchanged (or empty)."""
    assert build_mcp_meta() == {}
    assert build_mcp_meta({"foo": "bar"}) == {"foo": "bar"}


def test_build_mcp_meta_does_not_share_state_across_calls() -> None:
    a = build_mcp_meta({"x": 1})
    a["y"] = 2
    b = build_mcp_meta()
    assert "y" not in b


# --------------------------------------------------------------- notifications


class _FakeServer:
    """Minimal ``MCPServerWithClientSession`` shim for the handler under test."""

    def __init__(self) -> None:
        self.name = "fake"
        self.invalidated = 0

    def invalidate_tools_cache(self) -> None:
        self.invalidated += 1


async def test_handler_invalidates_cache_on_tools_list_changed() -> None:
    server = _FakeServer()
    handler = make_message_handler(server)  # type: ignore[arg-type]

    # Use a real namespace, not MagicMock, so attribute lookup is honest
    # — MagicMock auto-creates ``root``/``method`` and would silently
    # match the wrong branch in ``_extract_notification_method``.
    class _Notif:
        method = "notifications/tools/list_changed"

    await handler(_Notif())

    assert server.invalidated == 1


async def test_handler_ignores_other_notifications() -> None:
    server = _FakeServer()
    handler = make_message_handler(server)  # type: ignore[arg-type]

    class _Notif:
        method = "notifications/something_else"

    await handler(_Notif())

    assert server.invalidated == 0


async def test_handler_swallows_exception_inside_handler() -> None:
    """A handler that raises must not crash the session."""

    class _BadServer:
        name = "bad"

        def invalidate_tools_cache(self) -> None:
            raise RuntimeError("oops")

    handler = make_message_handler(_BadServer())  # type: ignore[arg-type]

    class _Notif:
        method = "notifications/tools/list_changed"

    await handler(_Notif())  # MUST NOT raise


def test_extract_method_handles_root_wrapper() -> None:
    """``message.root.method`` is the new MCP-SDK wrapper shape."""

    class _Inner:
        method = "notifications/tools/list_changed"

    class _Outer:
        root = _Inner()

    assert _extract_notification_method(_Outer()) == "notifications/tools/list_changed"


def test_extract_method_returns_none_for_unknown_shape() -> None:
    class _Empty:
        pass

    assert _extract_notification_method(_Empty()) is None


# ------------------------------------------- pytest plugin (asyncio_mode=auto)


@pytest.mark.asyncio
async def test_async_marker_explicit() -> None:
    """Sanity: explicit @pytest.mark.asyncio works alongside async_mode=auto."""
    assert True


# ------------------------------------------- run_hooks_bridge WARNING level


async def test_fire_on_mcp_error_logs_hook_failure_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: hook failures inside fire_on_mcp_error were swallowed at
    DEBUG level, making broken error-hooks undetectable in production.
    They MUST now log at WARNING with exc_info so operators see them.
    """
    hooks = MagicMock()
    hooks.on_mcp_error = AsyncMock(side_effect=RuntimeError("hook exploded"))
    ctx = MagicMock()

    active_run_hooks.set(hooks)
    active_run_context.set(ctx)
    try:
        with caplog.at_level(logging.WARNING, logger="troopai.adk.mcp.run_hooks_bridge"):
            await fire_on_mcp_error("my-server", ValueError("original error"))
    finally:
        active_run_hooks.set(None)
        active_run_context.set(None)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) > 0, "Expected WARNING log for hook failure"
    assert warning_records[0].exc_info is not None, "Expected exc_info on WARNING record"


async def test_fire_on_mcp_connect_logs_hook_failure_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Symmetry check: on_mcp_connect hook failures must also log at WARNING."""
    hooks = MagicMock()
    hooks.on_mcp_connect = AsyncMock(side_effect=RuntimeError("hook boom"))
    ctx = MagicMock()

    active_run_hooks.set(hooks)
    active_run_context.set(ctx)
    try:
        with caplog.at_level(logging.WARNING, logger="troopai.adk.mcp.run_hooks_bridge"):
            await fire_on_mcp_connect("my-server")
    finally:
        active_run_hooks.set(None)
        active_run_context.set(None)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) > 0
