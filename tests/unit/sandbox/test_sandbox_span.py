"""Tests for ``sandbox_span`` factory (P38)."""

from __future__ import annotations

from troopai.adk.tracing.spans import sandbox_span


class TestSandboxSpanFactory:
    def test_minimal_construction(self) -> None:
        # NoOp tracer is the default; backend_id passes through.
        span = sandbox_span(backend_id="unix_local", disabled=True)
        # disabled=True returns NoOpSpan[SandboxSpanData] so .data
        # IS the SandboxSpanData (no CustomSpanData wrap).
        assert span.data.backend_id == "unix_local"  # type: ignore[union-attr]

    def test_full_construction_disabled(self) -> None:
        span = sandbox_span(
            backend_id="docker",
            command="ls /tmp",
            exit_code=0,
            duration_ms=42,
            disabled=True,
        )
        assert span.data.backend_id == "docker"  # type: ignore[union-attr]
        assert span.data.command == "ls /tmp"  # type: ignore[union-attr]

    def test_default_tracer_path_returns_some_span(self) -> None:
        # With the NoOp default tracer + disabled=False, the factory
        # routes through custom_span and returns a NoOpSpan that
        # wraps CustomSpanData. The factory doesn't raise.
        span = sandbox_span(backend_id="docker", disabled=False)
        # We can't unambiguously read backend_id off the wrapped
        # CustomSpanData without inspecting export(); just confirm
        # the call returns and the span has SOME data.
        assert span is not None
        assert span.data is not None
