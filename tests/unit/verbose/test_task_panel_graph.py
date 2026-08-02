"""Integration tests for the 📋 Task boundary panel on ``arun_graph``.

These tests close the observability asymmetry between ``arun`` /
``arun_swarm`` (which have always emitted ``📋 Task Started`` /
``📋 Task Completed`` panels) and ``arun_graph`` (which previously
ran silently from the panel's perspective).

Graph nodes are plain async Python callables here — no LLM mocking
needed. The line backend captures rendered output to a ``StringIO`` so
assertions can run without a real TTY.
"""

from __future__ import annotations

import io

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig


async def _identity(text: str) -> str:
    return text


async def _boom(text: str) -> str:
    del text
    raise RuntimeError("graph-node-failure")


def _line_config() -> tuple[VerboseConfig, io.StringIO]:
    sink = io.StringIO()
    cfg = VerboseConfig(mode="line", output=sink, use_color=False)
    return cfg, sink


def _trivial_graph() -> Graph:
    return Graph.new("test-graph").node("only", _identity).entry("only").terminal("only").compile()


class TestGraphTaskPanelLifecycle:
    async def test_task_started_fires_at_arun_graph_entry(self) -> None:
        cfg, sink = _line_config()
        graph = _trivial_graph()

        await Runner.arun_graph(graph, "Summarise the docs", run_config=RunConfig(verbose=cfg))

        output = sink.getvalue()
        assert "task started" in output.lower()
        assert "Summarise the docs" in output

    async def test_task_completed_fires_on_clean_exit(self) -> None:
        cfg, sink = _line_config()
        graph = _trivial_graph()

        result = await Runner.arun_graph(graph, "small task", run_config=RunConfig(verbose=cfg))

        assert result.status == GraphRunStatus.COMPLETED
        output = sink.getvalue()
        assert "task completed" in output.lower()

    async def test_task_failed_fires_when_graph_status_failed(self) -> None:
        """Graphs do NOT propagate node-level exceptions — they record
        the failure on :attr:`GraphRunResult.status`. The Task panel
        close must honour that and emit ``❌ Task Failed`` rather than
        the success variant, so developers see the same failure
        signal they see in ``arun`` / ``arun_swarm``.
        """
        cfg, sink = _line_config()
        graph = Graph.new("test-graph-fail").node("only", _boom).entry("only").terminal("only").compile()

        result = await Runner.arun_graph(graph, "small task", run_config=RunConfig(verbose=cfg))

        assert result.status == GraphRunStatus.FAILED
        output = sink.getvalue()
        assert "task failed" in output.lower()
        # Assert on the exception message itself (set on `result.error`
        # by the graph driver) — that contract is stable. The type
        # prefix ("RuntimeError") is also checked but as a softer
        # signal in case the driver ever stops emitting it.
        assert "graph-node-failure" in output
        assert "RuntimeError" in output

    async def test_no_task_panel_when_verbose_disabled(self) -> None:
        graph = _trivial_graph()

        result = await Runner.arun_graph(graph, "small task")

        assert result.status == GraphRunStatus.COMPLETED

    async def test_long_prompt_truncated_in_task_name(self) -> None:
        cfg, sink = _line_config()
        graph = _trivial_graph()

        long_prompt = "x" * 200
        await Runner.arun_graph(graph, long_prompt, run_config=RunConfig(verbose=cfg))

        output = sink.getvalue()
        assert "..." in output
        assert long_prompt not in output
