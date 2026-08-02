"""End-to-end graph resume: a mid-flight graph restored from a
checkpoint runs to its correct terminal output."""

from __future__ import annotations

from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.config import GraphConfig
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.join import JoinSemantics
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.graph_loop import run_graph_loop


def _linear_graph() -> Graph:
    return (
        Graph.new("resume-e2e")
        .node("a", lambda: "a-done")
        .node("b", lambda: "b-done")
        .node("c", lambda: "c-done")
        .edge("a", "b")
        .edge("b", "c")
        .entry("a")
        .terminal("c")
        .compile()
    )


def _linear_graph_capped(max_supersteps: int) -> Graph:
    return (
        Graph.new("resume-e2e")
        .node("a", lambda: "a-done")
        .node("b", lambda: "b-done")
        .node("c", lambda: "c-done")
        .edge("a", "b")
        .edge("b", "c")
        .entry("a")
        .terminal("c")
        .with_config(GraphConfig(max_supersteps=max_supersteps))
        .compile()
    )


async def test_resume_completes_midflight_dag() -> None:
    cp = InMemoryCheckpointer()
    g_capped = _linear_graph_capped(1)
    ctx: RunContext = RunContext(context=None)
    first = await run_graph_loop(
        graph=g_capped,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="t1",
    )
    assert first.status.value == "max_supersteps"
    g_full = _linear_graph()
    restored = await cp.load("t1", g_full)
    assert restored is not None
    ctx2: RunContext = RunContext(context=None)
    second = await run_graph_loop(
        graph=g_full,
        user_prompt="go",
        context=ctx2,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="t1",
        initial_state=restored,
    )
    assert second.status.value == "completed"
    assert second.final_output == "c-done"


async def test_resume_does_not_reexecute_completed_node() -> None:
    calls: dict[str, int] = {"a": 0, "b": 0, "c": 0}

    def mk(name: str):
        def _fn() -> str:
            calls[name] += 1
            return f"{name}-done"

        return _fn

    cp = InMemoryCheckpointer()
    capped = (
        Graph.new("resume-idem")
        .node("a", mk("a"))
        .node("b", mk("b"))
        .node("c", mk("c"))
        .edge("a", "b")
        .edge("b", "c")
        .entry("a")
        .terminal("c")
        .with_config(GraphConfig(max_supersteps=2))
        .compile()
    )
    ctx: RunContext = RunContext(context=None)
    first = await run_graph_loop(
        graph=capped,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="t2",
    )
    assert first.status.value == "max_supersteps"
    assert calls == {"a": 1, "b": 1, "c": 0}
    full = (
        Graph.new("resume-idem")
        .node("a", mk("a"))
        .node("b", mk("b"))
        .node("c", mk("c"))
        .edge("a", "b")
        .edge("b", "c")
        .entry("a")
        .terminal("c")
        .compile()
    )
    restored = await cp.load("t2", full)
    assert restored is not None
    ctx2: RunContext = RunContext(context=None)
    res = await run_graph_loop(
        graph=full,
        user_prompt="go",
        context=ctx2,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="t2",
        initial_state=restored,
    )
    assert res.final_output == "c-done"
    assert calls == {"a": 1, "b": 1, "c": 1}


async def test_arun_graph_from_checkpoint_resumes() -> None:
    from troopai.adk.run.runner import Runner

    cp = InMemoryCheckpointer()
    capped = _linear_graph_capped(1)
    first = await Runner.arun_graph(capped, "go", hooks=[cp], thread_id="api1")
    assert first.status.value == "max_supersteps"
    full = _linear_graph()
    res = await Runner.arun_graph_from_checkpoint(full, checkpointer=cp, thread_id="api1")
    assert res.status.value == "completed"
    assert res.final_output == "c-done"


async def test_arun_graph_from_checkpoint_missing_thread_raises() -> None:
    import pytest

    from troopai.adk.run.runner import Runner

    cp = InMemoryCheckpointer()
    g = _linear_graph()
    with pytest.raises(ValueError, match="no checkpoint"):
        await Runner.arun_graph_from_checkpoint(g, checkpointer=cp, thread_id="does-not-exist")


async def test_cumulative_supersteps_survive_resume() -> None:
    cp = InMemoryCheckpointer()
    capped = _linear_graph_capped(1)
    ctx: RunContext = RunContext(context=None)
    first = await run_graph_loop(
        graph=capped,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="t3",
    )
    assert first.status.value == "max_supersteps"
    restored = await cp.load("t3", capped)
    assert restored is not None
    assert restored.superstep == 1
    ctx2: RunContext = RunContext(context=None)
    res = await run_graph_loop(
        graph=capped,
        user_prompt="go",
        context=ctx2,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="t3",
        initial_state=restored,
    )
    assert res.status.value == "max_supersteps"


def test_run_graph_from_checkpoint_resumes_sync() -> None:
    from troopai.adk.run.runner import Runner

    cp = InMemoryCheckpointer()
    capped = _linear_graph_capped(1)
    first = Runner.run_graph(capped, "go", hooks=[cp], thread_id="sync1")
    assert first.status.value == "max_supersteps"
    full = _linear_graph()
    res = Runner.run_graph_from_checkpoint(full, checkpointer=cp, thread_id="sync1")
    assert res.status.value == "completed"
    assert res.final_output == "c-done"


async def test_sqlite_checkpointer_end_to_end_resume(tmp_path) -> None:
    from troopai.adk.graphs.checkpointers.sqlite import SQLiteCheckpointer
    from troopai.adk.run.runner import Runner

    db = str(tmp_path / "run.db")
    cp = SQLiteCheckpointer(db)
    capped = _linear_graph_capped(2)
    first = await Runner.arun_graph(capped, "go", hooks=[cp], thread_id="sql1")
    assert first.status.value == "max_supersteps"
    await cp.close()

    cp2 = SQLiteCheckpointer(db)
    full = _linear_graph()
    res = await Runner.arun_graph_from_checkpoint(full, checkpointer=cp2, thread_id="sql1")
    assert res.status.value == "completed"
    assert res.final_output == "c-done"
    await cp2.close()


async def test_resume_completes_cyclic_graph() -> None:
    """Resume works correctly for a bounded cyclic graph.

    Topology (3 non-terminal nodes + 1 terminal):

        start(entry) ---> worker ---> gate ---[output=="continue"]--> worker (OR-join back-edge)
                                           ---[output=="exit"]-----> done(terminal)

    ``worker`` uses OR-join so the first superstep (where only ``start`` arrives)
    is enough to fire it — the cycle back-edge from ``gate`` is not yet present.
    ``gate`` inspects ``calls["worker"]`` and returns ``"continue"`` until
    ``calls["worker"] >= loop_count``, then returns ``"exit"``.

    loop_count=3, cap=4 supersteps.

    Expected superstep trace (first run, cap=4):
      SS1: start fires (superstep 0→1)
      SS2: worker fires, calls["worker"]=1 (superstep 1→2)
      SS3: gate fires → "continue", calls["gate"]=1 (superstep 2→3)
      SS4: worker fires, calls["worker"]=2 (superstep 3→4)
      [cap check: 4 >= 4 → MAX_SUPERSTEPS]

    At checkpoint: calls={"worker": 2, "gate": 1}, superstep=4.

    Checkpoint state analysis for selective re-fire:
      produced_at  = {start:1, worker:4, gate:3}
      versions_seen = {worker:{start:2, gate:4}, gate:{worker:3}}
    So on resume:
      worker→gate: produced=4 > consumed=3 → re-deliver (unconditional)
      gate→worker: produced=3 <= consumed=4 → skip
      gate→done: produced=3 > consumed=-1 → predicate on stored gate result
                 (gate output was "continue" at SS3) → False → record_skip

    Expected resume trace (SS5 onward):
      SS5: gate fires → "continue", calls["gate"]=2 (SS4→5)
      SS6: worker fires, calls["worker"]=3 (SS5→6)
      SS7: gate fires → "exit", calls["gate"]=3 (SS6→7)
      SS8: done fires → terminal → COMPLETED (SS7→8)

    Final totals: calls={"worker": 3, "gate": 3}.
    No node fires more or fewer times than expected — selective re-fire is
    correct across cycle boundaries.
    """
    loop_count = 3
    calls: dict[str, int] = {"worker": 0, "gate": 0}

    def mk_worker() -> str:
        calls["worker"] += 1
        return f"worker-iter-{calls['worker']}"

    def mk_gate() -> str:
        calls["gate"] += 1
        return "continue" if calls["worker"] < loop_count else "exit"

    cp = InMemoryCheckpointer()

    def _build_cyclic(max_supersteps: int | None = None) -> Graph:
        builder = (
            Graph.new("cyclic-resume")
            .node("start", lambda: "start-done")
            .node("worker", mk_worker, join=JoinSemantics.OR)
            .node("gate", mk_gate)
            .node("done", lambda: "done")
            .edge("start", "worker")
            .edge("worker", "gate")
            .edge("gate", "worker", when=lambda r: r.output == "continue")
            .edge("gate", "done", when=lambda r: r.output == "exit")
            .entry("start")
            .terminal("done")
        )
        if max_supersteps is not None:
            builder = builder.with_config(GraphConfig(max_supersteps=max_supersteps))
        return builder.compile()

    capped = _build_cyclic(max_supersteps=4)
    ctx: RunContext = RunContext(context=None)
    first = await run_graph_loop(
        graph=capped,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="cyclic1",
    )
    assert first.status.value == "max_supersteps", (
        f"Expected first run to hit cap; got {first.status.value!r}. calls={calls}, supersteps={first.total_supersteps}"
    )
    assert calls == {"worker": 2, "gate": 1}, f"After capped run expected worker=2, gate=1; got {calls}"

    full = _build_cyclic()
    restored = await cp.load("cyclic1", full)
    assert restored is not None, "Checkpoint must be present after first run"

    ctx2: RunContext = RunContext(context=None)
    second = await run_graph_loop(
        graph=full,
        user_prompt="go",
        context=ctx2,
        config=DEFAULT_RUN_CONFIG,
        hooks=[cp],
        thread_id="cyclic1",
        initial_state=restored,
    )
    assert second.status.value == "completed", (
        f"Resume must reach terminal; got {second.status.value!r}. calls={calls}, supersteps={second.total_supersteps}"
    )
    assert second.final_output == "done", f"Terminal output must be 'done'; got {second.final_output!r}"
    assert calls == {"worker": 3, "gate": 3}, (
        f"Total executions must be worker=3, gate=3; got {calls}. "
        "Over-execution means completed nodes re-fired; under-execution means "
        "the cycle never resumed correctly."
    )


async def test_builder_resume_from_round_trips() -> None:
    from troopai.adk.run.runner import Runner

    cp = InMemoryCheckpointer()
    capped = (
        Graph.new("builder-resume")
        .node("a", lambda: "a-done")
        .node("b", lambda: "b-done")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .with_config(GraphConfig(max_supersteps=1))
        .compile()
    )
    first = await Runner.arun_graph(capped, "go", hooks=[cp], thread_id="bld1")
    assert first.status.value == "max_supersteps"
    full = (
        Graph.new("builder-resume")
        .node("a", lambda: "a-done")
        .node("b", lambda: "b-done")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )
    res = await Runner.configure().graph(full).resume_from(cp, "bld1").arun()
    assert res.status.value == "completed"
    assert res.final_output == "b-done"
