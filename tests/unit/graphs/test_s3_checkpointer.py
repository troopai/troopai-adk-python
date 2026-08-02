"""``S3Checkpointer`` — tests via moto (in-process AWS S3 mock).

``moto`` intercepts all boto3 calls in-process; no real AWS credentials or
network access are required. Tests do NOT skip.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.s3 import S3Checkpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BUCKET = "ckpt"
_REGION = "us-east-1"


def _g() -> Graph:
    return (
        Graph.new("s3-cp")
        .node("a", lambda: "a")
        .node("b", lambda: "b")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )


def _checkpoint(state: GraphState, superstep: int = 0) -> GraphCheckpoint:
    thread_id = state.thread_id if state.thread_id is not None else "t1"
    return GraphCheckpoint(
        thread_id=thread_id,
        graph_id=state.graph.id,
        state=state.to_dict(),
        superstep=superstep,
    )


def _make_bucket() -> None:
    """Create the test bucket inside an active mock_aws context."""
    boto3.client("s3", region_name=_REGION).create_bucket(Bucket=_BUCKET)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_save_load_round_trip() -> None:
    """save then load returns a GraphState with the correct superstep."""
    with mock_aws():
        _make_bucket()
        cp = S3Checkpointer(bucket=_BUCKET, region=_REGION)
        g = _g()
        state = GraphState(graph=g, thread_id="t1")
        state.superstep = 3
        await cp.save(_checkpoint(state, superstep=3))
        loaded = await cp.load("t1", g)
        assert loaded is not None
        assert loaded.superstep == 3


async def test_load_missing_returns_none() -> None:
    """load on an unknown thread_id returns None."""
    with mock_aws():
        _make_bucket()
        cp = S3Checkpointer(bucket=_BUCKET, region=_REGION)
        result = await cp.load("does-not-exist", _g())
        assert result is None


async def test_list_and_delete() -> None:
    """After save list_checkpoints contains the thread_id; after delete it is gone and load returns None."""
    with mock_aws():
        _make_bucket()
        cp = S3Checkpointer(bucket=_BUCKET, region=_REGION)
        g = _g()
        state = GraphState(graph=g, thread_id="t1")
        await cp.save(_checkpoint(state, superstep=0))
        assert await cp.list_checkpoints() == ["t1"]
        await cp.delete("t1")
        assert await cp.list_checkpoints() == []
        assert await cp.load("t1", g) is None


async def test_graph_id_mismatch_raises() -> None:
    """load with a mismatched graph.id raises ValueError."""
    with mock_aws():
        _make_bucket()
        cp = S3Checkpointer(bucket=_BUCKET, region=_REGION)
        g = _g()
        state = GraphState(graph=g, thread_id="t3")
        await cp.save(
            GraphCheckpoint(
                thread_id="t3",
                graph_id="wrong-graph",
                state=state.to_dict(),
                superstep=0,
            )
        )
        with pytest.raises(ValueError, match="does not match"):
            await cp.load("t3", g)


async def test_last_write_wins() -> None:
    """S3 checkpointer accepts a second save over the same thread_id without error."""
    with mock_aws():
        _make_bucket()
        cp = S3Checkpointer(bucket=_BUCKET, region=_REGION)
        g = _g()
        state = GraphState(graph=g, thread_id="t-lww")
        state.superstep = 1
        await cp.save(_checkpoint(state, superstep=1))
        state.superstep = 2
        # Must NOT raise — last-write-wins, no conflict check.
        await cp.save(_checkpoint(state, superstep=2))
        loaded = await cp.load("t-lww", g)
        assert loaded is not None
        assert loaded.superstep == 2


async def test_list_checkpoints_pagination() -> None:
    """list_checkpoints must return all thread_ids even when >1000 objects are stored.

    list_objects_v2 returns at most 1000 keys per call; the implementation
    must follow pagination tokens until IsTruncated is false.
    """
    prefix = "graph-checkpoints/"
    with mock_aws():
        _make_bucket()
        client = boto3.client("s3", region_name=_REGION)
        # Put 1001 objects directly — faster than going through the checkpointer.
        for i in range(1001):
            client.put_object(Bucket=_BUCKET, Key=f"{prefix}{i}.json", Body=b"{}")
        cp = S3Checkpointer(bucket=_BUCKET, region=_REGION)
        ids = await cp.list_checkpoints()
        assert len(ids) == 1001
