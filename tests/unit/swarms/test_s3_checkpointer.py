"""``S3SwarmCheckpointer`` — tests via moto (in-process AWS S3 mock).

``moto`` intercepts all boto3 calls in-process; no real AWS credentials or
network access are required. Tests do NOT skip.
"""

from __future__ import annotations

from typing import Any, cast

import boto3
from moto import mock_aws

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.s3 import S3SwarmCheckpointer
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState, SwarmStateDict
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BUCKET = "swarm-ckpt"
_REGION = "us-east-1"


def _make_swarm() -> Swarm:
    """Single-member swarm fixture for checkpointer tests."""
    member = Agent(name="m1", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


def _make_state(swarm: Swarm, turns: int = 1) -> SwarmState:
    state = SwarmState(
        swarm=swarm,
        current_agent=swarm.members[0],
        current_agent_name=swarm.members[0].name,
    )
    state.total_turns = turns
    return state


def _ckpt(thread_id: str, state: SwarmState, turn: int) -> SwarmCheckpoint:
    return SwarmCheckpoint(
        thread_id=thread_id,
        state=cast(dict[str, Any], state.to_dict()),
        turn=turn,
    )


def _make_bucket() -> None:
    """Create the test bucket inside an active mock_aws context."""
    boto3.client("s3", region_name=_REGION).create_bucket(Bucket=_BUCKET)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_save_load_round_trip() -> None:
    """save then load returns a SwarmCheckpoint with the correct turn; rehydration succeeds."""
    with mock_aws():
        _make_bucket()
        swarm = _make_swarm()
        state = _make_state(swarm, turns=5)
        cp = S3SwarmCheckpointer(bucket=_BUCKET, region=_REGION)

        await cp.save(_ckpt("t1", state, state.total_turns))

        loaded = await cp.load("t1", swarm)
        assert loaded is not None
        assert loaded.turn == 5

        # Rehydrate via SwarmState.from_dict — the caller's contract.
        rehydrated = SwarmState.from_dict(cast(SwarmStateDict, loaded.state), swarm)
        assert rehydrated.total_turns == 5
        assert rehydrated.current_agent_name == "m1"


async def test_load_missing_returns_none() -> None:
    """load on an unknown thread_id returns None."""
    with mock_aws():
        _make_bucket()
        swarm = _make_swarm()
        cp = S3SwarmCheckpointer(bucket=_BUCKET, region=_REGION)
        result = await cp.load("does-not-exist", swarm)
        assert result is None


async def test_list_and_delete() -> None:
    """After save list_checkpoints contains the thread_id; after delete it is gone and load returns None."""
    with mock_aws():
        _make_bucket()
        swarm = _make_swarm()
        state = _make_state(swarm, turns=1)
        cp = S3SwarmCheckpointer(bucket=_BUCKET, region=_REGION)

        await cp.save(_ckpt("t1", state, 1))
        assert await cp.list_checkpoints() == ["t1"]

        await cp.delete("t1")
        assert await cp.list_checkpoints() == []
        assert await cp.load("t1", swarm) is None


async def test_last_write_wins() -> None:
    """S3 swarm checkpointer accepts a second save over the same thread_id without error."""
    with mock_aws():
        _make_bucket()
        swarm = _make_swarm()
        state = _make_state(swarm, turns=1)
        cp = S3SwarmCheckpointer(bucket=_BUCKET, region=_REGION)

        await cp.save(_ckpt("t-lww", state, 1))
        state.total_turns = 2
        # Must NOT raise — last-write-wins, no conflict check.
        await cp.save(_ckpt("t-lww", state, 2))

        loaded = await cp.load("t-lww", swarm)
        assert loaded is not None
        assert loaded.turn == 2


async def test_list_checkpoints_pagination() -> None:
    """list_checkpoints must return all thread_ids even when >1000 objects are stored.

    list_objects_v2 returns at most 1000 keys per call; the implementation
    must follow pagination tokens until IsTruncated is false.
    """
    prefix = "swarm-checkpoints/"
    with mock_aws():
        _make_bucket()
        client = boto3.client("s3", region_name=_REGION)
        # Put 1001 objects directly — faster than going through the checkpointer.
        for i in range(1001):
            client.put_object(Bucket=_BUCKET, Key=f"{prefix}{i}.json", Body=b"{}")
        cp = S3SwarmCheckpointer(bucket=_BUCKET, region=_REGION)
        ids = await cp.list_checkpoints()
        assert len(ids) == 1001
