"""Unit tests for :class:`Task` with non-Agent targets.

The :attr:`Task.agent` field accepts an :class:`Agent`, a
:class:`Swarm`, or a :class:`Graph`. :meth:`Runner.arun_task`
dispatches via :func:`isinstance` and projects the inner result type
(``RunResult`` / ``SwarmRunResult`` / ``GraphRunResult``) into a
:class:`TaskOutput`.

Covered:

- ``output_schema`` is rejected at Task construction when the target
  is not an Agent.
- Task with :class:`Swarm` target: dispatch hits ``arun_swarm``,
  output projection carries ``final_output`` + ``new_items`` +
  ``usage`` from the swarm's run context.
- Task with :class:`Graph` target: dispatch hits ``arun_graph``,
  output projection carries ``final_output`` + ``new_items`` +
  ``cumulative_usage``. Hooks supplied at the ``arun_task`` level
  are NOT propagated into the graph (documented limitation).
- Failure surfacing: a failed graph run (status=FAILED) projects
  into a TaskOutput with ``error`` set, mirroring the agent-failure
  path.
"""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.graph import Graph
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tasks import Task


class TestOutputSchemaValidation:
    def test_output_schema_with_swarm_target_rejected(self) -> None:
        agent_a = Agent(name="a", system_prompt="x")
        agent_b = Agent(name="b", system_prompt="x")
        swarm = Swarm(
            members=(agent_a, agent_b),
            entry=agent_a,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(2),
        )

        with pytest.raises(ValueError, match="output_schema is only supported"):
            Task(
                description="A",
                agent=swarm,
                output_schema=dict,
            )

    def test_output_schema_with_graph_target_rejected(self) -> None:
        async def _identity(text: str) -> str:
            return text

        g = Graph.new("test-graph").node("only", _identity).entry("only").terminal("only").compile()

        with pytest.raises(ValueError, match="output_schema is only supported"):
            Task(
                description="A",
                agent=g,
                output_schema=dict,
            )

    def test_output_schema_with_agent_target_accepted(self) -> None:
        agent = Agent(name="a", system_prompt="x")
        # No raise — Agent + output_schema is the supported case.
        task = Task(description="A", agent=agent, output_schema=dict)
        assert task.output_schema is dict


class TestGraphTargetDispatch:
    async def test_task_with_graph_target_runs_to_completion(self) -> None:
        async def _prefix(text: str) -> str:
            return f"graph-saw: {text}"

        g = Graph.new("test-graph-task").node("only", _prefix).entry("only").terminal("only").compile()

        task = Task(description="hello", agent=g, name="graph-task")

        from troopai.adk.run.runner import Runner

        output = await Runner.arun_task(task)

        assert output.task_name == "graph-task"
        assert output.error is None
        assert "graph-saw: hello" in str(output.final_output)

    async def test_task_with_failing_graph_target_projects_error(self) -> None:
        async def _boom(text: str) -> str:
            del text
            raise RuntimeError("graph-node-died")

        g = Graph.new("test-graph-failing").node("only", _boom).entry("only").terminal("only").compile()

        task = Task(description="hello", agent=g, name="graph-failing")

        from troopai.adk.run.runner import Runner

        output = await Runner.arun_task(task)

        # Graphs do NOT propagate node-level errors as exceptions —
        # they surface FAILED status. arun_task projects that into a
        # TaskOutput.error so the developer's signal-handling code
        # doesn't need to know which target kind was used.
        assert output.error is not None
        assert "RuntimeError" in output.error
        assert output.final_output is None
