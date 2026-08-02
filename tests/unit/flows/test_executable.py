"""Unit tests for :class:`FlowExecutable` Graph adapter."""

from __future__ import annotations

from pydantic import BaseModel

from troopai.adk.flows import Flow, FlowExecutable, flow_listen, flow_start
from troopai.adk.graphs.adapters import to_executable
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext


class _State(BaseModel):
    log: list[str] = []


class _LinearFlow(Flow[_State]):
    @flow_start
    async def begin(self) -> None:
        self.state.log.append("begin")

    @flow_listen(begin)
    async def end(self) -> None:
        self.state.log.append("end")


class TestFlowExecutable:
    async def test_invoke_returns_node_result(self) -> None:
        adapter: FlowExecutable[None] = FlowExecutable(flow=_LinearFlow(_State))
        ctx: RunContext[None] = RunContext.make(None)
        node_input = ExecutableInput(
            content=[],
            from_node=None,
            edge_label=None,
        )
        result = await adapter.invoke(node_input, ctx, RunConfig())
        assert result.metadata["adapter"] == "flow"
        assert result.metadata["status"] == "completed"
        # output is the typed state
        assert isinstance(result.output, _State)
        assert result.output.log == ["begin", "end"]


class TestToExecutableBranch:
    def test_to_executable_wraps_flow(self) -> None:
        flow = _LinearFlow(_State)
        ex = to_executable(flow)
        assert isinstance(ex, FlowExecutable)
        assert ex.flow is flow
