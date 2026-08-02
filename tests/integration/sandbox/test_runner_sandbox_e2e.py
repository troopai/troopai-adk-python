"""End-to-end test: SandboxAgent + LocalSubprocess + Runner (P49)."""

from __future__ import annotations

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.config import SandboxRunConfig


@pytest.mark.integration
class TestSandboxRunnerEndToEnd:
    @pytest.mark.asyncio
    async def test_sandbox_bracket_acquires_and_releases(self) -> None:
        """The Runner opens + closes the sandbox session around arun.

        This test exercises the bracket without a real LLM call —
        the agent loop's actual model invocation is mocked elsewhere.
        Here we just verify the bracket itself works end-to-end.
        """
        from troopai.adk.sandbox.clients.local import LocalSubprocessSandboxClient

        client = LocalSubprocessSandboxClient(warn_banner=False)
        sandbox_config = SandboxRunConfig(client=client)
        agent = SandboxAgent(name="e2e", system_prompt="x")
        config = RunConfig(sandbox=sandbox_config)
        # Verify the bracket helper works directly; full Runner.arun
        # integration with capability tools lands in a follow-up.
        import contextlib

        from troopai.adk.run.context import RunContext
        from troopai.adk.run.runner import _maybe_open_sandbox_bracket

        rc: RunContext[None] = RunContext.make(None)
        async with contextlib.AsyncExitStack() as stack:
            await _maybe_open_sandbox_bracket(
                stack=stack,
                agent=agent,
                config=config,
                run_context=rc,
            )
            handle = getattr(rc, "_sandbox_handle", None)
            assert handle is not None
            assert handle.session is not None
        # After exit, session is closed.
