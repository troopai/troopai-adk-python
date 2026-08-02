"""Minimal agent running as a Restate durable workflow.

Demonstrates:
    - Wrapping an agent's LLM with RestateLLM so every LLM call is journaled
      inside a Restate handler (crash-recoverable, replay-safe)
    - Defining a Restate workflow whose main handler calls Runner.arun()
    - The HITL pattern via RestateHumanReply + a Restate durable promise
      (ctx.promise). Durable promises are a workflow-only primitive: the
      WorkflowContext / WorkflowSharedContext expose them; a plain service
      Context does not — which is why this is a Workflow, not a service.

Prerequisites:
    pip install "troopai-adk-python[restate]"
    # Start the Restate server:
    # docker run --rm -p 8080:8080 -p 9070:9070 docker.restate.dev/restatedev/restate

Run with:
    python examples/restate/basic_agent.py
    # Register this deployment with the server (one-time):
    #   curl localhost:9070/deployments --json '{"uri": "http://localhost:9080"}'
    # Start a run (the path segment after the workflow name is a caller-chosen
    # workflow id that keys the durable instance):
    #   curl localhost:8080/AgentService/run-1/run --json '{"prompt": "Summarise ..."}'
    # The run blocks on human review; resolve it so the run completes:
    #   curl localhost:8080/AgentService/run-1/submit_review \
    #        --json '{"node_id": "review", "value": "yes"}'

References:
    Restate Python SDK: https://docs.restate.dev/develop/python
    Restate ctx.run journaling:
    https://docs.restate.dev/develop/python/durable-execution#journaling-results
    Restate durable promises:
    https://docs.restate.dev/develop/python/durable-execution#promises
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import restate  # type: ignore[import-not-found]  # optional dep — install the [restate] extra
except ImportError as _exc:
    raise SystemExit("restate not installed. Run: pip install 'troopai-adk-python[restate]'") from _exc

# ---------------------------------------------------------------------------
# Step 1 — define the agent (Agent = config, not execution)
# ---------------------------------------------------------------------------

from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM
from troopai.adk.workflows.engine import ModelActivityConfig
from troopai.adk.workflows.restate import RestateLLM, TroopAIRestateService

_inner_llm = LiteLLM(model="gpt-4o-mini")

# ---------------------------------------------------------------------------
# Step 2 — wrap the LLM with RestateLLM so calls are journaled by Restate
# ---------------------------------------------------------------------------

# RestateLLM checks restate.current_context() at call time:
#   - Inside a Restate handler  → calls are routed through ctx.run() and journaled.
#   - Outside a handler          → calls go directly to the wrapped LLM (no overhead).
_llm = RestateLLM(
    wrapped=_inner_llm,
    activity_config=ModelActivityConfig(),
)

_agent = Agent(
    name="summariser",
    system_prompt="Summarise the user's message in one sentence.",
    llm=_llm,
)

# ---------------------------------------------------------------------------
# Step 3 — define the Restate workflow (one @main + shared @handler handlers)
# ---------------------------------------------------------------------------

from troopai.adk.run.runner import Runner

_helpers = TroopAIRestateService()

# A Restate Workflow: its main handler receives a WorkflowContext (which,
# unlike a plain service Context, exposes durable promises via ctx.promise),
# and shared handlers receive a WorkflowSharedContext bound to the same
# instance. The HITL step below blocks the main handler on a promise that the
# shared submit_review handler resolves.
agent_workflow = restate.Workflow("AgentService")  # pyright: ignore[reportAttributeAccessIssue]  # restate provided by the [restate] extra at runtime


@agent_workflow.main()
async def run(ctx: restate.WorkflowContext, req: dict[str, str]) -> dict[str, str]:  # pyright: ignore[reportAttributeAccessIssue]
    """Run the agent durably, then pause for human review via a durable promise.

    Each invocation is a durable execution keyed by the caller-supplied
    workflow id. If the process crashes mid-run, Restate replays the journal:
    LLM calls that completed are not retried; pending ones resume.

    Args:
        ctx: Restate WorkflowContext — passed to RestateLLM automatically via
            restate.current_context(), and the source of the durable promise.
        req: Request payload. Expected key: "prompt" (str).

    Returns:
        Response dict with "output" (final text, prefixed when the reviewer
        rejects) and "human_reply" (the raw reviewer value) keys.
    """
    prompt = req.get("prompt", "")
    logger.info("AgentService.run: prompt=%r", prompt)

    # Runner.arun() → _agent.llm.acomplete() → RestateLLM.acomplete()
    # → ctx.run("invoke_model", …) → result is journaled.
    result = await Runner.arun(_agent, prompt)
    draft = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
    logger.info("AgentService.run: draft produced, awaiting human review")

    # Durably block until submit_review (or the admin API) resolves the promise.
    # The promise payload must be a dict with at least "node_id" and "value".
    reply = await _helpers.wait_for_human_reply(ctx, promise_name="review_reply")
    logger.info(
        "AgentService.run: human reply received, node_id=%r value=%r",
        reply.node_id,
        reply.value,
    )

    approved = reply.value.lower().startswith("y")
    final_output = draft if approved else f"[REJECTED by reviewer] {draft}"
    return {"output": final_output, "human_reply": reply.value}


@agent_workflow.handler()
async def submit_review(ctx: restate.WorkflowSharedContext, decision: dict[str, str]) -> dict[str, str]:  # pyright: ignore[reportAttributeAccessIssue]
    """Resolve the durable promise the main handler is blocked on.

    A shared handler runs against the same workflow instance and can resolve a
    promise the main handler awaits — the Restate equivalent of delivering a
    human-in-the-loop signal.

    Args:
        ctx: Restate WorkflowSharedContext for the running workflow instance.
        decision: The reviewer payload. Must carry the RestateHumanReply shape
            wait_for_human_reply expects: at least "node_id" and "value" keys.

    Returns:
        A small acknowledgement dict.
    """
    logger.info("AgentService.submit_review: resolving 'review_reply' promise")
    await ctx.promise("review_reply").resolve(decision)
    return {"status": "review submitted"}


# ---------------------------------------------------------------------------
# Step 4 — register the workflow and start the Restate HTTP endpoint
# ---------------------------------------------------------------------------

app = restate.app(services=[agent_workflow])  # pyright: ignore[reportAttributeAccessIssue]

if __name__ == "__main__":
    import asyncio

    import hypercorn.asyncio  # type: ignore[import-not-found]  # optional dep: pip install hypercorn
    import hypercorn.config  # type: ignore[import-not-found]

    # Restate requires hypercorn (or another ASGI server).
    # Install: pip install hypercorn
    config = hypercorn.config.Config()
    config.bind = ["0.0.0.0:9080"]

    logger.info("Starting Restate service on :9080 — connect via Restate server at :8080")
    asyncio.run(hypercorn.asyncio.serve(app, config))  # type: ignore[arg-type]
