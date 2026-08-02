"""Agent-internal HITL bridge — propagate agent deferrals up to the Flow.

A flow step body that calls
:func:`arun_flow_agent(flow, agent, input_prompt)` gets a
:class:`RunResult` like a plain :func:`Runner.arun` call when the
agent completes; if the agent's run defers (a tool with
``requires_approval=True`` short-circuits the agent loop), the
bridge raises :class:`FlowAgentDeferred` carrying the agent's
serialised :class:`RunState`. The executor catches that exception in
``_process_batch_results``, builds a :class:`FlowDeferredStep` with
``agent_run_state`` populated, halts the flow with
``status="deferred"``, and returns a :class:`FlowCheckpoint` to the
caller.

The developer records decisions on the deferred agent state via the
existing tool-layer surface — :meth:`RunState.approve` /
:meth:`RunState.reject` — then resumes the flow via
:meth:`Runner.arun_flow_from_checkpoint(flow, checkpoint, agent_resolutions=...)`
which threads the resolved state back to the inner agent run.

Sits as a free function (not a method on :class:`Flow`) because
:class:`Flow` is configuration and the Runner is execution — its base
class never exposes ``run`` / ``arun`` methods. The bridge is a helper
the developer
calls from inside a step body, exactly like
:func:`Runner.arun(...)`.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from troopai.adk.flows.exceptions import FlowAgentDeferred, FlowDefinitionError

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.flows.flow import Flow
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.state import RunState
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.types.run.run_result import RunResult


async def arun_flow_agent(
    flow: Flow[Any],
    agent: Agent,
    input_prompt: UserPrompt,
    *,
    defer_key: str | None = None,
    context: RunContext[Any] | None = None,
    run_config: RunConfig | None = None,
) -> RunResult[Any]:
    """Run an :class:`Agent` inside a flow step with HITL propagation.

    Drop-in replacement for ``await Runner.arun(agent, input_prompt)``
    inside a flow step body. The difference: when the agent's run
    returns ``requires_action=True`` (a tool deferred), this helper
    raises :class:`FlowAgentDeferred` so the
    :class:`FlowExecutor` halts the flow with a checkpoint instead
    of returning a half-baked :class:`RunResult` to the step body.

    On resume from a flow checkpoint, the consumer supplies an
    ``agent_resolutions`` mapping (``defer_key → RunState JSON``) to
    :meth:`Runner.arun_flow_from_checkpoint`. The flow's
    ``_pending_agent_resolutions`` map carries those into the
    resumed run; this helper pops the matching entry, hands it back
    to :func:`Runner.arun` to continue the agent loop, and returns
    the final :class:`RunResult` (or re-raises
    :class:`FlowAgentDeferred` if the agent defers a second time).

    Args:
        flow: The :class:`Flow` instance the calling step belongs to
            (typically ``self`` from inside a step body).
        agent: The :class:`Agent` to run.
        input_prompt: The user prompt for the agent run.
        defer_key: Optional stable key to identify this particular
            agent invocation when a single step runs multiple
            agents. When ``None``, the framework walks the call
            stack and cross-checks against the flow's registry to
            find the enclosing step's method name. The walk fails
            loudly (``FlowDefinitionError``) when no registered
            step is found on the stack — pass an explicit
            ``defer_key`` whenever the call site is more than one
            frame away from the step body (helper methods, nested
            free functions, ``asyncio.gather`` siblings, etc.).
        context: Optional :class:`RunContext` to attach to the
            agent run. Pass ``flow.run_context`` to share cumulative
            usage with the flow.
        run_config: Optional ``RunConfig`` forwarded to
            :func:`Runner.arun`.

    Returns:
        The final :class:`RunResult` when the agent run completes
        normally.

    Raises:
        FlowAgentDeferred: When the inner agent run defers via a
            tool ``requires_approval`` gate. Caught by the
            :class:`FlowExecutor` and never seen by step bodies.
    """
    from troopai.adk.run.runner import Runner

    if defer_key is not None:
        resolved_key = defer_key
        # Stack walk skipped for resolved_key: caller supplied an explicit
        # defer_key, bypassing the expensive and fragile stack inspection
        # for helper methods / asyncio.gather siblings. The actual step
        # name is still discovered via a non-raising walk (best-effort)
        # so the executor's last_invocation_triggers lookup succeeds. When
        # the walk finds nothing (call made across an async boundary) we fall
        # back to defer_key — but only a deferral consumes step_name, and the
        # requires_action branch below rejects an unregistered fallback there
        # so a poisoned checkpoint can never be written.
        inferred = _infer_calling_step_name(flow)
        step_name = inferred if inferred is not None else defer_key
    else:
        inferred = _infer_calling_step_name(flow)
        if inferred is None:
            raise FlowDefinitionError(
                "arun_flow_agent: unable to infer the enclosing flow step name. Pass an "
                "explicit defer_key= and call the bridge directly from a registered "
                "@flow_start / @flow_listen / @flow_router method body (not a helper).",
            )
        resolved_key = inferred
        step_name = inferred

    pending_state_json = flow.consume_pending_agent_resolution(resolved_key)
    if pending_state_json is not None:
        run_state = _decode_pending_run_state(pending_state_json, step_name=step_name, defer_key=resolved_key)
        result = await Runner.arun(agent, run_state, context=context, run_config=run_config)
    else:
        result = await Runner.arun(agent, input_prompt, context=context, run_config=run_config)

    if result.requires_action:
        # Only NOW — when an agent deferral is actually about to be captured —
        # require step_name to be a registered step. On the success path an
        # unresolved step name is harmless, but a deferral writes step_name into
        # the checkpoint's pending_steps, and an unregistered name (defer_key
        # used as a fallback after cross-async-boundary inference failed) makes
        # resume unrunnable. Raise here rather than silently poisoning it.
        if step_name not in flow.get_registry().step_names():
            raise FlowDefinitionError(
                f"arun_flow_agent: cannot identify the enclosing flow step for the agent "
                f"deferral (defer_key={resolved_key!r}). Step-name inference failed — the "
                f"bridge was called off the step's call stack (an async boundary such as "
                f"asyncio.gather / create_task) — and defer_key does not name a registered "
                f"@flow_start / @flow_listen / @flow_router step, so it cannot be written to "
                f"the resume checkpoint. Call arun_flow_agent from the step body (keeping the "
                f"step frame on the stack), or pass defer_key equal to the enclosing step's "
                f"method name.",
            )
        _raise_agent_deferral(result, step_name=step_name, defer_key=resolved_key)
    return result


def _raise_agent_deferral(
    result: RunResult[Any],
    *,
    step_name: str,
    defer_key: str,
) -> None:
    """Surface an agent-level deferral as :class:`FlowAgentDeferred`.

    Validates the Agent contract (``state is not None`` when
    ``requires_action`` is True) and the serialisability of the
    captured :class:`RunState`. Both failure modes raise
    :class:`FlowDefinitionError` so resume never sees an
    empty-or-invalid state.
    """
    if result.state is None:
        raise FlowDefinitionError(
            f"arun_flow_agent: agent run for step {step_name!r} returned "
            f"requires_action=True with state=None. The Agent contract "
            f"guarantees a populated RunState in this case; this is a "
            f"framework boundary violation.",
        )
    try:
        run_state_data = json.dumps(result.state.to_dict())
    except (TypeError, ValueError) as exc:
        raise FlowDefinitionError(
            f"arun_flow_agent: failed to serialise agent RunState for step "
            f"{step_name!r} defer_key={defer_key!r}: {type(exc).__name__}: {exc}. "
            f"Ensure the agent's ``context`` is JSON-encodable; non-serialisable "
            f"types (datetimes, sets, custom classes) must be normalised before "
            f"the deferral happens.",
        ) from exc
    raise FlowAgentDeferred(
        step_name=step_name,
        defer_key=defer_key,
        run_state_data=run_state_data,
    )


def _decode_pending_run_state(
    payload: str,
    *,
    step_name: str,
    defer_key: str,
) -> RunState:
    """Rehydrate a :class:`RunState` from the resume-time JSON payload.

    Surfaces malformed or tampered payloads as
    :class:`FlowDefinitionError` rather than silently producing a
    half-populated state.  Validates the load-bearing keys
    (``conversation_history``, ``current_agent_name``) so a stale or
    truncated payload cannot slip past the resume path.

    Imported lazily so the ``Runner`` import in :func:`arun_flow_agent`
    stays in one place.
    """
    from troopai.adk.run.state import RunState

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FlowDefinitionError(
            f"arun_flow_agent: pending agent resolution for step "
            f"{step_name!r} defer_key={defer_key!r} is not valid JSON: {exc}.",
        ) from exc
    if not isinstance(data, dict):
        raise FlowDefinitionError(
            f"arun_flow_agent: pending agent resolution for defer_key="
            f"{defer_key!r} must decode to a dict, got {type(data).__name__}.",
        )
    for required in ("conversation_history", "current_agent_name"):
        if required not in data:
            raise FlowDefinitionError(
                f"arun_flow_agent: pending agent resolution for defer_key="
                f"{defer_key!r} is missing required key {required!r}. "
                f"Keys present: {sorted(data.keys())!r}.",
            )
    return RunState.from_dict(data)


def _infer_calling_step_name(flow: Flow[Any]) -> str | None:
    """Look up the enclosing flow step's method name, validated against the registry.

    Walks the call stack one frame at a time looking for the first
    frame whose ``self`` is the supplied :class:`Flow` instance AND
    whose function name matches a registered step. Returns ``None``
    when no match is found — caller raises :class:`FlowDefinitionError`
    rather than guessing, so an un-resumable deferred step is never
    silently produced.

    Cross-checking against the registry prevents helper methods on
    the Flow subclass from being mistaken for the step (a Flow with
    `async def _call_agent(self)` calling :func:`arun_flow_agent`
    would otherwise capture ``_call_agent`` rather than the actual
    step name).
    """
    from troopai.adk.flows.flow import Flow as FlowClass

    registry = flow.get_registry()
    known = registry.starts | set(registry.listeners) | set(registry.routers)

    frame = inspect.currentframe()
    while frame is not None:
        local_self = frame.f_locals.get("self")
        if isinstance(local_self, FlowClass) and frame.f_code.co_name in known:
            return frame.f_code.co_name
        frame = frame.f_back
    return None
