(run/runner-profile)=

# Runner Profiles

`Runner.configure()` creates an immutable `RunnerProfile` for defaults you
reuse across multiple runs. A profile does not execute by itself. Bind it to a
target, then call the target runner:

```python
from troopai.adk import Runner

profile = (
    Runner.configure()
    .model("claude-haiku-4-5-20251001")
    .verbose()
    .limits(tokens=50_000)
    .context({"tenant": "acme"})
)

result = await profile.agent(support_agent).max_turns(6).arun("Help reset billing access.")
```

Profiles are additive to the existing direct APIs. Use
`Runner.arun(agent, ...)` for one-off calls; use a profile when several runs
share the same model, limits, tracing, tenant, context management, or context
value. Target runners delegate back to the corresponding `Runner` execution
classmethods, such as `Runner.arun`, `Runner.arun_swarm`,
`Runner.arun_graph`, `Runner.arun_task`, `Runner.arun_task_pipeline`,
`Runner.arun_task_group`, and `Runner.arun_flow`, so profiles do not introduce
a second execution path.

## Immutability

Every fluent call returns a new object:

```python
base = Runner.configure().model("claude-haiku-4-5-20251001")

loud = base.verbose()
quiet = base.verbose(enabled=False)
```

`base`, `loud`, and `quiet` can be reused independently. Reading
`profile.run_config` returns a copy, so mutating that value does not change the
profile.

## Targets

Bind the profile to the primitive you want to execute:

```python
agent_run = profile.agent(agent).max_turns(6)
swarm_run = profile.swarm(swarm).max_total_turns(30)
graph_run = profile.graph(graph).thread("case-123")
task_run = profile.task(task)
pipeline_run = profile.pipeline(pipeline)
group_run = profile.task_group(task_group)
flow_run = profile.flow(flow).config(flow_config)
```

Each target runner keeps target-specific options local:

```python
agent_result = await agent_run.arun("Draft the customer reply.")
swarm_result = await swarm_run.arun("Resolve the incident.")
graph_result = await graph_run.arun("Review the claim.")
task_output = await task_run.arun()
pipeline_result = await pipeline_run.arun()
group_result = await group_run.arun()
flow_result = await flow_run.arun()
```

Streaming stays explicit:

```python
agent_stream = await profile.agent(agent).arun("Write a summary.", stream=True)
graph_stream = await profile.graph(graph).arun("Start", stream=True)
task_stream = await profile.task(task).arun_streamed()
pipeline_stream = profile.pipeline(pipeline).arun_streamed()
flow_stream = profile.flow(flow).arun_streamed()
```

## Scope

`RunnerProfile` stores `RunConfig` defaults and the user context value. Hooks,
sessions, memory, checkpointers, graph thread ids, and flow configs stay on
target runners because their types and semantics differ by primitive:

```python
agent_runner = profile.agent(agent).hooks(run_hooks).session(session).memory(memory)
swarm_runner = profile.swarm(swarm).checkpointer(swarm_checkpointer)
graph_runner = profile.graph(graph).hooks([graph_hooks]).thread("thread-123")
flow_runner = profile.flow(flow).config(flow_config)
```

Flow execution uses `FlowConfig`, not `RunConfig`. A profile's context value is
passed to `Runner.arun_flow(...)`; profile `RunConfig` defaults are not
auto-injected into step bodies. For that reason, `FlowRunner` exposes
`.context(...)` and `.config(...)`, not `RunConfig` mutators such as
`.model(...)` or `.limits(...)`.

## Full Config

Use `with_config()` when a field does not have a convenience method:

```python
from troopai.adk import RunConfig, Runner

profile = Runner.configure().with_config(
    RunConfig(
        tracing_enabled=True,
        max_parallel_tools=3,
    )
)
```

Convenience methods copy into the profile's `RunConfig`:

```python
profile = (
    Runner.configure()
    .model("claude-haiku-4-5-20251001")
    .tracing(enabled=True, metadata={"service": "support"})
    .tenant("acme")
    .max_total_turns(40)
    .fail_on_tool_error(False)
)
```
