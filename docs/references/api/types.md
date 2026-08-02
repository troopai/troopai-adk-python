(references/api/types)=

# Types

Provider-agnostic wire and history types shared across the framework.

## Run items

```{eval-rst}
.. autodata:: troopai.adk.types.RunItem

.. autoclass:: troopai.adk.types.RunItemBase
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.UserItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.SystemItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MessageOutputItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ReasoningItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolCallItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolCallOutputItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolApprovalItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolSearchCallItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolSearchOutputItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.HandoffCallItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.HandoffOutputItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPListToolsItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPApprovalRequestItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPApprovalResponseItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.CompactionItem
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ItemHelpers
   :members:
   :show-inheritance:
   :exclude-members: type
```

## Result

`RunResult` is documented in prose because its source docstring does
not render through autodoc.

```{eval-rst}
.. py:class:: troopai.adk.types.RunResult

   Result of a completed (or interrupted) agent run. Contains the
   final output (if the run completed), all items generated during
   execution, and supports HITL interruptions via
   ``deferred_requests``.

   Fields:

   - ``final_output`` — the final output from the agent, or ``None``
     if interrupted for approval.
   - ``user_prompt`` — the original user prompt passed to the run.
   - ``new_items`` — Layer 3 ``RunItem`` values generated during this
     run (messages, tool calls, results).
   - ``context`` — the run context with usage tracking.
   - ``last_agent`` — the last agent that was active.
   - ``recovered`` — ``True`` when an error handler produced
     ``final_output`` after the run raised; recovered runs skip
     session and memory persistence.
   - ``deferred_requests`` — tools captured for approval or external
     execution; ``None`` if the run completed.
   - ``state`` — serializable state for resuming interrupted runs.
   - ``guardrail_results`` — per-phase agent-level guardrail audit
     trail (``input`` and ``output`` slots).
   - ``guardrail_audit`` — per-action guardrail audit records across
     every level (agent, tool, flow), captured as hashes, never raw
     payloads.
   - ``swarm_yield`` — set only by the swarm driver when an agent turn
     yielded control; ``None`` on every plain ``Runner.arun()`` path.
   - ``sandbox_usage`` — aggregate sandbox resource and cost usage,
     or ``None`` when no sandbox session ran.

   Members:

   - ``requires_action`` — property; ``True`` when human approval or
     external action is pending.
   - ``interruptions`` — property; tool calls awaiting human approval,
     as a flat list.
   - ``last_response_id`` — property; the ``response_id`` of the most
     recent LLM response in this run, or ``None``.
   - ``release_agents(*, release_new_items=True)`` — drop strong
     references to agents and, optionally, run items.
   - ``to_input_list()`` — convert to a Layer 1 input list for a
     continued conversation.
   - ``final_output_as(output_type)`` — cast the final output to the
     expected type.

   .. rubric:: Example

   .. code-block:: python

      result = await Runner.arun(agent, "Delete user 123")
      if result.requires_action:
          for req in list(result.deferred_requests.approvals):
              if await confirm(f"Approve {req.tool_name}?"):
                  result.state.approve(req)
              else:
                  result.state.reject(req, "Denied")
          result = await Runner.arun(agent, result.state)
```

## Built-in tool call and result types

```{eval-rst}
.. autoclass:: troopai.adk.types.WebSearchToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.WebSearchToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.WebSearchResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.FileSearchToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.FileSearchToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.FileSearchResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.CodeInterpreterToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.CodeInterpreterToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.CodeInterpreterOutput
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ComputerToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ComputerToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ComputerAction
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ImageGenerationToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ImageGenerationToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ShellToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ShellToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ApplyPatchToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ApplyPatchToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolSearchToolCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolSearchToolCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ToolSearchResultEntry
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPListTools
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPListToolsTool
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPCall
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPCallResult
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPApprovalRequest
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.MCPApprovalResponse
   :members:
   :show-inheritance:
   :exclude-members: type
```

## Tracing span data

```{eval-rst}
.. autoclass:: troopai.adk.types.SpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.AgentSpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.FunctionSpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.GenerationSpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.GuardrailSpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.HandoffSpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.ResponseSpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.types.CustomSpanData
   :members:
   :show-inheritance:
   :exclude-members: type

.. autodata:: troopai.adk.types.AnySpanData
```

How the type layers fit together is explained in the
[Types guide](../../types/types.md).
