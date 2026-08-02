(references/api/a2a)=

# A2A

Agent-to-Agent protocol support: independent agents talking to each other
as peers, as protocol clients and as protocol servers.

The `a2a-sdk` package is an optional extra
(`pip install 'troopai-adk-python[a2a]'`). When it is not installed,
every name below is bound to `None` so downstream code can skip A2A
wiring gracefully.

## Client side

```{eval-rst}
.. autoclass:: troopai.adk.a2a.A2AAgent
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2ARunner
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2AClient
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2ARunResult
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2AStreamEvent
   :members:
   :show-inheritance:
   :exclude-members: type
```

## Server side

```{eval-rst}
.. autoclass:: troopai.adk.a2a.A2AServer
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2AExecutor
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.a2a.build_starlette_app
```

## Long-running tasks

```{eval-rst}
.. autoclass:: troopai.adk.a2a.A2AContinuationToken
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2ATaskStatus
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.a2a.A2ATaskStateLiteral

.. autoclass:: troopai.adk.a2a.TaskStore
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.InMemoryTaskStore
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.SQLiteTaskStore
   :members:
   :show-inheritance:
```

## Composition

```{eval-rst}
.. autoclass:: troopai.adk.a2a.A2AExecutableAdapter
   :members:
   :show-inheritance:
```

## Exceptions

```{eval-rst}
.. autoclass:: troopai.adk.a2a.A2AError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2AProtocolError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2ATransportError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2ATaskError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2ATaskCancelledError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.a2a.A2ATaskInterruptedError
   :members:
   :show-inheritance:
```

Usage lives in the [A2A guide](../../a2a/a2a.md).
