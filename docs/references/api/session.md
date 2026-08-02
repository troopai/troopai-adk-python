(references/api/session)=

# Session

Conversation persistence for agent runs.

## Core

```{eval-rst}
.. autoclass:: troopai.adk.session.Session
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.session.SessionSettings
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.session.SessionEvent
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.session.create_session_event

.. autoclass:: troopai.adk.session.State
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.session.SessionStore
   :members:
   :show-inheritance:
```

## Implementations

```{eval-rst}
.. autoclass:: troopai.adk.session.BufferedSession
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.session.SQLiteSession
   :members:
   :show-inheritance:
```

## Multi-session stores

```{eval-rst}
.. autoclass:: troopai.adk.session.MultiSessions
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.session.SQLiteMultiSessions
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.session.SessionInfo
   :members:
   :show-inheritance:
```

Usage lives in the [Session guide](../../session/session.md).
