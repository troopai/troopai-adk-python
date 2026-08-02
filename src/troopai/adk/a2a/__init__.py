"""Agent-to-Agent (A2A) protocol support for the TroopAI ADK.

The A2A protocol (https://a2a-protocol.org/) standardises how independent
autonomous agents talk to each other as peers — distinct from MCP, which
standardises how an agent talks to its tools.

This package ships two halves of the same protocol:

* **Client side** — :class:`A2AAgent` is a ``BaseAgent`` peer-class
  that wraps a remote A2A endpoint as **pure config**. Execute via
  :class:`A2ARunner` for direct peer-style orchestration, or use
  :meth:`A2AAgent.as_tool` to plug the remote into a local Agent's
  ``tools`` list (the same dual surface ``Agent`` itself has).
* **Server side** — :class:`A2AServer` wraps a local Agent +
  developer-authored ``AgentCard`` into a config object that
  :func:`build_starlette_app` turns into an ASGI app the developer's
  own ``uvicorn`` (or hypercorn / granian) instance serves.

Long-running tasks are first-class via :class:`A2AContinuationToken`:
``await A2ARunner.arun(agent, prompt, background=True)`` returns the
typed token; ``await A2ARunner.arun(agent, prompt, continuation_token=token)``
resumes from any process, and ``await A2ARunner.poll_task(agent, token)``
returns a one-shot status snapshot without waiting.

The ``a2a-sdk`` package is an optional extra. Install with::

    pip install 'troopai-adk-python[a2a]'

When the extra is missing, every public name in this module is ``None``;
downstream code can branch on ``A2AAgent is None`` to skip A2A wiring
gracefully. The mechanism mirrors :mod:`troopai.adk.mcp` exactly.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-check-time: assume the dep is installed so static analysis
    # narrows on the real types. Downstream code that wants to detect
    # missing extras at runtime should compare against ``None``
    # explicitly.
    from .a2a_agent import A2AAgent, A2ARunResult, A2AStreamEvent
    from .a2a_client import A2AClient
    from .a2a_continuation_token import (
        A2AContinuationToken,
        A2ATaskStateLiteral,
        A2ATaskStatus,
    )
    from .a2a_runner import A2ARunner
    from .adapters import A2AExecutableAdapter
    from .app_factory import build_starlette_app
    from .exceptions import (
        A2AError,
        A2AProtocolError,
        A2ATaskCancelledError,
        A2ATaskError,
        A2ATaskInterruptedError,
        A2ATransportError,
    )
    from .executor import A2AExecutor
    from .server import A2AServer
    from .task_store import InMemoryTaskStore, SQLiteTaskStore, TaskStore
else:
    try:
        from .a2a_agent import A2AAgent, A2ARunResult, A2AStreamEvent
        from .a2a_client import A2AClient
        from .a2a_continuation_token import (
            A2AContinuationToken,
            A2ATaskStateLiteral,
            A2ATaskStatus,
        )
        from .a2a_runner import A2ARunner
        from .adapters import A2AExecutableAdapter
        from .app_factory import build_starlette_app
        from .exceptions import (
            A2AError,
            A2AProtocolError,
            A2ATaskCancelledError,
            A2ATaskError,
            A2ATaskInterruptedError,
            A2ATransportError,
        )
        from .executor import A2AExecutor
        from .server import A2AServer
        from .task_store import InMemoryTaskStore, SQLiteTaskStore, TaskStore
    except ImportError as _exc:
        # Only swallow "top-level a2a package not installed". Any other
        # ImportError (a typo inside our modules, a transitive dep
        # blowing up) MUST surface rather than be masked as "extra
        # missing". ``ModuleNotFoundError.name`` holds the exact
        # top-level module the interpreter failed to find — much more
        # reliable than substring matching the exception message.
        if getattr(_exc, "name", None) != "a2a":
            raise
        A2AAgent = None
        A2ARunResult = None
        A2AStreamEvent = None
        A2AClient = None
        A2AContinuationToken = None
        A2ATaskStateLiteral = None
        A2ATaskStatus = None
        A2AError = None
        A2AProtocolError = None
        A2ATaskCancelledError = None
        A2ATaskError = None
        A2ATaskInterruptedError = None
        A2ATransportError = None
        A2AExecutor = None
        A2AExecutableAdapter = None
        A2ARunner = None
        A2AServer = None
        build_starlette_app = None
        InMemoryTaskStore = None
        SQLiteTaskStore = None
        TaskStore = None

__all__ = [
    "A2AAgent",
    "A2AClient",
    "A2AContinuationToken",
    "A2AError",
    "A2AExecutableAdapter",
    "A2AExecutor",
    "A2AProtocolError",
    "A2ARunResult",
    "A2ARunner",
    "A2AServer",
    "A2AStreamEvent",
    "A2ATaskCancelledError",
    "A2ATaskError",
    "A2ATaskInterruptedError",
    "A2ATaskStateLiteral",
    "A2ATaskStatus",
    "A2ATransportError",
    "InMemoryTaskStore",
    "SQLiteTaskStore",
    "TaskStore",
    "build_starlette_app",
]
