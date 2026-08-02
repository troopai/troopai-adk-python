# A2A Examples

Runnable examples for the Agent-to-Agent (A2A) protocol integration.

| Example | Purpose |
|---|---|
| `client_basic.py` | Minimal client: call a remote A2A agent, print the result. |
| `client_streaming.py` | Stream incremental text deltas from a remote agent. |
| `client_background.py` | Submit a long-running task; poll for completion via the typed continuation token. |
| `client_as_tool.py` | Wrap a remote agent as a `FunctionTool` and let a local agent's LLM invoke it mid-turn. |
| `client_in_graph.py` | Place a remote A2A agent as a node inside a local `Graph` alongside local nodes. |
| `server_basic.py` | Expose a local Agent over A2A; full uvicorn boot with manually-authored AgentCard. |

Install the optional extra first:

```bash
pip install 'troopai-adk-python[a2a]'
```

Each script is self-contained and runs with `python examples/a2a/<file>.py`.
The client examples assume a server is reachable at the URL passed
on the command line (default: `http://localhost:8080`); start
`server_basic.py` in another terminal first.

See `docs/a2a/a2a.md` for the user-facing walkthrough.
