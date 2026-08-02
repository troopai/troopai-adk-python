# Sandbox Examples

| File | What it shows |
|---|---|
| `local_subprocess_basic.py` | Minimal: SandboxAgent + ShellCapability + LocalSubprocess backend |
| `manifest_materialization.py` | Manifest `File`/`Dir`/`LocalFile` materialized into the workspace before the agent loop; least-privilege `SandboxPathGrant` for the host copy — no API key |

Each example loads `.env` first via `python-dotenv` (optional extra).

To run:

```bash
python examples/sandbox/local_subprocess_basic.py
```
