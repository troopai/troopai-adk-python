# Cost-Aware Sandbox Selection

By default a run pins a specific backend via `SandboxRunConfig(client=...)`.
Cost-aware selection is an opt-in alternative: supply a `SandboxSelector`
and a list of `SandboxCandidate` objects and the framework chooses the backend
at run time based on capabilities and cost.

## When to use it

Use cost-aware selection when:

- Multiple backends are available (local, Docker, hosted) and you want the
  cheapest one that meets the run's requirements.
- Different deployment environments have different backends available, and
  the calling code should not hard-code a choice.
- You want to fall back to a free local backend when a priced hosted backend
  is unavailable or over-budget.

An explicit `client=` always beats the selector — use it when the backend
is fixed and selection overhead is unwanted.

## Core types

### `SandboxSelector`

```
SandboxSelector.select(
    candidates: list[SandboxCandidate],
    requirements: SandboxRequirements,
) -> SandboxCandidate
```

Abstract base class. `select` returns the chosen candidate or raises
`SandboxSelectionError` when no candidate satisfies the requirements.

### `CheapestFirstSelector`

The built-in selector. It:

1. Raises `SandboxSelectionError` immediately if `candidates` is empty.
2. Filters `candidates` to those whose `client.capabilities.satisfies(requirements)` is `True`.
3. Raises `SandboxSelectionError` if the filtered list is empty (none qualify).
4. Returns the candidate with the lowest `client.cost.rate_key()`.
   - `free=True` on a cost descriptor yields `rate_key() == 0`, which always wins.
   - A candidate whose `client.cost` is `None` sorts after all priced candidates
     (treated as `float("inf")`).
   - Ties are broken by position in the original list (the first qualifying
     candidate is preferred).

### `SandboxCandidate`

A frozen dataclass pairing a client with creation options:

```python
@dataclasses.dataclass(frozen=True)
class SandboxCandidate:
    client: BaseSandboxClient
    options: BaseSandboxClientOptions | None = None
```

`options` is `None` when the client supports default session options.

### `SandboxRequirements`

What a run needs from a backend. All fields default to their most
permissive value so an empty `SandboxRequirements()` matches every backend.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `network` | `bool` | `False` | Run needs outbound network access |
| `persistent` | `bool` | `False` | Run needs a persistent (non-ephemeral) workspace |
| `min_cpu` | `int \| None` | `None` | Minimum CPU count the backend must provide |
| `min_memory_mb` | `int \| None` | `None` | Minimum memory (MiB) the backend must provide |
| `region` | `str \| None` | `None` | Required region; must appear in the backend's `regions` list |

### `SandboxBackendCapabilities`

What a backend offers. Each backend client declares its capabilities as a
class attribute. Defaults are conservative (no network, ephemeral, limits
unknown) so a backend that declares nothing only satisfies an empty
requirement.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `network` | `bool` | `False` | Backend grants outbound network access |
| `persistent` | `bool` | `False` | Backend offers a persistent workspace |
| `max_cpu` | `int \| None` | `None` | Maximum CPU count (`None` = unknown) |
| `max_memory_mb` | `int \| None` | `None` | Maximum memory in MiB (`None` = unknown) |
| `regions` | `tuple[str, ...]` | `()` | Regions the backend can run in |

`capabilities.satisfies(requirements)` returns `True` when the backend
meets every field the requirements state.

## Config wiring

Set `selector`, `candidates`, and `requirements` on `SandboxRunConfig`.
The `candidates` list must be non-empty when a `selector` is provided —
`__post_init__` raises `ValueError` if it is empty.

```python
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.selector import CheapestFirstSelector, SandboxCandidate
from troopai.adk.types.sandbox.cost import SandboxRequirements

sandbox_config = SandboxRunConfig(
    selector=CheapestFirstSelector(),
    candidates=[pricey_e2b, free_local],
    requirements=SandboxRequirements(network=True),
)
```

## Acquisition precedence

The Runner resolves a session in this order, stopping at the first match:

1. `session` — caller provides a live session; the Runner reuses it.
2. `session_state` — Runner reconnects via `client.resume(state)`.
3. `client` — Runner creates a fresh session from the explicit client.
4. `selector` + `candidates` — Runner calls `selector.select(candidates, requirements)`
   and creates a fresh session from the chosen candidate.

An explicit `client=` therefore takes precedence over the selector.
Set only one of `client=` or `selector=`/`candidates=` per config.

## Example

```python
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.clients.hosted.e2b.e2b_client import E2bSandboxClient
from troopai.adk.sandbox.clients.local import LocalSubprocessSandboxClient
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.selector import CheapestFirstSelector, SandboxCandidate
from troopai.adk.types.sandbox.cost import SandboxRequirements

local_client = LocalSubprocessSandboxClient()
e2b_client = E2bSandboxClient()

candidates = [
    SandboxCandidate(client=e2b_client),   # $0.06/min, network=True, persistent=True
    SandboxCandidate(client=local_client), # free, network=True, persistent=False
]

requirements = SandboxRequirements(network=True)

# Offline: verify the selection result directly.
chosen = CheapestFirstSelector().select(candidates, requirements)
# chosen.client is local_client — free wins over $0.06/min

# Full run: selector drives backend acquisition inside Runner.arun.
agent = SandboxAgent(
    name="coder",
    system_prompt="You are a sandboxed coder.",
    capabilities=[ShellCapability()],
)

run_config = RunConfig(
    sandbox=SandboxRunConfig(
        selector=CheapestFirstSelector(),
        candidates=candidates,
        requirements=requirements,
    )
)

result = await Runner.arun(
    agent,
    "Run `echo hello` and report the output.",
    run_config=run_config,
)
```

## Custom selectors

Subclass `SandboxSelector` and implement `select`. A selector that prefers
a specific region, for example:

```python
from troopai.adk.sandbox.selector import SandboxCandidate, SandboxSelector
from troopai.adk.types.sandbox.cost import SandboxRequirements
from troopai.adk.exceptions.exceptions import SandboxSelectionError

class RegionPreferenceSelector(SandboxSelector):
    def __init__(self, preferred_region: str) -> None:
        self._preferred_region = preferred_region

    def select(
        self,
        candidates: list[SandboxCandidate],
        requirements: SandboxRequirements,
    ) -> SandboxCandidate:
        eligible = [
            c for c in candidates
            if c.client.capabilities.satisfies(requirements)
        ]
        if len(eligible) == 0:
            raise SandboxSelectionError("No candidate satisfies requirements")
        preferred = [
            c for c in eligible
            if self._preferred_region in c.client.capabilities.regions
        ]
        return preferred[0] if len(preferred) > 0 else eligible[0]
```

See [cost.md](cost.md) for the rate card types and per-backend rates.
See [observability.md](observability.md) for hooks and usage accumulation.
