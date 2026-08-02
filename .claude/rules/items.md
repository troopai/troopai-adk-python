---
paths:
  - "src/troopai/adk/types/items/**/*.py"
  - "src/troopai/adk/items/**/*.py"
---

# Items Layer — CRITICAL

- Every item MUST have `raw: T`. `raw` MUST NEVER be `None` — items model
  wire-format snapshots; absence belongs in the union (`Item | None`), NOT
  inside an item.
- NEVER add extracted fields that duplicate `raw` — access data via `raw`.
- NEVER add convenience properties on item classes — transformation logic
  lives in `ItemHelpers` (keeps it type-checked and replaceable; properties
  couple presentation to storage).
- NEVER dict-like access (`.get()`, `[]`, `.items()`) — use `raw` attribute
  access.

## Adding a New Item

1. Subclass `RunItemBase[T]` with a concrete raw type.
2. Declare a unique `type: Literal["..."]` discriminator.
3. Implement `to_param()` returning `LLMInputContentItem`.
4. Add to the `RunItem` union.

Extra framework fields (`source`, `target`, `approved`) are allowed
alongside `raw`. A `None` raw means we lost the source — that's a bug, not a
state to model.

## Self-Check

1. `raw: T` declared (non-Optional)?
2. Unique `type: Literal["..."]` discriminator?
3. `to_param()` implemented and added to the `RunItem` union?
