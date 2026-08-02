---
name: add-run-item
description: Procedure to add a new RunItem (Layer 3 conversation-history entry) to the TroopAI ADK. Use when modeling a new kind of turn artifact (tool call, message, handoff, provider-native output, etc.).
---

# Add a RunItem

Constraints live in `.claude/rules/items.md` and `type-layers.md` (load
when you edit `types/items/`). This is the ordered procedure.

## 1. Define the item

In `src/troopai/adk/types/items/`:

- Subclass `RunItemBase[T]` with a concrete `raw: T` (the Layer 1
  wire-format snapshot). `raw` is REQUIRED and NEVER `None` — absence
  belongs in the union (`Item | None`), not inside an item.
- Declare a unique `type: Literal["..."]` discriminator (unique across
  ALL items).
- `@dataclass(frozen=True)`. Extra framework fields (`source`, `target`,
  `approved`, `agent_name`) are allowed alongside `raw`.
- NO convenience properties and NO extracted fields that duplicate
  `raw` — transformation logic lives in `ItemHelpers`
  (`types/items/items.py`), accessed via `raw` attribute access only
  (never `.get()` / `[]`).

## 2. Implement `to_param()`

Return the matching `LLMInputContentItem` (Layer 1). This is how the
item replays into the next turn's input.

## 3. Add to the union

Add the new class to the `RunItem` union in
`src/troopai/adk/types/items/items.py`. Update any `ItemHelpers`
conversion (`response_to_run_items`, `run_items_to_params`,
`message_to_run_items`) that must now produce/consume it.

## 4. Layer discipline

The item is Layer 3; `raw` is Layer 1. NEVER let a `types/chat/`
(Layer 2 wire) type into the item or any developer-facing surface —
conversion stays inside the LLM implementation.

## 5. Complete + verify

Tests in `tests/unit/` exercising construction, `to_param()`
round-trip, and `ItemHelpers` mapping; update `types/items/CLAUDE.md`'s
item listing. Run the `code-hygiene-gate` skill. No `NotImplementedError`.
