"""LLM prompt templates for the memory pipeline.

Two-phase pipeline:

* **Extraction** (extract): system prompt + per-rollout user prompt.
  The LLM emits ``RolloutExtractionArtifacts`` (rollout_slug +
  rollout_summary + raw_memory). Prompt is signal-gated: if the
  rollout has no high-signal observations, the LLM SHOULD emit
  empty strings so the result can be silently dropped.
* **Consolidation** (consolidate): system prompt + selection-diff user
  prompt. The LLM reads + rewrites ``MEMORY.md`` and
  ``memory_summary.md`` inside the sandbox; it doesn't return
  structured output.

The templates are kept short on purpose: a long instruction block
costs prompt-cache space on every flush. Verbose guidance lives
in ``docs/sandbox/memory.md``; this module only contains the part
the LLM must see at run-time.
"""

from __future__ import annotations

from troopai.adk.sandbox.memory.storage import ConsolidationInputSelection

__all__ = [
    "MEMORY_CONSOLIDATION_SYSTEM_PROMPT_TEMPLATE",
    "MEMORY_READ_PROMPT_TEMPLATE",
    "ROLLOUT_EXTRACTION_SYSTEM_PROMPT_TEMPLATE",
    "ROLLOUT_EXTRACTION_USER_PROMPT_TEMPLATE",
    "render_memory_consolidation_prompt",
    "render_memory_read_prompt",
    "render_rollout_extraction_prompt",
    "render_rollout_extraction_user_prompt",
]


ROLLOUT_EXTRACTION_SYSTEM_PROMPT_TEMPLATE: str = """\
You extract durable signal from a single agent rollout (one run).
Emit a JSON object with three fields: ``rollout_slug``,
``rollout_summary``, ``raw_memory``.

* ``rollout_slug`` — file-safe ([a-z0-9_-]+, ≤80 chars) descriptor
  of the run's TASK. Examples: ``add-pii-redaction``, ``debug-flaky-test``.
* ``rollout_summary`` — a multi-line markdown summary that another
  agent could read to understand what happened: the user's
  request, the steps the agent took, what succeeded, what failed,
  what artifacts landed in the workspace.
* ``raw_memory`` — distilled high-signal observations the NEXT
  agent must know. Format: a short YAML frontmatter block (keys:
  rollout_id, updated_at, rollout_path, rollout_summary_file,
  terminal_state) followed by markdown bullets under "Preferences",
  "Reusable knowledge", "Failures", "References".

GATING RULES:
* No-op runs (the agent did nothing useful, errored on the first
  turn, or duplicated work covered by an earlier memory) — emit
  empty strings for all three fields. The pipeline will discard.
* Never include credentials, tokens, or raw user PII. Redact with
  ``<redacted>`` if you must reference the structure.
* Match newline style of the input.

{extra_prompt}\
"""


ROLLOUT_EXTRACTION_USER_PROMPT_TEMPLATE: str = """\
Terminal metadata (JSON):
{terminal_metadata_json}

---

Rollout contents (JSONL — one JSON object per turn):
{rollout_contents}

---

Emit the JSON object now.\
"""


MEMORY_CONSOLIDATION_SYSTEM_PROMPT_TEMPLATE: str = """\
You consolidate the agent's accumulated raw memories into a stable
durable handbook the next run can read.

Files you may read AND write inside the sandbox:
* ``{memory_root}/MEMORY.md`` — durable knowledge handbook. Group
  by task. Three subsections per task: "User preferences",
  "Reusable knowledge", "Failures".
* ``{memory_root}/memory_summary.md`` — short summary injected
  into the agent system prompt every run. Sections: "User Profile",
  "User preferences", "General Tips", "What's in Memory".
* ``{memory_root}/skills/<skill_name>/`` — optional reusable
  procedure folders. Each contains a ``SKILL.md`` (frontmatter
  with name + description + when_to_use; markdown body with the
  procedure).

Workflow:
1. Read the current MEMORY.md + memory_summary.md.
2. Read ``{memory_root}/raw_memories.md`` (the freshly-merged
   the current input).
3. Note the diff against the prior consolidation (see "Removed"
   below — entries no longer in scope).
4. Rewrite MEMORY.md and memory_summary.md in place. Add new
   high-signal items, drop or summarize removed items, leave
   stable items intact.
5. Optionally create / update a ``skills/`` packet for any
   distinct reusable procedure you can identify.

Selection diff:
  Retained (still in scope): {retained_count}
  Removed (evicted from scope): {removed_count}
{removed_block}

{extra_prompt}\
"""


MEMORY_READ_PROMPT_TEMPLATE: str = """\
You have durable cross-run memory at ``{memory_dir}``.
Summary (read this every run before you decide what to do):

---
{memory_summary}
---

When you spot an out-of-date entry mid-run AND ``live_update`` is
enabled, you MAY edit ``{memory_dir}/MEMORY.md`` /
``{memory_dir}/memory_summary.md`` directly via your shell or
``apply_patch`` tool.\
"""


def render_rollout_extraction_prompt(*, extra_prompt: str | None = None) -> str:
    """Render the extraction system prompt with an optional caller-supplied appendix."""
    appendix = (
        ""
        if extra_prompt is None or len(extra_prompt.strip()) == 0
        else f"\nADDITIONAL GUIDANCE:\n{extra_prompt.strip()}\n"
    )
    return ROLLOUT_EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(extra_prompt=appendix)


def render_rollout_extraction_user_prompt(
    *,
    terminal_metadata_json: str,
    rollout_contents: str,
) -> str:
    """Render the extraction user message with the rollout's metadata + JSONL body."""
    return ROLLOUT_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        terminal_metadata_json=terminal_metadata_json,
        rollout_contents=rollout_contents,
    )


def render_memory_consolidation_prompt(
    *,
    memory_root: str,
    selection: ConsolidationInputSelection,
    extra_prompt: str | None = None,
) -> str:
    """Render the consolidation system prompt — includes the selection diff."""
    appendix = (
        ""
        if extra_prompt is None or len(extra_prompt.strip()) == 0
        else f"\nADDITIONAL GUIDANCE:\n{extra_prompt.strip()}\n"
    )
    if len(selection.removed) == 0:
        removed_block = ""
    else:
        bullets = "\n".join(
            f"  - {item.rollout_id} ({item.terminal_state}) — {item.rollout_summary_file}" for item in selection.removed
        )
        removed_block = f"\nRemoved entries:\n{bullets}\n"
    return MEMORY_CONSOLIDATION_SYSTEM_PROMPT_TEMPLATE.format(
        memory_root=memory_root,
        retained_count=len(selection.retained_rollout_ids),
        removed_count=len(selection.removed),
        removed_block=removed_block,
        extra_prompt=appendix,
    )


def render_memory_read_prompt(*, memory_dir: str, memory_summary: str) -> str:
    """Render the prompt fragment injected into the agent's system prompt."""
    return MEMORY_READ_PROMPT_TEMPLATE.format(
        memory_dir=memory_dir,
        memory_summary=memory_summary.rstrip(),
    )
