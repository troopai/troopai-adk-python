"""Tests for JITContextAwareTool, NoteStore, and HistoryAwareToolContext."""

import asyncio
from unittest.mock import MagicMock

from troopai.adk.context.directives import DirectiveStore
from troopai.adk.tools.builtin.jit_context_aware_tool import (
    ContextStatsContextAwareTool,
    InMemoryNoteStore,
    JITContextAwareTool,
    ManageContextAwareTool,
    RecallNotesContextAwareTool,
    SaveNoteContextAwareTool,
    SearchHistoryContextAwareTool,
    _extract_text_from_item,
    _item_type_to_role,
)
from troopai.adk.tools.tool_context import (
    ExecutionAwareToolContext,
    HistoryAwareToolContext,
    ToolContext,
)


def _make_jit_tools(
    store: InMemoryNoteStore | None = None,
    search: bool = True,
    stats: bool = True,
) -> list[JITContextAwareTool]:
    """Test helper: create JIT tools with shared store and directives."""
    effective_store = store or InMemoryNoteStore()
    shared_directives = DirectiveStore()
    tools: list[JITContextAwareTool] = [
        SaveNoteContextAwareTool(store=effective_store),
        RecallNotesContextAwareTool(store=effective_store),
        ManageContextAwareTool(store=effective_store),
    ]
    if search:
        tools.append(SearchHistoryContextAwareTool(store=effective_store))
    if stats:
        tools.append(ContextStatsContextAwareTool(store=effective_store))
    for t in tools:
        object.__setattr__(t, "directives", shared_directives)
    return tools


# =====================================================================
# NoteStore tests
# =====================================================================


class TestInMemoryNoteStore:
    """Tests for InMemoryNoteStore."""

    def test_save_and_recall(self) -> None:
        store = InMemoryNoteStore()
        entry = store.save("key1", "content1", 3, turn=1)
        assert entry.key == "key1"
        assert entry.content == "content1"
        assert entry.importance == 3
        assert entry.turn == 1

        notes = store.recall()
        assert len(notes) == 1
        assert notes[0].key == "key1"

    def test_save_upsert(self) -> None:
        store = InMemoryNoteStore()
        store.save("key1", "original", 3, turn=1)
        store.save("key1", "updated", 5, turn=2)

        notes = store.recall()
        assert len(notes) == 1
        assert notes[0].content == "updated"
        assert notes[0].importance == 5
        assert notes[0].turn == 2

    def test_recall_with_query(self) -> None:
        store = InMemoryNoteStore()
        store.save("user_prefs", "prefers dark mode", 4, turn=1)
        store.save("task_progress", "step 2 of 5 done", 3, turn=2)
        store.save("api_key_note", "key stored in vault", 5, turn=3)

        results = store.recall("user")
        assert len(results) == 1
        assert results[0].key == "user_prefs"

        results = store.recall("step")
        assert len(results) == 1
        assert results[0].key == "task_progress"

    def test_recall_query_case_insensitive(self) -> None:
        store = InMemoryNoteStore()
        store.save("Important", "Critical finding", 5, turn=1)

        results = store.recall("important")
        assert len(results) == 1

        results = store.recall("CRITICAL")
        assert len(results) == 1

    def test_recall_empty(self) -> None:
        store = InMemoryNoteStore()
        assert store.recall() == []
        assert store.recall("query") == []

    def test_recall_sorted_by_importance(self) -> None:
        store = InMemoryNoteStore()
        store.save("low", "low priority", 1, turn=1)
        store.save("high", "high priority", 5, turn=2)
        store.save("mid", "medium priority", 3, turn=3)

        notes = store.recall()
        assert [n.key for n in notes] == ["high", "mid", "low"]

    def test_delete(self) -> None:
        store = InMemoryNoteStore()
        store.save("key1", "content", 3, turn=1)
        assert store.delete("key1") is True
        assert store.count() == 0
        assert store.delete("key1") is False

    def test_count(self) -> None:
        store = InMemoryNoteStore()
        assert store.count() == 0
        store.save("a", "content", 3, turn=1)
        assert store.count() == 1
        store.save("b", "content", 3, turn=2)
        assert store.count() == 2

    def test_keys(self) -> None:
        store = InMemoryNoteStore()
        store.save("alpha", "a", 3, turn=1)
        store.save("beta", "b", 3, turn=2)
        assert sorted(store.keys()) == ["alpha", "beta"]

    def test_max_notes_eviction(self) -> None:
        store = InMemoryNoteStore(capacity=3)
        store.save("a", "content", 1, turn=1)
        store.save("b", "content", 3, turn=2)
        store.save("c", "content", 5, turn=3)
        store.save("d", "content", 4, turn=4)  # Should evict "a" (lowest importance)

        assert store.count() == 3
        keys = store.keys()
        assert "a" not in keys
        assert "b" in keys
        assert "c" in keys
        assert "d" in keys

    def test_importance_clamped(self) -> None:
        store = InMemoryNoteStore()
        entry = store.save("low", "test", 0, turn=1)
        assert entry.importance == 1

        entry = store.save("high", "test", 10, turn=2)
        assert entry.importance == 5


# =====================================================================
# HistoryAwareToolContext tests
# =====================================================================


class TestHistoryAwareToolContext:
    """Tests for HistoryAwareToolContext."""

    def test_isinstance_chain(self) -> None:
        ctx = HistoryAwareToolContext(
            tool_name="search",
            tool_call_id="call_1",
            tool_arguments={},
            raw_arguments="{}",
        )
        assert isinstance(ctx, HistoryAwareToolContext)
        assert isinstance(ctx, ExecutionAwareToolContext)
        assert isinstance(ctx, ToolContext)

    def test_with_context_preserves_history(self) -> None:
        history = (MagicMock(), MagicMock())
        ctx = HistoryAwareToolContext(
            tool_name="search",
            tool_call_id="call_1",
            tool_arguments={},
            raw_arguments="{}",
            context="old",
            turns=3,
            messages=10,
            tokens=5000,
            history=history,
        )
        new_ctx = ctx.with_context("new")
        assert isinstance(new_ctx, HistoryAwareToolContext)
        assert new_ctx.context == "new"
        assert new_ctx.history is history
        assert new_ctx.turns == 3
        assert new_ctx.tokens == 5000

    def test_history_defaults_to_empty_tuple(self) -> None:
        ctx = HistoryAwareToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
        )
        assert ctx.history == ()

    def test_history_is_tuple(self) -> None:
        ctx = HistoryAwareToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
            history=(MagicMock(),),
        )
        assert isinstance(ctx.history, tuple)


# =====================================================================
# JITContextAwareTool expansion tests
# =====================================================================


class TestJITContextAwareToolExpansion:
    """Tests for JITContextAwareTool.tools."""

    def test_default_config_five_tools(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        assert len(tools) == 5
        names = [t.name for t in tools]
        assert "save_note" in names
        assert "recall_notes" in names
        assert "manage_context" in names
        assert "search_history" in names
        assert "context_stats" in names

    def test_no_stats_four_tools(self) -> None:
        tools = _make_jit_tools(stats=False)
        # tools already set
        assert len(tools) == 4
        names = [t.name for t in tools]
        assert "context_stats" not in names
        assert "manage_context" in names

    def test_no_history_search_four_tools(self) -> None:
        tools = _make_jit_tools(search=False)
        # tools already set
        assert len(tools) == 4
        names = [t.name for t in tools]
        assert "search_history" not in names
        assert "manage_context" in names

    def test_minimal_config_three_tools(self) -> None:
        tools = _make_jit_tools(stats=False, search=False)
        # tools already set
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "save_note" in names
        assert "recall_notes" in names
        assert "manage_context" in names

    def test_search_history_is_history_aware(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        search = next(t for t in tools if t.name == "search_history")
        assert search.history_aware is True

    def test_context_stats_is_execution_aware(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        stats = next(t for t in tools if t.name == "context_stats")
        assert stats.execution_aware is True
        assert stats.history_aware is False

    def test_save_note_is_basic_context(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        save = next(t for t in tools if t.name == "save_note")
        assert save.execution_aware is False
        assert save.history_aware is False

    def test_subclasses_builtin_tool(self) -> None:
        from troopai.adk.tools.builtin.builtin_tool import BuiltinTool

        tools = _make_jit_tools()
        assert isinstance(tools[0], BuiltinTool)

    def test_custom_note_store(self) -> None:
        custom_store = InMemoryNoteStore(capacity=10)
        tools = _make_jit_tools(store=custom_store)
        assert tools[0].store is custom_store

    def test_tools_share_note_store(self) -> None:
        """All generated tools should operate on the same NoteStore."""
        tools = _make_jit_tools()
        # tools already set

        # Save a note via save_note handler
        save = next(t for t in tools if t.name == "save_note")
        recall = next(t for t in tools if t.name == "recall_notes")

        ctx = ToolContext(
            tool_name="save_note",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )

        result = asyncio.run(save.on_invoke(ctx, '{"key":"test","content":"hello"}'))
        assert "Saved note 'test'" in result

        result = asyncio.run(recall.on_invoke(ctx, "{}"))
        assert "test" in result
        assert "hello" in result


# =====================================================================
# Handler tests
# =====================================================================


class TestSaveNoteHandler:
    """Tests for save_note handler."""

    def test_save_basic(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        save = next(t for t in tools if t.name == "save_note")
        ctx = ToolContext(
            tool_name="save_note",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )
        result = asyncio.run(save.on_invoke(ctx, '{"key":"k1","content":"hello"}'))
        assert "Saved note 'k1'" in result
        assert "1 note(s)" in result

    def test_save_with_importance(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        save = next(t for t in tools if t.name == "save_note")
        ctx = ToolContext(
            tool_name="save_note",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )
        result = asyncio.run(save.on_invoke(ctx, '{"key":"k1","content":"hello","importance":5}'))
        assert "importance: 5" in result


class TestRecallNotesHandler:
    """Tests for recall_notes handler."""

    def test_recall_empty(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        recall = next(t for t in tools if t.name == "recall_notes")
        ctx = ToolContext(
            tool_name="recall_notes",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )
        result = asyncio.run(recall.on_invoke(ctx, "{}"))
        assert "No notes saved yet" in result

    def test_recall_with_notes(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        save = next(t for t in tools if t.name == "save_note")
        recall = next(t for t in tools if t.name == "recall_notes")
        ctx = ToolContext(
            tool_name="t",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )
        asyncio.run(save.on_invoke(ctx, '{"key":"prefs","content":"dark mode"}'))
        result = asyncio.run(recall.on_invoke(ctx, "{}"))
        assert "prefs" in result
        assert "dark mode" in result

    def test_recall_with_query_no_match(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        save = next(t for t in tools if t.name == "save_note")
        recall = next(t for t in tools if t.name == "recall_notes")
        ctx = ToolContext(
            tool_name="t",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )
        asyncio.run(save.on_invoke(ctx, '{"key":"a","content":"hello"}'))
        result = asyncio.run(recall.on_invoke(ctx, '{"query":"xyz"}'))
        assert "No notes matching 'xyz'" in result


class TestSearchHistoryHandler:
    """Tests for search_history handler."""

    def test_search_with_matches(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        search = next(t for t in tools if t.name == "search_history")

        # Create mock RunItems with type and raw attributes
        item1 = MagicMock()
        item1.type = "user"
        item1.raw = {"content": "Tell me about quantum computing"}

        item2 = MagicMock()
        item2.type = "message_output"
        item2.raw = MagicMock()
        item2.raw.content = "Quantum computing uses qubits..."

        ctx = HistoryAwareToolContext(
            tool_name="search_history",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            history=(item1, item2),
        )
        result = asyncio.run(search.on_invoke(ctx, '{"pattern":"quantum"}'))
        assert "quantum" in result.lower()
        assert "2 match" in result

    def test_search_no_matches(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        search = next(t for t in tools if t.name == "search_history")

        item = MagicMock()
        item.type = "user"
        item.raw = {"content": "Hello world"}

        ctx = HistoryAwareToolContext(
            tool_name="search_history",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            history=(item,),
        )
        result = asyncio.run(search.on_invoke(ctx, '{"pattern":"xyz"}'))
        assert "No messages matching" in result

    def test_search_empty_history(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        search = next(t for t in tools if t.name == "search_history")

        ctx = HistoryAwareToolContext(
            tool_name="search_history",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            history=(),
        )
        result = asyncio.run(search.on_invoke(ctx, '{"pattern":"test"}'))
        assert "No conversation history" in result

    def test_search_missing_pattern(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        search = next(t for t in tools if t.name == "search_history")

        ctx = HistoryAwareToolContext(
            tool_name="search_history",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            history=(),
        )
        result = asyncio.run(search.on_invoke(ctx, "{}"))
        assert "Error" in result


class TestContextStatsHandler:
    """Tests for context_stats handler."""

    def test_stats_output(self) -> None:
        tools = _make_jit_tools()
        # tools already set
        stats = next(t for t in tools if t.name == "context_stats")

        usage = MagicMock()
        usage.input_tokens = 10000
        usage.output_tokens = 5000

        ctx = ExecutionAwareToolContext(
            tool_name="context_stats",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            usage=usage,
            turns=5,
            messages=20,
            tokens=45000,
        )
        result = asyncio.run(stats.on_invoke(ctx, "{}"))
        assert "45,000" in result
        assert "Turns completed: 5" in result
        assert "Messages in history: 20" in result
        assert "Notes saved: 0" in result
        assert "15,000" in result  # total = 10000 + 5000


# =====================================================================
# manage_context handler tests
# =====================================================================


class TestManageContextHandler:
    """Tests for manage_context handler."""

    def test_compact_directive(self) -> None:
        from troopai.adk.context.directives import CompactDirective

        tools = _make_jit_tools()
        manage = next(t for t in tools if t.name == "manage_context")
        # Use ExecutionAwareToolContext so msg_count > 1 (guard requires >=2 messages)
        ctx = ExecutionAwareToolContext(
            tool_name="manage_context",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            usage=None,
            turns=0,
            messages=10,
            tokens=0,
        )
        result = asyncio.run(manage.on_invoke(ctx, '{"action":"compact","preserve":5}'))
        assert "Scheduled" in result
        assert "compact" in result
        assert "5" in result
        assert manage.directives.count == 1
        directive = manage.directives.consume()[0]
        assert isinstance(directive, CompactDirective)
        assert directive.preserve == 5

    def test_drop_directive(self) -> None:
        from troopai.adk.context.directives import DropDirective

        tools = _make_jit_tools()
        manage = next(t for t in tools if t.name == "manage_context")
        # Use ExecutionAwareToolContext so msg_count > 1 (guard requires >=2 messages)
        ctx = ExecutionAwareToolContext(
            tool_name="manage_context",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            usage=None,
            turns=0,
            messages=10,
            tokens=0,
        )
        result = asyncio.run(manage.on_invoke(ctx, '{"action":"drop","preserve":3}'))
        assert "Scheduled" in result
        assert "drop" in result
        assert manage.directives.count == 1
        directive = manage.directives.consume()[0]
        assert isinstance(directive, DropDirective)
        assert directive.preserve == 3

    def test_unknown_action(self) -> None:
        tools = _make_jit_tools()
        manage = next(t for t in tools if t.name == "manage_context")
        # Use ExecutionAwareToolContext so msg_count > 1 (guard requires >=2 messages)
        ctx = ExecutionAwareToolContext(
            tool_name="manage_context",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            usage=None,
            turns=0,
            messages=10,
            tokens=0,
        )
        result = asyncio.run(manage.on_invoke(ctx, '{"action":"unknown","preserve":5}'))
        assert "Error" in result
        assert manage.directives.count == 0

    def test_directives_shared_across_tools(self) -> None:
        from troopai.adk.context.directives import DirectiveStore

        tools = _make_jit_tools()
        manage = next(t for t in tools if t.name == "manage_context")
        assert isinstance(manage.directives, DirectiveStore)
        # All tools share the same directives
        for t in tools:
            assert t.directives is manage.directives


# =====================================================================
# FunctionTool.history_aware flag tests
# =====================================================================


class TestFunctionToolHistoryAwareFlag:
    """Tests for the history_aware flag on FunctionTool."""

    def test_history_aware_implies_execution_aware(self) -> None:
        from troopai.adk.tools.function_tool import FunctionTool

        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object", "properties": {}},
            history_aware=True,
        )
        assert tool.history_aware is True
        assert tool.execution_aware is True

    def test_default_false(self) -> None:
        from troopai.adk.tools.function_tool import FunctionTool

        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object", "properties": {}},
        )
        assert tool.history_aware is False


# =====================================================================
# Helper function tests
# =====================================================================


class TestEvictTiebreaker:
    """Eviction tiebreaker must drop the OLDEST note within the lowest tier."""

    def test_evict_removes_oldest_among_equal_importance(self) -> None:
        """When several notes tie at the lowest importance, the oldest is evicted."""
        from troopai.adk.tools.builtin.jit_context_aware_tool import NoteEntry

        store = InMemoryNoteStore(capacity=3)
        # Three notes, all importance=1, with explicit ascending created_at so
        # timing jitter cannot affect ordering. "oldest" has the smallest stamp.
        store._notes["oldest"] = NoteEntry("oldest", "a", importance=1, created_at=100.0, turn=1)
        store._notes["middle"] = NoteEntry("middle", "b", importance=1, created_at=200.0, turn=2)
        store._notes["newest"] = NoteEntry("newest", "c", importance=1, created_at=300.0, turn=3)

        store._evict()

        keys = store.keys()
        # The OLDEST low-importance note must be removed, freshest kept.
        assert "oldest" not in keys
        assert "middle" in keys
        assert "newest" in keys

    def test_evict_prefers_lowest_importance_then_oldest(self) -> None:
        """Importance dominates; created_at only breaks ties within a tier."""
        from troopai.adk.tools.builtin.jit_context_aware_tool import NoteEntry

        store = InMemoryNoteStore(capacity=3)
        # A newer-but-lower-importance note must be evicted before an older
        # higher-importance one.
        store._notes["old_high"] = NoteEntry("old_high", "a", importance=5, created_at=100.0, turn=1)
        store._notes["new_low"] = NoteEntry("new_low", "b", importance=1, created_at=300.0, turn=2)

        store._evict()

        keys = store.keys()
        assert "new_low" not in keys
        assert "old_high" in keys


class TestDetectJitDirectivesOrdering:
    """_detect_jit_directives must find the store that actually receives writes."""

    def test_manage_context_not_first_still_detected(self) -> None:
        """manage_context placed after another JIT tool must not be a silent no-op."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.loop import _detect_jit_directives

        shared_store = InMemoryNoteStore()
        save = SaveNoteContextAwareTool(store=shared_store)
        manage = ManageContextAwareTool(store=shared_store)
        # Natural developer ordering: manage_context is NOT first, and each tool
        # owns its own independent DirectiveStore (no shared-store workaround).
        agent = Agent(name="ctx-agent", system_prompt="manage your context", tools=[save, manage])

        # The LLM schedules a compact via manage_context.
        ctx = ExecutionAwareToolContext(
            tool_name="manage_context",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
            usage=None,
            turns=0,
            messages=10,
            tokens=0,
        )
        result = asyncio.run(manage.on_invoke(ctx, '{"action":"compact","preserve":5}'))
        assert "Scheduled" in result

        # The Runner watches whatever _detect_jit_directives returns. It must be
        # the store that received the directive, not SaveNote's empty store.
        detected = _detect_jit_directives(agent)
        assert detected is manage.directives
        assert detected is not None
        assert detected.count == 1

    def test_falls_back_to_first_jit_when_no_manage_tool(self) -> None:
        """Agents without a ManageContextAwareTool still resolve a JIT store."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.loop import _detect_jit_directives

        store = InMemoryNoteStore()
        save = SaveNoteContextAwareTool(store=store)
        recall = RecallNotesContextAwareTool(store=store)
        agent = Agent(name="notes-agent", system_prompt="take notes", tools=[save, recall])

        detected = _detect_jit_directives(agent)
        assert detected is save.directives

    def test_no_jit_tools_returns_none(self) -> None:
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.loop import _detect_jit_directives

        agent = Agent(name="bare-agent", system_prompt="do nothing", tools=[])
        assert _detect_jit_directives(agent) is None


class TestItemHelpers:
    """Tests for _extract_text_from_item and _item_type_to_role."""

    def test_extract_from_dict_raw(self) -> None:
        item = MagicMock()
        item.raw = {"content": "hello world"}
        assert _extract_text_from_item(item) == "hello world"

    def test_extract_from_basemodel_raw(self) -> None:
        item = MagicMock()
        item.raw = MagicMock()
        item.raw.content = "model output"
        assert _extract_text_from_item(item) == "model output"

    def test_extract_from_tool_result(self) -> None:
        item = MagicMock()
        item.raw = MagicMock(spec=[])
        item.raw.content = None
        item.raw.output = "tool result text"
        # When content is not a string, fall through to output
        assert "tool result text" in _extract_text_from_item(item)

    def test_extract_from_none_raw(self) -> None:
        item = MagicMock()
        item.raw = None
        assert _extract_text_from_item(item) == ""

    def test_role_mapping(self) -> None:
        assert _item_type_to_role("user") == "user"
        assert _item_type_to_role("message_output") == "assistant"
        assert _item_type_to_role("tool_call_output") == "tool"
        assert _item_type_to_role("system") == "system"
        assert _item_type_to_role("unknown_type") == "unknown"


# =====================================================================
# Finding 3: malformed JSON in invoke closures
# =====================================================================


class TestInvokeJsonDecodeHandling:
    """Finding 3: json.loads in _invoke must return error string, not crash."""

    def test_save_note_bad_json_returns_error(self) -> None:
        tools = _make_jit_tools()
        save = next(t for t in tools if t.name == "save_note")
        ctx = ToolContext(tool_name="save_note", tool_call_id="c1", tool_arguments={}, raw_arguments="")
        result = asyncio.run(save.on_invoke(ctx, "{not valid json}"))
        assert "JSON parse error" in result

    def test_recall_notes_bad_json_returns_error(self) -> None:
        tools = _make_jit_tools()
        recall = next(t for t in tools if t.name == "recall_notes")
        ctx = ToolContext(tool_name="recall_notes", tool_call_id="c1", tool_arguments={}, raw_arguments="")
        result = asyncio.run(recall.on_invoke(ctx, "{bad}"))
        assert "JSON parse error" in result

    def test_manage_context_bad_json_returns_error(self) -> None:
        tools = _make_jit_tools()
        manage = next(t for t in tools if t.name == "manage_context")
        ctx = ToolContext(tool_name="manage_context", tool_call_id="c1", tool_arguments={}, raw_arguments="")
        result = asyncio.run(manage.on_invoke(ctx, "{bad}"))
        assert "JSON parse error" in result

    def test_search_history_bad_json_returns_error(self) -> None:
        tools = _make_jit_tools()
        search = next(t for t in tools if t.name == "search_history")
        ctx = ToolContext(tool_name="search_history", tool_call_id="c1", tool_arguments={}, raw_arguments="")
        result = asyncio.run(search.on_invoke(ctx, "{bad}"))
        assert "JSON parse error" in result


# =====================================================================
# Finding 8: implicit truthiness on importance arg
# =====================================================================


class TestSaveNoteImportanceTruthiness:
    """Finding 8: args.get('importance') or default treats 0 as missing."""

    def test_importance_zero_not_swapped_for_default(self) -> None:
        """LLM-supplied importance=0 must reach store.save as 0, not default=3."""
        store = InMemoryNoteStore()
        tools = [SaveNoteContextAwareTool(store=store, importance=3)]
        save = tools[0]
        ctx = ToolContext(tool_name="save_note", tool_call_id="c1", tool_arguments={}, raw_arguments="")
        # importance=0 is explicitly supplied; schema ge=1 may clamp it,
        # but the point is: save is called with 0, not 3.
        import json

        raw = json.dumps({"key": "k", "content": "v", "importance": 0})
        # store.save clamps to max(1, min(5, 0)) = 1; we verify default (3) is NOT used
        asyncio.run(save.on_invoke(ctx, raw))
        notes = store.recall()
        assert len(notes) == 1
        # importance must NOT be the default (3); it must be clamped from the supplied 0
        assert notes[0].importance != 3

    def test_importance_none_uses_default(self) -> None:
        """When importance key is absent, the default is used."""
        store = InMemoryNoteStore()
        save = SaveNoteContextAwareTool(store=store, importance=4)
        ctx = ToolContext(tool_name="save_note", tool_call_id="c1", tool_arguments={}, raw_arguments="")
        import json

        raw = json.dumps({"key": "k", "content": "v"})
        asyncio.run(save.on_invoke(ctx, raw))
        notes = store.recall()
        assert notes[0].importance == 4


# =====================================================================
# Finding 9: preserve-clamp off-by-one
# =====================================================================


class TestManageContextPreserveClamp:
    """Finding 9: max(1, msg_count - 1) yields preserve == msg_count when msg_count == 1."""

    def test_single_message_returns_early_error(self) -> None:
        """When msg_count <= 1 the tool must return an error, not schedule a no-op."""
        tools = _make_jit_tools()
        manage = next(t for t in tools if t.name == "manage_context")
        ctx = ExecutionAwareToolContext(
            tool_name="manage_context",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="",
            usage=None,
            turns=0,
            messages=1,
            tokens=0,
        )
        result = asyncio.run(manage.on_invoke(ctx, '{"action":"compact","preserve":1}'))
        assert "Not enough" in result or "at least 2" in result.lower()

    def test_two_messages_clamps_preserve_to_one(self) -> None:
        """With msg_count == 2 and preserve == 5, preserve is clamped to 1."""
        tools = _make_jit_tools()
        manage = next(t for t in tools if t.name == "manage_context")
        ctx = ExecutionAwareToolContext(
            tool_name="manage_context",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="",
            usage=None,
            turns=0,
            messages=2,
            tokens=0,
        )
        result = asyncio.run(manage.on_invoke(ctx, '{"action":"compact","preserve":5}'))
        assert "Scheduled" in result
        # preserve must be clamped to msg_count - 1 = 1
        from troopai.adk.context.directives import CompactDirective

        directive = manage.directives.consume()[0]
        assert isinstance(directive, CompactDirective)
        assert directive.preserve == 1
