"""Tests for :mod:`troopai.adk.workflows.temporal.activity`.

Covers:
- ``ModelActivityInput`` serialization and roundtrip via ``dataclasses.asdict``.
- Model registry: ``register_model`` / ``get_model``.
- ``get_model`` raises ``KeyError`` with a helpful message listing available models.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC
from unittest.mock import MagicMock

import pytest

from troopai.adk.llms.llm import LLM
from troopai.adk.workflows.temporal.activity import (
    ModelActivityInput,
    get_model,
    register_model,
)


class TestModelActivityInputSerializes:
    def test_model_activity_input_serializes(self) -> None:
        """``dataclasses.asdict`` produces the expected key/value mapping."""
        inp = ModelActivityInput(
            model_name="gpt-4o",
            messages_json='[{"role": "user", "content": "Hello"}]',
            tools_json="[]",
            config_json='{"temperature": 0.7}',
        )

        result = dataclasses.asdict(inp)

        assert result["model_name"] == "gpt-4o"
        assert result["messages_json"] == '[{"role": "user", "content": "Hello"}]'
        assert result["tools_json"] == "[]"
        assert result["config_json"] == '{"temperature": 0.7}'
        assert result["output_schema_json"] == ""


class TestModelActivityInputRoundtrips:
    def test_model_activity_input_roundtrips(self) -> None:
        """``asdict`` → ``ModelActivityInput(**d)`` reconstructs an equal instance."""
        original = ModelActivityInput(
            model_name="claude-sonnet-4-20250514",
            messages_json='[{"type": "message", "role": "user", "content": "Hi"}]',
            tools_json='[{"name": "search", "description": "Web search"}]',
            config_json='{"temperature": 0.5, "max_output_tokens": 256}',
            output_schema_json='{"type": "object"}',
        )

        as_dict = dataclasses.asdict(original)
        reconstructed = ModelActivityInput(**as_dict)

        assert reconstructed == original


class TestRegisterAndGetModel:
    def test_register_and_get_model(self) -> None:
        """A registered LLM can be retrieved by the same name."""
        mock_llm = MagicMock(spec=LLM)
        register_model("test-model-register", mock_llm)

        retrieved = get_model("test-model-register")

        assert retrieved is mock_llm

    def test_register_overwrites_previous_entry(self) -> None:
        """Re-registering the same name replaces the previous LLM."""
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM

        first_llm = MagicMock(spec=LLM)
        second_llm = MagicMock(spec=LLM)

        register_model("test-model-overwrite", first_llm)
        register_model("test-model-overwrite", second_llm)

        assert get_model("test-model-overwrite") is second_llm


class TestInvokeModelActivityTimestampNullification:
    """invoke_model_activity must not include a datetime in its return dict.

    Regression: ``dataclasses.asdict(response)`` includes the ``timestamp``
    field.  If the response carries a non-None ``datetime``, Temporal's JSON
    DataConverter raises a serialization error because ``datetime`` is not
    JSON-serializable without a custom payload codec.

    Fix: set ``timestamp=None`` before serializing (consistent with
    ``_dict_to_llm_response`` which always reconstructs ``timestamp=None``).
    """

    async def test_invoke_model_activity_result_is_json_serializable(self) -> None:
        """invoke_model_activity returns a JSON-serializable dict.

        Even when the underlying LLM response carries a datetime timestamp,
        the activity return value must be JSON-serializable so Temporal's default
        DataConverter can encode it.  The fix: nullify timestamp before asdict().
        """
        import json
        from datetime import datetime
        from unittest.mock import AsyncMock, MagicMock, patch

        import pytest

        pytest.importorskip("temporalio")
        from temporalio import activity

        from troopai.adk.llms.llm import LLM
        from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
        from troopai.adk.workflows.temporal.activity import (
            ModelActivityInput,
            invoke_model_activity,
            register_model,
        )

        # Register a mock LLM that returns a response with a real datetime.
        fake_response = LLMResponse(
            response_id="r1",
            model="gpt-4o",
            response=[LLMResponseText(type="text", text="hello")],
            finish_reason="stop",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        mock_llm = MagicMock(spec=LLM)
        mock_llm.acomplete = AsyncMock(return_value=fake_response)
        register_model("__test_timestamp_model__", mock_llm)

        inp = ModelActivityInput(
            model_name="__test_timestamp_model__",
            messages_json='["hi"]',
            tools_json="[]",
            config_json="{}",
        )

        # Patch activity.info() to return a minimal ActivityInfo-like object.
        fake_info = MagicMock()
        fake_info.attempt = 1
        fake_info.heartbeat_timeout = None

        with patch.object(activity, "info", return_value=fake_info):
            result_dict = await invoke_model_activity(inp)

        # The returned dict must be JSON-serializable (no datetime objects).
        try:
            json.dumps(result_dict)
        except TypeError as exc:
            raise AssertionError(
                f"invoke_model_activity returned a non-JSON-serializable dict: {exc}. "
                "The timestamp field must be set to None before dataclasses.asdict()."
            ) from exc

        # timestamp must be None in the returned dict.
        assert result_dict.get("timestamp") is None, (
            f"Expected timestamp=None in result dict, got {result_dict.get('timestamp')!r}"
        )


class TestActivityImportErrorNarrowing:
    """The bare except ImportError around @activity.defn must only swallow the
    'temporalio not installed' case.

    Regression: a broad ``except ImportError: pass`` silently swallowed any
    ImportError raised inside the try block (from a partial install or an
    unrelated dep), leaving ``invoke_model_activity`` undefined.  Workers
    would then silently have no activity registered, causing silent hangs.

    Fix: narrow to ``except ImportError as exc: if 'temporalio' not in str(exc): raise``.
    """

    def test_activity_module_source_has_narrowed_except(self) -> None:
        """The activity.py source must narrow the ImportError to temporalio-only.

        Regression: ``except ImportError: pass`` swallowed all ImportErrors.
        Verify the source contains the narrowing guard rather than a bare pass.
        """
        import inspect

        import troopai.adk.workflows.temporal.activity as activity_mod

        source = inspect.getsource(activity_mod)
        # The narrowed guard must be present.
        assert "temporalio" in source, "activity.py source must reference 'temporalio' in guard"
        assert "not in str(exc)" in source or "not in str(" in source, (
            "activity.py must narrow the except ImportError using a string check on the error"
        )
        # The old bare 'except ImportError:\n    pass' must NOT be present.
        assert "except ImportError:\n    pass" not in source, (
            "activity.py must not have a bare 'except ImportError: pass' that swallows all errors"
        )


class TestGetModelMissingRaisesKeyError:
    def test_get_model_missing_raises_keyerror(self) -> None:
        """``get_model`` raises ``KeyError`` when the name is not registered."""
        with pytest.raises(KeyError):
            get_model("__nonexistent_model_xyz__")

    def test_get_model_missing_message_lists_available(self) -> None:
        """The ``KeyError`` message names the missing key and lists available models."""
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM

        register_model("available-model-a", MagicMock(spec=LLM))

        with pytest.raises(KeyError) as exc_info:
            get_model("__definitely_not_registered__")

        message = str(exc_info.value)
        assert "__definitely_not_registered__" in message
        assert "available-model-a" in message
