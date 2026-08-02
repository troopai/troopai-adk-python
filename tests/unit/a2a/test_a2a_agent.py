"""Tests for ``A2AAgent`` — peer-agent class for remote A2A endpoints.

``A2AAgent`` is pure config — execution lives on
:class:`troopai.adk.a2a.a2a_runner.A2ARunner` (covered by
``test_a2a_runner.py``). This module verifies:

* ``A2AAgent`` is a ``BaseAgent`` subclass with the expected inherited
  fields plus the A2A-specific URL / agent_card / etc.
* ``__post_init__`` validates the URL and raises ``ValueError`` on empty.
* ``as_tool()`` returns a ``FunctionTool`` with snake-cased default
  name, propagates ``max_result_tokens`` / ``timeout``, dispatches
  via ``A2ARunner.arun`` when invoked, and supports overrides.
* The internal ``_client`` is lazily constructed on first use and
  reused thereafter, and ``close()`` releases it idempotently.
* The ``_snake_case`` helper handles the common name shapes correctly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip the entire module if the optional `a2a` extra is missing.
pytest.importorskip("a2a.client")

from troopai.adk.a2a import (
    A2AAgent,
    A2AClient,
    A2ARunResult,
)
from troopai.adk.agents import BaseAgent
from troopai.adk.tools.function_tool import FunctionTool


class TestA2AAgentConstruction:
    def test_is_a_baseagent(self) -> None:
        agent = A2AAgent(name="remote", url="http://example.com")
        assert isinstance(agent, BaseAgent)

    def test_inherited_fields_default_correctly(self) -> None:
        agent = A2AAgent(name="remote", url="http://example.com")
        assert agent.name == "remote"
        assert agent.description is None
        assert len(agent.tools) == 0

    def test_a2a_specific_field_defaults(self) -> None:
        agent = A2AAgent(name="remote", url="http://example.com")
        assert agent.url == "http://example.com"
        assert agent.agent_card is None
        assert agent.timeout == 30.0
        assert agent.interceptors == []
        assert agent.client_config is None

    def test_no_run_method_on_agent(self) -> None:
        # Pure-config invariant: A2AAgent must not carry execution
        # methods. Local Agent / Swarm / Graph all delegate to Runner;
        # A2AAgent delegates to A2ARunner. A regression that re-adds
        # ``run`` here would silently bypass the A2ARunner type guard.
        agent = A2AAgent(name="x", url="http://example.com")
        assert not hasattr(agent, "run")
        assert not hasattr(agent, "poll_task")
        assert not hasattr(agent, "cancel_task")

    def test_no_transport_or_http_client_field(self) -> None:
        # A2AAgent exposes no ``transport`` field (protocol bindings live on
        # ClientConfig.supported_protocol_bindings) and no ``http_client``
        # field (the client lives on ClientConfig.httpx_client). Both are
        # reached via ``client_config``; this guards against accidental
        # re-introduction.
        agent = A2AAgent(name="x", url="http://example.com")
        assert not hasattr(agent, "transport")
        assert not hasattr(agent, "http_client")

    def test_empty_url_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="url MUST be a non-empty"):
            A2AAgent(name="x", url="")

    def test_url_default_factor_is_empty_string(self) -> None:
        # ``url`` has a default of "" purely so dataclass field
        # ordering works alongside the inherited default-having
        # BaseAgent fields. The actual contract is "must be set" —
        # enforced at __post_init__ time.
        with pytest.raises(ValueError):
            A2AAgent(name="x")  # type: ignore[call-arg]


class TestSnakeCaseDefaultToolName:
    """Verify the default tool_name in as_tool() is snake-case-derived.

    Tested through the public ``as_tool()`` API rather than reaching
    into the private helper directly — keeps the test resilient to
    refactors that move the helper or change its name.
    """

    @pytest.mark.parametrize(
        "agent_name,expected_tool_name",
        [
            ("Research Bot", "research_bot"),
            ("HTTPClient", "http_client"),
            ("camelCase", "camel_case"),
            ("PascalCase", "pascal_case"),
            ("simple", "simple"),
            ("with123numbers", "with123numbers"),
            ("my-special.agent", "my_special_agent"),
            ("ParseURL2HTML", "parse_url2_html"),
        ],
    )
    def test_default_tool_name_variants(self, agent_name: str, expected_tool_name: str) -> None:
        agent = A2AAgent(name=agent_name, url="http://example.com")
        assert agent.as_tool().name == expected_tool_name


class TestLazyClientInit:
    async def test_first_call_constructs_client(self) -> None:
        # The client is built lazily — construction stays synchronous
        # so a developer can build an A2AAgent without an event loop.
        agent = A2AAgent(name="remote", url="http://example.com")
        with patch("troopai.adk.a2a.a2a_agent.A2AClient") as mock_cls:
            mock_cls.return_value = MagicMock(spec=A2AClient)
            client_a = await agent.get_client()
            client_b = await agent.get_client()
            assert client_a is client_b
            # Should have been constructed exactly once.
            assert mock_cls.call_count == 1

    async def test_close_releases_client(self) -> None:
        agent = A2AAgent(name="remote", url="http://example.com")
        with patch("troopai.adk.a2a.a2a_agent.A2AClient") as mock_cls:
            mock_client = MagicMock(spec=A2AClient)
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client
            await agent.get_client()
            await agent.close()
            mock_client.close.assert_awaited_once()
            # After close, the cached client is gone — next call would
            # construct a new one.
            await agent.get_client()
            assert mock_cls.call_count == 2

    async def test_close_idempotent(self) -> None:
        agent = A2AAgent(name="remote", url="http://example.com")
        # No client constructed; close should be a no-op.
        await agent.close()
        await agent.close()


class TestAsTool:
    def test_default_name_is_snake_case(self) -> None:
        agent = A2AAgent(name="Research Bot", url="http://example.com")
        tool = agent.as_tool()
        assert tool.name == "research_bot"

    def test_default_description_falls_back_to_agent_description(self) -> None:
        agent = A2AAgent(
            name="x",
            description="Looks things up.",
            url="http://example.com",
        )
        assert agent.as_tool().description == "Looks things up."

    def test_default_description_when_no_agent_description(self) -> None:
        agent = A2AAgent(name="x", url="http://example.com")
        desc = agent.as_tool().description
        assert desc is not None
        assert "Delegate" in desc
        assert "'x'" in desc

    def test_overrides(self) -> None:
        agent = A2AAgent(name="x", url="http://example.com")
        tool = agent.as_tool(
            tool_name="custom_name",
            tool_description="Custom description.",
            max_result_tokens=500,
            timeout=10.0,
        )
        assert tool.name == "custom_name"
        assert tool.description == "Custom description."
        assert tool.max_result_tokens == 500
        assert tool.timeout == 10.0

    def test_returns_function_tool(self) -> None:
        agent = A2AAgent(name="x", url="http://example.com")
        assert isinstance(agent.as_tool(), FunctionTool)

    def test_has_default_input_schema(self) -> None:
        agent = A2AAgent(name="x", url="http://example.com")
        schema = agent.as_tool().schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "input" in schema["properties"]
        assert schema["required"] == ["input"]

    async def test_on_invoke_calls_runner_and_returns_text(self) -> None:
        agent = A2AAgent(name="x", url="http://example.com")
        # Mock at the ``get_client`` boundary — short-circuits before
        # dispatch reaches the network. End-to-end ``A2ARunner``
        # coverage (isinstance guard, mutex constraints, flag
        # dispatch) lives in ``test_a2a_runner.py``.
        mock_client = MagicMock(spec=A2AClient)
        mock_client.send_message = AsyncMock(
            return_value=A2ARunResult(text="remote-answer", task_id="t1", context_id="c1")
        )
        with patch.object(agent, "get_client", AsyncMock(return_value=mock_client)):
            tool = agent.as_tool()
            assert tool.on_invoke is not None
            # Pretend the LLM produced this JSON.
            result = await tool.on_invoke(MagicMock(), '{"input": "What is 2+2?"}')
        assert result == "remote-answer"
        mock_client.send_message.assert_awaited_once_with("What is 2+2?", context_id=None, continuation_token=None)

    async def test_on_invoke_handles_bare_string_input(self) -> None:
        # Some LLMs serialise bare strings instead of JSON objects.
        # Tool must still pass something useful through.
        agent = A2AAgent(name="x", url="http://example.com")
        mock_client = MagicMock(spec=A2AClient)
        mock_client.send_message = AsyncMock(return_value=A2ARunResult(text="ok", task_id="t1", context_id="c1"))
        with patch.object(agent, "get_client", AsyncMock(return_value=mock_client)):
            tool = agent.as_tool()
            assert tool.on_invoke is not None
            await tool.on_invoke(MagicMock(), "raw bare string")
        mock_client.send_message.assert_awaited_once_with("raw bare string", context_id=None, continuation_token=None)
