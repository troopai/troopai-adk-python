"""Tests for dumping an Agent back to a config dict (CFG-7, lossy)."""

from __future__ import annotations

from troopai.adk.agents.agent import Agent
from troopai.adk.config.dump import dump_agent
from troopai.adk.prompts.system_prompt import SystemPrompt
from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
from troopai.adk.types.config.agent_config import AgentConfig

from .sample_symbols import SampleOutput


class TestDumpScalars:
    def test_name_and_prompt(self) -> None:
        d = dump_agent(Agent(name="a", system_prompt="hi"))
        assert d["name"] == "a"
        assert d["system_prompt"] == "hi"

    def test_description_when_set(self) -> None:
        d = dump_agent(Agent(name="a", system_prompt="p", description="does x"))
        assert d["description"] == "does x"

    def test_description_omitted_when_none(self) -> None:
        assert "description" not in dump_agent(Agent(name="a", system_prompt="p"))

    def test_structured_system_prompt(self) -> None:
        d = dump_agent(Agent(name="a", system_prompt=SystemPrompt(role="You review.")))
        assert d["system_prompt"]["role"] == "You review."

    def test_output_schema_ref(self) -> None:
        d = dump_agent(Agent(name="a", system_prompt="p", output_schema=AgentOutputSchema(SampleOutput)))
        assert d["output_schema"]["ref"] == "tests.unit.config.sample_symbols:SampleOutput"
        assert d["output_schema"]["enforcement"] == "strict"

    def test_dumped_config_revalidates(self) -> None:
        d = dump_agent(Agent(name="a", system_prompt="p", description="d"))
        assert AgentConfig.model_validate(d).name == "a"


class TestDumpLLM:
    def test_string_llm_with_llm_config(self) -> None:
        from troopai.adk.llms.llm_config import LLMConfig

        d = dump_agent(Agent(name="a", system_prompt="p", llm="gpt-4o", llm_config=LLMConfig(temperature=0.5)))
        assert d["llm"] == "gpt-4o"
        assert d["llm_config"]["temperature"] == 0.5

    def test_string_llm_no_config(self) -> None:
        d = dump_agent(Agent(name="a", system_prompt="p", llm="gpt-4o"))
        assert d["llm"] == "gpt-4o"
        assert "llm_config" not in d

    def test_provider_instance_dumps_block_without_api_key(self) -> None:
        from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
        from troopai.adk.llms.anthropic.anthropic_model import AnthropicLLM

        llm = AnthropicLLM(model="claude-sonnet-4-5", api_key="secret")
        config = AnthropicConfig(temperature=0.5, auto_cache_control=True)
        d = dump_agent(Agent(name="a", system_prompt="p", llm=llm, llm_config=config))
        assert d["llm"]["provider"] == "anthropic"
        assert d["llm"]["model"] == "claude-sonnet-4-5"
        assert "api_key" not in d["llm"]
        assert d["llm"]["config"]["temperature"] == 0.5
        assert d["llm"]["config"]["auto_cache_control"] is True
        assert "llm_config" not in d


class TestDumpToolsAndRoundTrip:
    def test_hosted_tool_dumped(self) -> None:
        from troopai.adk.tools.hosted.web_search_tool import WebSearchTool

        d = dump_agent(Agent(name="a", system_prompt="p", tools=[WebSearchTool(max_uses=3)]))
        assert d["tools"] == [{"type": "web_search", "args": {"max_uses": 3}}]

    def test_function_tool_omitted(self) -> None:
        from .sample_symbols import sample_tool

        d = dump_agent(Agent(name="a", system_prompt="p", tools=[sample_tool]))
        assert "tools" not in d

    def test_dump_exported_from_config(self) -> None:
        from troopai.adk.config import dump_agent as exported

        assert exported is dump_agent

    def test_round_trip_via_build(self) -> None:
        from troopai.adk.config import build_agent

        original = {
            "name": "support",
            "system_prompt": "Help users.",
            "llm": "gpt-4o",
            "llm_config": {"temperature": 0.5},
            "tools": [{"type": "web_search", "args": {"max_uses": 2}}],
        }
        agent = build_agent(AgentConfig.model_validate(original))
        dumped = dump_agent(agent)
        assert dumped["name"] == "support"
        assert dumped["llm"] == "gpt-4o"
        assert dumped["llm_config"]["temperature"] == 0.5
        assert dumped["tools"] == [{"type": "web_search", "args": {"max_uses": 2}}]
        AgentConfig.model_validate(dumped)


class TestDumpReviewFixes:
    def test_compact_enforcement_preserved(self) -> None:
        from troopai.adk.schemas import SchemaEnforcement

        schema = AgentOutputSchema(SampleOutput, schema_enforcement=SchemaEnforcement.COMPACT)
        d = dump_agent(Agent(name="a", system_prompt="p", output_schema=schema))
        assert d["output_schema"]["enforcement"] == "compact"

    def test_none_enforcement_preserved(self) -> None:
        from troopai.adk.schemas import SchemaEnforcement

        schema = AgentOutputSchema(SampleOutput, schema_enforcement=SchemaEnforcement.NONE)
        d = dump_agent(Agent(name="a", system_prompt="p", output_schema=schema))
        assert d["output_schema"]["enforcement"] == "none"

    def test_default_factory_field_omitted(self) -> None:
        from troopai.adk.tools.hosted.file_search_tool import FileSearchTool

        d = dump_agent(Agent(name="a", system_prompt="p", tools=[FileSearchTool()]))
        assert d["tools"] == [{"type": "file_search", "args": {}}]

    def test_stop_on_first_tool_dumped(self) -> None:
        d = dump_agent(Agent(name="a", system_prompt="p", tool_use_behavior="stop_on_first_tool"))
        assert d["tool_use_behavior"] == "stop_on_first_tool"

    def test_stop_at_tools_dumped_and_revalidates(self) -> None:
        from troopai.adk.types.tools.tool_use_behavior import StopAtTools

        d = dump_agent(Agent(name="a", system_prompt="p", tool_use_behavior=StopAtTools(stop_at_tool_names=["done"])))
        assert d["tool_use_behavior"] == {"stop_at_tool_names": ["done"]}
        AgentConfig.model_validate(d)

    def test_eager_skill_activation_dumped(self) -> None:
        from troopai.adk.skills.activation import SkillActivation

        d = dump_agent(Agent(name="a", system_prompt="p", skill_activation=SkillActivation.EAGER))
        assert d["skill_activation"] == "eager"

    def test_provider_instance_no_config_omits_config(self) -> None:
        from troopai.adk.llms.anthropic.anthropic_model import AnthropicLLM

        d = dump_agent(Agent(name="a", system_prompt="p", llm=AnthropicLLM(model="claude-sonnet-4-5", api_key="s")))
        assert d["llm"] == {"provider": "anthropic", "model": "claude-sonnet-4-5"}
        assert "llm_config" not in d


class TestDumpSecretOmission:
    def test_hosted_mcp_secrets_omitted(self) -> None:
        from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool

        tool = HostedMCPTool(
            server_label="docs",
            server_url="https://mcp.example.com",
            authorization="Bearer sk-secret",
            headers={"Authorization": "Bearer sk-secret-2"},
        )
        d = dump_agent(Agent(name="a", system_prompt="p", tools=[tool]))
        args = d["tools"][0]["args"]
        assert args["server_label"] == "docs"
        assert args["server_url"] == "https://mcp.example.com"
        assert "authorization" not in args
        assert "headers" not in args

    def test_llm_config_extra_maps_omitted(self) -> None:
        from troopai.adk.llms.llm_config import LLMConfig

        config = LLMConfig(temperature=0.5, extra_headers={"Authorization": "Bearer sk"}, extra_body={"api_key": "sk"})
        d = dump_agent(Agent(name="a", system_prompt="p", llm="gpt-4o", llm_config=config))
        assert d["llm_config"] == {"temperature": 0.5}
