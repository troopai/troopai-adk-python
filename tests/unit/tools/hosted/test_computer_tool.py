"""Construction + protocol tests for ComputerTool, Computer, SafetyCheck.

These cover the typed-surface introduction. Loop dispatch (executing
``Computer`` actions in response to provider-emitted ``computer_call``
items) is a follow-up commit.
"""

from __future__ import annotations

import dataclasses

import pytest

from troopai.adk.tools.hosted import (
    Computer,
    ComputerTool,
    HostedTool,
    SafetyCheck,
)


class StubComputer:
    """Minimal Computer implementation satisfying the runtime check."""

    async def screenshot(self) -> str:
        return "base64-png"

    async def click(self, x: int, y: int, button: str = "left") -> None:
        return None

    async def double_click(self, x: int, y: int) -> None:
        return None

    async def type(self, text: str) -> None:
        return None

    async def keypress(self, keys: list[str]) -> None:
        return None

    async def move(self, x: int, y: int) -> None:
        return None

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        return None

    async def drag(self, path: list[tuple[int, int]]) -> None:
        return None

    async def wait(self, seconds: float) -> None:
        return None


class TestComputerToolConstruction:
    def test_default_attributes(self) -> None:
        computer = StubComputer()
        tool = ComputerTool(computer=computer)
        assert tool.computer is computer
        assert tool.display_width == 1024
        assert tool.display_height == 768
        assert tool.environment == "browser"
        assert tool.on_safety_check is None
        # Cost-and-safety-conservative default: approval required.
        assert tool.requires_approval is True

    def test_custom_display_and_environment(self) -> None:
        tool = ComputerTool(
            computer=StubComputer(),
            display_width=1920,
            display_height=1080,
            environment="linux",
        )
        assert tool.display_width == 1920
        assert tool.display_height == 1080
        assert tool.environment == "linux"

    def test_invalid_display_width_raises(self) -> None:
        with pytest.raises(ValueError, match="display_width"):
            ComputerTool(computer=StubComputer(), display_width=0)
        with pytest.raises(ValueError, match="display_width"):
            ComputerTool(computer=StubComputer(), display_width=-100)

    def test_invalid_display_height_raises(self) -> None:
        with pytest.raises(ValueError, match="display_height"):
            ComputerTool(computer=StubComputer(), display_height=0)
        with pytest.raises(ValueError, match="display_height"):
            ComputerTool(computer=StubComputer(), display_height=-100)

    def test_computer_none_raises(self) -> None:
        # Explicit None guard runs before the Protocol isinstance check.
        with pytest.raises(TypeError, match="must not be None"):
            ComputerTool(computer=None)  # type: ignore[arg-type]

    def test_computer_must_satisfy_protocol(self) -> None:
        # Missing methods → TypeError at construction.
        class Incomplete:
            async def screenshot(self) -> str:
                return ""

        with pytest.raises(TypeError, match="Computer protocol"):
            ComputerTool(computer=Incomplete())  # type: ignore[arg-type]

    def test_inherits_from_hosted_tool(self) -> None:
        tool = ComputerTool(computer=StubComputer())
        assert isinstance(tool, HostedTool)

    def test_supported_providers(self) -> None:
        # Anthropic coverage is deferred — its computer-use vocabulary
        # (triple_click, hold_key, cursor_position, …) requires
        # Computer Protocol extensions that haven't shipped yet.
        assert ComputerTool.SUPPORTED_PROVIDERS == ("openai-responses",)


class TestSafetyCheck:
    def test_minimal(self) -> None:
        sc = SafetyCheck(action="click", reason="user clicked submit")
        assert sc.action == "click"
        assert sc.reason == "user clicked submit"
        assert sc.target is None

    def test_full(self) -> None:
        sc = SafetyCheck(
            action="type",
            reason="filling password field",
            target="login-form.password",
        )
        assert sc.target == "login-form.password"

    def test_frozen(self) -> None:
        sc = SafetyCheck(action="click", reason="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            sc.action = "type"  # type: ignore[misc]


class TestComputerProtocol:
    """The Protocol is ``@runtime_checkable``; verify both isinstance
    checks and rejection of incomplete implementations.
    """

    def test_stub_satisfies_protocol(self) -> None:
        stub = StubComputer()
        assert isinstance(stub, Computer)

    def test_incomplete_rejected(self) -> None:
        class Incomplete:
            async def screenshot(self) -> str:
                return ""

        assert not isinstance(Incomplete(), Computer)


class TestSafetyCheckCallback:
    """The ``on_safety_check`` callback is purely user-supplied; this
    test pins that the callback is stored as-is for later dispatch."""

    def test_callback_stored(self) -> None:
        captured: list[SafetyCheck] = []

        def check(_ctx: object, sc: SafetyCheck) -> bool:
            captured.append(sc)
            return True

        tool = ComputerTool(computer=StubComputer(), on_safety_check=check)
        assert tool.on_safety_check is check

    def test_async_callback_stored(self) -> None:
        async def check(_ctx: object, _sc: SafetyCheck) -> bool:
            return True

        tool = ComputerTool(computer=StubComputer(), on_safety_check=check)
        assert tool.on_safety_check is check


class TestComputerToolOpenAIResponsesConversion:
    def test_converts_to_computer_use_preview(self) -> None:
        # Regression: ComputerTool documents OpenAI Responses support, but the
        # Responses converter had no branch, so it raised
        # UnsupportedHostedToolError on every use. It must now emit a
        # ``computer_use_preview`` wire tool honouring geometry + environment.
        from troopai.adk.llms.openai.openai_responses_converter import OpenAIResponsesConverter

        tool = ComputerTool(
            computer=StubComputer(),
            display_width=1280,
            display_height=800,
            environment="linux",
        )
        wire = OpenAIResponsesConverter.convert_tools([tool])
        assert wire == [
            {
                "type": "computer_use_preview",
                "display_width": 1280,
                "display_height": 800,
                "environment": "linux",
            }
        ]
