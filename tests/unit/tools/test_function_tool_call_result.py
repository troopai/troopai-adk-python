from troopai.adk.types.output import FunctionToolCallResult


class TestFunctionToolCallResult:
    def test_construction(self) -> None:
        result = FunctionToolCallResult(call_id="call_123", output="success")
        assert result.call_id == "call_123"
        assert result.output == "success"

    def test_equality(self) -> None:
        a = FunctionToolCallResult(call_id="call_1", output="ok")
        b = FunctionToolCallResult(call_id="call_1", output="ok")
        assert a == b

    def test_attribute_access(self) -> None:
        result = FunctionToolCallResult(call_id="call_x", output="data")
        assert hasattr(result, "call_id")
        assert hasattr(result, "output")
        assert not hasattr(result, "__getitem__")
