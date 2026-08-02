from troopai.adk.types.tracing import TracingConvention


def test_convention_members_and_values():
    assert TracingConvention.DEFAULT.value == "default"
    assert TracingConvention.OPENINFERENCE.value == "openinference"
    assert {c.value for c in TracingConvention} == {"default", "openinference"}
