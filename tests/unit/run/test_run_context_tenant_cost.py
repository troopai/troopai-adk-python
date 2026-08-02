from troopai.adk.run.context import RunContext


def test_runcontext_tenant_and_cost_defaults():
    ctx = RunContext.make(None)
    assert ctx.tenant_id is None
    assert ctx.cost_usd == 0.0


def test_runcontext_make_sets_tenant_and_accumulates_cost():
    ctx = RunContext.make({"user": "u"}, tenant_id="acme")
    assert ctx.cost_usd == 0.0
    assert ctx.tenant_id == "acme"
    ctx.cost_usd += 0.25
    ctx.cost_usd += 0.10
    assert ctx.cost_usd == 0.35
