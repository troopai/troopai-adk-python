import logging

from troopai.adk.tracing.logging import EVENT_AGENT_TURN_START, log_event


def test_log_event_carries_tenant_id(caplog):
    logger = logging.getLogger("troopai.tenant.test")
    with caplog.at_level(logging.INFO, logger="troopai.tenant.test"):
        log_event(logger, EVENT_AGENT_TURN_START, agent_name="a", tenant_id="acme")
    assert len(caplog.records) == 1
    assert caplog.records[0].tenant_id == "acme"
    assert caplog.records[0].agent_name == "a"
