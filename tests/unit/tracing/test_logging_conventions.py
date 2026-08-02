import logging

from troopai.adk.tracing.logging import EVENT_AGENT_TURN_START, log_event


def test_log_event_emits_structured_fields(caplog):
    logger = logging.getLogger("troopai.test")
    with caplog.at_level(logging.INFO, logger="troopai.test"):
        log_event(logger, EVENT_AGENT_TURN_START, agent_name="triage", turn=1)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.event == "agent.turn.start"
    assert record.agent_name == "triage"
    assert record.turn == 1


def test_log_event_respects_level(caplog):
    logger = logging.getLogger("troopai.test2")
    with caplog.at_level(logging.WARNING, logger="troopai.test2"):
        log_event(logger, EVENT_AGENT_TURN_START, level=logging.DEBUG, agent_name="x")
    # DEBUG below the WARNING threshold → not captured
    assert len(caplog.records) == 0


def test_log_event_forwards_level(caplog):
    logger = logging.getLogger("troopai.test3")
    with caplog.at_level(logging.WARNING, logger="troopai.test3"):
        log_event(logger, EVENT_AGENT_TURN_START, level=logging.WARNING)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
