"""Shared pytest fixtures for the test suite."""

import logging

import pytest


@pytest.fixture(autouse=True)
def _enable_log_propagation():
    """Ensure troopai.adk loggers propagate to root so caplog captures them.

    The production YAML config sets propagate=no (correct for file handlers),
    but pytest's caplog fixture relies on propagation to capture records.
    """
    logger = logging.getLogger("troopai.adk")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original
