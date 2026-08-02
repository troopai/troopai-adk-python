"""Helicone gateway — LLM base_url redirection + auth header.

Helicone is an LLM gateway/proxy, NOT a span exporter: route LLM traffic
through its gateway by setting the provider ``base_url`` and the
``Helicone-Auth`` header. Wire the returned values into your LLM config.
Docs: https://docs.helicone.ai/getting-started/integration-method/gateway
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DEFAULT_HELICONE_GATEWAY = "https://gateway.helicone.ai"


@dataclass(frozen=True)
class HeliconeGatewayConfig:
    """Values to wire into an LLM config to route traffic through Helicone.

    Attributes:
        base_url: Gateway base URL to set as the provider ``base_url``.
        headers: Headers to attach to LLM requests (carries ``Helicone-Auth``).
    """

    base_url: str
    """Gateway base URL to set as the provider ``base_url``."""

    headers: dict[str, str]
    """Headers to attach to LLM requests (carries ``Helicone-Auth``)."""


def setup_helicone(*, api_key: str, base_url: str = _DEFAULT_HELICONE_GATEWAY) -> HeliconeGatewayConfig:
    """Return the gateway base_url + auth header for routing LLM calls.

    Args:
        api_key: Helicone API key (non-empty). Load from the environment.
        base_url: Helicone gateway base URL.

    Returns:
        A :class:`HeliconeGatewayConfig` carrying the ``base_url`` and
        ``headers`` to wire into your LLM provider config.

    Raises:
        ValueError: When ``api_key`` is empty.
    """
    if len(api_key) == 0:
        raise ValueError("helicone api_key must be non-empty")
    logger.debug("Helicone gateway configured at %s", base_url)
    return HeliconeGatewayConfig(base_url=base_url, headers={"Helicone-Auth": f"Bearer {api_key}"})
