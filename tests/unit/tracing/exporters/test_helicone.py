import pytest

from troopai.adk.tracing.exporters.helicone import setup_helicone


def test_helicone_returns_base_url_and_auth_header():
    cfg = setup_helicone(api_key="hl-secret")
    assert cfg.base_url.startswith("https://")
    assert cfg.base_url == "https://gateway.helicone.ai"
    assert cfg.headers["Helicone-Auth"] == "Bearer hl-secret"


def test_helicone_rejects_empty_key():
    with pytest.raises(ValueError, match="helicone api_key must be non-empty"):
        setup_helicone(api_key="")


def test_helicone_custom_base_url():
    cfg = setup_helicone(api_key="k", base_url="https://custom.example.com")
    assert cfg.base_url == "https://custom.example.com"
    assert cfg.headers["Helicone-Auth"] == "Bearer k"
