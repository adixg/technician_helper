import pytest

from technician_helper.config import Settings


def test_require_raises_without_hf_token():
    settings = Settings(hf_token=None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        settings.require()


def test_require_passes_with_hf_token():
    settings = Settings(hf_token="hf_dummy")
    settings.require()  # weaviate check off by default -> no raise


def test_require_reports_weaviate_when_unreachable(monkeypatch):
    monkeypatch.setattr("technician_helper.clients.weaviate_ready", lambda: False)
    settings = Settings(hf_token="hf_dummy")
    with pytest.raises(RuntimeError, match="Weaviate is not reachable"):
        settings.require(weaviate=True)


def test_reliability_defaults_present():
    settings = Settings(hf_token="x")
    assert settings.llm_timeout > 0
    assert settings.llm_max_attempts >= 1
    assert settings.weaviate_connect_attempts >= 1
