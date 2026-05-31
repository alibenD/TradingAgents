from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.validators import validate_model


def test_codex_oauth_has_no_static_api_key_prompt():
    assert get_api_key_env("codex-oauth") is None


def test_factory_creates_codex_oauth_client(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    from tradingagents.llm_clients.codex_auth import save_codex_tokens

    save_codex_tokens({"access_token": "at", "refresh_token": "rt"})

    client = create_llm_client("codex-oauth", "gpt-5.4")

    assert client.get_provider_name() == "codex-oauth"


def test_codex_model_catalog_contains_gpt54():
    labels = [label for label, _ in get_model_options("codex-oauth", "deep")]
    assert any("GPT-5.4" in label for label in labels)


def test_codex_validator_allows_forward_compatible_models():
    assert validate_model("codex-oauth", "gpt-5.4")
    assert validate_model("codex-oauth", "custom-codex-model")

