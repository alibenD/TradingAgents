import json
import time

import pytest

from tradingagents.llm_clients import codex_auth
from tradingagents.llm_clients import codex_login


def _jwt_with_exp(exp_epoch: int) -> str:
    import base64

    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp_epoch}).encode("utf-8"))
        .rstrip(b"=")
        .decode("utf-8")
    )
    return f"header.{payload}.sig"


def test_auth_path_uses_tradingagents_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / "ta-home"))
    assert codex_auth.codex_auth_path() == tmp_path / "ta-home" / "auth" / "codex_oauth.json"


def test_save_and_read_tokens_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    codex_auth.save_codex_tokens({"access_token": "at", "refresh_token": "rt"})

    state = codex_auth.read_codex_tokens()

    assert state["tokens"]["access_token"] == "at"
    assert state["tokens"]["refresh_token"] == "rt"
    assert state["auth_mode"] == "chatgpt"


def test_read_missing_tokens_raises_relogin(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))

    with pytest.raises(codex_auth.CodexAuthError) as exc:
        codex_auth.read_codex_tokens()

    assert exc.value.relogin_required is True
    assert exc.value.code == "codex_auth_missing"


def test_expiring_access_token_refreshes(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    codex_auth.save_codex_tokens(
        {
            "access_token": _jwt_with_exp(int(time.time()) - 10),
            "refresh_token": "old-refresh",
        }
    )
    called = {"count": 0}

    def fake_refresh(access_token, refresh_token, timeout_seconds=20.0):
        called["count"] += 1
        assert refresh_token == "old-refresh"
        return {"access_token": "new-access", "refresh_token": "new-refresh"}

    monkeypatch.setattr(codex_auth, "refresh_codex_oauth", fake_refresh)

    creds = codex_auth.resolve_codex_runtime_credentials()

    assert called["count"] == 1
    assert creds["api_key"] == "new-access"
    assert creds["base_url"] == codex_auth.DEFAULT_CODEX_BASE_URL


def test_refresh_error_parses_nested_openai_error(monkeypatch):
    class FakeResponse:
        status_code = 401

        def json(self):
            return {
                "error": {
                    "code": "refresh_token_reused",
                    "message": "refresh token reused",
                }
            }

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    with pytest.raises(codex_auth.CodexAuthError) as exc:
        codex_auth.refresh_codex_oauth("old-access", "old-refresh")

    assert exc.value.code == "refresh_token_reused"
    assert exc.value.relogin_required is True


def test_refresh_wraps_network_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise codex_auth.httpx.ConnectError("tls eof")

    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    with pytest.raises(codex_auth.CodexAuthError) as exc:
        codex_auth.refresh_codex_oauth("old-access", "old-refresh")

    assert exc.value.code == "codex_refresh_request_failed"
    assert exc.value.relogin_required is False


def test_device_code_request_wraps_network_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise codex_auth.httpx.ConnectError("tls eof")

    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    with pytest.raises(codex_auth.CodexAuthError) as exc:
        codex_auth.request_device_code()

    assert exc.value.code == "device_code_request_failed"
    assert exc.value.relogin_required is True


def test_request_device_code_prefers_https_proxy_over_all_proxy(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"user_code": "ABCD-EFGH", "device_auth_id": "device-1"}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897/")
    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    payload = codex_auth.request_device_code()

    assert payload["user_code"] == "ABCD-EFGH"
    assert captured["url"] == codex_auth.CODEX_DEVICE_USER_CODE_URL
    assert captured["kwargs"]["proxy"] == "http://127.0.0.1:7897/"
    assert captured["kwargs"]["trust_env"] is False


def test_request_device_code_normalizes_socks_proxy_scheme(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"user_code": "ABCD-EFGH", "device_auth_id": "device-1"}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    payload = codex_auth.request_device_code()

    assert payload["user_code"] == "ABCD-EFGH"
    assert captured["kwargs"]["proxy"] == "socks5://127.0.0.1:7897/"
    assert captured["kwargs"]["trust_env"] is False


def test_build_httpx_client_for_url_prefers_https_proxy(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897/")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897/")

    client = codex_auth.build_httpx_client_for_url("https://auth.openai.com/oauth/token")

    transport = getattr(client, "_transport", None)
    assert transport is not None
    assert client._trust_env is False
    client.close()


def test_device_code_login_saves_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    calls = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url == codex_auth.CODEX_DEVICE_USER_CODE_URL:
            return FakeResponse(
                200,
                {
                    "user_code": "ABCD-EFGH",
                    "device_auth_id": "device-1",
                    "interval": "3",
                },
            )
        if url == codex_auth.CODEX_DEVICE_TOKEN_URL:
            return FakeResponse(
                200,
                {
                    "authorization_code": "auth-code",
                    "code_verifier": "verifier",
                },
            )
        if url == codex_auth.CODEX_OAUTH_TOKEN_URL:
            assert kwargs["data"]["grant_type"] == "authorization_code"
            assert kwargs["data"]["redirect_uri"] == codex_auth.CODEX_DEVICE_REDIRECT_URI
            return FakeResponse(
                200,
                {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                },
            )
        raise AssertionError(f"unexpected URL: {url}")

    seen_codes = []
    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    tokens = codex_auth.device_code_login(
        sleep_fn=lambda _: None,
        on_user_code=lambda code, url: seen_codes.append((code, url)),
    )

    assert tokens == {"access_token": "access-token", "refresh_token": "refresh-token"}
    assert seen_codes == [("ABCD-EFGH", codex_auth.CODEX_DEVICE_AUTHORIZE_URL)]
    state = codex_auth.read_codex_tokens()
    assert state["tokens"]["access_token"] == "access-token"
    assert [url for url, _ in calls] == [
        codex_auth.CODEX_DEVICE_USER_CODE_URL,
        codex_auth.CODEX_DEVICE_TOKEN_URL,
        codex_auth.CODEX_OAUTH_TOKEN_URL,
    ]


def test_device_poll_retries_transient_network_errors(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "authorization_code": "auth-code",
                "code_verifier": "verifier",
            }

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise codex_auth.httpx.ConnectError("tls eof")
        return FakeResponse()

    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    result = codex_auth.poll_device_authorization(
        device_auth_id="device",
        user_code="code",
        timeout_seconds=20,
        sleep_fn=lambda _: None,
    )

    assert result == {"authorization_code": "auth-code", "code_verifier": "verifier"}
    assert calls["count"] == 2


def test_token_exchange_wraps_network_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise codex_auth.httpx.ConnectError("tls eof")

    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)

    with pytest.raises(codex_auth.CodexAuthError) as exc:
        codex_auth.exchange_device_authorization(
            authorization_code="auth-code",
            code_verifier="verifier",
        )

    assert exc.value.code == "token_exchange_request_failed"
    assert exc.value.relogin_required is True


def test_codex_login_status_when_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))

    assert codex_login.main(["status"]) == 1

    out = capsys.readouterr().out
    assert "codex-oauth: not logged in" in out
    assert "access_token" not in out
    assert "refresh_token" not in out


def test_codex_login_status_when_present(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path))
    codex_auth.save_codex_tokens({"access_token": "at", "refresh_token": "rt"})

    assert codex_login.main(["status"]) == 0

    out = capsys.readouterr().out
    assert "codex-oauth: logged in" in out
    assert "at" not in out
    assert "rt" not in out
