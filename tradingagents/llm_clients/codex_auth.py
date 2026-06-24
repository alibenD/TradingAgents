"""Codex OAuth token storage and refresh for TradingAgents.

This module owns TradingAgents' Codex OAuth state. It intentionally does not
read or write Hermes auth stores, browser credential stores, or Codex CLI auth
files. A separate TradingAgents-owned session avoids refresh-token rotation
conflicts with Codex CLI, VS Code, or Hermes.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
CODEX_DEVICE_USER_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
CODEX_DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
CODEX_DEVICE_AUTHORIZE_URL = "https://auth.openai.com/codex/device"
CODEX_DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"


class CodexAuthError(RuntimeError):
    """Raised when Codex OAuth credentials are missing, invalid, or expired."""

    def __init__(self, message: str, *, code: str, relogin_required: bool = False):
        super().__init__(message)
        self.code = code
        self.relogin_required = relogin_required


def tradingagents_home() -> Path:
    """Return the TradingAgents user state directory."""
    return Path(os.getenv("TRADINGAGENTS_HOME", "~/.tradingagents")).expanduser()


def codex_auth_path() -> Path:
    """Return the TradingAgents-owned Codex OAuth credential path."""
    return tradingagents_home() / "auth" / "codex_oauth.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_proxy_url(raw_proxy: str) -> str:
    """Normalize proxy URLs for httpx compatibility."""
    proxy = str(raw_proxy or "").strip()
    if proxy.startswith("socks://"):
        return "socks5://" + proxy[len("socks://") :]
    return proxy


def _proxy_for_url(url: str) -> str | None:
    """Resolve a proxy URL for the target request.

    Prefer scheme-specific proxy environment variables over ``ALL_PROXY`` so a
    generic SOCKS proxy does not override a working HTTP(S) proxy.
    """
    scheme = urlparse(url).scheme.lower()
    candidates: list[str | None] = []
    if scheme == "https":
        candidates.extend((os.getenv("HTTPS_PROXY"), os.getenv("https_proxy")))
    elif scheme == "http":
        candidates.extend((os.getenv("HTTP_PROXY"), os.getenv("http_proxy")))
    candidates.extend((os.getenv("ALL_PROXY"), os.getenv("all_proxy")))

    for raw_proxy in candidates:
        proxy = _normalize_proxy_url(str(raw_proxy or "").strip())
        if proxy:
            return proxy
    return None


def _httpx_post(url: str, **kwargs: Any) -> httpx.Response:
    """POST with proxy handling that is resilient to common local proxy envs."""
    request_kwargs = dict(kwargs)
    proxy = _proxy_for_url(url)
    if proxy:
        request_kwargs["proxy"] = proxy
        request_kwargs["trust_env"] = False
    return httpx.post(url, **request_kwargs)


def build_httpx_client_for_url(
    url: str,
    *,
    timeout: float | None = None,
) -> httpx.Client:
    """Build an httpx client with TradingAgents proxy normalization."""
    kwargs: dict[str, Any] = {}
    proxy = _proxy_for_url(url)
    if proxy:
        kwargs["proxy"] = proxy
        kwargs["trust_env"] = False
    if timeout is not None:
        kwargs["timeout"] = timeout
    return httpx.Client(**kwargs)


def save_codex_tokens(tokens: dict[str, str], *, last_refresh: str | None = None) -> None:
    """Atomically save TradingAgents-owned Codex OAuth tokens."""
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not access_token or not refresh_token:
        raise CodexAuthError(
            "Codex OAuth tokens must include access_token and refresh_token.",
            code="codex_auth_invalid_tokens",
            relogin_required=True,
        )

    path = codex_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "provider": "codex-oauth",
        "auth_mode": "chatgpt",
        "base_url": DEFAULT_CODEX_BASE_URL,
        "last_refresh": last_refresh or _utc_now(),
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def read_codex_tokens() -> dict[str, Any]:
    """Read TradingAgents-owned Codex OAuth tokens."""
    path = codex_auth_path()
    if not path.is_file():
        raise CodexAuthError(
            "No TradingAgents Codex OAuth credentials found. "
            "Run `python -m tradingagents.llm_clients.codex_login login`.",
            code="codex_auth_missing",
            relogin_required=True,
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CodexAuthError(
            "TradingAgents Codex OAuth credentials are not valid JSON.",
            code="codex_auth_invalid_json",
            relogin_required=True,
        ) from exc

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise CodexAuthError(
            "TradingAgents Codex OAuth credentials are missing tokens.",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )

    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not access_token:
        raise CodexAuthError(
            "Codex auth is missing access_token.",
            code="codex_auth_missing_access_token",
            relogin_required=True,
        )
    if not refresh_token:
        raise CodexAuthError(
            "Codex auth is missing refresh_token.",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )
    return payload


def _jwt_exp(access_token: str) -> int | None:
    parts = access_token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8"))
    except Exception:
        return None
    exp = data.get("exp")
    return int(exp) if isinstance(exp, (int, float)) else None


def access_token_is_expiring(
    access_token: str,
    skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> bool:
    """Return True when a JWT access token expires within ``skew_seconds``."""
    exp = _jwt_exp(access_token)
    if exp is None:
        return False
    return exp <= int(time.time()) + int(skew_seconds)


def refresh_codex_oauth(
    access_token: str,
    refresh_token: str,
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, str]:
    """Refresh Codex OAuth tokens without printing or logging token values."""
    del access_token
    refresh_token = str(refresh_token or "").strip()
    if not refresh_token:
        raise CodexAuthError(
            "Codex auth is missing refresh_token.",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    try:
        response = _httpx_post(
            CODEX_OAUTH_TOKEN_URL,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
            timeout=max(5.0, float(timeout_seconds)),
        )
    except httpx.HTTPError as exc:
        raise CodexAuthError(
            f"Codex token refresh request failed: {exc}",
            code="codex_refresh_request_failed",
            relogin_required=False,
        ) from exc

    if response.status_code != 200:
        code = "codex_refresh_failed"
        message = f"Codex token refresh failed with status {response.status_code}."
        relogin_required = response.status_code in {401, 403}
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    nested_code = err.get("code") or err.get("type")
                    nested_msg = err.get("message")
                    if isinstance(nested_code, str) and nested_code.strip():
                        code = nested_code.strip()
                    if isinstance(nested_msg, str) and nested_msg.strip():
                        message = f"Codex token refresh failed: {nested_msg.strip()}"
                elif isinstance(err, str) and err.strip():
                    code = err.strip()
                    err_desc = body.get("error_description") or body.get("message")
                    if isinstance(err_desc, str) and err_desc.strip():
                        message = f"Codex token refresh failed: {err_desc.strip()}"
        except Exception:
            pass

        if code in {"invalid_grant", "invalid_token", "invalid_request", "refresh_token_reused"}:
            relogin_required = True
        raise CodexAuthError(message, code=code, relogin_required=relogin_required)

    try:
        payload = response.json()
    except Exception as exc:
        raise CodexAuthError(
            "Codex token refresh returned invalid JSON.",
            code="codex_refresh_invalid_json",
            relogin_required=True,
        ) from exc

    new_access = str(payload.get("access_token", "") or "").strip()
    if not new_access:
        raise CodexAuthError(
            "Codex token refresh response was missing access_token.",
            code="codex_refresh_missing_access_token",
            relogin_required=True,
        )
    new_refresh = str(payload.get("refresh_token", "") or "").strip() or refresh_token
    return {"access_token": new_access, "refresh_token": new_refresh}


def request_device_code(*, timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Request a Codex OAuth device code."""
    try:
        response = _httpx_post(
            CODEX_DEVICE_USER_CODE_URL,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json={"client_id": CODEX_OAUTH_CLIENT_ID},
            timeout=max(5.0, float(timeout_seconds)),
        )
    except httpx.HTTPError as exc:
        raise CodexAuthError(
            f"Codex device-code request failed: {exc}",
            code="device_code_request_failed",
            relogin_required=True,
        ) from exc
    if response.status_code != 200:
        raise CodexAuthError(
            f"Codex device-code request failed with status {response.status_code}.",
            code="device_code_request_error",
            relogin_required=True,
        )
    payload = response.json()
    user_code = str(payload.get("user_code", "") or "").strip()
    device_auth_id = str(payload.get("device_auth_id", "") or "").strip()
    if not user_code or not device_auth_id:
        raise CodexAuthError(
            "Codex device-code response was missing user_code or device_auth_id.",
            code="device_code_incomplete",
            relogin_required=True,
        )
    return payload


def poll_device_authorization(
    *,
    device_auth_id: str,
    user_code: str,
    interval_seconds: float = 5.0,
    timeout_seconds: float = 900.0,
    request_timeout_seconds: float = 20.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Poll Codex OAuth device authorization until the user completes login."""
    deadline = time.monotonic() + timeout_seconds
    interval = max(3.0, float(interval_seconds))
    while time.monotonic() < deadline:
        try:
            response = _httpx_post(
                CODEX_DEVICE_TOKEN_URL,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                json={"device_auth_id": device_auth_id, "user_code": user_code},
                timeout=max(5.0, float(request_timeout_seconds)),
            )
        except httpx.HTTPError:
            sleep_fn(interval)
            continue
        if response.status_code == 200:
            payload = response.json()
            authorization_code = str(payload.get("authorization_code", "") or "").strip()
            code_verifier = str(payload.get("code_verifier", "") or "").strip()
            if not authorization_code or not code_verifier:
                raise CodexAuthError(
                    "Codex device authorization completed but exchange fields were missing.",
                    code="device_code_incomplete_exchange",
                    relogin_required=True,
                )
            return {"authorization_code": authorization_code, "code_verifier": code_verifier}
        if response.status_code not in {403, 404}:
            raise CodexAuthError(
                f"Codex device authorization poll failed with status {response.status_code}.",
                code="device_code_poll_error",
                relogin_required=True,
            )
        sleep_fn(interval)
    raise CodexAuthError(
        "Timed out waiting for Codex device authorization.",
        code="device_code_timeout",
        relogin_required=True,
    )


def exchange_device_authorization(
    *,
    authorization_code: str,
    code_verifier: str,
    timeout_seconds: float = 20.0,
) -> dict[str, str]:
    """Exchange a completed device authorization for Codex OAuth tokens."""
    try:
        response = _httpx_post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": CODEX_DEVICE_REDIRECT_URI,
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            timeout=max(5.0, float(timeout_seconds)),
        )
    except httpx.HTTPError as exc:
        raise CodexAuthError(
            f"Codex token exchange request failed: {exc}",
            code="token_exchange_request_failed",
            relogin_required=True,
        ) from exc
    if response.status_code != 200:
        raise CodexAuthError(
            f"Codex token exchange failed with status {response.status_code}.",
            code="token_exchange_error",
            relogin_required=True,
        )
    payload = response.json()
    access_token = str(payload.get("access_token", "") or "").strip()
    refresh_token = str(payload.get("refresh_token", "") or "").strip()
    if not access_token:
        raise CodexAuthError(
            "Codex token exchange response was missing access_token.",
            code="token_exchange_no_access_token",
            relogin_required=True,
        )
    if not refresh_token:
        raise CodexAuthError(
            "Codex token exchange response was missing refresh_token.",
            code="token_exchange_no_refresh_token",
            relogin_required=True,
        )
    return {"access_token": access_token, "refresh_token": refresh_token}


def device_code_login(
    *,
    timeout_seconds: float = 900.0,
    request_timeout_seconds: float = 20.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_user_code: Callable[[str, str], None] | None = None,
) -> dict[str, str]:
    """Run the Codex OAuth device-code login flow and persist tokens."""
    device = request_device_code(timeout_seconds=request_timeout_seconds)
    user_code = str(device["user_code"]).strip()
    device_auth_id = str(device["device_auth_id"]).strip()
    raw_interval = device.get("interval", 5)
    try:
        interval = float(raw_interval)
    except (TypeError, ValueError):
        interval = 5.0

    if on_user_code:
        on_user_code(user_code, CODEX_DEVICE_AUTHORIZE_URL)

    exchange = poll_device_authorization(
        device_auth_id=device_auth_id,
        user_code=user_code,
        interval_seconds=interval,
        timeout_seconds=timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        sleep_fn=sleep_fn,
    )
    tokens = exchange_device_authorization(
        authorization_code=exchange["authorization_code"],
        code_verifier=exchange["code_verifier"],
        timeout_seconds=request_timeout_seconds,
    )
    save_codex_tokens(tokens)
    return tokens


def resolve_codex_runtime_credentials(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return a usable Codex bearer token and backend URL for runtime calls."""
    state = read_codex_tokens()
    tokens = dict(state["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()

    if force_refresh or access_token_is_expiring(access_token):
        refreshed = refresh_codex_oauth(access_token, str(tokens.get("refresh_token", "") or ""))
        tokens.update(refreshed)
        save_codex_tokens(tokens)
        access_token = tokens["access_token"]

    return {
        "provider": "codex-oauth",
        "base_url": str(state.get("base_url") or DEFAULT_CODEX_BASE_URL).rstrip("/"),
        "api_key": access_token,
        "source": "tradingagents-auth-store",
        "auth_mode": "chatgpt",
        "last_refresh": state.get("last_refresh"),
    }
