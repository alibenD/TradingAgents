"""CLI helpers for TradingAgents Codex OAuth login."""

from __future__ import annotations

import argparse
import sys
import webbrowser

from .codex_auth import (
    CODEX_DEVICE_AUTHORIZE_URL,
    CodexAuthError,
    access_token_is_expiring,
    device_code_login,
    read_codex_tokens,
)


def _print_status() -> int:
    try:
        state = read_codex_tokens()
    except CodexAuthError as exc:
        print(f"codex-oauth: not logged in ({exc.code})", flush=True)
        print("Run: python -m tradingagents.llm_clients.codex_login login", flush=True)
        return 1

    access_token = str(state["tokens"].get("access_token", "") or "")
    if access_token_is_expiring(access_token):
        print("codex-oauth: credentials found, access token is expiring and will refresh on use", flush=True)
    else:
        print("codex-oauth: logged in", flush=True)
    print(f"Auth file: {state.get('provider', 'codex-oauth')} credentials in TradingAgents home", flush=True)
    return 0


def _print_user_code(user_code: str, authorize_url: str) -> None:
    print("", flush=True)
    print("Sign in to OpenAI Codex for TradingAgents", flush=True)
    print(f"Open: {authorize_url}", flush=True)
    print(f"Code: {user_code}", flush=True)
    print("", flush=True)
    try:
        webbrowser.open(authorize_url)
    except Exception:
        pass


def _login() -> int:
    try:
        device_code_login(on_user_code=_print_user_code)
    except KeyboardInterrupt:
        print("\nCodex OAuth login cancelled.", flush=True)
        return 130
    except CodexAuthError as exc:
        print(f"Codex OAuth login failed ({exc.code}): {exc}", flush=True)
        return 1
    print("codex-oauth: login successful", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage TradingAgents Codex OAuth login")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Check Codex OAuth login status")
    subparsers.add_parser("login", help="Run Codex OAuth device-code login")
    args = parser.parse_args(argv)

    if args.command == "status":
        return _print_status()
    if args.command == "login":
        return _login()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
