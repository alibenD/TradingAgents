"""Tests for the external-auth LLM provider.

The provider is designed for user-controlled OpenAI-compatible gateways that
need a bearer token from an external command. These tests intentionally avoid
real network calls and real OAuth/Codex credentials.
"""

from __future__ import annotations

import subprocess

import pytest

from tradingagents.llm_clients.factory import create_llm_client


def test_factory_dispatches_external_auth_client():
    client = create_llm_client(
        provider="external-auth",
        model="gateway-model",
        base_url="https://gateway.example/v1",
    )

    assert client.__class__.__name__ == "ExternalAuthClient"


def test_external_auth_requires_backend_url(monkeypatch):
    from tradingagents.llm_clients.external_auth_client import ExternalAuthClient

    monkeypatch.setenv("TRADINGAGENTS_LLM_AUTH_COMMAND", "fetch-token")
    client = ExternalAuthClient(model="gateway-model")

    with pytest.raises(ValueError, match="TRADINGAGENTS_LLM_BACKEND_URL"):
        client.get_llm()


def test_external_auth_requires_auth_command(monkeypatch):
    from tradingagents.llm_clients.external_auth_client import ExternalAuthClient

    monkeypatch.delenv("TRADINGAGENTS_LLM_AUTH_COMMAND", raising=False)
    client = ExternalAuthClient(
        model="gateway-model",
        base_url="https://gateway.example/v1",
    )

    with pytest.raises(ValueError, match="TRADINGAGENTS_LLM_AUTH_COMMAND"):
        client.get_llm()


def test_auth_command_stdout_becomes_api_key(monkeypatch):
    import tradingagents.llm_clients.external_auth_client as mod

    captured_kwargs = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setenv("TRADINGAGENTS_LLM_AUTH_COMMAND", "fetch-token")
    monkeypatch.setattr(mod, "_run_auth_command", lambda command, timeout: "token-123")
    monkeypatch.setattr(mod, "NormalizedChatOpenAI", FakeChat)

    client = mod.ExternalAuthClient(
        model="gateway-model",
        base_url="https://gateway.example/v1",
        temperature=0.0,
        callbacks=["cb"],
    )

    llm = client.get_llm()

    assert isinstance(llm, FakeChat)
    assert captured_kwargs["model"] == "gateway-model"
    assert captured_kwargs["base_url"] == "https://gateway.example/v1"
    assert captured_kwargs["api_key"] == "token-123"
    assert captured_kwargs["temperature"] == 0.0
    assert captured_kwargs["callbacks"] == ["cb"]


def test_auth_command_failure_does_not_leak_stderr():
    from tradingagents.llm_clients.external_auth_client import _run_auth_command

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=7,
            cmd=args[0],
            output="secret stdout",
            stderr="secret stderr",
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError) as exc_info:
            _run_auth_command("fetch-token", timeout=10)

    message = str(exc_info.value)
    assert "exit code 7" in message
    assert "secret" not in message
