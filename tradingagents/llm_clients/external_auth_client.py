"""External bearer-token provider for OpenAI-compatible gateways.

This provider is intentionally generic. It does not read Codex, Hermes, or
browser auth stores. Instead, users provide a command that prints a short-lived
bearer token to stdout, similar to Codex's command-backed provider pattern.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Optional

from .base_client import BaseLLMClient
from .openai_client import NormalizedChatOpenAI, _PASSTHROUGH_KWARGS
from .validators import validate_model


AUTH_COMMAND_ENV = "TRADINGAGENTS_LLM_AUTH_COMMAND"
AUTH_COMMAND_TIMEOUT_ENV = "TRADINGAGENTS_LLM_AUTH_COMMAND_TIMEOUT"
DEFAULT_AUTH_COMMAND_TIMEOUT = 10


def _auth_command_timeout() -> int:
    raw = os.environ.get(AUTH_COMMAND_TIMEOUT_ENV)
    if not raw:
        return DEFAULT_AUTH_COMMAND_TIMEOUT
    try:
        timeout = int(raw)
    except ValueError as exc:
        raise ValueError(f"{AUTH_COMMAND_TIMEOUT_ENV} must be an integer number of seconds") from exc
    if timeout <= 0:
        raise ValueError(f"{AUTH_COMMAND_TIMEOUT_ENV} must be greater than zero")
    return timeout


def _run_auth_command(command: str, timeout: int) -> str:
    """Run the configured token command and return a stripped bearer token.

    stdout/stderr are deliberately excluded from error messages because token
    helpers often print sensitive diagnostics or accidentally echo secrets.
    """
    try:
        completed = subprocess.run(
            shlex.split(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{AUTH_COMMAND_ENV} executable was not found; check the command path."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{AUTH_COMMAND_ENV} timed out after {timeout} seconds."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{AUTH_COMMAND_ENV} failed with exit code {exc.returncode}; "
            "stdout and stderr were suppressed."
        ) from exc

    token = completed.stdout.strip()
    if not token:
        raise RuntimeError(f"{AUTH_COMMAND_ENV} returned an empty token.")
    return token.splitlines()[0].strip()


class ExternalAuthClient(BaseLLMClient):
    """Client for user-controlled OpenAI-compatible gateways with external auth."""

    provider = "external-auth"

    def get_llm(self) -> Any:
        """Return a ChatOpenAI-compatible instance using a command-backed token."""
        self.warn_if_unknown_model()

        base_url = self.base_url or os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL")
        if not base_url:
            raise ValueError(
                "The external-auth provider requires TRADINGAGENTS_LLM_BACKEND_URL "
                "or an explicit backend_url/base_url."
            )

        auth_command = os.environ.get(AUTH_COMMAND_ENV)
        if not auth_command:
            raise ValueError(
                f"The external-auth provider requires {AUTH_COMMAND_ENV} to point "
                "to a command that prints a bearer token to stdout."
            )

        llm_kwargs = {
            "model": self.model,
            "base_url": base_url,
            "api_key": _run_auth_command(auth_command, _auth_command_timeout()),
        }

        for key in _PASSTHROUGH_KWARGS:
            if key == "api_key":
                continue
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Allow custom gateway model IDs."""
        return validate_model(self.provider, self.model)
