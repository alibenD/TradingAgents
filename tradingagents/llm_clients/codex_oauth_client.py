"""TradingAgents LLM client for ChatGPT/Codex OAuth-backed Codex backend."""

from __future__ import annotations

from typing import Any

from .base_client import BaseLLMClient
from .codex_auth import resolve_codex_runtime_credentials
from .codex_responses_model import CodexResponsesChatModel
from .validators import validate_model


_PASSTHROUGH_KWARGS = (
    "timeout",
    "max_retries",
    "callbacks",
    "reasoning_effort",
)


class CodexOAuthClient(BaseLLMClient):
    """Client for TradingAgents-owned ChatGPT/Codex OAuth sessions."""

    provider = "codex-oauth"

    def get_llm(self) -> Any:
        """Return a LangChain chat model using the Codex Responses backend."""
        self.warn_if_unknown_model()
        creds = resolve_codex_runtime_credentials()
        llm_kwargs = {
            "model": self.model,
            "base_url": creds["base_url"],
            "api_key": creds["api_key"],
        }
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return CodexResponsesChatModel(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)

