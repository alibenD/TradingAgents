#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -U pip
  .venv/bin/python -m pip install -e .
fi

source .venv/bin/activate

export TRADINGAGENTS_LLM_PROVIDER="${TRADINGAGENTS_LLM_PROVIDER:-codex-oauth}"
export TRADINGAGENTS_DEEP_THINK_LLM="${TRADINGAGENTS_DEEP_THINK_LLM:-gpt-5.4}"
export TRADINGAGENTS_QUICK_THINK_LLM="${TRADINGAGENTS_QUICK_THINK_LLM:-gpt-5.4-mini}"
export TRADINGAGENTS_OUTPUT_LANGUAGE="${TRADINGAGENTS_OUTPUT_LANGUAGE:-Chinese}"

python -m tradingagents.llm_clients.codex_login status || {
  echo
  echo "Codex OAuth credentials are missing or invalid."
  echo "Run: python -m tradingagents.llm_clients.codex_login login"
  exit 2
}

python -m cli.main "$@"
