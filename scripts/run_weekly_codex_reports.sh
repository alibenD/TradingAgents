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

CONFIG_PATH="${TRADINGAGENTS_WEEKLY_CONFIG:-$REPO_ROOT/configs/watchlists/weekly_deep_portfolio_research.example.yaml}"

python -m tradingagents.llm_clients.codex_login status
python -m tradingagents.automation.weekly_runner --config "$CONFIG_PATH" "$@"
