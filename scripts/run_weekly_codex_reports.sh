#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${TRADINGAGENTS_WEEKLY_CONFIG:-$REPO_ROOT/configs/watchlists/weekly_deep_portfolio_research.example.yaml}"

exec "$SCRIPT_DIR/bringup_tradingagents_codex.sh" --mode weekly --config "$CONFIG_PATH" -- "$@"
