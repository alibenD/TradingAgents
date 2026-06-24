#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="cli"
CONFIG_PATH="${TRADINGAGENTS_WEEKLY_CONFIG:-$REPO_ROOT/configs/watchlists/weekly_deep_portfolio_research.example.yaml}"
FORCE_REINSTALL=0

usage() {
  cat <<'EOF'
Usage: ./scripts/bringup_tradingagents_codex.sh [options] [-- extra-args]

Options:
  --mode MODE           One of: cli, weekly, login, status, doctor
  --config PATH         Weekly runner config path
  --force-reinstall     Reinstall dependencies even if .venv already exists
  --help                Show this message

Examples:
  ./scripts/bringup_tradingagents_codex.sh
  ./scripts/bringup_tradingagents_codex.sh --mode login
  ./scripts/bringup_tradingagents_codex.sh --mode weekly --config ./my-watchlist.yaml
  ./scripts/bringup_tradingagents_codex.sh --mode doctor
EOF
}

EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:?missing value for --config}"
      shift 2
      ;;
    --force-reinstall)
      FORCE_REINSTALL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 127
  fi
}

need_cmd python3

sanitize_proxy_env() {
  local http_proxy_value="${HTTP_PROXY:-${http_proxy:-}}"
  local https_proxy_value="${HTTPS_PROXY:-${https_proxy:-}}"
  local all_proxy_value="${ALL_PROXY:-${all_proxy:-}}"
  local keep_all_proxy="${TRADINGAGENTS_KEEP_ALL_PROXY:-0}"

  if [[ "$keep_all_proxy" == "1" ]]; then
    return
  fi

  # Old proxy managers often export ALL_PROXY=socks://... alongside working
  # HTTP(S)_PROXY values. Several Python HTTP stacks reject the bare
  # "socks://" scheme, so prefer the explicit HTTP/HTTPS proxy settings when
  # they are already present.
  if [[ -n "$all_proxy_value" && ( -n "$http_proxy_value" || -n "$https_proxy_value" ) ]]; then
    echo "[bringup] Unsetting ALL_PROXY/all_proxy and using HTTP_PROXY/HTTPS_PROXY"
    unset ALL_PROXY || true
    unset all_proxy || true
  fi
}

ensure_venv() {
  if [[ ! -d ".venv" ]]; then
    echo "[bringup] Creating virtual environment in $REPO_ROOT/.venv"
    python3 -m venv .venv
    .venv/bin/python -m pip install -U pip setuptools wheel
    .venv/bin/python -m pip install -e . --no-build-isolation
    return
  fi

  if (( FORCE_REINSTALL )); then
    echo "[bringup] Reinstalling TradingAgents and dependencies"
    .venv/bin/python -m pip install -U pip setuptools wheel
    .venv/bin/python -m pip install -e . --no-build-isolation
    return
  fi

  # Always resync the editable package to the current checkout so branch
  # switches do not leave stale code in the virtual environment.
  echo "[bringup] Syncing current checkout into .venv"
  .venv/bin/python -m pip install -e . --no-deps --no-build-isolation
}

ensure_env_file() {
  if [[ ! -f ".env" && -f ".env.example" ]]; then
    echo "[bringup] Creating .env from .env.example"
    cp .env.example .env
  fi
}

export_defaults() {
  export TRADINGAGENTS_LLM_PROVIDER="${TRADINGAGENTS_LLM_PROVIDER:-codex-oauth}"
  export TRADINGAGENTS_DEEP_THINK_LLM="${TRADINGAGENTS_DEEP_THINK_LLM:-gpt-5.4}"
  export TRADINGAGENTS_QUICK_THINK_LLM="${TRADINGAGENTS_QUICK_THINK_LLM:-gpt-5.4-mini}"
  export TRADINGAGENTS_OUTPUT_LANGUAGE="${TRADINGAGENTS_OUTPUT_LANGUAGE:-Chinese}"
}

activate_venv() {
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

print_doctor() {
  echo "[doctor] repo=$REPO_ROOT"
  echo "[doctor] python=$(python --version 2>&1)"
  echo "[doctor] provider=${TRADINGAGENTS_LLM_PROVIDER}"
  echo "[doctor] deep_think_model=${TRADINGAGENTS_DEEP_THINK_LLM}"
  echo "[doctor] quick_think_model=${TRADINGAGENTS_QUICK_THINK_LLM}"
  echo "[doctor] output_language=${TRADINGAGENTS_OUTPUT_LANGUAGE}"
  echo "[doctor] env_file=$REPO_ROOT/.env"
  echo "[doctor] HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-<unset>}}"
  echo "[doctor] HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-<unset>}}"
  echo "[doctor] ALL_PROXY=${ALL_PROXY:-${all_proxy:-<unset>}}"
  python -m tradingagents.llm_clients.codex_login status
}

ensure_login() {
  if python -m tradingagents.llm_clients.codex_login status >/dev/null; then
    return
  fi

  echo "[bringup] Codex OAuth login required"
  python -m tradingagents.llm_clients.codex_login login
}

ensure_venv
ensure_env_file
activate_venv
sanitize_proxy_env
export_defaults

case "$MODE" in
  doctor)
    print_doctor
    ;;
  status)
    python -m tradingagents.llm_clients.codex_login status
    ;;
  login)
    python -m tradingagents.llm_clients.codex_login login
    ;;
  cli)
    ensure_login
    exec python -m cli.main "${EXTRA_ARGS[@]}"
    ;;
  weekly)
    ensure_login
    exec python -m tradingagents.automation.weekly_runner --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
