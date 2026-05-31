"""Weekly portfolio research runner for TradingAgents.

The runner is intentionally non-interactive: it reads a YAML config, runs
TradingAgentsGraph for each configured symbol, and writes report artifacts to a
knowledge-vault directory instead of the source repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf
import yaml

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


DEFAULT_CONFIG_PATH = (
    Path("configs/watchlists/weekly_deep_portfolio_research.example.yaml")
)


@dataclass(frozen=True)
class SymbolJob:
    symbol: str
    group: str
    profile: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class SymbolResult:
    symbol: str
    group: str
    status: str
    rating: str = "Unknown"
    decision: str = ""
    report_path: str = ""
    data_quality: str = "unknown"
    error: str = ""
    duration_seconds: float = 0.0


class SymbolTimeoutError(TimeoutError):
    """Raised when one symbol exceeds the configured wall-time limit."""


def classify_execution_error(error: str) -> str:
    text = (error or "").lower()
    if any(
        marker in text
        for marker in (
            "codexautherror",
            "codex_auth_",
            "oauth",
            "unauthorized",
            "invalid_grant",
            "invalid token",
            "relogin",
            "status 401",
            "status 403",
            " 401",
            " 403",
        )
    ):
        return "oauth"
    if any(
        marker in text
        for marker in (
            "quota",
            "rate limit",
            "ratelimit",
            "too many requests",
            "status 429",
            " 429",
            "insufficient_quota",
        )
    ):
        return "quota"
    return "symbol"


def should_stop_for_execution_error(result: SymbolResult, execution: dict[str, Any]) -> bool:
    if result.status == "ok":
        return False
    category = classify_execution_error(result.error)
    if category == "oauth":
        return bool(execution.get("stop_on_oauth_error", True))
    if category == "quota":
        return bool(execution.get("stop_on_quota_error", True))
    return False


def load_weekly_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Weekly config must be a mapping: {config_path}")
    return config


def resolve_as_of_date(config: dict[str, Any], override: str | None = None) -> date:
    if override:
        return datetime.strptime(override, "%Y-%m-%d").date()

    schedule = config.get("schedule") or {}
    timezone_name = schedule.get("timezone") or "Asia/Shanghai"
    as_of = (config.get("dates") or {}).get("as_of", "run_date")
    if as_of == "run_date":
        return datetime.now(ZoneInfo(timezone_name)).date()
    if isinstance(as_of, str):
        return datetime.strptime(as_of, "%Y-%m-%d").date()
    raise ValueError(f"Unsupported dates.as_of value: {as_of!r}")


def latest_trading_day(symbol: str, as_of: date) -> str:
    end_date = as_of + timedelta(days=1)
    start_date = as_of - timedelta(days=14)
    history = yf.Ticker(symbol).history(start=start_date.isoformat(), end=end_date.isoformat())
    if history.empty:
        raise ValueError(f"No OHLCV data found for {symbol} on or before {as_of.isoformat()}.")
    index = history.index
    if hasattr(index, "tz_localize") and getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    dates = [d.date() for d in index if d.date() <= as_of]
    if not dates:
        raise ValueError(f"No OHLCV rows on or before {as_of.isoformat()} for {symbol}.")
    return max(dates).isoformat()


def iter_symbol_jobs(config: dict[str, Any]) -> list[SymbolJob]:
    profiles = config.get("research_profiles") or {}
    metadata = config.get("metadata") or {}
    symbols_by_group = config.get("symbols") or {}
    jobs: list[SymbolJob] = []
    for group, symbols in symbols_by_group.items():
        profile = profiles.get(group)
        if not isinstance(profile, dict):
            raise ValueError(f"Missing research profile for symbol group {group!r}.")
        for symbol in symbols or []:
            symbol = str(symbol).strip().upper()
            if symbol:
                jobs.append(
                    SymbolJob(
                        symbol=symbol,
                        group=str(group),
                        profile=profile,
                        metadata=metadata.get(symbol, {}),
                    )
                )
    if not jobs:
        raise ValueError("Weekly config has no symbols to analyze.")
    return jobs


def weekly_run_dir(config: dict[str, Any], as_of: date) -> Path:
    output = config.get("output") or {}
    vault_root = Path(output["vault_root"]).expanduser()
    weekly_dir = Path(output["weekly_dir"])
    return vault_root / weekly_dir / str(as_of.year) / as_of.isoformat()


def symbol_index_dir(config: dict[str, Any]) -> Path:
    output = config.get("output") or {}
    vault_root = Path(output["vault_root"]).expanduser()
    return vault_root / Path(output["symbol_dir"])


def tradingagents_base_config(
    config: dict[str, Any],
    profile: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    llm = config.get("llm") or {}
    execution = config.get("execution") or {}
    state_root = run_dir / "_TradingAgentsState"
    result = DEFAULT_CONFIG.copy()
    result.update(
        {
            "llm_provider": llm.get("provider", "codex-oauth"),
            "quick_think_llm": llm.get("quick_model", "gpt-5.4-mini"),
            "deep_think_llm": llm.get("deep_model", "gpt-5.4"),
            "output_language": config.get("output_language") or "Chinese",
            "max_debate_rounds": int(profile.get("research_depth", 1)),
            "max_risk_discuss_rounds": int(profile.get("research_depth", 1)),
            "analyst_concurrency_limit": int(execution.get("analyst_concurrency_limit", 1)),
            "checkpoint_enabled": bool(execution.get("checkpoint_enabled", True)),
            "news_article_limit": int(profile.get("news_article_limit", 20)),
            "global_news_article_limit": int(profile.get("global_news_article_limit", 10)),
            "global_news_lookback_days": int(profile.get("global_news_lookback_days", 7)),
            "results_dir": str(state_root / "logs"),
            "data_cache_dir": str(state_root / "cache"),
            "memory_log_path": str(state_root / "memory/trading_memory.md"),
        }
    )
    if llm.get("reasoning_effort"):
        result["openai_reasoning_effort"] = llm["reasoning_effort"]
    if "temperature" in llm:
        result["temperature"] = llm["temperature"]
    return result


def markdown_section(title: str, content: str | None) -> str:
    content = (content or "").strip()
    if not content:
        content = "_No content produced._"
    return f"## {title}\n\n{content}\n"


def render_symbol_report(
    *,
    job: SymbolJob,
    state: dict[str, Any],
    decision: str,
    as_of: date,
    market_data_date: str,
    config: dict[str, Any],
) -> str:
    news_window = ((config.get("dates") or {}).get("news_window") or {})
    lookback = int(news_window.get("lookback_days", 7))
    news_start = as_of - timedelta(days=lookback)
    theme = job.metadata.get("theme", "")
    parts = [
        f"# {job.symbol} Weekly TradingAgents Research\n",
        f"- Group: `{job.group}`",
        f"- Theme: {theme or 'Unspecified'}",
        f"- Report as of: {as_of.isoformat()} 18:00 Asia/Shanghai",
        f"- Market data through: {market_data_date} close",
        f"- News window: {news_start.isoformat()} to {as_of.isoformat()}",
        f"- Decision: `{decision}`",
        "- Source policy: use TradingAgents tool-backed data where available; "
        "treat uncited dynamic numbers as `needs-source` until manually verified.",
        "",
        markdown_section("Market Report", state.get("market_report")),
        markdown_section("Sentiment Report", state.get("sentiment_report")),
        markdown_section("News Report", state.get("news_report")),
        markdown_section("Fundamentals Report", state.get("fundamentals_report")),
        markdown_section("Research Manager Plan", state.get("investment_plan")),
        markdown_section("Trader Plan", state.get("trader_investment_plan")),
        markdown_section("Final Portfolio Decision", state.get("final_trade_decision")),
    ]
    return "\n".join(parts)


def write_symbol_report(run_dir: Path, job: SymbolJob, report: str) -> Path:
    path = run_dir / f"{safe_ticker_component(job.symbol).upper()}.md"
    path.write_text(report, encoding="utf-8")
    return path


def update_symbol_index(config: dict[str, Any], symbol: str, report_path: Path, as_of: date) -> None:
    index_dir = symbol_index_dir(config)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{safe_ticker_component(symbol).upper()}.md"
    vault_root = Path((config.get("output") or {})["vault_root"]).expanduser()
    try:
        rel = report_path.relative_to(vault_root).with_suffix("")
    except ValueError:
        rel = report_path.with_suffix("")
    link = f"- {as_of.isoformat()}: [[{rel.as_posix()}]]"
    if index_path.exists():
        text = index_path.read_text(encoding="utf-8")
        if link in text:
            return
        if "## Weekly Reports" in text:
            text = text.rstrip() + "\n" + link + "\n"
        else:
            text = text.rstrip() + "\n\n## Weekly Reports\n" + link + "\n"
    else:
        text = f"# {symbol}\n\n## Weekly Reports\n{link}\n"
    index_path.write_text(text, encoding="utf-8")


def _timeout_handler(signum: int, frame: Any) -> None:
    del signum, frame
    raise SymbolTimeoutError("symbol analysis exceeded configured timeout")


def run_symbol(job: SymbolJob, config: dict[str, Any], as_of: date, run_dir: Path) -> SymbolResult:
    started = time.monotonic()
    execution = config.get("execution") or {}
    timeout_minutes = float(execution.get("per_symbol_timeout_minutes", 45))
    previous_handler = None
    try:
        if timeout_minutes > 0:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max(1, int(timeout_minutes * 60)))
        market_data_date = latest_trading_day(job.symbol, as_of)
        graph_config = tradingagents_base_config(config, job.profile, run_dir)
        graph = TradingAgentsGraph(
            selected_analysts=list(job.profile.get("analysts") or ["market", "news"]),
            config=graph_config,
            debug=False,
        )
        state, decision = graph.propagate(job.symbol, as_of.isoformat(), asset_type="stock")
        report = render_symbol_report(
            job=job,
            state=state,
            decision=decision,
            as_of=as_of,
            market_data_date=market_data_date,
            config=config,
        )
        output = config.get("output") or {}
        report_path = None
        if output.get("write_symbol_reports", True):
            report_path = write_symbol_report(run_dir, job, report)
        history = config.get("history") or {}
        if report_path is not None and history.get("enabled", True) and history.get("update_symbol_index", True):
            update_symbol_index(config, job.symbol, report_path, as_of)
        rating = parse_rating(state.get("final_trade_decision", ""))
        return SymbolResult(
            symbol=job.symbol,
            group=job.group,
            status="ok",
            rating=rating,
            decision=decision,
            report_path=str(report_path or ""),
            data_quality="generated",
            duration_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        code = getattr(exc, "code", None)
        error = f"{exc.__class__.__name__}: {exc}"
        if code:
            error = f"{error} [{code}]"
        return SymbolResult(
            symbol=job.symbol,
            group=job.group,
            status="failed",
            error=error,
            duration_seconds=time.monotonic() - started,
        )
    finally:
        if timeout_minutes > 0:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


def write_portfolio_matrix(run_dir: Path, results: list[SymbolResult]) -> Path:
    path = run_dir / "Portfolio Matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "group",
                "status",
                "rating",
                "decision",
                "data_quality",
                "duration_seconds",
                "report_path",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "symbol": result.symbol,
                    "group": result.group,
                    "status": result.status,
                    "rating": result.rating,
                    "decision": result.decision,
                    "data_quality": result.data_quality,
                    "duration_seconds": f"{result.duration_seconds:.1f}",
                    "report_path": result.report_path,
                    "error": result.error,
                }
            )
    return path


def write_weekly_summary(run_dir: Path, as_of: date, results: list[SymbolResult]) -> Path:
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    lines = [
        f"# Weekly Portfolio Research - {as_of.isoformat()}",
        "",
        "## Run Summary",
        "",
        f"- Completed symbols: {len(ok)}",
        f"- Failed symbols: {len(failed)}",
        "",
        "## Portfolio Matrix",
        "",
        "| Symbol | Group | Status | Rating | Decision | Report |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        report_link = ""
        if result.report_path:
            report_link = f"[{result.symbol}]({Path(result.report_path).name})"
        lines.append(
            f"| {result.symbol} | {result.group} | {result.status} | "
            f"{result.rating} | {result.decision} | {report_link} |"
        )
    if failed:
        lines.extend(["", "## Run Issues", ""])
        for result in failed:
            lines.append(f"- `{result.symbol}` failed: {result.error}")
    path = run_dir / "Weekly Portfolio Research.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_run_metadata(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    as_of: date,
    results: list[SymbolResult],
) -> Path:
    payload = {
        "config_path": str(config_path),
        "as_of_date": as_of.isoformat(),
        "generated_at": datetime.now(ZoneInfo((config.get("schedule") or {}).get("timezone", "Asia/Shanghai"))).isoformat(),
        "llm": config.get("llm"),
        "source_policy": config.get("source_policy"),
        "results": [result.__dict__ for result in results],
    }
    path = run_dir / "Run Metadata.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def filter_jobs(
    jobs: list[SymbolJob],
    *,
    symbols: list[str] | None = None,
    max_symbols: int | None = None,
) -> list[SymbolJob]:
    if symbols:
        wanted = {symbol.upper() for symbol in symbols}
        jobs = [job for job in jobs if job.symbol in wanted]
    if max_symbols is not None:
        jobs = jobs[:max_symbols]
    return jobs


def run_weekly(
    config_path: str | Path,
    *,
    as_of_override: str | None = None,
    symbols: list[str] | None = None,
    max_symbols: int | None = None,
) -> int:
    config_path = Path(config_path).expanduser()
    config = load_weekly_config(config_path)
    as_of = resolve_as_of_date(config, as_of_override)
    run_dir = weekly_run_dir(config, as_of)
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = filter_jobs(iter_symbol_jobs(config), symbols=symbols, max_symbols=max_symbols)
    if not jobs:
        raise ValueError("No configured symbols matched the requested run filter.")
    results: list[SymbolResult] = []
    execution = config.get("execution") or {}
    delay = float(execution.get("delay_between_symbols_seconds", 0))
    continue_on_failure = bool(execution.get("continue_on_symbol_failure", True))
    retry_count = max(0, int(execution.get("retry_count", 0)))

    for index, job in enumerate(jobs):
        print(f"[{index + 1}/{len(jobs)}] Running {job.symbol} ({job.group})...", flush=True)
        result = run_symbol(job, config, as_of, run_dir)
        for attempt in range(1, retry_count + 1):
            if result.status == "ok":
                break
            if should_stop_for_execution_error(result, execution):
                break
            print(
                f"[{job.symbol}] failed attempt {attempt}/{retry_count + 1}: {result.error}; retrying...",
                flush=True,
            )
            result = run_symbol(job, config, as_of, run_dir)
        results.append(result)
        print(f"[{job.symbol}] {result.status} {result.decision or result.error}", flush=True)
        if result.status != "ok" and (
            not continue_on_failure or should_stop_for_execution_error(result, execution)
        ):
            break
        if delay > 0 and index < len(jobs) - 1:
            time.sleep(delay)

    output = config.get("output") or {}
    if output.get("write_portfolio_matrix", True):
        write_portfolio_matrix(run_dir, results)
    if output.get("write_weekly_summary", True):
        write_weekly_summary(run_dir, as_of, results)
    if output.get("write_run_meta", True):
        write_run_metadata(run_dir, config_path, config, as_of, results)
    print(f"Weekly report directory: {run_dir}", flush=True)
    return 0 if all(result.status == "ok" for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run weekly TradingAgents portfolio research")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to weekly YAML config",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Override report as-of date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol subset to run, e.g. NVDA,AMD,SPY",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Run only the first N configured symbols",
    )
    args = parser.parse_args(argv)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()] if args.symbols else None
    return run_weekly(
        args.config,
        as_of_override=args.as_of,
        symbols=symbols,
        max_symbols=args.max_symbols,
    )


if __name__ == "__main__":
    raise SystemExit(main())
