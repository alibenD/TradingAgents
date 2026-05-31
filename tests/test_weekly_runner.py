from datetime import date
from pathlib import Path

from tradingagents.automation import weekly_runner


def test_public_weekly_runner_files_do_not_embed_personal_absolute_paths():
    repo_root = Path(__file__).resolve().parents[1]
    public_files = [
        repo_root / "tradingagents/automation/weekly_runner.py",
        repo_root / "scripts/run_weekly_codex_reports.sh",
        repo_root / "configs/watchlists/weekly_deep_portfolio_research.example.yaml",
    ]
    forbidden_fragments = [
        "/Users/aliben/",
        "Project/TalentOctopus",
        "TalentOctopus",
    ]

    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in text


def test_iter_symbol_jobs_uses_group_profiles():
    config = {
        "research_profiles": {
            "core_holdings": {"analysts": ["market"], "research_depth": 5},
            "benchmarks": {"analysts": ["market"], "research_depth": 3},
        },
        "symbols": {
            "core_holdings": ["NVDA"],
            "benchmarks": ["SPY", "QQQ"],
        },
        "metadata": {"NVDA": {"theme": "AI Semiconductors"}},
    }

    jobs = weekly_runner.iter_symbol_jobs(config)

    assert [job.symbol for job in jobs] == ["NVDA", "SPY", "QQQ"]
    assert jobs[0].group == "core_holdings"
    assert jobs[0].profile["research_depth"] == 5
    assert jobs[0].metadata["theme"] == "AI Semiconductors"


def test_weekly_run_dir_points_to_vault(tmp_path):
    config = {
        "output": {
            "vault_root": str(tmp_path),
            "weekly_dir": "03-Research/Investing/TradingAgents/Weekly",
        }
    }

    path = weekly_runner.weekly_run_dir(config, date(2026, 6, 7))

    assert path == tmp_path / "03-Research/Investing/TradingAgents/Weekly/2026/2026-06-07"


def test_filter_jobs_by_symbols_and_max_count():
    config = {
        "research_profiles": {"watchlist": {"analysts": ["market"], "research_depth": 1}},
        "symbols": {"watchlist": ["AAPL", "MSFT", "GOOG"]},
    }
    jobs = weekly_runner.iter_symbol_jobs(config)

    assert [job.symbol for job in weekly_runner.filter_jobs(jobs, symbols=["msft", "goog"])] == [
        "MSFT",
        "GOOG",
    ]
    assert [job.symbol for job in weekly_runner.filter_jobs(jobs, max_symbols=2)] == ["AAPL", "MSFT"]


def test_writes_weekly_outputs_and_symbol_index(tmp_path):
    config = {
        "output": {
            "vault_root": str(tmp_path),
            "weekly_dir": "Weekly",
            "symbol_dir": "Symbols",
        },
        "history": {"update_symbol_index": True},
    }
    run_dir = tmp_path / "Weekly/2026/2026-06-07"
    run_dir.mkdir(parents=True)
    result = weekly_runner.SymbolResult(
        symbol="NVDA",
        group="core_holdings",
        status="ok",
        rating="Hold",
        decision="Hold",
        report_path=str(run_dir / "NVDA.md"),
    )
    (run_dir / "NVDA.md").write_text("# NVDA\n", encoding="utf-8")

    matrix = weekly_runner.write_portfolio_matrix(run_dir, [result])
    summary = weekly_runner.write_weekly_summary(run_dir, date(2026, 6, 7), [result])
    weekly_runner.update_symbol_index(config, "NVDA", Path(result.report_path), date(2026, 6, 7))

    assert matrix.is_file()
    assert summary.is_file()
    assert "| NVDA | core_holdings | ok | Hold | Hold | [NVDA](NVDA.md) |" in summary.read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "Symbols/NVDA.md").is_file()
    assert "[[Weekly/2026/2026-06-07/NVDA]]" in (tmp_path / "Symbols/NVDA.md").read_text(
        encoding="utf-8"
    )


def test_run_weekly_retries_failed_symbol(monkeypatch, tmp_path):
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text(
        """
schedule:
  timezone: Asia/Shanghai
dates:
  as_of: "2026-06-07"
research_profiles:
  watchlist:
    analysts: [market]
    research_depth: 1
symbols:
  watchlist: [AAPL]
execution:
  retry_count: 1
  delay_between_symbols_seconds: 0
  continue_on_symbol_failure: true
output:
  vault_root: "~"
  weekly_dir: "weekly-runner-test"
  symbol_dir: "weekly-runner-test-symbols"
history:
  update_symbol_index: false
""",
        encoding="utf-8",
    )
    attempts = []

    def fake_run_symbol(job, config, as_of, run_dir):
        del config, as_of, run_dir
        attempts.append(job.symbol)
        if len(attempts) == 1:
            return weekly_runner.SymbolResult(symbol=job.symbol, group=job.group, status="failed", error="temporary")
        return weekly_runner.SymbolResult(symbol=job.symbol, group=job.group, status="ok", decision="Hold")

    monkeypatch.setattr(weekly_runner, "run_symbol", fake_run_symbol)
    monkeypatch.setattr(weekly_runner, "weekly_run_dir", lambda config, as_of: tmp_path / "run")
    monkeypatch.setattr(weekly_runner, "symbol_index_dir", lambda config: tmp_path / "symbols")

    assert weekly_runner.run_weekly(config_path) == 0
    assert attempts == ["AAPL", "AAPL"]


def test_run_weekly_stops_without_retry_on_oauth_failure(monkeypatch, tmp_path):
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text(
        """
schedule:
  timezone: Asia/Shanghai
dates:
  as_of: "2026-06-07"
research_profiles:
  watchlist:
    analysts: [market]
    research_depth: 1
symbols:
  watchlist: [AAPL, MSFT]
execution:
  retry_count: 2
  delay_between_symbols_seconds: 0
  continue_on_symbol_failure: true
  stop_on_oauth_error: true
output:
  vault_root: "~"
  weekly_dir: "weekly-runner-test"
  symbol_dir: "weekly-runner-test-symbols"
history:
  update_symbol_index: false
""",
        encoding="utf-8",
    )
    attempts = []

    def fake_run_symbol(job, config, as_of, run_dir):
        del config, as_of, run_dir
        attempts.append(job.symbol)
        return weekly_runner.SymbolResult(
            symbol=job.symbol,
            group=job.group,
            status="failed",
            error="CodexAuthError: No TradingAgents Codex OAuth credentials found. [codex_auth_missing]",
        )

    monkeypatch.setattr(weekly_runner, "run_symbol", fake_run_symbol)
    monkeypatch.setattr(weekly_runner, "weekly_run_dir", lambda config, as_of: tmp_path / "run")
    monkeypatch.setattr(weekly_runner, "symbol_index_dir", lambda config: tmp_path / "symbols")

    assert weekly_runner.run_weekly(config_path) == 1
    assert attempts == ["AAPL"]


def test_run_weekly_respects_aggregate_output_flags(monkeypatch, tmp_path):
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text(
        """
schedule:
  timezone: Asia/Shanghai
dates:
  as_of: "2026-06-07"
research_profiles:
  watchlist:
    analysts: [market]
    research_depth: 1
symbols:
  watchlist: [AAPL]
execution:
  delay_between_symbols_seconds: 0
output:
  vault_root: "~"
  weekly_dir: "weekly-runner-test"
  symbol_dir: "weekly-runner-test-symbols"
  write_weekly_summary: false
  write_portfolio_matrix: false
  write_run_meta: false
history:
  update_symbol_index: false
""",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    def fake_run_symbol(job, config, as_of, run_dir):
        del config, as_of, run_dir
        return weekly_runner.SymbolResult(symbol=job.symbol, group=job.group, status="ok", decision="Hold")

    monkeypatch.setattr(weekly_runner, "run_symbol", fake_run_symbol)
    monkeypatch.setattr(weekly_runner, "weekly_run_dir", lambda config, as_of: run_dir)
    monkeypatch.setattr(weekly_runner, "symbol_index_dir", lambda config: tmp_path / "symbols")

    assert weekly_runner.run_weekly(config_path) == 0
    assert not (run_dir / "Weekly Portfolio Research.md").exists()
    assert not (run_dir / "Portfolio Matrix.csv").exists()
    assert not (run_dir / "Run Metadata.json").exists()
