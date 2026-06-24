from pathlib import Path


def test_codex_launcher_delegates_to_bringup_cli_mode():
    script = Path("scripts/start_tradingagents_codex.sh").read_text(encoding="utf-8")

    assert 'bringup_tradingagents_codex.sh" --mode cli' in script
    assert 'python -m cli.main "$@"' not in script


def test_readme_documents_current_typer_root_command():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python -m cli.main analyze" not in readme


def test_weekly_launcher_delegates_to_bringup_weekly_mode():
    script = Path("scripts/run_weekly_codex_reports.sh").read_text(encoding="utf-8")

    assert 'bringup_tradingagents_codex.sh" --mode weekly' in script


def test_bringup_script_exposes_doctor_mode():
    script = Path("scripts/bringup_tradingagents_codex.sh").read_text(encoding="utf-8")

    assert "--mode doctor" in script
    assert "Syncing current checkout into .venv" in script
