from pathlib import Path


def test_codex_launcher_calls_current_typer_root_command():
    script = Path("scripts/start_tradingagents_codex.sh").read_text(encoding="utf-8")

    assert 'python -m cli.main "$@"' in script
    assert 'python -m cli.main analyze "$@"' not in script


def test_readme_documents_current_typer_root_command():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python -m cli.main analyze" not in readme
