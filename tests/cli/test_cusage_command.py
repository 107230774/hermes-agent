from unittest.mock import MagicMock, patch


def test_cusage_registered_in_command_registry():
    from hermes_cli.commands import COMMANDS

    assert "/cusage" in COMMANDS



def test_cusage_uses_chat_console_when_tui_is_live():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.console = MagicMock()
    cli._app = object()
    cli._session_db = object()

    live_console = MagicMock()

    fake_engine = MagicMock()
    fake_engine.generate.return_value = {"empty": False}
    fake_engine.format_terminal.return_value = "Codex Usage\nSessions: 2"

    with patch("cli.ChatConsole", return_value=live_console), \
         patch("agent.codex_usage.CodexUsageEngine", return_value=fake_engine):
        cli._handle_cusage_command("/cusage today")

    fake_engine.generate.assert_called_once_with(days=None, source=None, today=True)
    live_console.print.assert_called()
    cli.console.print.assert_not_called()
