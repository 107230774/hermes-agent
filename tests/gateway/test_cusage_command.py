"""Tests for gateway /cusage command."""

import asyncio
from unittest.mock import MagicMock, patch



def test_gateway_cusage_returns_formatted_report():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = MagicMock()
    event.get_command_args.return_value = "today --source telegram"

    fake_engine = MagicMock()
    fake_engine.generate.return_value = {"empty": False}
    fake_engine.format_terminal.return_value = "Codex Usage\nWindow: today so far"

    with patch("gateway.run.SessionDB") as mock_db, \
         patch("agent.codex_usage.CodexUsageEngine", return_value=fake_engine):
        result = asyncio.run(GatewayRunner._handle_cusage_command(runner, event))

    mock_db.assert_called_once()
    fake_engine.generate.assert_called_once_with(days=None, source="telegram", today=True)
    assert "Codex Usage" in result



def test_gateway_cusage_rejects_bad_args():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = MagicMock()
    event.get_command_args.return_value = "--days nope"

    result = asyncio.run(GatewayRunner._handle_cusage_command(runner, event))

    assert "Invalid /cusage arguments" in result
