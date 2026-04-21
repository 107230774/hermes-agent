"""Tests for agent.codex_usage — aggregate Codex usage across sessions/platforms."""

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_codex_usage.db")
    yield session_db
    session_db.close()


@pytest.fixture()
def populated_db(db):
    now = time.time()
    day = 86400

    # Codex session in CLI
    db.create_session("codex-cli", source="cli", model="gpt-5.4")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 2 * day, "codex-cli"))
    db.update_token_counts(
        "codex-cli",
        input_tokens=1000,
        output_tokens=250,
        cache_read_tokens=300,
        model="gpt-5.4",
        billing_provider="openai-codex",
        billing_base_url="https://chatgpt.com/backend-api/codex",
        billing_mode="subscription_included",
        cost_status="included",
    )
    for role in ("user", "assistant", "tool"):
        db.append_message("codex-cli", role=role, content=f"{role}-message")

    # Codex session in Telegram
    db.create_session("codex-telegram", source="telegram", model="gpt-5.4")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 1 * day, "codex-telegram"))
    db.update_token_counts(
        "codex-telegram",
        input_tokens=2000,
        output_tokens=500,
        cache_read_tokens=100,
        cache_write_tokens=50,
        model="gpt-5.4",
        billing_provider="openai-codex",
        billing_base_url="https://chatgpt.com/backend-api/codex",
        billing_mode="subscription_included",
        cost_status="included",
    )
    for role in ("user", "assistant"):
        db.append_message("codex-telegram", role=role, content=f"{role}-message")

    # Realtime/today Codex session
    db.create_session("codex-today", source="cli", model="gpt-5.4")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 60, "codex-today"))
    db.update_token_counts(
        "codex-today",
        input_tokens=400,
        output_tokens=80,
        cache_read_tokens=20,
        model="gpt-5.4",
        billing_provider="openai-codex",
        billing_base_url="https://chatgpt.com/backend-api/codex",
        billing_mode="subscription_included",
        cost_status="included",
    )
    for role in ("user", "assistant"):
        db.append_message("codex-today", role=role, content=f"today-{role}")

    # Older Codex session to test days filter
    db.create_session("codex-old", source="discord", model="gpt-5.2-codex")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 40 * day, "codex-old"))
    db.update_token_counts(
        "codex-old",
        input_tokens=900,
        output_tokens=100,
        model="gpt-5.2-codex",
        billing_provider="openai-codex",
        billing_base_url="https://chatgpt.com/backend-api/codex",
        billing_mode="subscription_included",
        cost_status="included",
    )
    db.append_message("codex-old", role="user", content="old")

    # Non-Codex session should be excluded
    db.create_session("openrouter", source="cli", model="openai/gpt-4o")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 1 * day, "openrouter"))
    db.update_token_counts(
        "openrouter",
        input_tokens=9999,
        output_tokens=999,
        model="openai/gpt-4o",
        billing_provider="openrouter",
        billing_base_url="https://openrouter.ai/api/v1",
        cost_status="estimated",
    )
    db.append_message("openrouter", role="user", content="ignore me")

    db._conn.commit()
    return db


def test_generate_all_time_aggregates_codex_only(populated_db):
    from agent.codex_usage import CodexUsageEngine

    report = CodexUsageEngine(populated_db).generate()

    assert report["empty"] is False
    assert report["overview"]["session_count"] == 4
    assert report["overview"]["message_count"] == 8
    assert report["overview"]["input_tokens"] == 4300
    assert report["overview"]["output_tokens"] == 930
    assert report["overview"]["cache_read_tokens"] == 420
    assert report["overview"]["cache_write_tokens"] == 50
    assert report["overview"]["accounted_tokens"] == 5700
    assert report["overview"]["billing_status"] == "included"

    by_source = {row["source"]: row for row in report["sources"]}
    assert by_source["cli"]["session_count"] == 2
    assert by_source["cli"]["accounted_tokens"] == 2050
    assert by_source["telegram"]["accounted_tokens"] == 2650
    assert by_source["discord"]["session_count"] == 1


def test_generate_supports_days_source_and_today_filters(populated_db):
    from agent.codex_usage import CodexUsageEngine

    engine = CodexUsageEngine(populated_db)

    last_week = engine.generate(days=7)
    assert last_week["overview"]["session_count"] == 3
    assert last_week["overview"]["input_tokens"] == 3400

    cli_only = engine.generate(source="cli")
    assert cli_only["overview"]["session_count"] == 2
    assert cli_only["overview"]["accounted_tokens"] == 2050

    today_only = engine.generate(today=True)
    assert today_only["overview"]["session_count"] == 1
    assert today_only["overview"]["accounted_tokens"] == 500
    assert today_only["window_label"] == "today so far"


def test_format_terminal_shows_sources_models_billing_and_reference_pricing(populated_db):
    from agent.codex_usage import CodexUsageEngine

    engine = CodexUsageEngine(populated_db)
    weekly = engine.generate(days=7)
    text = engine.format_terminal(weekly)
    today_report = engine.generate(today=True)
    today_text = engine.format_terminal(today_report)

    assert "Codex Usage" in text
    assert "Window: last 7 days" in text
    assert "Billing: included" in text
    assert "By source:" in text
    assert "cli" in text
    assert "telegram" in text
    assert "By model:" in text
    assert "gpt-5.4" in text
    assert "Reference pricing" in text
    assert "$2.50/M input" in text
    assert "$0.25/M cached input" in text
    assert "$15.00/M output" in text
    assert "Benchmark cost" in text
    assert today_report["reference_pricing"]["model"] == "gpt-5.4"
    assert today_report["reference_pricing"]["cost_basis_status"] == "complete"
    assert "Window: today so far" in today_text


def test_parse_cusage_args_supports_days_source_and_today():
    from agent.codex_usage import parse_cusage_args

    assert parse_cusage_args("") == {"days": None, "source": None, "today": False}
    assert parse_cusage_args("7") == {"days": 7, "source": None, "today": False}
    assert parse_cusage_args("today") == {"days": None, "source": None, "today": True}
    assert parse_cusage_args("now") == {"days": None, "source": None, "today": True}
    assert parse_cusage_args("--today --source telegram") == {"days": None, "source": "telegram", "today": True}
    assert parse_cusage_args("--days 30 --source telegram") == {"days": 30, "source": "telegram", "today": False}

    with pytest.raises(ValueError):
        parse_cusage_args("--days nope")
    with pytest.raises(ValueError):
        parse_cusage_args("today 7")
