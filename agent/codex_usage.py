"""Aggregate Codex usage across Hermes sessions and platforms."""

from __future__ import annotations

import shlex
import time
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

_CODEX_PROVIDER = "openai-codex"
_CODEX_BASE_URL_LIKE = "%chatgpt.com/backend-api/codex%"
_OFFICIAL_PRICING_SOURCE_URL = "https://developers.openai.com/api/docs/pricing"
_OFFICIAL_REFERENCE_PRICING: Dict[str, Dict[str, Any]] = {
    "gpt-5.4": {
        "display_name": "gpt-5.4 (<272K context length)",
        "input_per_million": Decimal("2.50"),
        "cache_read_per_million": Decimal("0.25"),
        "output_per_million": Decimal("15.00"),
        "source_url": _OFFICIAL_PRICING_SOURCE_URL,
        "source_note": "OpenAI 官方 API Pricing 页；当前 ChatGPT Codex 路由仍按 included 处理，这里只做参考价换算。",
    },
}


def parse_cusage_args(raw_args: str) -> Dict[str, Optional[Any]]:
    """Parse /cusage args.

    Supported forms:
      /cusage
      /cusage 7
      /cusage today
      /cusage now
      /cusage --today
      /cusage --days 7
      /cusage --source telegram
      /cusage --today --source telegram
      /cusage --days 7 --source telegram
    """
    days: Optional[int] = None
    source: Optional[str] = None
    today = False
    parts = shlex.split(raw_args or "")
    i = 0
    while i < len(parts):
        part = parts[i].strip().lower()
        if part == "--days":
            if today:
                raise ValueError("cannot combine --days with today/now")
            if i + 1 >= len(parts):
                raise ValueError("--days requires an integer value")
            try:
                days = int(parts[i + 1])
            except ValueError as exc:
                raise ValueError("--days must be an integer") from exc
            i += 2
            continue
        if part == "--today":
            if days is not None:
                raise ValueError("cannot combine --today with --days")
            today = True
            i += 1
            continue
        if part == "--source":
            if i + 1 >= len(parts):
                raise ValueError("--source requires a platform name")
            source = parts[i + 1].strip().lower() or None
            i += 2
            continue
        if part in {"today", "now"}:
            if days is not None:
                raise ValueError("cannot combine today/now with day counts")
            today = True
            i += 1
            continue
        if part.startswith("--"):
            raise ValueError(f"unknown flag: {part}")
        if days is None and not today:
            try:
                days = int(part)
            except ValueError as exc:
                raise ValueError(f"unexpected argument: {parts[i]}") from exc
            i += 1
            continue
        raise ValueError(f"unexpected argument: {parts[i]}")

    if days is not None and days <= 0:
        raise ValueError("--days must be greater than 0")

    return {"days": days, "source": source, "today": today}


class CodexUsageEngine:
    """Query Codex-backed session usage from SessionDB."""

    _SESSION_COLS = (
        "id, source, model, started_at, ended_at, message_count, tool_call_count, "
        "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
        "billing_provider, billing_base_url, billing_mode, cost_status"
    )

    def __init__(self, db):
        self.db = db
        self._conn = db._conn

    def _get_sessions(self, cutoff: Optional[float] = None, source: Optional[str] = None) -> List[Dict[str, Any]]:
        where = [
            "(lower(COALESCE(billing_provider, '')) = ? OR lower(COALESCE(billing_base_url, '')) LIKE ?)"
        ]
        params: List[Any] = [_CODEX_PROVIDER, _CODEX_BASE_URL_LIKE]
        if cutoff is not None:
            where.append("started_at >= ?")
            params.append(cutoff)
        if source:
            where.append("lower(source) = ?")
            params.append(source.lower())
        sql = (
            f"SELECT {self._SESSION_COLS} FROM sessions "
            f"WHERE {' AND '.join(where)} ORDER BY started_at DESC"
        )
        return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    @staticmethod
    def _accounted_tokens(row: Dict[str, Any]) -> int:
        return sum(
            int(row.get(key) or 0)
            for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")
        )

    @staticmethod
    def _billing_status(sessions: List[Dict[str, Any]]) -> str:
        statuses = {str(s.get("billing_mode") or s.get("cost_status") or "unknown") for s in sessions}
        statuses.discard("")
        if not statuses:
            return "unknown"
        if statuses == {"subscription_included"} or statuses == {"included"} or statuses == {"subscription_included", "included"}:
            return "included"
        if len(statuses) == 1:
            return next(iter(statuses))
        return "mixed"

    def generate(
        self,
        days: Optional[int] = None,
        source: Optional[str] = None,
        *,
        today: bool = False,
    ) -> Dict[str, Any]:
        if today and days is not None:
            raise ValueError("cannot combine today=True with days")

        if today:
            now = datetime.now().astimezone()
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff = day_start.timestamp()
            window_label = "today so far"
        else:
            cutoff = None if days is None else time.time() - (days * 86400)
            window_label = "all time" if days is None else f"last {days} days"

        sessions = self._get_sessions(cutoff=cutoff, source=source)
        if not sessions:
            return {
                "days": days,
                "source_filter": source,
                "today": today,
                "window_label": window_label,
                "empty": True,
                "overview": {},
                "sources": [],
                "models": [],
                "reference_pricing": None,
            }

        overview = {
            "session_count": len(sessions),
            "message_count": sum(int(s.get("message_count") or 0) for s in sessions),
            "tool_call_count": sum(int(s.get("tool_call_count") or 0) for s in sessions),
            "input_tokens": sum(int(s.get("input_tokens") or 0) for s in sessions),
            "output_tokens": sum(int(s.get("output_tokens") or 0) for s in sessions),
            "cache_read_tokens": sum(int(s.get("cache_read_tokens") or 0) for s in sessions),
            "cache_write_tokens": sum(int(s.get("cache_write_tokens") or 0) for s in sessions),
            "accounted_tokens": sum(self._accounted_tokens(s) for s in sessions),
            "billing_status": self._billing_status(sessions),
            "first_seen_at": min(float(s.get("started_at") or 0) for s in sessions),
            "last_seen_at": max(float(s.get("started_at") or 0) for s in sessions),
        }

        source_rows: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "source": "",
                "session_count": 0,
                "message_count": 0,
                "tool_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "accounted_tokens": 0,
            }
        )
        model_rows: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "model": "",
                "session_count": 0,
                "message_count": 0,
                "tool_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "accounted_tokens": 0,
            }
        )

        for session in sessions:
            source_key = str(session.get("source") or "unknown")
            model_key = str(session.get("model") or "(unknown model)")
            accounted = self._accounted_tokens(session)
            for row, key_name, key in ((source_rows[source_key], "source", source_key), (model_rows[model_key], "model", model_key)):
                row[key_name] = key
                row["session_count"] += 1
                row["message_count"] += int(session.get("message_count") or 0)
                row["tool_call_count"] += int(session.get("tool_call_count") or 0)
                row["input_tokens"] += int(session.get("input_tokens") or 0)
                row["output_tokens"] += int(session.get("output_tokens") or 0)
                row["cache_read_tokens"] += int(session.get("cache_read_tokens") or 0)
                row["cache_write_tokens"] += int(session.get("cache_write_tokens") or 0)
                row["accounted_tokens"] += accounted

        sort_key = lambda row: (-row["accounted_tokens"], str(row.get("source") or row.get("model") or ""))
        sources = sorted(source_rows.values(), key=sort_key)
        models = sorted(model_rows.values(), key=sort_key)
        result = {
            "days": days,
            "source_filter": source,
            "today": today,
            "window_label": window_label,
            "empty": False,
            "generated_at": time.time(),
            "overview": overview,
            "sources": sources,
            "models": models,
        }
        result["reference_pricing"] = self._build_reference_pricing(result)
        return result

    @staticmethod
    def _fmt_ts(value: float) -> str:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M") if value else "n/a"

    @staticmethod
    def _fmt_int(value: int) -> str:
        return f"{int(value):,}"

    @staticmethod
    def _fmt_money(value: Decimal) -> str:
        quantized = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return f"${quantized}"

    def _build_reference_pricing(self, report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        models = report.get("models") or []
        if len(models) != 1:
            return None

        model = str(models[0].get("model") or "")
        pricing = _OFFICIAL_REFERENCE_PRICING.get(model)
        if not pricing:
            return None

        overview = report.get("overview") or {}
        input_tokens = int(overview.get("input_tokens") or 0)
        cache_read_tokens = int(overview.get("cache_read_tokens") or 0)
        cache_write_tokens = int(overview.get("cache_write_tokens") or 0)
        output_tokens = int(overview.get("output_tokens") or 0)

        benchmark_cost = (
            Decimal(input_tokens) * pricing["input_per_million"] / Decimal("1000000")
            + Decimal(cache_read_tokens) * pricing["cache_read_per_million"] / Decimal("1000000")
            + Decimal(output_tokens) * pricing["output_per_million"] / Decimal("1000000")
        )
        status = "complete" if cache_write_tokens == 0 else "partial"
        note = pricing["source_note"]
        if cache_write_tokens:
            note += " 当前有 cache write tokens，但官方页未单列其价格，基准总价未计入该部分。"

        return {
            "model": model,
            "display_name": pricing["display_name"],
            "input_per_million": pricing["input_per_million"],
            "cache_read_per_million": pricing["cache_read_per_million"],
            "output_per_million": pricing["output_per_million"],
            "benchmark_cost_usd": benchmark_cost,
            "cost_basis_status": status,
            "source_url": pricing["source_url"],
            "note": note,
        }

    def format_terminal(self, report: Dict[str, Any]) -> str:
        window = report.get("window_label")
        if not window:
            if report.get("today"):
                window = "today so far"
            else:
                window = "all time" if report.get("days") is None else f"last {report.get('days')} days"
        if report.get("empty"):
            source_filter = report.get("source_filter")
            suffix = f" (source={source_filter})" if source_filter else ""
            return f"Codex Usage\nWindow: {window}{suffix}\nNo Codex usage data found."

        overview = report["overview"]
        source_filter = report.get("source_filter")
        lines = [
            "Codex Usage",
            f"Window: {window}" + (f" | source={source_filter}" if source_filter else ""),
            f"First seen: {self._fmt_ts(overview['first_seen_at'])}",
            f"Last seen:  {self._fmt_ts(overview['last_seen_at'])}",
            f"Sessions: {self._fmt_int(overview['session_count'])}",
            f"Messages: {self._fmt_int(overview['message_count'])}",
            f"Tool calls: {self._fmt_int(overview['tool_call_count'])}",
            f"Input tokens: {self._fmt_int(overview['input_tokens'])}",
            f"Cache read tokens: {self._fmt_int(overview['cache_read_tokens'])}",
            f"Cache write tokens: {self._fmt_int(overview['cache_write_tokens'])}",
            f"Output tokens: {self._fmt_int(overview['output_tokens'])}",
            f"Accounted tokens: {self._fmt_int(overview['accounted_tokens'])}",
            f"Billing: {overview['billing_status']}",
        ]
        reference = report.get("reference_pricing")
        if reference:
            benchmark_label = "Benchmark cost" if reference.get("cost_basis_status") == "complete" else "Benchmark cost (partial)"
            lines.extend([
                f"Reference pricing: {reference['display_name']} | $2.50/M input | $0.25/M cached input | $15.00/M output",
                f"{benchmark_label}: {self._fmt_money(reference['benchmark_cost_usd'])}",
                f"Reference note: {reference['note']}",
                f"Reference source: {reference['source_url']}",
            ])
        lines.extend([
            "",
            "By source:",
        ])
        total_accounted = max(int(overview["accounted_tokens"] or 0), 1)
        for row in report.get("sources", [])[:10]:
            pct = row["accounted_tokens"] / total_accounted * 100
            lines.append(
                f"- {row['source']}: {self._fmt_int(row['session_count'])} sessions, "
                f"{self._fmt_int(row['message_count'])} messages, "
                f"{self._fmt_int(row['accounted_tokens'])} tokens ({pct:.0f}%)"
            )
        lines.extend(["", "By model:"])
        for row in report.get("models", [])[:10]:
            pct = row["accounted_tokens"] / total_accounted * 100
            lines.append(
                f"- {row['model']}: {self._fmt_int(row['session_count'])} sessions, "
                f"{self._fmt_int(row['accounted_tokens'])} tokens ({pct:.0f}%)"
            )
        return "\n".join(lines)

    def format_gateway(self, report: Dict[str, Any]) -> str:
        return self.format_terminal(report)
