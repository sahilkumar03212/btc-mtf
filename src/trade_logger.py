"""
Trade logger — records every decision and trade to CSV for analysis.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path
from config import LOG_DIR


DECISION_LOG = LOG_DIR / "decisions.csv"
TRADE_LOG = LOG_DIR / "trades.csv"

DECISION_HEADERS = [
    "timestamp", "price", "p_up", "expected_return",
    "action", "size_usd", "stop_loss", "take_profit", "reason",
]

TRADE_HEADERS = [
    "timestamp", "side", "entry_price", "exit_price",
    "pnl_pct", "pnl_usd", "bars_held", "exit_reason",
]


def _ensure_headers(filepath, headers):
    if not filepath.exists():
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def log_decision(timestamp, price, p_up, expected_return, decision):
    """Log every 15-minute decision (BUY, SELL, or HOLD)."""
    _ensure_headers(DECISION_LOG, DECISION_HEADERS)
    with open(DECISION_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            round(price, 2),
            round(p_up, 4),
            round(expected_return * 100, 4),
            decision["action"],
            decision["size_usd"],
            decision.get("stop_loss", ""),
            decision.get("take_profit", ""),
            decision["reason"],
        ])


def log_trade(exit_reason, trade_result):
    """Log a completed trade (entry → exit)."""
    _ensure_headers(TRADE_LOG, TRADE_HEADERS)
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            trade_result["side"],
            trade_result["entry_price"],
            trade_result["exit_price"],
            trade_result["pnl_pct"],
            trade_result["pnl_usd"],
            trade_result["bars_held"],
            exit_reason,
        ])
