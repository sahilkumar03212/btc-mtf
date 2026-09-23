"""
Position manager — tracks open positions, SL/TP, and daily risk.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from config import MAX_HOLD_BARS, MAX_DAILY_LOSS_PCT, LOG_DIR


STATE_FILE = LOG_DIR / "position_state.json"


class PositionManager:
    def __init__(self):
        self.position = None  # "long", "short", or None
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.size_usd = 0.0
        self.btc_quantity = 0.0
        self.bars_held = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = None
        self._load_state()

    def _load_state(self):
        """Restore state from disk (survives restarts)."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.position = state.get("position")
                self.entry_price = state.get("entry_price", 0)
                self.entry_time = state.get("entry_time")
                self.stop_loss = state.get("stop_loss", 0)
                self.take_profit = state.get("take_profit", 0)
                self.size_usd = state.get("size_usd", 0)
                self.btc_quantity = state.get("btc_quantity", 0)
                self.bars_held = state.get("bars_held", 0)
                self.daily_pnl = state.get("daily_pnl", 0)
                self.daily_trades = state.get("daily_trades", 0)
                self.last_reset_date = state.get("last_reset_date")
                print(f"  Restored position state: {self.position or 'flat'}")
            except Exception as e:
                print(f"  [WARN] Could not load state: {e}")

    def _save_state(self):
        """Persist state to disk."""
        state = {
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "size_usd": self.size_usd,
            "btc_quantity": self.btc_quantity,
            "bars_held": self.bars_held,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "last_reset_date": self.last_reset_date,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @property
    def has_position(self):
        return self.position is not None

    def reset_daily_if_needed(self):
        """Reset daily PnL tracker at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
            self._save_state()

    def is_circuit_breaker_active(self, account_balance):
        """Stop trading if daily loss exceeds threshold."""
        if account_balance <= 0:
            return True
        if self.daily_pnl < 0 and abs(self.daily_pnl) > account_balance * MAX_DAILY_LOSS_PCT:
            return True
        return False

    def open_position(self, side, entry_price, size_usd, btc_qty, stop_loss, take_profit):
        """Record a new position."""
        self.position = side  # "long" or "short"
        self.entry_price = entry_price
        self.entry_time = datetime.now(timezone.utc).isoformat()
        self.size_usd = size_usd
        self.btc_quantity = btc_qty
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.bars_held = 0
        self._save_state()

    def close_position(self, exit_price):
        """Close current position and record PnL."""
        if not self.has_position:
            return 0.0

        if self.position == "long":
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - exit_price) / self.entry_price

        pnl_usd = pnl_pct * self.size_usd
        self.daily_pnl += pnl_usd
        self.daily_trades += 1

        result = {
            "side": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct * 100, 4),
            "pnl_usd": round(pnl_usd, 2),
            "bars_held": self.bars_held,
        }

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.size_usd = 0.0
        self.btc_quantity = 0.0
        self.bars_held = 0
        self._save_state()

        return result

    def check_exit_conditions(self, current_price):
        """
        Check if we should exit: SL hit, TP hit, or max hold time.
        Returns: "stop_loss", "take_profit", "max_hold", or None.
        """
        if not self.has_position:
            return None

        self.bars_held += 1
        self._save_state()

        if self.position == "long":
            if current_price <= self.stop_loss:
                return "stop_loss"
            if current_price >= self.take_profit:
                return "take_profit"
        elif self.position == "short":
            if current_price >= self.stop_loss:
                return "stop_loss"
            if current_price <= self.take_profit:
                return "take_profit"

        if self.bars_held >= MAX_HOLD_BARS:
            return "max_hold"

        return None
