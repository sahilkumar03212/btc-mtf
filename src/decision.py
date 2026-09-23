"""
Decision layer — determines whether to BUY, SELL, or HOLD
based on model predictions.

Strategy: use the classifier probability directly.
If P(up) > threshold → BUY. If P(up) < (1 - threshold) → SELL.
SL/TP use fixed, realistic percentages for 15m–2h BTC trades.
"""
from config import (
    P_UP_BUY_THRESHOLD,
    P_UP_SELL_THRESHOLD,
    MAX_POSITION_PCT,
    ROUND_TRIP_COST,
)

# ── Fixed risk-management parameters ──────────────────────────
# These are realistic for 15m–2h BTC swing trades
SL_PCT = 0.01    # 1.0% stop loss
TP_PCT = 0.02    # 2.0% take profit  (2:1 reward-to-risk ratio)


def decide(p_up, expected_return, current_price, account_balance, has_position):
    """
    Core trading decision based on classifier probability.

    Returns a dict:
        action: "BUY", "SELL", or "HOLD"
        size_usd: dollar amount to trade (0 if HOLD)
        stop_loss: SL price (None if HOLD)
        take_profit: TP price (None if HOLD)
        reason: human-readable explanation
    """
    # Rule 1: Already in a position — don't stack
    if has_position:
        return {
            "action": "HOLD",
            "size_usd": 0,
            "stop_loss": None,
            "take_profit": None,
            "reason": "Already in a position",
        }

    # Rule 2: Direction check — use probability directly
    if p_up > P_UP_BUY_THRESHOLD:
        action = "BUY"
    elif p_up < P_UP_SELL_THRESHOLD:
        action = "SELL"
    else:
        return {
            "action": "HOLD",
            "size_usd": 0,
            "stop_loss": None,
            "take_profit": None,
            "reason": f"p_up={p_up:.4f} in dead zone [{P_UP_SELL_THRESHOLD}, {P_UP_BUY_THRESHOLD}]",
        }

    # Position sizing — fixed fraction of account
    size_usd = account_balance * MAX_POSITION_PCT

    # Stop-loss and take-profit — fixed realistic percentages
    if action == "BUY":
        stop_loss = current_price * (1 - SL_PCT)
        take_profit = current_price * (1 + TP_PCT)
    else:
        stop_loss = current_price * (1 + SL_PCT)
        take_profit = current_price * (1 - TP_PCT)

    return {
        "action": action,
        "size_usd": round(size_usd, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "reason": f"p_up={p_up:.4f}, E[ret]={expected_return*100:+.4f}%, size=${size_usd:.2f}",
    }
