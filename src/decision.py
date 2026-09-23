"""
Decision layer — determines whether to BUY, SELL, or HOLD
based on model predictions.

Strategy: use the classifier probability directly.
If P(up) > threshold → BUY. If P(up) < (1 - threshold) → SELL.
The regressor magnitude is used only for position sizing.
"""
from config import (
    P_UP_BUY_THRESHOLD,
    P_UP_SELL_THRESHOLD,
    MAX_POSITION_PCT,
    STOP_LOSS_MULTIPLIER,
    TAKE_PROFIT_MULTIPLIER,
    ROUND_TRIP_COST,
)


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

    # Stop-loss and take-profit based on ATR-like distance
    # Use round-trip cost as minimum distance
    sl_dist = max(abs(expected_return), ROUND_TRIP_COST) * STOP_LOSS_MULTIPLIER
    tp_dist = max(abs(expected_return), ROUND_TRIP_COST) * TAKE_PROFIT_MULTIPLIER

    if action == "BUY":
        stop_loss = current_price * (1 - sl_dist)
        take_profit = current_price * (1 + tp_dist)
    else:
        stop_loss = current_price * (1 + sl_dist)
        take_profit = current_price * (1 - tp_dist)

    return {
        "action": action,
        "size_usd": round(size_usd, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "reason": f"p_up={p_up:.4f}, E[ret]={expected_return*100:+.4f}%, size=${size_usd:.2f}",
    }
