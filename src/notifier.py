"""
Telegram notifications — sends comprehensive alerts to your phone every cycle.

Setup (one-time, 2 minutes):
  1. Open Telegram, search for @BotFather
  2. Send /newbot, give it a name (e.g. "BTC Bot Alerts")
  3. Copy the API token it gives you → paste in config.py
  4. Search for your new bot in Telegram and press "Start"
  5. Go to https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
  6. Find your chat_id in the response → paste in config.py
"""
import requests
from datetime import datetime, timezone
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_telegram(message):
    """Send a message via Telegram bot. Fails silently if not configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"  [WARN] Telegram send failed: {resp.text}")
    except Exception as e:
        print(f"  [WARN] Telegram error: {e}")


def notify_bot_started():
    """Notify when the bot starts running."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    send_telegram(
        f"🤖 <b>BTC Trading Bot Started</b>\n"
        f"Paper trading mode active.\n"
        f"Monitoring every 15 minutes.\n"
        f"Time: {now}"
    )


def notify_cycle_report(
    price, p_up, exp_return, action, reason,
    balance, position, daily_pnl, daily_trades,
    size_usd=None, stop_loss=None, take_profit=None,
    entry_price=None, unrealized_pnl_pct=None, bars_held=None,
):
    """
    Send a full status report every 15-minute cycle.
    This is the main notification — tells you EVERYTHING.
    """
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    # Action emoji
    if action == "BUY":
        action_emoji = "🟢"
    elif action == "SELL":
        action_emoji = "🔴"
    else:
        action_emoji = "⏸"

    # Build the message
    lines = []
    lines.append(f"{'─' * 20}")
    lines.append(f"⏰ <b>{now} — Cycle Report</b>")
    lines.append(f"{'─' * 20}")

    # ── Market ──
    lines.append(f"")
    lines.append(f"📊 <b>Market</b>")
    lines.append(f"BTC Price: ${price:,.2f}")

    # ── Model Prediction ──
    lines.append(f"")
    lines.append(f"🧠 <b>Model Prediction</b>")
    lines.append(f"P(up): {p_up:.4f}")
    lines.append(f"E[return]: {exp_return * 100:+.4f}%")

    # ── Decision ──
    lines.append(f"")
    lines.append(f"{action_emoji} <b>Decision: {action}</b>")
    lines.append(f"Reason: {reason}")

    # ── Trade Details (only if BUY or SELL) ──
    if action in ("BUY", "SELL") and size_usd is not None:
        lines.append(f"")
        lines.append(f"💼 <b>Trade Details</b>")
        lines.append(f"Size: ${size_usd:.2f}")
        if stop_loss:
            lines.append(f"Stop Loss: ${stop_loss:,.2f}")
        if take_profit:
            lines.append(f"Take Profit: ${take_profit:,.2f}")

    # ── Open Position ──
    lines.append(f"")
    if position and entry_price:
        lines.append(f"📌 <b>Open Position</b>")
        lines.append(f"Side: {position.upper()}")
        lines.append(f"Entry: ${entry_price:,.2f}")
        if unrealized_pnl_pct is not None:
            pnl_emoji = "📈" if unrealized_pnl_pct >= 0 else "📉"
            lines.append(f"Unrealized PnL: {pnl_emoji} {unrealized_pnl_pct:+.2f}%")
        if bars_held is not None:
            lines.append(f"Bars Held: {bars_held}/8")
    else:
        lines.append(f"📌 <b>Position: FLAT</b> (no open trade)")

    # ── Account ──
    lines.append(f"")
    lines.append(f"💰 <b>Account</b>")
    lines.append(f"Balance: ${balance:,.2f} USDT")
    pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
    lines.append(f"Daily PnL: {pnl_emoji} ${daily_pnl:+.2f}")
    lines.append(f"Trades Today: {daily_trades}")

    send_telegram("\n".join(lines))


def notify_trade_opened(action, price, size_usd, stop_loss, take_profit, p_up, exp_return):
    """Notify when a new trade is opened."""
    emoji = "🟢" if action == "BUY" else "🔴"
    msg = (
        f"{emoji} <b>{action} BTC</b> @ ${price:,.2f}\n"
        f"Size: ${size_usd:.2f}\n"
        f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}\n"
        f"P(up): {p_up:.3f} | E[ret]: {exp_return*100:+.4f}%"
    )
    send_telegram(msg)


def notify_trade_closed(side, entry_price, exit_price, pnl_pct, pnl_usd, bars_held, exit_reason):
    """Notify when a trade is closed."""
    emoji = "💰" if pnl_usd > 0 else "💸"
    msg = (
        f"{emoji} <b>CLOSED {side.upper()}</b>\n"
        f"Entry: ${entry_price:,.2f} → Exit: ${exit_price:,.2f}\n"
        f"PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n"
        f"Held: {bars_held} bars | Reason: {exit_reason}"
    )
    send_telegram(msg)


def notify_daily_summary(daily_pnl, daily_trades, balance):
    """Send end-of-day summary."""
    emoji = "📈" if daily_pnl >= 0 else "📉"
    msg = (
        f"{emoji} <b>Daily Summary</b>\n"
        f"PnL: ${daily_pnl:+.2f}\n"
        f"Trades: {daily_trades}\n"
        f"Balance: ${balance:,.2f}"
    )
    send_telegram(msg)


def notify_error(error_msg):
    """Notify when something goes wrong so you know the bot is struggling."""
    send_telegram(f"⚠️ <b>Bot Error</b>\n{error_msg}")
