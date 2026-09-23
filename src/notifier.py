"""
Telegram notifications — sends alerts to your phone when the bot trades.

Setup (one-time, 2 minutes):
  1. Open Telegram, search for @BotFather
  2. Send /newbot, give it a name (e.g. "BTC Bot Alerts")
  3. Copy the API token it gives you → paste in config.py
  4. Search for your new bot in Telegram and press "Start"
  5. Go to https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
  6. Find your chat_id in the response → paste in config.py
"""
import requests
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
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"  [WARN] Telegram send failed: {resp.text}")
    except Exception as e:
        print(f"  [WARN] Telegram error: {e}")


def notify_trade_opened(action, price, size_usd, stop_loss, take_profit, p_up, exp_return):
    """Notify when a new trade is opened."""
    emoji = "🟢" if action == "BUY" else "🔴"
    msg = (
        f"{emoji} *{action} BTC* @ \\${price:,.2f}\n"
        f"Size: \\${size_usd:.2f}\n"
        f"SL: \\${stop_loss:,.2f} | TP: \\${take_profit:,.2f}\n"
        f"P(up): {p_up:.3f} | E\\[ret\\]: {exp_return*100:+.4f}%"
    )
    send_telegram(msg)


def notify_trade_closed(side, entry_price, exit_price, pnl_pct, pnl_usd, bars_held, exit_reason):
    """Notify when a trade is closed."""
    emoji = "💰" if pnl_usd > 0 else "💸"
    msg = (
        f"{emoji} *CLOSED {side.upper()}*\n"
        f"Entry: \\${entry_price:,.2f} → Exit: \\${exit_price:,.2f}\n"
        f"PnL: {pnl_pct:+.2f}% (\\${pnl_usd:+.2f})\n"
        f"Held: {bars_held} bars | Reason: {exit_reason}"
    )
    send_telegram(msg)


def notify_bot_started():
    """Notify when the bot starts running."""
    send_telegram("🤖 *BTC Trading Bot Started*\nPaper trading mode active. Monitoring every 15 minutes.")


def notify_daily_summary(daily_pnl, daily_trades, balance):
    """Send end-of-day summary."""
    emoji = "📈" if daily_pnl >= 0 else "📉"
    msg = (
        f"{emoji} *Daily Summary*\n"
        f"PnL: \\${daily_pnl:+.2f}\n"
        f"Trades: {daily_trades}\n"
        f"Balance: \\${balance:,.2f}"
    )
    send_telegram(msg)


def notify_hold(price, p_up, reason):
    """Send a status update when the bot decides to HOLD."""
    msg = (
        f"⏸ *HOLD* @ \\${price:,.2f}\n"
        f"P(up): {p_up:.3f}\n"
        f"Reason: {reason}"
    )
    send_telegram(msg)
