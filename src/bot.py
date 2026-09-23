"""
BTC Trading Bot — Main Loop
============================
Runs every 15 minutes:
  1. Fetch live candles (15m + 1h + 4h)
  2. Compute features
  3. Run XGBoost prediction (p_up + expected_return)
  4. Apply decision logic (BUY / SELL / HOLD)
  5. Check existing position (SL / TP / max hold)
  6. Execute trade if signaled
  7. Log everything

Usage:
    python src/bot.py
"""
import time
import sys
from datetime import datetime, timezone

from config import LOOP_INTERVAL_SECONDS, MODE
from live_data import get_exchange, build_live_dataframe
from predictor import Predictor
from decision import decide
from executor import Executor
from position_manager import PositionManager
from trade_logger import log_decision, log_trade
from notifier import notify_bot_started, notify_trade_opened, notify_trade_closed, notify_daily_summary, notify_hold


def print_banner():
    print("=" * 60)
    print("  BTC/USDT Trading Bot")
    print(f"  Mode: {'PAPER TRADING (Testnet)' if MODE == 'testnet' else 'LIVE TRADING'}")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)


def run_one_cycle(exchange, predictor, executor, pm):
    """Execute one complete bot cycle."""
    now = datetime.now(timezone.utc)
    print(f"\n{'─' * 50}")
    print(f"[{now.strftime('%H:%M:%S UTC')}] Starting cycle...")

    # ── 0. Daily reset ──
    pm.reset_daily_if_needed()

    # ── 1. Get current price ──
    current_price = executor.get_current_price()
    if current_price is None:
        print("  [SKIP] Could not get current price")
        return
    print(f"  BTC price: ${current_price:,.2f}")

    # ── 2. Check existing position for exit conditions ──
    if pm.has_position:
        exit_reason = pm.check_exit_conditions(current_price)
        if exit_reason:
            print(f"  [EXIT] {exit_reason} triggered (held {pm.bars_held} bars)")
            # Execute the close
            if pm.position == "long":
                order = executor.place_sell(pm.btc_quantity)
            else:
                order = executor.place_buy(pm.size_usd)

            trade_result = pm.close_position(current_price)
            if trade_result:
                log_trade(exit_reason, trade_result)
                notify_trade_closed(
                    side=trade_result["side"],
                    entry_price=trade_result["entry_price"],
                    exit_price=trade_result["exit_price"],
                    pnl_pct=trade_result["pnl_pct"],
                    pnl_usd=trade_result["pnl_usd"],
                    bars_held=trade_result["bars_held"],
                    exit_reason=exit_reason,
                )
                print(f"  Closed: PnL = {trade_result['pnl_pct']:+.2f}% (${trade_result['pnl_usd']:+.2f})")

    # ── 3. Check circuit breaker ──
    balance = executor.get_balance()
    account_balance = balance["total"]
    if pm.is_circuit_breaker_active(account_balance):
        print(f"  [CIRCUIT BREAKER] Daily loss limit reached. Skipping.")
        return

    # ── 4. Fetch live data and compute features ──
    try:
        features_row, ohlcv_row, bar_ts, feature_cols = build_live_dataframe(exchange)
        print(f"  Latest bar: {bar_ts}")
    except Exception as e:
        print(f"  [ERROR] Data fetch failed: {e}")
        return

    # ── 5. Run model prediction ──
    try:
        p_up, exp_return = predictor.predict(features_row, feature_cols)
        print(f"  Prediction: P(up)={p_up:.3f}, E[return]={exp_return*100:+.4f}%")
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        return

    # ── 6. Decision layer ──
    decision_result = decide(
        p_up=p_up,
        expected_return=exp_return,
        current_price=current_price,
        account_balance=account_balance,
        has_position=pm.has_position,
    )
    action = decision_result["action"]
    print(f"  Decision: {action} — {decision_result['reason']}")

    # Log every decision
    log_decision(bar_ts, current_price, p_up, exp_return, decision_result)
    
    if action == "HOLD":
        notify_hold(current_price, p_up, decision_result["reason"])

    # ── 7. Execute trade ──
    if action == "BUY":
        order = executor.place_buy(decision_result["size_usd"])
        if order:
            filled_price = float(order.get("average", current_price) or current_price)
            btc_qty = float(order.get("filled", decision_result["size_usd"] / current_price))
            pm.open_position(
                side="long",
                entry_price=filled_price,
                size_usd=decision_result["size_usd"],
                btc_qty=btc_qty,
                stop_loss=decision_result["stop_loss"],
                take_profit=decision_result["take_profit"],
            )
            notify_trade_opened(
                action="BUY",
                price=filled_price,
                size_usd=decision_result["size_usd"],
                stop_loss=decision_result["stop_loss"],
                take_profit=decision_result["take_profit"],
                p_up=p_up,
                exp_return=exp_return,
            )
            print(f"  ✓ Opened LONG at ${filled_price:,.2f}")
            print(f"    SL=${decision_result['stop_loss']:,.2f}  TP=${decision_result['take_profit']:,.2f}")

    elif action == "SELL" and pm.has_position and pm.position == "long":
        # Close an existing long position
        order = executor.place_sell(pm.btc_quantity)
        if order:
            filled_price = float(order.get("average", current_price) or current_price)
            trade_result = pm.close_position(filled_price)
            if trade_result:
                log_trade("signal_sell", trade_result)
                notify_trade_closed(
                    side=trade_result["side"],
                    entry_price=trade_result["entry_price"],
                    exit_price=trade_result["exit_price"],
                    pnl_pct=trade_result["pnl_pct"],
                    pnl_usd=trade_result["pnl_usd"],
                    bars_held=trade_result["bars_held"],
                    exit_reason="signal_sell",
                )
                print(f"  ✓ Closed LONG: PnL = {trade_result['pnl_pct']:+.2f}%")

    # ── 8. Status summary ──
    print(f"  Balance: ${account_balance:,.2f} USDT")
    print(f"  Position: {pm.position or 'flat'}")
    print(f"  Daily PnL: ${pm.daily_pnl:+.2f} ({pm.daily_trades} trades today)")


def main():
    print_banner()

    # Initialize components
    exchange = get_exchange()
    predictor = Predictor()
    executor = Executor(exchange)
    pm = PositionManager()

    print(f"\nBot initialized. Entering main loop (every {LOOP_INTERVAL_SECONDS}s)...")
    print("Press Ctrl+C to stop.\n")
    
    notify_bot_started()

    last_bar = None

    while True:
        try:
            # Only act when a new 15m candle has formed
            now = datetime.now(timezone.utc)
            current_bar = now.replace(
                minute=(now.minute // 15) * 15, second=0, microsecond=0
            )

            if current_bar != last_bar:
                # Wait 10 seconds for the candle to close on the exchange
                time.sleep(10)
                run_one_cycle(exchange, predictor, executor, pm)
                last_bar = current_bar

            time.sleep(LOOP_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n\nBot stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[FATAL ERROR] {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    main()
