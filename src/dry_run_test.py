"""
Dry-run test — simulates the full bot pipeline using historical data.
No API keys needed. Verifies every component works end-to-end.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from features import add_features
from predictor import Predictor
from decision import decide
from position_manager import PositionManager
from trade_logger import log_decision, log_trade
from config import ROOT, LOG_DIR

DATA = ROOT / "data"


def test_data_loading():
    """Test 1: Load and align multi-timeframe data."""
    print("=" * 60)
    print("TEST 1: Data Loading & Feature Engineering")
    print("=" * 60)

    df_15m = pd.read_parquet(DATA / "btc_15m.parquet")
    base = add_features(df_15m, "15m")
    ohlcv = base[["open", "high", "low", "close", "volume"]].copy()

    for tf in ["1h", "4h"]:
        df_hi = pd.read_parquet(DATA / f"btc_{tf}.parquet")
        hi_feats = add_features(df_hi, tf)
        cols = [c for c in hi_feats.columns if c.startswith(f"{tf}_")]
        aligned = hi_feats[cols].shift(1).reindex(base.index, method="ffill")
        base = base.join(aligned, how="left")

    feature_cols = [c for c in base.columns if c not in {"open", "high", "low", "close", "volume"}]

    # Take a sample row (simulate "latest bar")
    sample_idx = -10  # 10 bars from the end
    features_row = base[feature_cols].iloc[sample_idx]
    ohlcv_row = ohlcv.iloc[sample_idx]
    timestamp = base.index[sample_idx]

    nan_count = features_row.isna().sum()
    total = len(features_row)

    print(f"  ✓ Loaded {len(base)} 15m bars")
    print(f"  ✓ {len(feature_cols)} features computed")
    print(f"  ✓ Sample bar: {timestamp}")
    print(f"  ✓ Close price: ${ohlcv_row['close']:,.2f}")
    print(f"  ✓ NaN features: {nan_count}/{total}")

    if nan_count > 0:
        nans = features_row[features_row.isna()].index.tolist()
        print(f"  ⚠ NaN features: {nans[:5]}{'...' if len(nans) > 5 else ''}")

    assert nan_count == 0, f"Found {nan_count} NaN features — pipeline has a gap"
    print("  ✓ PASS: No NaN features\n")
    return features_row, ohlcv_row, timestamp, feature_cols, base, ohlcv


def test_predictor(features_row, feature_cols):
    """Test 2: Load XGBoost models and run prediction."""
    print("=" * 60)
    print("TEST 2: XGBoost Predictor")
    print("=" * 60)

    predictor = Predictor()
    p_up, exp_return = predictor.predict(features_row, feature_cols)

    print(f"  ✓ P(up)           = {p_up:.4f}")
    print(f"  ✓ E[return]       = {exp_return*100:+.4f}%")

    assert 0 <= p_up <= 1, f"p_up={p_up} out of range [0, 1]"
    assert -0.1 < exp_return < 0.1, f"exp_return={exp_return} seems unreasonable"
    print("  ✓ PASS: Predictions are reasonable\n")
    return predictor, p_up, exp_return


def test_decision(p_up, exp_return, price):
    """Test 3: Decision layer."""
    print("=" * 60)
    print("TEST 3: Decision Layer")
    print("=" * 60)

    result = decide(
        p_up=p_up,
        expected_return=exp_return,
        current_price=price,
        account_balance=10000.0,  # simulated $10k account
        has_position=False,
    )

    print(f"  ✓ Action:      {result['action']}")
    print(f"  ✓ Size:        ${result['size_usd']}")
    print(f"  ✓ Stop-loss:   {result['stop_loss']}")
    print(f"  ✓ Take-profit: {result['take_profit']}")
    print(f"  ✓ Reason:      {result['reason']}")

    assert result["action"] in ("BUY", "SELL", "HOLD"), f"Invalid action: {result['action']}"
    print("  ✓ PASS: Decision layer works\n")
    return result


def test_position_manager():
    """Test 4: Position manager lifecycle."""
    print("=" * 60)
    print("TEST 4: Position Manager")
    print("=" * 60)

    pm = PositionManager()
    assert not pm.has_position, "Should start flat"
    print("  ✓ Starts flat")

    pm.open_position("long", 60000.0, 200.0, 0.00333, 59000.0, 62000.0)
    assert pm.has_position, "Should have position"
    assert pm.position == "long"
    print("  ✓ Opened long at $60,000")

    exit_check = pm.check_exit_conditions(60500.0)
    assert exit_check is None, "Should not exit yet"
    print(f"  ✓ Price $60,500 — no exit (bars_held={pm.bars_held})")

    exit_check = pm.check_exit_conditions(58500.0)
    assert exit_check == "stop_loss", "Should hit stop loss"
    print("  ✓ Price $58,500 — stop loss triggered")

    result = pm.close_position(58500.0)
    assert not pm.has_position
    print(f"  ✓ Closed: PnL = {result['pnl_pct']:+.2f}% (${result['pnl_usd']:+.2f})")

    # Clean up test state
    state_file = LOG_DIR / "position_state.json"
    if state_file.exists():
        state_file.unlink()

    print("  ✓ PASS: Position manager works\n")


def test_simulation(predictor, base, ohlcv, feature_cols):
    """Test 5: Run a mini-simulation on the last 200 bars."""
    print("=" * 60)
    print("TEST 5: Mini Simulation (last 200 bars)")
    print("=" * 60)

    feature_data = base[feature_cols]
    mask = feature_data.notna().all(axis=1)
    feature_data = feature_data.loc[mask]
    ohlcv_sim = ohlcv.loc[mask]

    # Take last 200 bars
    n = min(200, len(feature_data))
    feature_data = feature_data.iloc[-n:]
    ohlcv_sim = ohlcv_sim.iloc[-n:]

    decisions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    p_ups = []
    exp_rets = []

    for i in range(n):
        row = feature_data.iloc[i]
        price = ohlcv_sim["close"].iloc[i]

        p_up, exp_ret = predictor.predict(row, feature_cols)
        p_ups.append(p_up)
        exp_rets.append(exp_ret)

        result = decide(p_up, exp_ret, price, 10000.0, has_position=False)
        decisions[result["action"]] += 1

    p_ups = np.array(p_ups)
    exp_rets = np.array(exp_rets) * 100

    print(f"  Simulated {n} bars: {ohlcv_sim.index[0]} → {ohlcv_sim.index[-1]}")
    print(f"  Price range: ${ohlcv_sim['close'].min():,.2f} — ${ohlcv_sim['close'].max():,.2f}")
    print(f"")
    print(f"  Prediction stats:")
    print(f"    P(up):     mean={p_ups.mean():.3f}  std={p_ups.std():.3f}  min={p_ups.min():.3f}  max={p_ups.max():.3f}")
    print(f"    E[return]: mean={exp_rets.mean():+.4f}%  std={exp_rets.std():.4f}%")
    print(f"")
    print(f"  Decision breakdown:")
    print(f"    BUY:  {decisions['BUY']:3d} ({decisions['BUY']/n*100:.1f}%)")
    print(f"    SELL: {decisions['SELL']:3d} ({decisions['SELL']/n*100:.1f}%)")
    print(f"    HOLD: {decisions['HOLD']:3d} ({decisions['HOLD']/n*100:.1f}%)")
    print(f"")
    print(f"  ✓ PASS: Simulation complete\n")


def main():
    print("\n" + "🔧 " * 20)
    print("  DRY RUN TEST — Full Pipeline Verification")
    print("🔧 " * 20 + "\n")

    # Test 1
    features_row, ohlcv_row, timestamp, feature_cols, base, ohlcv = test_data_loading()

    # Test 2
    predictor, p_up, exp_return = test_predictor(features_row, feature_cols)

    # Test 3
    test_decision(p_up, exp_return, ohlcv_row["close"])

    # Test 4
    test_position_manager()

    # Test 5
    test_simulation(predictor, base, ohlcv, feature_cols)

    # Final summary
    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\n  The pipeline is ready. To start paper trading:")
    print("  1. Add Binance Testnet API keys to src/config.py")
    print("  2. Run: python3 src/bot.py")
    print()


if __name__ == "__main__":
    main()
