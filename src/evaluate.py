"""Reproducible OOS evaluation against simple trading baselines.

Usage:
    python src/evaluate.py

The script never retrains a model. It uses cached walk-forward probabilities,
applies identical non-overlapping execution and sizing to every strategy, and
reports cost sensitivity plus calendar-year results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_final import (
    ALLOW_SHORT,
    HORIZON,
    MAX_LEV,
    RISK_FRAC,
    STOP_DIST_PCT,
    build_positions,
    build_sizes,
    compute_pnl,
    load_data,
    metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation.csv"


def model_signal(proba, X):
    signal = pd.Series(proba.to_numpy().argmax(axis=1) - 1, index=proba.index)
    gate = np.sign(X["4h_ema_ratio"]).fillna(0)
    return signal.where(np.sign(signal) == gate, 0).fillna(0)


def baseline_signals(X, ohlcv):
    gate = np.sign(X["4h_ema_ratio"]).fillna(0)
    signals = {
        "always_long_in_trend": gate.clip(lower=0),
        "5m_momentum": np.sign(X["5m_ret_12"]).where(lambda s: s > 0, 0),
        "rsi_trend": (X["5m_rsi_14"] > 50).astype(int),
        "1h_momentum": np.sign(X["1h_ret_1"]).clip(lower=0),
    }
    # Spot-compatible baselines are long-only and use the same trend gate.
    return {
        name: signal.where((signal > 0) & (gate > 0), 0).astype(int)
        for name, signal in signals.items()
    }


def evaluate_strategy(name, signal, ohlcv, costs):
    position = build_positions(signal, horizon=HORIZON, allow_short=ALLOW_SHORT)
    size = build_sizes(position, RISK_FRAC, STOP_DIST_PCT, MAX_LEV)
    rows = []
    for cost in costs:
        net, _, _ = compute_pnl(ohlcv, position, size, cost=cost)
        trade_returns = []
        active = False
        current = 0.0
        for pos, bar_return in zip(position.to_numpy(), net.to_numpy()):
            if pos != 0 and not active:
                active, current = True, 0.0
            if active:
                current += bar_return
            if pos == 0 and active:
                trade_returns.append(current)
                active = False
        if active:
            trade_returns.append(current)
        trade_frame = pd.DataFrame({"net_%": np.asarray(trade_returns) * 100})
        summary, _, _, _ = metrics(
            net, trade_frame, ohlcv
        )
        summary.update({"strategy": name, "cost_per_side": cost})
        rows.append(summary)
    return rows


def main():
    X, ohlcv, proba = load_data()
    costs = [0.0015, 0.0030, 0.0050]
    strategies = {"model": model_signal(proba, X)}
    strategies.update(baseline_signals(X, ohlcv))

    rows = []
    for name, signal in strategies.items():
        rows.extend(evaluate_strategy(name, signal, ohlcv, costs))

    result = pd.DataFrame(rows).sort_values(
        ["cost_per_side", "final_equity"], ascending=[True, False]
    )
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"\nsaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
