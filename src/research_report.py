"""Generate a practical, leakage-safe research report from cached OOS output.

This is deliberately diagnostic: it does not select a profitable threshold on
the full sample or place orders. It answers whether a signal has survived
costs, by year and by confidence bucket, and emits the latest paper signal.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_final import (
    CONF_TH,
    HORIZON,
    MAX_LEV,
    RISK_FRAC,
    STOP_DIST_PCT,
    build_positions,
    build_sizes,
    compute_pnl,
    load_data,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research_report.csv"
PAPER_SIGNAL = ROOT / "paper_signal.csv"
COSTS = (0.0015, 0.0030)
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)


def make_signal(proba, trend, threshold):
    signal = pd.Series(0, index=proba.index, dtype=int)
    signal[(proba["p_long"] >= threshold) &
           (proba["p_long"] > proba["p_short"])] = 1
    signal[(proba["p_short"] >= threshold) &
           (proba["p_short"] > proba["p_long"])] = -1
    signal = signal.where(np.sign(signal) == np.sign(trend), 0)
    return signal.fillna(0)


def trade_returns(position, net):
    returns, active, current = [], False, 0.0
    for pos, value in zip(position.to_numpy(), net.to_numpy()):
        if pos != 0 and not active:
            active, current = True, 0.0
        if active:
            current += value
        if pos == 0 and active:
            returns.append(current)
            active = False
    if active:
        returns.append(current)
    return np.asarray(returns)


def main():
    X, ohlcv, proba = load_data()
    trend = np.sign(X["4h_ema_ratio"]).fillna(0)
    rows = []

    for threshold in THRESHOLDS:
        signal = make_signal(proba, trend, threshold)
        position = build_positions(signal, horizon=HORIZON, allow_short=False)
        size = build_sizes(position, RISK_FRAC, STOP_DIST_PCT, MAX_LEV)
        for cost in COSTS:
            net, _, _ = compute_pnl(ohlcv, position, size, cost=cost)
            equity = (1 + net).cumprod()
            for year, group in net.groupby(net.index.year):
                year_equity = (1 + group).cumprod()
                rows.append({
                    "threshold": threshold,
                    "cost_per_side": cost,
                    "year": int(year),
                    "signals": int((signal[group.index] != 0).sum()),
                    "return": float(year_equity.iloc[-1] - 1),
                    "max_drawdown": float(
                        (year_equity / year_equity.cummax() - 1).min()
                    ),
                })

    report = pd.DataFrame(rows)
    report.to_csv(REPORT, index=False)

    latest = proba.index[-1]
    latest_signal = make_signal(proba, trend, CONF_TH).loc[latest]
    paper = pd.DataFrame([{
        "timestamp": latest,
        "signal": int(latest_signal),
        "action": {1: "LONG", -1: "SHORT", 0: "FLAT"}[int(latest_signal)],
        "confidence": float(max(proba.loc[latest, "p_long"],
                                proba.loc[latest, "p_short"])),
        "p_short": float(proba.loc[latest, "p_short"]),
        "p_no": float(proba.loc[latest, "p_no"]),
        "p_long": float(proba.loc[latest, "p_long"]),
        "trend_gate": float(trend.loc[latest]),
        "status": "RESEARCH_ONLY",
    }])
    paper.to_csv(PAPER_SIGNAL, index=False)

    print(report.groupby(["threshold", "cost_per_side"])[
        ["return", "max_drawdown", "signals"]
    ].sum().to_string())
    print(f"\nlatest paper signal:\n{paper.to_string(index=False)}")
    print(f"\nsaved -> {REPORT}\n saved -> {PAPER_SIGNAL}")


if __name__ == "__main__":
    main()
