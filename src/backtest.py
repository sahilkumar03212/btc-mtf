"""
Honest backtest with per-trade logging.
Non-overlapping positions, bar-by-bar PnL.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path

from merge import load_and_merge
from labels import make_labels
from train import walk_forward

# ---------- config ----------
FEE         = 0.001
SLIP        = 0.0005
COST        = FEE + SLIP
HORIZON     = 12
ALLOW_SHORT = False
RISK_FRAC   = 0.01
STOP_ATR    = 1.5
MAX_LEV     = 1.0
LOG_TRADES  = True          # <-- toggle per-trade logging
LOG_LIMIT   = None          # None = log all; set e.g. 50 to cap output

OUT = Path(__file__).resolve().parents[1]

PARAMS = dict(
    objective="multiclass", num_class=3,
    learning_rate=0.03, num_leaves=31,
    min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1,
    verbose=-1,
)

# ---------- signals ----------
def build_signals(X, y, Xf, n_splits=6, embargo=12):
    preds = pd.Series(0.0, index=X.index)
    for Xtr, ytr, Xte, yte in walk_forward(Xf, y, n_splits=n_splits, embargo=embargo):
        m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr.map({-1: 0, 0: 1, 1: 2})),
                      num_boost_round=400)
        preds.loc[Xte.index] = m.predict(Xte).argmax(axis=1) - 1
    return preds

def apply_gate(sig, X):
    gate = np.sign(X["4h_ema_ratio"]).fillna(0)
    return sig.where(np.sign(sig) == gate, 0).fillna(0)

# ---------- positions ----------
def signals_to_positions(sig, horizon=HORIZON, allow_short=True):
    pos = np.zeros(len(sig), dtype=int)
    s = sig.values
    i = 0
    n = len(s)
    while i < n:
        if s[i] != 0 and (allow_short or s[i] > 0):
            end = min(i + 1, n)
            exit_ = min(i + 1 + horizon, n)
            pos[end:exit_] = s[i]
            i = exit_
        else:
            i += 1
    return pd.Series(pos, index=sig.index, name="position")

def position_sizes(position, ohlcv,
                   risk_frac=RISK_FRAC, stop_atr=STOP_ATR, max_lev=MAX_LEV):
    atr = (ohlcv["high"] - ohlcv["low"]).rolling(14).mean()
    stop_dist = (stop_atr * atr / ohlcv["close"]).clip(lower=1e-4)
    size = (risk_frac / stop_dist).clip(upper=max_lev).fillna(0)
    entry_mask = position.diff().fillna(0) != 0
    entry_sizes = size.where(entry_mask).ffill()
    return (entry_sizes * position.abs()).fillna(0)

# ---------- PnL ----------
def compute_pnl(ohlcv, position, size):
    ret = np.log(ohlcv["close"]).diff().fillna(0)
    pos_prev = position.shift(1).fillna(0)
    size_prev = size.shift(1).fillna(0)
    gross = pos_prev * ret * size_prev
    turnover = (position * size - pos_prev * size_prev).abs()
    cost = turnover * COST
    return gross - cost, gross, cost

# ---------- trade log ----------
def log_trades(position, ohlcv, net, cost, size):
    """Print one line per trade: entry, exit, direction, size, PnL."""
    p = position.values
    idx = position.index
    trades = []

    i = 0
    n = len(p)
    while i < n:
        if p[i] != 0:
            start = i
            d = p[i]
            while i < n and p[i] == d:
                i += 1
            end = i - 1
            # trade window is [start, end]
            entry_ts = idx[start]
            exit_ts  = idx[min(end + 1, n - 1)]
            entry_px = ohlcv["close"].iloc[start]
            exit_px  = ohlcv["close"].iloc[min(end + 1, n - 1)]
            sz       = size.iloc[start]
            # sum net over the trade window
            pnl      = net.iloc[start:end + 1].sum()
            cst      = cost.iloc[start:end + 1].sum()
            raw      = pnl + cst
            trades.append({
                "entry":     entry_ts,
                "exit":      exit_ts,
                "dir":       "LONG" if d > 0 else "SHORT",
                "entry_px":  round(float(entry_px), 2),
                "exit_px":   round(float(exit_px), 2),
                "size":      round(float(sz), 4),
                "gross_%":   round(float(raw) * 100, 4),
                "cost_%":    round(float(cst) * 100, 4),
                "net_%":     round(float(pnl) * 100, 4),
            })
        else:
            i += 1
    return pd.DataFrame(trades)

# ---------- metrics ----------
def compute_metrics(net, position, ohlcv):
    bars_per_year = 288 * 365
    mu, sd = net.mean(), net.std() + 1e-12
    sharpe_bar = mu / sd * np.sqrt(bars_per_year)

    entry_idx = position.diff().fillna(0) != 0
    trade_ret = []
    in_trade = False
    cur = 0.0
    for p, r in zip(position.values, net.values):
        if p != 0 and not in_trade:
            in_trade = True
            cur = 0.0
        if in_trade:
            cur += r
        if p == 0 and in_trade:
            trade_ret.append(cur)
            in_trade = False
    trade_ret = np.array(trade_ret) if trade_ret else np.array([0.0])

    hit_rate = (trade_ret > 0).mean()
    gw = trade_ret[trade_ret > 0].sum()
    gl = -trade_ret[trade_ret < 0].sum()
    pf = gw / gl if gl > 0 else np.inf
    sharpe_trade = trade_ret.mean() / (trade_ret.std() + 1e-12) * np.sqrt(365 * 24 / 12)

    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    return {
        "final_equity": round(float(equity.iloc[-1]), 4),
        "total_return": round(float(equity.iloc[-1] - 1), 4),
        "sharpe_bar": round(float(sharpe_bar), 2),
        "sharpe_trade": round(float(sharpe_trade), 2),
        "max_drawdown": round(float(dd.min()), 4),
        "trades": int(len(trade_ret)),
        "hit_rate": round(float(hit_rate), 3),
        "profit_factor": round(float(pf), 2),
    }, equity, dd, trade_ret

# ---------- plot ----------
def plot_all(equity, dd, net, position, trade_ret, out_path=OUT / "equity_fixed.png"):
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
    equity.plot(ax=axes[0], color="tab:blue"); axes[0].set_title("Equity"); axes[0].grid(alpha=0.3)
    dd.plot(ax=axes[1], color="tab:red"); axes[1].set_title("Drawdown"); axes[1].grid(alpha=0.3)
    (1 + net).cumprod().plot(ax=axes[2], color="tab:green"); axes[2].set_title("Cumulative net return"); axes[2].grid(alpha=0.3)
    axes[3].hist(trade_ret * 100, bins=50, color="tab:purple"); axes[3].set_title("Per-trade return (%)"); axes[3].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.show()
    print(f"saved -> {out_path}")

def buy_and_hold(ohlcv):
    ret = ohlcv["close"].pct_change().fillna(0)
    eq = (1 + ret).cumprod()
    sharpe = ret.mean() / (ret.std() + 1e-12) * np.sqrt(288 * 365)
    return eq, sharpe

# ---------- main ----------
def run():
    X, ohlcv = load_and_merge()
    y = make_labels(ohlcv, horizon=HORIZON, cost=0.0015)
    mask = X.notna().all(axis=1) & y.notna()
    X, y, ohlcv = X[mask], y[mask], ohlcv[mask]
    feat_cols = [c for c in X.columns
                 if c not in ("open", "high", "low", "close", "volume")]
    Xf = X[feat_cols]

    print(f"Data: {len(X)} bars, {X.index.min()} -> {X.index.max()}")

    raw_sig = build_signals(X, y, Xf)
    sig = apply_gate(raw_sig, X)
    print(f"raw signals: {(raw_sig != 0).sum()}   after gate: {(sig != 0).sum()}")

    position = signals_to_positions(sig, horizon=HORIZON, allow_short=ALLOW_SHORT)
    size = position_sizes(position, ohlcv)
    net, gross, cost = compute_pnl(ohlcv, position, size)

    # ---------- trade log ----------
    if LOG_TRADES:
        trades_df = log_trades(position, ohlcv, net, cost, size)
        print(f"\n=== TRADE LOG ({len(trades_df)} trades) ===")
        if LOG_LIMIT:
            print(trades_df.head(LOG_LIMIT).to_string(index=False))
            print(f"... ({len(trades_df) - LOG_LIMIT} more)")
        else:
            print(trades_df.to_string(index=False))
        trades_df.to_csv(OUT / "trade_log.csv", index=False)
        print(f"\nsaved trade log -> {OUT / 'trade_log.csv'}")

    metrics, equity, dd, trade_ret = compute_metrics(net, position, ohlcv)
    print("\n=== STRATEGY ===")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v}")

    bh_eq, bh_sharpe = buy_and_hold(ohlcv)
    print("\n=== BUY & HOLD ===")
    print(f"  final_equity    : {bh_eq.iloc[-1]:.4f}")
    print(f"  sharpe_bar      : {bh_sharpe:.2f}")

    plot_all(equity, dd, net, position, trade_ret)
    return metrics

if __name__ == "__main__":
    run()
