"""
Honest backtest with proper fractional position sizing.
Reads cached oos_proba.parquet — runs in ~5 seconds.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from merge import load_and_merge
from labels import make_labels

# ============================================================
# CONFIG — tweak these
# ============================================================
FEE          = 0.001       # 0.1% per side (Binance spot taker)
SLIP         = 0.0005      # 0.05% per side
COST         = FEE + SLIP  # 0.15% per side

HORIZON      = 12          # bars per trade (matches label)
ALLOW_SHORT  = False       # spot = False, futures = True

RISK_FRAC    = 0.005       # 0.5% of equity risked per trade
STOP_DIST_PCT= 0.01        # 1% stop distance (price space)
MAX_LEV      = 1.0         # cap position size fraction

# signal source: "argmax" or "threshold"
SIGNAL_MODE  = "threshold"
CONF_TH      = 0.70        # only trade unusually confident predictions

LOG_LIMIT    = 30

OUT         = Path(__file__).resolve().parents[1]
PROBA_PATH  = OUT / "models" / "oos_proba.parquet"

# ============================================================
# CORE
# ============================================================
def load_data():
    X, ohlcv = load_and_merge()
    y = make_labels(ohlcv, horizon=HORIZON, cost=0.0015)
    mask = X.notna().all(axis=1) & y.notna()
    X, ohlcv = X[mask], ohlcv[mask]

    proba = pd.read_parquet(PROBA_PATH)
    required = {"p_short", "p_no", "p_long"}
    if not required.issubset(proba.columns):
        raise ValueError(f"{PROBA_PATH} must contain {sorted(required)}")
    common = X.index.intersection(proba.index)
    X, ohlcv, proba = X.loc[common], ohlcv.loc[common], proba.loc[common]
    return X, ohlcv, proba

def build_signal(proba, X, mode="argmax", conf_th=0.55):
    if mode == "argmax":
        sig = pd.Series(proba.values.argmax(axis=1) - 1, index=proba.index)
    elif mode == "threshold":
        p_long  = proba["p_long"].fillna(0)
        p_short = proba["p_short"].fillna(0)
        sig = pd.Series(0, index=proba.index)
        sig[(p_long  >= conf_th) & (p_long  > p_short)] =  1
        sig[(p_short >= conf_th) & (p_short > p_long)]  = -1
    else:
        raise ValueError(mode)

    # regime gate: only trade in direction of 4h trend
    gate = np.sign(X["4h_ema_ratio"]).fillna(0)
    sig = sig.where(np.sign(sig) == gate, 0).fillna(0)
    return sig

def build_positions(sig, horizon=HORIZON, allow_short=True):
    """One position at a time, held for exactly `horizon` bars."""
    pos = np.zeros(len(sig), dtype=int)
    s = sig.values
    i = 0
    while i < len(s):
        if s[i] != 0 and (allow_short or s[i] > 0):
            end = min(i + 1, len(s))
            ex  = min(i + 1 + horizon, len(s))
            pos[end:ex] = s[i]
            i = ex
        else:
            i += 1
    return pd.Series(pos, index=sig.index)

def build_sizes(position, risk_frac, stop_dist_pct, max_lev):
    """
    Constant fractional size per trade.
    size_frac = RISK_FRAC / STOP_DIST_PCT, capped at MAX_LEV.
    """
    base_size = min(risk_frac / stop_dist_pct, max_lev)
    # only size bars that have an active position
    return (position.abs() * base_size).fillna(0)

def compute_pnl(ohlcv, position, size, cost=COST):
    ret       = np.log(ohlcv["close"]).diff().fillna(0)
    pos_prev  = position.shift(1).fillna(0)
    size_prev = size.shift(1).fillna(0)

    gross = pos_prev * ret * size_prev
    # Charge turnover at both entry and exit using the notional on each side.
    turnover = (position * size - pos_prev * size_prev).abs()
    costs = turnover * cost
    net = gross - costs
    return net, gross, costs

def trade_log(position, ohlcv, net, size):
    p = position.values
    idx = ohlcv.index
    trades = []
    i = 0
    while i < len(p):
        if p[i] != 0:
            start = i
            d = p[i]
            while i < len(p) and p[i] == d:
                i += 1
            end = i - 1
            trades.append({
                "entry":    idx[start],
                "exit":     idx[min(end + 1, len(p) - 1)],
                "dir":      "LONG" if d > 0 else "SHORT",
                "entry_px": round(float(ohlcv["close"].iloc[start]), 2),
                "exit_px":  round(float(ohlcv["close"].iloc[min(end + 1, len(p) - 1)]), 2),
                "size":     round(float(size.iloc[start]), 4),
                "net_%":    round(float(net.iloc[start:end + 1].sum()) * 100, 4),
            })
        else:
            i += 1
    return pd.DataFrame(trades)

def metrics(net, tdf, ohlcv):
    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    sharpe_bar = net.mean() / (net.std() + 1e-12) * np.sqrt(288 * 365)

    trade_ret = tdf["net_%"].values / 100 if len(tdf) else np.array([0.0])
    hit = (trade_ret > 0).mean() if len(trade_ret) else 0.0
    gw = trade_ret[trade_ret > 0].sum()
    gl = -trade_ret[trade_ret < 0].sum()
    pf = gw / gl if gl > 0 else np.inf
    sharpe_trade = (trade_ret.mean() / (trade_ret.std() + 1e-12)
                    * np.sqrt(365 * 24 / (HORIZON * 5 / 60)))

    return {
        "final_equity":  round(float(equity.iloc[-1]), 4),
        "total_return":  round(float(equity.iloc[-1] - 1), 4),
        "sharpe_bar":    round(float(sharpe_bar), 2),
        "sharpe_trade":  round(float(sharpe_trade), 2),
        "max_drawdown":  round(float(dd.min()), 4),
        "trades":        int(len(tdf)),
        "hit_rate":      round(float(hit), 3),
        "profit_factor": round(float(pf), 2),
        "avg_trade_%":   round(float(trade_ret.mean() * 100), 4),
        "avg_win_%":     round(float(trade_ret[trade_ret > 0].mean() * 100), 4)
                          if (trade_ret > 0).any() else 0.0,
        "avg_loss_%":    round(float(trade_ret[trade_ret < 0].mean() * 100), 4)
                          if (trade_ret < 0).any() else 0.0,
    }, equity, dd, trade_ret

def yearly_metrics(net, equity):
    """Return calendar-year returns and drawdowns for regime diagnostics."""
    rows = []
    for year, group in net.groupby(net.index.year):
        eq = (1 + group).cumprod()
        rows.append({
            "year": int(year),
            "return": round(float(eq.iloc[-1] - 1), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "bars": int(len(group)),
        })
    return pd.DataFrame(rows)

def plot(equity, dd, net, trade_ret, out_path=OUT / "equity_final.png"):
    fig, ax = plt.subplots(4, 1, figsize=(12, 11))
    equity.plot(ax=ax[0], color="tab:blue"); ax[0].set_title("Equity"); ax[0].grid(alpha=0.3)
    dd.plot(ax=ax[1], color="tab:red"); ax[1].set_title("Drawdown"); ax[1].grid(alpha=0.3)
    (1 + net).cumprod().plot(ax=ax[2], color="tab:green"); ax[2].set_title("Cumulative net"); ax[2].grid(alpha=0.3)
    pd.Series(trade_ret * 100).hist(ax=ax[3], bins=60, color="tab:purple")
    ax[3].axvline(0, color="black", lw=0.8, ls="--")
    ax[3].set_title("Per-trade return (%)"); ax[3].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.show()
    print(f"saved -> {out_path}")

def buy_and_hold(ohlcv):
    ret = ohlcv["close"].pct_change().fillna(0)
    eq = (1 + ret).cumprod()
    sharpe = ret.mean() / (ret.std() + 1e-12) * np.sqrt(288 * 365)
    return eq, sharpe

# ============================================================
# MAIN
# ============================================================
def run(risk_frac=RISK_FRAC, stop_dist_pct=STOP_DIST_PCT,
        signal_mode=SIGNAL_MODE, conf_th=CONF_TH, cost=COST, verbose=True):

    X, ohlcv, proba = load_data()
    sig = build_signal(proba, X, mode=signal_mode, conf_th=conf_th)
    position = build_positions(sig, horizon=HORIZON, allow_short=ALLOW_SHORT)
    size = build_sizes(position, risk_frac, stop_dist_pct, MAX_LEV)

    net, gross, costs = compute_pnl(ohlcv, position, size, cost=cost)
    tdf = trade_log(position, ohlcv, net, size)
    m, equity, dd, trade_ret = metrics(net, tdf, ohlcv)

    if verbose:
        print(f"\n=== CONFIG ===")
        print(f"  RISK_FRAC       : {risk_frac}")
        print(f"  STOP_DIST_PCT   : {stop_dist_pct}")
        print(f"  base size/trade : {min(risk_frac/stop_dist_pct, MAX_LEV):.3f}")
        print(f"  ALLOW_SHORT     : {ALLOW_SHORT}")
        print(f"  SIGNAL_MODE     : {signal_mode}")
        print(f"  COST_PER_SIDE   : {cost}")

        print(f"\n=== TRADE LOG ({len(tdf)} trades) ===")
        print(tdf.head(LOG_LIMIT).to_string(index=False))
        if len(tdf) > LOG_LIMIT:
            print(f"... ({len(tdf)-LOG_LIMIT} more)")
        tdf.to_csv(OUT / "trade_log.csv", index=False)

        print(f"\n=== STRATEGY ===")
        for k, v in m.items():
            print(f"  {k:16s}: {v}")
        print("\n=== YEARLY ===")
        print(yearly_metrics(net, equity).to_string(index=False))

        bh_eq, bh_sh = buy_and_hold(ohlcv)
        print(f"\n=== BUY & HOLD ===")
        print(f"  final_equity  : {bh_eq.iloc[-1]:.4f}")
        print(f"  sharpe_bar    : {bh_sh:.2f}")

        plot(equity, dd, net, trade_ret)

    return m

if __name__ == "__main__":
    run()
