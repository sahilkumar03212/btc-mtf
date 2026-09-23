"""
Backtest the hierarchical model:
  regime (4h) + direction (1h) + trigger (5m) -> signal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from merge import load_and_merge

FEE, SLIP   = 0.001, 0.0005
COST        = FEE + SLIP
HORIZON     = 12
ALLOW_SHORT = False
RISK_FRAC   = 0.005
STOP_PCT    = 0.01
MAX_LEV     = 1.0
TRIG_TH     = 0.50
REGIME_TH   = 0.40
DIR_TH      = 0.40

ROOT   = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT    = ROOT

def load_levels():
    r = pd.read_parquet(MODELS / "oos_4h_regime.parquet")
    d = pd.read_parquet(MODELS / "oos_1h_direction.parquet")
    t = pd.read_parquet(MODELS / "oos_5m_trigger.parquet")["p_enter"]

    X, ohlcv = load_and_merge()
    idx = ohlcv.index

    r_b = r.shift(1).reindex(idx, method="ffill")
    d_b = d.shift(1).reindex(idx, method="ffill")
    t_b = t.reindex(idx, method="ffill")

    return X, ohlcv, r_b, d_b, t_b

def build_signal(r_b, d_b, t_b, allow_short=False):
    sig = pd.Series(0, index=r_b.index, dtype=int)

    up_regime   = (r_b["p_up"]   >= REGIME_TH) & (r_b["p_up"]   > r_b["p_down"])
    down_regime = (r_b["p_down"] >= REGIME_TH) & (r_b["p_down"] > r_b["p_up"])
    long_dir    = (d_b["p_long"]  >= DIR_TH) & (d_b["p_long"]  > d_b["p_short"])
    short_dir   = (d_b["p_short"] >= DIR_TH) & (d_b["p_short"] > d_b["p_long"])
    trig_ok     = t_b >= TRIG_TH

    long_mask  = up_regime  & long_dir  & trig_ok
    short_mask = down_regime & short_dir & trig_ok

    sig[long_mask] = 1
    if allow_short:
        sig[short_mask] = -1
    return sig

def build_positions(sig, horizon=HORIZON):
    pos = np.zeros(len(sig), dtype=int)
    s = sig.values
    i = 0
    while i < len(s):
        if s[i] != 0:
            end = min(i + 1, len(s))
            ex  = min(i + 1 + horizon, len(s))
            pos[end:ex] = s[i]
            i = ex
        else:
            i += 1
    return pd.Series(pos, index=sig.index)

def pnl(ohlcv, position, size_frac):
    ret       = np.log(ohlcv["close"]).diff().fillna(0)
    pos_prev  = position.shift(1).fillna(0)
    sz_prev   = size_frac.shift(1).fillna(0)
    gross = pos_prev * ret * sz_prev
    turnover = (position * size_frac - pos_prev * sz_prev).abs()
    cost = turnover * COST
    return gross - cost

def sizes(position):
    base = min(RISK_FRAC / STOP_PCT, MAX_LEV)
    return (position.abs() * base).fillna(0)

def metrics(net, position, ohlcv):
    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    sharpe = net.mean() / (net.std() + 1e-12) * np.sqrt(288 * 365)

    p = position.values
    trade_rets = []
    in_t = False; cur = 0.0
    for pos, r in zip(p, net.values):
        if pos != 0 and not in_t:
            in_t = True; cur = 0.0
        if in_t: cur += r
        if pos == 0 and in_t:
            trade_rets.append(cur); in_t = False
    trade_rets = np.array(trade_rets) if trade_rets else np.array([0.0])

    hit = (trade_rets > 0).mean()
    gw = trade_rets[trade_rets > 0].sum()
    gl = -trade_rets[trade_rets < 0].sum()
    pf = gw / gl if gl > 0 else np.inf

    return {
        "final_equity": round(float(equity.iloc[-1]), 4),
        "sharpe_bar":   round(float(sharpe), 2),
        "max_drawdown": round(float(dd.min()), 4),
        "trades":       int(len(trade_rets)),
        "hit_rate":     round(float(hit), 3),
        "profit_factor":round(float(pf), 2),
    }, equity, dd, trade_rets

def buy_and_hold(ohlcv):
    ret = ohlcv["close"].pct_change().fillna(0)
    return (1 + ret).cumprod()

def run(allow_short=ALLOW_SHORT, trig_th=TRIG_TH,
        regime_th=REGIME_TH, dir_th=DIR_TH, verbose=True):
    global TRIG_TH, REGIME_TH, DIR_TH
    TRIG_TH, REGIME_TH, DIR_TH = trig_th, regime_th, dir_th

    X, ohlcv, r_b, d_b, t_b = load_levels()
    sig = build_signal(r_b, d_b, t_b, allow_short=allow_short)
    position = build_positions(sig)
    size_frac = sizes(position)
    net = pnl(ohlcv, position, size_frac)

    m, equity, dd, trade_ret = metrics(net, position, ohlcv)

    if verbose:
        print(f"=== CONFIG: trig>={trig_th} regime>={regime_th} dir>={dir_th} short={allow_short} ===")
        print(f"signals fired: {(sig != 0).sum()} / {len(sig)}")
        print(f"\n=== STRATEGY ===")
        for k, v in m.items():
            print(f"  {k:14s}: {v}")
        bh = buy_and_hold(ohlcv)
        print(f"\n=== BUY & HOLD ===")
        print(f"  final_equity  : {bh.iloc[-1]:.4f}")

        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        equity.plot(ax=ax[0], color="tab:blue"); ax[0].set_title("Equity"); ax[0].grid(alpha=0.3)
        dd.plot(ax=ax[1], color="tab:red"); ax[1].set_title("Drawdown"); ax[1].grid(alpha=0.3)
        pd.Series(trade_ret * 100).hist(ax=ax[2], bins=60, color="tab:purple")
        ax[2].set_title("Per-trade return (%)"); ax[2].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT / "equity_hierarchical.png", dpi=120); plt.show()

    return m

if __name__ == "__main__":
    run()
