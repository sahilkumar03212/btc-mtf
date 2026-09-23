
"""
Option B v2 backtest: hard fusion + soft fusion modes.
Reads cached predictions from hierarchical_v2.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from merge import load_and_merge

FEE, SLIP = 0.001, 0.0005
COST      = FEE + SLIP
HORIZON   = 12
RISK_FRAC = 0.005
STOP_PCT  = 0.01
MAX_LEV   = 1.0

# hard-fusion thresholds
REGIME_TH     = 0.55
REGIME_MARGIN = 0.20
DIR_TH        = 0.45
TRIG_TH       = 0.50

# soft-fusion threshold
SCORE_TH      = 0.15

ROOT   = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT    = ROOT

def load_levels():
    r = pd.read_parquet(MODELS / "oos_4h_regime_v2.parquet")
    d = pd.read_parquet(MODELS / "oos_1h_direction_v2.parquet")
    t = pd.read_parquet(MODELS / "oos_5m_trigger_v2.parquet")["p_enter"]
    X, ohlcv = load_and_merge()
    idx = ohlcv.index
    return (
        X, ohlcv,
        r.shift(1).reindex(idx, method="ffill"),
        d.shift(1).reindex(idx, method="ffill"),
        t.reindex(idx, method="ffill"),
    )

def build_signal(r_b, d_b, t_b, mode="soft", allow_short=False):
    sig = pd.Series(0, index=r_b.index, dtype=int)

    if mode == "hard":
        up_regime   = (r_b["p_up"]   >= REGIME_TH) & ((r_b["p_up"]   - r_b["p_down"]) > REGIME_MARGIN)
        down_regime = (r_b["p_down"] >= REGIME_TH) & ((r_b["p_down"] - r_b["p_up"])   > REGIME_MARGIN)
        long_dir    = (d_b["p_long"]  >= DIR_TH) & (d_b["p_long"]  > d_b["p_short"])
        short_dir   = (d_b["p_short"] >= DIR_TH) & (d_b["p_short"] > d_b["p_long"])
        trig_ok     = t_b >= TRIG_TH
        sig[(up_regime & long_dir & trig_ok)] = 1
        if allow_short:
            sig[(down_regime & short_dir & trig_ok)] = -1
    else:  # soft fusion
        score_long  = r_b["p_up"]   * d_b["p_long"]  * t_b
        score_short = r_b["p_down"] * d_b["p_short"] * t_b
        sig[score_long  > SCORE_TH] = 1
        if allow_short:
            sig[score_short > SCORE_TH] = -1
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

def pnl(ohlcv, position, cost=COST):
    ret      = np.log(ohlcv["close"]).diff().fillna(0)
    pos_prev = position.shift(1).fillna(0)
    size     = (position.abs() * min(RISK_FRAC/STOP_PCT, MAX_LEV)).fillna(0)
    sz_prev  = size.shift(1).fillna(0)
    gross = pos_prev * ret * sz_prev
    turnover = (position * size - pos_prev * sz_prev).abs()
    fees  = turnover * cost
    return gross - fees

def metrics(net, position):
    equity = (1 + net).cumprod()
    dd     = equity / equity.cummax() - 1
    sharpe = net.mean() / (net.std() + 1e-12) * np.sqrt(288 * 365)
    p = position.values
    tr = []; in_t = False; cur = 0.0
    for pos, r in zip(p, net.values):
        if pos != 0 and not in_t: in_t = True; cur = 0.0
        if in_t: cur += r
        if pos == 0 and in_t: tr.append(cur); in_t = False
    tr = np.array(tr) if tr else np.array([0.0])
    hit = (tr > 0).mean()
    gw = tr[tr > 0].sum(); gl = -tr[tr < 0].sum()
    pf = gw/gl if gl > 0 else np.inf
    return {
        "final_equity": round(float(equity.iloc[-1]), 4),
        "sharpe_bar":   round(float(sharpe), 2),
        "max_drawdown": round(float(dd.min()), 4),
        "trades":       int(len(tr)),
        "hit_rate":     round(float(hit), 3),
        "profit_factor":round(float(pf), 2),
    }, equity, dd, tr

def run(mode="soft", allow_short=False, cost=COST, verbose=True, tag=""):
    X, ohlcv, r_b, d_b, t_b = load_levels()
    sig = build_signal(r_b, d_b, t_b, mode=mode, allow_short=allow_short)
    position = build_positions(sig)
    net = pnl(ohlcv, position, cost=cost)
    m, equity, dd, tr = metrics(net, position)

    if verbose:
        print(f"\n=== {tag or mode.upper()}  (cost={cost:.4f}, short={allow_short}) ===")
        print(f"signals fired: {(sig != 0).sum()} / {len(sig)}")
        for k, v in m.items():
            print(f"  {k:14s}: {v}")
        bh = (1 + ohlcv["close"].pct_change().fillna(0)).cumprod()
        print(f"  buy_and_hold  : {bh.iloc[-1]:.4f}")

        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        equity.plot(ax=ax[0], color="tab:blue"); ax[0].set_title(f"Equity ({tag or mode})"); ax[0].grid(alpha=0.3)
        dd.plot(ax=ax[1], color="tab:red"); ax[1].set_title("Drawdown"); ax[1].grid(alpha=0.3)
        pd.Series(tr * 100).hist(ax=ax[2], bins=60, color="tab:purple")
        ax[2].set_title("Per-trade return (%)"); ax[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"equity_hier_v2_{tag or mode}.png", dpi=120)
        plt.show()

    return m

if __name__ == "__main__":
    print("--- HARD fusion ---")
    run(mode="hard")
    print("\n--- SOFT fusion ---")
    run(mode="soft")
