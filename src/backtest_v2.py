import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from merge import load_and_merge

FEE_MAKER = 0.0002 # 0.02%
FEE_TAKER = 0.0010 # 0.1%
SLIP      = 0.0005 # 0.05%

RISK_FRAC     = 0.005
STOP_DIST_PCT = 0.01
MAX_LEV       = 1.0

ROOT   = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT    = ROOT

def simulate_mtm(X, ohlcv, proba, horizon, conf_gate, fee_mode, allow_short=False):
    gate = np.sign(X["4h_ema_ratio"]).fillna(0)
    p_long = proba["p_long"].values
    p_short = proba["p_short"].values
    
    cost_per_side = (FEE_MAKER if fee_mode == "maker" else FEE_TAKER) + SLIP
    
    atr = (ohlcv["high"] - ohlcv["low"]).rolling(14).mean().values
    closes = ohlcv["close"].values
    highs = ohlcv["high"].values
    lows = ohlcv["low"].values
    gates = gate.values
    
    n = len(closes)
    positions = np.zeros(n)
    sizes = np.zeros(n)
    
    in_trade = False
    direction = 0
    entry_idx = 0
    tp_price = 0.0
    sl_price = 0.0
    current_size = 0.0
    
    for i in range(n):
        if in_trade:
            bars_held = i - entry_idx
            hit_sl = False
            hit_tp = False
            
            if direction == 1:
                if lows[i] <= sl_price: hit_sl = True
                if highs[i] >= tp_price: hit_tp = True
            elif direction == -1:
                if highs[i] >= sl_price: hit_sl = True
                if lows[i] <= tp_price: hit_tp = True
                
            if hit_sl or hit_tp or bars_held >= horizon:
                in_trade = False
                
        if not in_trade:
            long_cond = (p_long[i] > conf_gate) and (p_long[i] > p_short[i]) and (gates[i] > 0)
            short_cond = (p_short[i] > conf_gate) and (p_short[i] > p_long[i]) and (gates[i] < 0)
            
            if long_cond or (allow_short and short_cond):
                direction = 1 if long_cond else -1
                entry_idx = i
                entry_price = closes[i]
                current_atr = atr[i]
                if not np.isnan(current_atr) and current_atr > 0:
                    sl_price = entry_price - direction * 1 * current_atr
                    tp_price = entry_price + direction * 2 * current_atr
                    
                    conf = p_long[i] if direction == 1 else p_short[i]
                    base_size = min(RISK_FRAC / STOP_DIST_PCT, MAX_LEV)
                    current_size = base_size * (conf - 0.33) / 0.67
                    in_trade = True
                    
        if in_trade:
            positions[i] = direction
            sizes[i] = current_size

    position = pd.Series(positions, index=ohlcv.index)
    size = pd.Series(sizes, index=ohlcv.index)
    
    ret = np.log(ohlcv["close"]).diff().fillna(0)
    pos_prev = position.shift(1).fillna(0)
    size_prev = size.shift(1).fillna(0)
    
    gross = pos_prev * ret * size_prev
    turnover = (position * size - pos_prev * size_prev).abs()
    costs = turnover * cost_per_side
    net = gross - costs
    
    return net, position, size

def metrics(net, position, ohlcv):
    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    sharpe = net.mean() / (net.std() + 1e-12) * np.sqrt(288 * 365)
    
    p = position.values
    tr = []
    in_t = False
    cur = 0.0
    for pos, r in zip(p, net.values):
        if pos != 0 and not in_t:
            in_t = True
            cur = 0.0
        if in_t:
            cur += r
        if pos == 0 and in_t:
            tr.append(cur)
            in_t = False
            
    tr = np.array(tr) if tr else np.array([0.0])
    hit = (tr > 0).mean()
    gw = tr[tr > 0].sum()
    gl = -tr[tr < 0].sum()
    pf = gw / gl if gl > 0 else np.inf
    
    return {
        "final_equity": round(float(equity.iloc[-1]), 4),
        "sharpe_bar": round(float(sharpe), 2),
        "max_drawdown": round(float(dd.min()), 4),
        "trades": int(len(tr)),
        "hit_rate": round(float(hit), 3),
        "profit_factor": round(float(pf), 2),
    }

def run_backtest_v2(horizon=48, cost=0.005, conf_gate=0.55, fee_mode="taker", verbose=False):
    X, ohlcv = load_and_merge()
    proba_path = MODELS / f"oos_proba_h{horizon}_c{cost}.parquet"
    if not proba_path.exists():
        if verbose:
            print(f"No proba found: {proba_path}")
        return None
        
    proba = pd.read_parquet(proba_path)
    common = X.index.intersection(proba.index)
    X, ohlcv, proba = X.loc[common], ohlcv.loc[common], proba.loc[common]
    
    net, position, size = simulate_mtm(X, ohlcv, proba, horizon, conf_gate, fee_mode)
    m = metrics(net, position, ohlcv)
    return m

if __name__ == "__main__":
    m = run_backtest_v2(horizon=48, cost=0.005, conf_gate=0.55, fee_mode="taker", verbose=True)
    if m:
        for k, v in m.items():
            print(f"{k}: {v}")
