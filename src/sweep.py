import pandas as pd
from pathlib import Path
from train_v2 import train_v2
from backtest_v2 import run_backtest_v2

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "sweep_results.csv"

def run_sweep():
    combos = [
        (24, 0.003),
        (48, 0.005),
        (72, 0.0075),
        (96, 0.01)
    ]
    
    conf_gates = [0.50, 0.55, 0.60, 0.65, 0.70]
    fee_modes = ["taker", "maker"]
    
    results = []
    
    for h, c in combos:
        print(f"\n--- Training Horizon {h}, Cost Threshold {c} ---")
        try:
            train_v2(horizon=h, cost=c)
        except Exception as e:
            print(f"Training failed for h={h}, c={c}: {e}")
            continue
            
        for cg in conf_gates:
            for fm in fee_modes:
                print(f"Backtesting h={h}, c={c}, conf={cg}, fee={fm}...")
                m = run_backtest_v2(horizon=h, cost=c, conf_gate=cg, fee_mode=fm)
                if m is not None:
                    res = {
                        "horizon": h,
                        "label_cost": c,
                        "conf_gate": cg,
                        "fee_mode": fm,
                    }
                    res.update(m)
                    results.append(res)
                    
    df = pd.DataFrame(results)
    df = df.sort_values(by="sharpe_bar", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSweep complete! Saved results to {OUT_CSV}")
    print("\nTop 5 Configs by Sharpe:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    run_sweep()
