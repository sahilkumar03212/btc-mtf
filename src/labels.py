import numpy as np
import pandas as pd

def make_labels(ohlcv: pd.DataFrame, horizon: int = 12, cost: float = 0.0015):
    c = ohlcv["close"]
    fwd = c.shift(-horizon) / c - 1
    # The last `horizon` rows have no observable outcome and must not become
    # artificial no-trade examples.
    y = pd.Series(np.nan, index=ohlcv.index, name="y", dtype=float)
    valid = fwd.notna()
    y.loc[valid] = np.select(
        [fwd.loc[valid] > cost, fwd.loc[valid] < -cost],
        [1, -1],
        default=0,
    )
    return y

def make_labels_sweep(ohlcv: pd.DataFrame, combos: list = None):
    if combos is None:
        combos = [
            (24, 0.003),
            (48, 0.005),
            (72, 0.0075),
            (96, 0.01)
        ]
    labels = {}
    for horizon, cost in combos:
        labels[f"h{horizon}_c{cost}"] = make_labels(ohlcv, horizon, cost)
    return pd.DataFrame(labels)
