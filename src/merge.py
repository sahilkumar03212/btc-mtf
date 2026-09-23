import pandas as pd
from pathlib import Path
from features import add_features

DATA = Path(__file__).resolve().parents[1] / "data"
TFS = ["5m", "1h", "4h"]
BASE = "5m"

def load_and_merge():
    feats = {}
    for tf in TFS:
        df = pd.read_parquet(DATA / f"btc_{tf}.parquet")
        feats[tf] = add_features(df, prefix=tf)

    base = feats[BASE].copy()
    base_ohlcv = base[["open","high","low","close","volume"]].copy()

    for tf in TFS:
        if tf == BASE:
            continue
        hi = feats[tf]
        cols = [c for c in hi.columns if c.startswith(f"{tf}_")]
        # Shift on the source timeframe before expanding it. Shifting after
        # forward-fill only delays the first row of a higher-TF candle; all
        # remaining lower-TF rows would still see the incomplete candle.
        aligned = hi[cols].shift(1).reindex(base.index, method="ffill")
        base = base.join(aligned, how="left")

    return base, base_ohlcv
