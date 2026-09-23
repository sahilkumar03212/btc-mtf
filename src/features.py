import pandas as pd, numpy as np

def get_rolling_slope(series, window):
    x = np.arange(1, window + 1)
    sum_x = x.sum()
    sum_x2 = (x**2).sum()
    denominator = window * sum_x2 - sum_x**2
    sum_xy = series.rolling(window).apply(lambda y: np.dot(y, x), raw=True)
    sum_y = series.rolling(window).sum()
    return (window * sum_xy - sum_x * sum_y) / denominator

def add_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("OHLCV index must be a DatetimeIndex")

    d = df.copy()
    c = d["close"]

    for lag in [1, 3, 6, 12, 24, 48]:
        d[f"{prefix}_ret_{lag}"] = c.pct_change(lag)

    d[f"{prefix}_vol_12"] = d[f"{prefix}_ret_1"].rolling(12).std()
    prev_close = c.shift(1)
    true_range = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d[f"{prefix}_atr_14"] = true_range.rolling(14).mean() / c

    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    d[f"{prefix}_rsi_14"] = 100 - 100 / (1 + up / (dn + 1e-9))

    ema_fast = c.ewm(span=12, adjust=False).mean()
    ema_slow = c.ewm(span=48, adjust=False).mean()
    d[f"{prefix}_ema_ratio"] = ema_fast / ema_slow - 1

    v = d["volume"]
    d[f"{prefix}_vol_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-9)

    hl = d["high"] - d["low"] + 1e-9
    d[f"{prefix}_body"] = (d["close"] - d["open"]) / hl
    d[f"{prefix}_upper_wick"] = (d["high"] - d[["open","close"]].max(axis=1)) / hl
    d[f"{prefix}_lower_wick"] = (d[["open","close"]].min(axis=1) - d["low"]) / hl

    # NEW FEATURES
    # garman_klass_vol
    log_hl = np.log(d["high"] / d["low"])
    log_co = np.log(d["close"] / d["open"])
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    d[f"{prefix}_garman_klass_vol"] = gk.rolling(24).mean().apply(np.sqrt)

    # range features
    high_24 = d["high"].rolling(24).max()
    low_24 = d["low"].rolling(24).min()
    d[f"{prefix}_range_pct"] = (high_24 - low_24) / c
    d[f"{prefix}_close_vs_range"] = (c - low_24) / (high_24 - low_24 + 1e-9)
    
    # volume_trend
    vol_sma_12 = d["volume"].rolling(12).mean()
    vol_sma_48 = d["volume"].rolling(48).mean()
    d[f"{prefix}_volume_trend"] = vol_sma_12 / (vol_sma_48 + 1e-9) - 1

    # obv slope
    obv = (np.sign(c.diff()) * d['volume']).fillna(0).cumsum()
    d[f"{prefix}_obv_slope"] = get_rolling_slope(obv, 24)

    # rsi slope
    d[f"{prefix}_rsi_slope"] = get_rolling_slope(d[f"{prefix}_rsi_14"], 12)

    # Cyclic encodings avoid treating 23:00 and 00:00 as far apart.
    hour = d.index.hour + d.index.minute / 60
    dow = d.index.dayofweek
    d[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    d[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return d
