"""
Live data fetcher — pulls real-time OHLCV from Binance via CCXT.
Handles both testnet and live modes.
"""
import ccxt
import pandas as pd
import time
from config import (
    MODE, SYMBOL,
    TESTNET_API_KEY, TESTNET_API_SECRET,
    LIVE_API_KEY, LIVE_API_SECRET,
    LOOKBACK_BARS, BASE_TIMEFRAME, CONTEXT_TIMEFRAMES,
)
from features import add_features


def get_exchange():
    """Create a CCXT Binance exchange instance."""
    if MODE == "testnet":
        exchange = ccxt.binance({
            "apiKey": TESTNET_API_KEY,
            "secret": TESTNET_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        exchange.set_sandbox_mode(True)
    else:
        exchange = ccxt.binance({
            "apiKey": LIVE_API_KEY,
            "secret": LIVE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    return exchange


def fetch_candles(exchange, timeframe, n_bars=100):
    """Fetch the last n_bars candles for a given timeframe."""
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=n_bars)
            df = pd.DataFrame(
                ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts").sort_index()
            return df
        except Exception as e:
            print(f"  [WARN] fetch_candles({timeframe}) attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {timeframe} candles after 3 attempts")


def build_live_dataframe(exchange):
    """
    Fetch 15m + 1h + 4h candles, compute features, and align.
    Returns (feature_row, ohlcv_row) for the LATEST completed bar.
    """
    # Fetch all timeframes
    candles = {}
    for tf in [BASE_TIMEFRAME] + CONTEXT_TIMEFRAMES:
        n = LOOKBACK_BARS.get(tf, 100)
        candles[tf] = fetch_candles(exchange, tf, n)
        print(f"  Fetched {tf}: {len(candles[tf])} bars, latest={candles[tf].index[-1]}")

    # Compute features on each timeframe
    feats = {}
    for tf, df in candles.items():
        feats[tf] = add_features(df, prefix=tf)

    # Use 15m as base
    base = feats[BASE_TIMEFRAME].copy()
    ohlcv = base[["open", "high", "low", "close", "volume"]].copy()

    # Align higher TFs — same logic as training (shift before ffill)
    for tf in CONTEXT_TIMEFRAMES:
        hi = feats[tf]
        cols = [c for c in hi.columns if c.startswith(f"{tf}_")]
        aligned = hi[cols].shift(1).reindex(base.index, method="ffill")
        base = base.join(aligned, how="left")

    # Drop OHLCV columns, keep only feature columns
    feature_cols = [
        c for c in base.columns
        if c not in {"open", "high", "low", "close", "volume"}
    ]

    # Return the LAST COMPLETE bar (second to last, since last may be forming)
    # Use iloc[-2] for the last fully closed candle
    if len(base) < 2:
        raise RuntimeError("Not enough data to extract a complete bar")

    features_row = base[feature_cols].iloc[-2]
    ohlcv_row = ohlcv.iloc[-2]
    timestamp = base.index[-2]

    return features_row, ohlcv_row, timestamp, feature_cols
