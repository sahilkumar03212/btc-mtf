import ccxt, pandas as pd, time
from pathlib import Path

EX = ccxt.binance({"enableRateLimit": True})
DATA = Path(__file__).resolve().parents[1] / "data"

def fetch_ohlcv(symbol="BTC/USDT", timeframe="5m", since_days=365):
    tf_ms = EX.parse_timeframe(timeframe) * 1000
    since = EX.milliseconds() - since_days * 24 * 60 * 60 * 1000
    rows = []
    while True:
        batch = EX.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if since > EX.milliseconds():
            break
        time.sleep(EX.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index()

def fetch_all(timeframes=("5m","15m","1h","4h"), since_days=365, symbol="BTC/USDT"):
    for tf in timeframes:
        path = DATA / f"btc_{tf}.parquet"
        if path.exists():
            print(f"skip {tf} (cached)")
            continue
        df = fetch_ohlcv(symbol, tf, since_days)
        df.to_parquet(path)
        print(f"saved {tf}: {len(df)} rows -> {path}")
