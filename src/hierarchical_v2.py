
"""
Option B v2: Hierarchical model.
L1 = 4h regime       (3-class)  {down, range, up}
L2 = 1h direction    (3-class)  {short, no, long}, uses L1 + 5m aggregates
L3 = 5m trigger      (binary)   conditional on L2 direction

Fixes vs v1:
  - L3 target is conditional on L2's direction, not generic |move|
  - Regime thresholds widened (1.5% over 24h)
  - L2 gets 5m-aggregate features
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from features import add_features
from train import walk_forward

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

PARAMS_3C = dict(objective="multiclass", num_class=3,
                 learning_rate=0.03, num_leaves=31,
                 min_data_in_leaf=200, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, verbose=-1)

PARAMS_BIN = dict(objective="binary",
                  learning_rate=0.03, num_leaves=31,
                  min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1)

# ---------------- L1: 4h regime ----------------
def train_level_1(n_splits=6, embargo=6):
    df = pd.read_parquet(DATA / "btc_4h.parquet")
    df = add_features(df, prefix="4h")
    fwd = df["close"].shift(-6) / df["close"] - 1
    # wider thresholds -> cleaner regime classes
    y = pd.Series(np.nan, index=df.index, name="y_regime", dtype=float)
    valid = fwd.notna()
    y.loc[valid] = np.select(
        [fwd.loc[valid] > 0.015, fwd.loc[valid] < -0.015],
        [2, 0], default=1,
    )

    cols = [c for c in df.columns if c.startswith("4h_")]
    X = df[cols]
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    proba = pd.DataFrame(index=X.index, columns=["p_down","p_range","p_up"], dtype=float)
    for Xtr, ytr, Xte, yte in walk_forward(X, y, n_splits=n_splits, embargo=embargo):
        m = lgb.train(PARAMS_3C, lgb.Dataset(Xtr, ytr), num_boost_round=300)
        proba.loc[Xte.index] = m.predict(Xte)
    proba.to_parquet(MODELS / "oos_4h_regime_v2.parquet")
    print(f"L1 done: {proba.shape}  class balance: {pd.Series(y).value_counts(normalize=True).round(3).to_dict()}")
    return proba

# ---------------- helpers ----------------
def broadcast_to(src_pred, target_index, src_cols):
    """Reindex higher-TF predictions to lower-TF index, one-bar-shifted."""
    shifted = src_pred[src_cols].shift(1)
    return shifted.reindex(target_index, method="ffill")

def aggregate_5m_to_1h():
    """Summarize the previous 1h of 5m activity for L2 features."""
    df = pd.read_parquet(DATA / "btc_5m.parquet")
    df = add_features(df, prefix="5m")
    agg = df.resample("1h").agg({
        "5m_ret_1":  ["mean", "std"],
        "5m_atr_14": "mean",
        "5m_rsi_14": "mean",
        "5m_vol_z":  "mean",
        "5m_body":   "mean",
    })
    agg.columns = [f"agg_{a}_{b}" for a, b in agg.columns]
    return agg.shift(1)

# ---------------- L2: 1h direction + 5m context ----------------
def train_level_2(regime_pred, n_splits=6, embargo=6):
    df = pd.read_parquet(DATA / "btc_1h.parquet")
    df = add_features(df, prefix="1h")
    fwd = df["close"].shift(-6) / df["close"] - 1
    y = pd.Series(np.nan, index=df.index, name="y_dir", dtype=float)
    valid = fwd.notna()
    y.loc[valid] = np.select(
        [fwd.loc[valid] > 0.003, fwd.loc[valid] < -0.003],
        [2, 0], default=1,
    )

    cols = [c for c in df.columns if c.startswith("1h_")]
    X = df[cols].copy()

    # regime preds broadcast to 1h
    X = X.join(broadcast_to(regime_pred, X.index, list(regime_pred.columns)))
    # 5m aggregates
    X = X.join(aggregate_5m_to_1h(), how="left")

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    proba = pd.DataFrame(index=X.index, columns=["p_short","p_no","p_long"], dtype=float)
    for Xtr, ytr, Xte, yte in walk_forward(X, y, n_splits=n_splits, embargo=embargo):
        m = lgb.train(PARAMS_3C, lgb.Dataset(Xtr, ytr), num_boost_round=300)
        proba.loc[Xte.index] = m.predict(Xte)
    proba.to_parquet(MODELS / "oos_1h_direction_v2.parquet")
    print(f"L2 done: {proba.shape}")
    return proba

# ---------------- L3: 5m trigger conditional on L2 ----------------
def train_level_3(regime_pred, dir_pred, n_splits=6, embargo=12):
    df = pd.read_parquet(DATA / "btc_5m.parquet")
    df = add_features(df, prefix="5m")
    fwd = df["close"].shift(-12) / df["close"] - 1

    dir_b = broadcast_to(dir_pred, df.index, list(dir_pred.columns))
    want_long  = dir_b["p_long"]  > dir_b["p_short"]
    want_short = dir_b["p_short"] > dir_b["p_long"]

    # CONDITIONAL target
    y = (
        (want_long  & (fwd >  0.0015)) |
        (want_short & (fwd < -0.0015))
    ).where(fwd.notna())
    y = pd.Series(y, index=df.index, name="y_trig_cond")

    cols = [c for c in df.columns if c.startswith("5m_")]
    X = df[cols].copy()
    X = X.join(broadcast_to(regime_pred, X.index, list(regime_pred.columns)))
    X = X.join(dir_b)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    print(f"L3 target balance: y=1 is {y.mean():.3f} of rows")

    proba = pd.Series(index=X.index, dtype=float, name="p_enter")
    for Xtr, ytr, Xte, yte in walk_forward(X, y, n_splits=n_splits, embargo=embargo):
        m = lgb.train(PARAMS_BIN, lgb.Dataset(Xtr, ytr), num_boost_round=300)
        proba.loc[Xte.index] = m.predict(Xte)
    proba.to_frame().to_parquet(MODELS / "oos_5m_trigger_v2.parquet")
    print(f"L3 done: {proba.shape}")
    return proba

def run_all():
    r = train_level_1()
    d = train_level_2(r)
    t = train_level_3(r, d)
    print("\nv2 training complete.")
    return r, d, t

if __name__ == "__main__":
    run_all()
