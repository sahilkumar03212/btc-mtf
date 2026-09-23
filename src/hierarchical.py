"""
Option B: Hierarchical model.
L1 = 4h regime classifier    -> p_regime (3-class)
L2 = 1h direction classifier -> p_dir (3-class), uses L1 preds
L3 = 5m trigger classifier   -> p_enter (binary), uses L1 + L2 preds

Saves OOS predictions for each level to models/.
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

# ---------------- level 1: 4h regime ----------------
def build_4h():
    df = pd.read_parquet(DATA / "btc_4h.parquet")
    df = add_features(df, prefix="4h")
    fwd = df["close"].shift(-6) / df["close"] - 1          # 24h ahead
    y = pd.Series(np.nan, index=df.index, name="y_regime", dtype=float)
    valid = fwd.notna()
    y.loc[valid] = np.select(
        [fwd.loc[valid] > 0.01, fwd.loc[valid] < -0.01],
        [2, 0], default=1,
    )
    return df, y

def train_level_1(n_splits=6, embargo=6):
    df, y = build_4h()
    cols = [c for c in df.columns if c.startswith("4h_")]
    X = df[cols]
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    proba = pd.DataFrame(index=X.index, columns=["p_down","p_range","p_up"], dtype=float)
    for Xtr, ytr, Xte, yte in walk_forward(X, y, n_splits=n_splits, embargo=embargo):
        m = lgb.train(PARAMS_3C, lgb.Dataset(Xtr, ytr), num_boost_round=300)
        p = m.predict(Xte)
        proba.loc[Xte.index] = p
    proba.to_parquet(MODELS / "oos_4h_regime.parquet")
    print(f"L1 done: {proba.shape}  ->  oos_4h_regime.parquet")
    return proba

# ---------------- helpers ----------------
def broadcast_to(src_pred: pd.DataFrame, target_index, src_cols):
    """Reindex a higher-TF prediction frame to a lower-TF index.
    Shift by 1 src bar first so only closed src bars are used."""
    shifted = src_pred[src_cols].shift(1)
    return shifted.reindex(target_index, method="ffill")

# ---------------- level 2: 1h direction ----------------
def build_1h():
    df = pd.read_parquet(DATA / "btc_1h.parquet")
    df = add_features(df, prefix="1h")
    fwd = df["close"].shift(-6) / df["close"] - 1          # 6h ahead
    y = pd.Series(np.nan, index=df.index, name="y_dir", dtype=float)
    valid = fwd.notna()
    y.loc[valid] = np.select(
        [fwd.loc[valid] > 0.003, fwd.loc[valid] < -0.003],
        [2, 0], default=1,
    )
    return df, y

def train_level_2(regime_pred, n_splits=6, embargo=6):
    df, y = build_1h()
    own_cols = [c for c in df.columns if c.startswith("1h_")]
    X = df[own_cols].copy()

    regime_cols = list(regime_pred.columns)
    regime_for_1h = broadcast_to(regime_pred, X.index, regime_cols)
    X = X.join(regime_for_1h)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    proba = pd.DataFrame(index=X.index, columns=["p_short","p_no","p_long"], dtype=float)
    for Xtr, ytr, Xte, yte in walk_forward(X, y, n_splits=n_splits, embargo=embargo):
        m = lgb.train(PARAMS_3C, lgb.Dataset(Xtr, ytr), num_boost_round=300)
        p = m.predict(Xte)
        proba.loc[Xte.index] = p
    proba.to_parquet(MODELS / "oos_1h_direction.parquet")
    print(f"L2 done: {proba.shape}  ->  oos_1h_direction.parquet")
    return proba

# ---------------- level 3: 5m trigger ----------------
def build_5m():
    df = pd.read_parquet(DATA / "btc_5m.parquet")
    df = add_features(df, prefix="5m")
    fwd = df["close"].shift(-12) / df["close"] - 1
    y = pd.Series(np.nan, index=df.index, name="y_trig", dtype=float)
    y.loc[fwd.notna()] = (fwd.loc[fwd.notna()].abs() > 0.0015).astype(int)
    return df, y

def train_level_3(regime_pred, dir_pred, n_splits=6, embargo=12):
    df, y = build_5m()
    own_cols = [c for c in df.columns if c.startswith("5m_")]
    X = df[own_cols].copy()

    regime_for_5m = broadcast_to(regime_pred, X.index, list(regime_pred.columns))
    X = X.join(regime_for_5m)

    dir_for_5m = broadcast_to(dir_pred, X.index, list(dir_pred.columns))
    X = X.join(dir_for_5m)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    proba = pd.Series(index=X.index, dtype=float, name="p_enter")
    for Xtr, ytr, Xte, yte in walk_forward(X, y, n_splits=n_splits, embargo=embargo):
        m = lgb.train(PARAMS_BIN, lgb.Dataset(Xtr, ytr), num_boost_round=300)
        proba.loc[Xte.index] = m.predict(Xte)
    proba.to_frame().to_parquet(MODELS / "oos_5m_trigger.parquet")
    print(f"L3 done: {proba.shape}  ->  oos_5m_trigger.parquet")
    return proba

# ---------------- orchestrate ----------------
def run_all():
    regime = train_level_1()
    direction = train_level_2(regime)
    trigger = train_level_3(regime, direction)
    print("\nAll levels complete.")
    return regime, direction, trigger

if __name__ == "__main__":
    run_all()
