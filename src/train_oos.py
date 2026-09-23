"""Train deterministic walk-forward probabilities for the corrected pipeline."""
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from labels import make_labels
from merge import load_and_merge
from train import walk_forward

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "oos_proba.parquet"

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "data_random_seed": 42,
    "verbosity": -1,
}


def train_oos(n_splits=6, embargo=12, num_boost_round=400):
    X, ohlcv = load_and_merge()
    y = make_labels(ohlcv, horizon=12, cost=0.0015)
    feature_cols = [
        c for c in X.columns
        if c not in {"open", "high", "low", "close", "volume"}
    ]
    X, y = X[feature_cols], y
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int).map({-1: 0, 0: 1, 1: 2})

    probabilities = pd.DataFrame(
        index=X.index, columns=["p_short", "p_no", "p_long"], dtype=float
    )
    for fold, (X_train, y_train, X_test, _) in enumerate(
        walk_forward(X, y, n_splits=n_splits, embargo=embargo)
    ):
        model = lgb.train(
            PARAMS,
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=num_boost_round,
        )
        probabilities.loc[X_test.index] = model.predict(X_test)
        print(f"fold {fold}: train={len(X_train)} test={len(X_test)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    probabilities.to_parquet(OUTPUT)
    print(f"saved -> {OUTPUT} ({len(probabilities)} rows)")
    return probabilities


if __name__ == "__main__":
    train_oos()
