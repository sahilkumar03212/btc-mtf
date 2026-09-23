import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import classification_report
from merge import load_and_merge
from labels import make_labels

def walk_forward(X, y, n_splits=6, embargo=12):
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have identical, sorted indexes")
    if not X.index.is_monotonic_increasing:
        raise ValueError("X and y must be sorted chronologically")
    if n_splits < 1 or embargo < 0:
        raise ValueError("n_splits must be positive and embargo non-negative")
    n = len(X)
    fold = n // (n_splits + 1)
    if fold < 1:
        raise ValueError("not enough rows for requested walk-forward splits")
    for i in range(1, n_splits + 1):
        tr_end = fold * i
        te_start = tr_end + embargo
        te_end = min(te_start + fold, n)
        if te_start >= n:
            break
        yield (
            X.iloc[:tr_end], y.iloc[:tr_end],
            X.iloc[te_start:te_end], y.iloc[te_start:te_end],
        )

def main():
    X, ohlcv = load_and_merge()
    y = make_labels(ohlcv, horizon=12, cost=0.0015)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    X = X.drop(columns=[c for c in X.columns if c in ("open","high","low","close","volume")])

    params = dict(
        objective="multiclass", num_class=3,
        learning_rate=0.03, num_leaves=31,
        min_data_in_leaf=200, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1,
        verbose=-1,
    )

    for i, (Xtr, ytr, Xte, yte) in enumerate(walk_forward(X, y)):
        ytr_m = ytr.map({-1:0, 0:1, 1:2})
        yte_m = yte.map({-1:0, 0:1, 1:2})
        m = lgb.train(params, lgb.Dataset(Xtr, ytr_m), num_boost_round=400)
        pred = m.predict(Xte).argmax(axis=1)
        print(f"--- fold {i} ---")
        print(classification_report(yte_m, pred, digits=3))

if __name__ == "__main__":
    main()
