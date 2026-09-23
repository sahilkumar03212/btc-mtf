import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from merge import load_and_merge
from labels import make_labels

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

PARAMS = dict(
    objective="multiclass", num_class=3,
    learning_rate=0.01,
    num_leaves=63,
    min_data_in_leaf=500,
    feature_fraction=0.6,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l1=0.1,
    lambda_l2=1.0,
    max_depth=7,
    min_gain_to_split=0.01,
    seed=42,
    feature_fraction_seed=42,
    bagging_seed=42,
    data_random_seed=42,
    verbosity=-1,
)

def walk_forward_with_val(X, y, n_splits=6, horizon=12, buffer=12):
    n = len(X)
    fold = n // (n_splits + 1)
    embargo = horizon + buffer
    for i in range(1, n_splits + 1):
        tr_end = fold * i
        te_start = tr_end + embargo
        te_end = min(te_start + fold, n)
        if te_start >= n:
            break
        
        # Use last 20% of the training window as validation
        val_size = int(fold * i * 0.2)
        real_tr_end = tr_end - val_size - embargo
        if real_tr_end <= 0:
            continue
            
        yield (
            X.iloc[:real_tr_end], y.iloc[:real_tr_end],
            X.iloc[tr_end - val_size:tr_end], y.iloc[tr_end - val_size:tr_end],
            X.iloc[te_start:te_end], y.iloc[te_start:te_end],
        )

def train_v2(horizon=48, cost=0.005):
    X, ohlcv = load_and_merge()
    y = make_labels(ohlcv, horizon=horizon, cost=cost)
    
    feature_cols = [c for c in X.columns if c not in {"open", "high", "low", "close", "volume"}]
    X = X[feature_cols]
    
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int).map({-1: 0, 0: 1, 1: 2})
    
    probabilities = pd.DataFrame(index=X.index, columns=["p_short", "p_no", "p_long"], dtype=float)
    
    for fold, (X_tr, y_tr, X_val, y_val, X_te, y_te) in enumerate(walk_forward_with_val(X, y, horizon=horizon)):
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        probabilities.loc[X_te.index] = model.predict(X_te)
        print(f"fold {fold}: train={len(X_tr)} val={len(X_val)} test={len(X_te)} best_iter={model.best_iteration}")
        
    out_path = MODELS / f"oos_proba_h{horizon}_c{cost}.parquet"
    MODELS.mkdir(parents=True, exist_ok=True)
    probabilities.to_parquet(out_path)
    print(f"saved -> {out_path} ({len(probabilities)} rows)")
    return probabilities, out_path

if __name__ == "__main__":
    train_v2(horizon=48, cost=0.005)
