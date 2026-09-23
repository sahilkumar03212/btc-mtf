"""
MSc Project: XGBoost Classifier + Regressor on 15m BTC/USDT
============================================================
Architecture:
  - XGBClassifier  -> P(price goes up in next 30 min)
  - XGBRegressor   -> E[return over next 30 min]
  - Decision layer -> trade only if |E[return]| > cost + margin

Walk-forward validation with 6 folds and embargo.
Prints per-fold metrics and saves OOS predictions + final models.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score,
)
from pathlib import Path
from features import add_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

# ── data loading ──────────────────────────────────────────────
def load_msc_data():
    """Load 15m base with 1h + 4h context, properly aligned."""
    print("Loading 15m base data...")
    df_15m = pd.read_parquet(DATA / "btc_15m.parquet")
    base = add_features(df_15m, "15m")
    ohlcv = base[["open", "high", "low", "close", "volume"]].copy()

    for tf in ["1h", "4h"]:
        print(f"  Aligning {tf} context...")
        df_hi = pd.read_parquet(DATA / f"btc_{tf}.parquet")
        hi_feats = add_features(df_hi, tf)
        cols = [c for c in hi_feats.columns if c.startswith(f"{tf}_")]
        # shift(1) on source TF BEFORE ffill — no lookahead
        aligned = hi_feats[cols].shift(1).reindex(base.index, method="ffill")
        base = base.join(aligned, how="left")

    return base, ohlcv

# ── labels ────────────────────────────────────────────────────
def get_labels(ohlcv, horizon=2):
    """
    Horizon = 2 candles on 15m = 30 minutes ahead.
    y_cls: 1 if return > 0, else 0  (binary direction)
    y_reg: raw % return              (magnitude)
    """
    fwd = ohlcv["close"].shift(-horizon) / ohlcv["close"] - 1

    y_cls = pd.Series(np.nan, index=ohlcv.index, name="y_cls")
    valid = fwd.notna()
    y_cls.loc[valid] = (fwd.loc[valid] > 0).astype(int)

    y_reg = fwd.rename("y_reg")
    return y_cls, y_reg

# ── walk-forward splitter ─────────────────────────────────────
def walk_forward(n, n_splits=6, embargo=12):
    """
    Expanding-window walk-forward.
    Embargo = 12 bars (3 hours) to prevent label leakage from the
    8-bar forward horizon plus a safety buffer.
    """
    fold = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        tr_end = fold * i
        te_start = tr_end + embargo
        te_end = min(te_start + fold, n)
        if te_start >= n:
            break
        yield np.arange(0, tr_end), np.arange(te_start, te_end)

# ── main training loop ────────────────────────────────────────
def train_models():
    X, ohlcv = load_msc_data()
    y_cls, y_reg = get_labels(ohlcv, horizon=8)

    feature_cols = [
        c for c in X.columns
        if c not in {"open", "high", "low", "close", "volume"}
    ]
    X_feats = X[feature_cols]

    # drop warm-up NaN rows
    mask = X_feats.notna().all(axis=1) & y_cls.notna()
    X_clean = X_feats.loc[mask]
    y_cls_clean = y_cls.loc[mask]
    y_reg_clean = y_reg.loc[mask]

    print(f"\nDataset: {len(X_clean)} rows × {len(feature_cols)} features")
    print(f"Class balance: up={y_cls_clean.mean():.3f}  down={1 - y_cls_clean.mean():.3f}")
    print(f"Return stats: mean={y_reg_clean.mean()*100:.4f}%  std={y_reg_clean.std()*100:.4f}%\n")

    # output containers
    preds = pd.DataFrame(
        index=X_clean.index,
        columns=["p_up", "exp_return"],
        dtype=float,
    )

    fold_metrics = []
    n_splits = 6

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward(len(X_clean), n_splits=n_splits, embargo=4)
    ):
        X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        yc_tr, yc_te = y_cls_clean.iloc[train_idx], y_cls_clean.iloc[test_idx]
        yr_tr, yr_te = y_reg_clean.iloc[train_idx], y_reg_clean.iloc[test_idx]

        # ── use last 15% of training as early-stopping validation ──
        val_size = int(len(X_tr) * 0.15)
        X_tr_fit, X_tr_val = X_tr.iloc[:-val_size], X_tr.iloc[-val_size:]
        yc_tr_fit, yc_tr_val = yc_tr.iloc[:-val_size], yc_tr.iloc[-val_size:]
        yr_tr_fit, yr_tr_val = yr_tr.iloc[:-val_size], yr_tr.iloc[-val_size:]

        print(f"═══ Fold {fold+1}/{n_splits} ═══")
        print(f"  train={len(X_tr_fit)}  val={len(X_tr_val)}  test={len(X_te)}")

        # ── Classifier ──
        cls_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            early_stopping_rounds=30,
        )
        cls_model.fit(
            X_tr_fit, yc_tr_fit,
            eval_set=[(X_tr_val, yc_tr_val)],
            verbose=False,
        )
        p_up = cls_model.predict_proba(X_te)[:, 1]
        preds.loc[X_te.index, "p_up"] = p_up

        cls_pred = (p_up > 0.5).astype(int)
        acc = accuracy_score(yc_te, cls_pred)
        print(f"  Classifier: acc={acc:.4f}  best_iter={cls_model.best_iteration}")

        # ── Regressor ──
        reg_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="mae",
            early_stopping_rounds=30,
        )
        reg_model.fit(
            X_tr_fit, yr_tr_fit,
            eval_set=[(X_tr_val, yr_tr_val)],
            verbose=False,
        )
        exp_ret = reg_model.predict(X_te)
        preds.loc[X_te.index, "exp_return"] = exp_ret

        mae = mean_absolute_error(yr_te, exp_ret)
        r2 = r2_score(yr_te, exp_ret)
        print(f"  Regressor:  MAE={mae*100:.4f}%  R²={r2:.4f}  best_iter={reg_model.best_iteration}")

        fold_metrics.append({
            "fold": fold + 1,
            "train": len(X_tr_fit),
            "test": len(X_te),
            "cls_acc": round(acc, 4),
            "cls_best_iter": cls_model.best_iteration,
            "reg_mae_pct": round(mae * 100, 4),
            "reg_r2": round(r2, 4),
            "reg_best_iter": reg_model.best_iteration,
        })

    # ── save outputs ──────────────────────────────────────────
    MODELS.mkdir(parents=True, exist_ok=True)

    out_path = MODELS / "oos_preds_xgb.parquet"
    preds.dropna().to_parquet(out_path)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_path = MODELS / "xgb_fold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # save final-fold models for later use (decision layer / paper trading)
    cls_model.save_model(str(MODELS / "xgb_classifier.json"))
    reg_model.save_model(str(MODELS / "xgb_regressor.json"))

    # ── print summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\nMean classifier accuracy : {metrics_df['cls_acc'].mean():.4f}")
    print(f"Mean regressor MAE       : {metrics_df['reg_mae_pct'].mean():.4f}%")
    print(f"Mean regressor R²        : {metrics_df['reg_r2'].mean():.4f}")

    # ── feature importance (from last fold) ────────────────────
    imp = pd.Series(
        cls_model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    print(f"\nTop 15 classifier features (last fold):")
    for feat, val in imp.head(15).items():
        print(f"  {feat:25s}  {val:.4f}")

    imp.to_csv(MODELS / "xgb_feature_importance.csv")

    print(f"\nSaved:")
    print(f"  OOS predictions  -> {out_path}")
    print(f"  Fold metrics     -> {metrics_path}")
    print(f"  Classifier model -> {MODELS / 'xgb_classifier.json'}")
    print(f"  Regressor model  -> {MODELS / 'xgb_regressor.json'}")
    print(f"  Feature imp.     -> {MODELS / 'xgb_feature_importance.csv'}")

    return preds, metrics_df, imp


if __name__ == "__main__":
    train_models()
