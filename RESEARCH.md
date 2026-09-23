# BTC Multi-Timeframe ML Trading — Research Log

**Project**: Multi-timeframe (5m + 15m + 1h + 4h) LightGBM classifier for BTC/USDT
**Data**: Binance monthly klines via data.binance.vision
**Period**: 2024-01-01 -> 2026-08-31 (~32 months, ~278k 5m bars)
**Last updated**: 2026-09-20

---

## 1. Project Goal

Build and honestly evaluate an ML model that predicts short-term BTC direction
using multi-timeframe features, and backtest it with realistic costs.

Not a goal: getting rich. Crypto ML research typically fails. The purpose
is to learn the pipeline and produce an honest Sharpe, not a fantasy one.

---

## 2. Data

| Timeframe | Rows    | Range                       |
|-----------|---------|-----------------------------|
| 5m        | 280,512 | 2024-01-01 -> 2026-08-31    |
| 15m       |  93,504 | same                        |
| 1h        |  23,376 | same                        |
| 4h        |   5,844 | same                        |

Source : https://data.binance.vision/data/spot/monthly/klines
Reason : Binance live API returns HTTP 451 from Colab (US IP block).
Note   : Timestamps auto-detected between ms (13 digits) and us (16 digits)
         because Binance changed the format in Jan 2025.

Storage: /content/drive/MyDrive/btc-mtf/data/btc_{tf}.parquet

---

## 3. Features (per timeframe)

Computed identically on each TF with a prefix (5m_, 15m_, 1h_, 4h_):

- Returns   : ret_1, ret_3, ret_6, ret_12
- Volatility: vol_12 (rolling std), atr_14 (normalized)
- RSI       : rsi_14
- Trend     : ema_ratio = EMA(12) / EMA(48) - 1
- Volume    : vol_z = z-score over 48 bars
- Candle    : body, upper_wick, lower_wick
- Time      : hour, dow

Total: 56 features (14 per TF * 4 TFs).

---

## 4. MTF Alignment (critical)

Higher-timeframe features aligned to 5m base index with:

    aligned = hi[cols].reindex(base.index, method="ffill").shift(1)

Why: ensures the 5m bar at time t only sees the last fully closed higher-TF
candle. Prevents lookahead.

Verified: a "double-shift" leak test on 4h_ret_1 (highest-importance feature)
showed no accuracy drop (0.615 vs 0.615). Alignment is clean.

---

## 5. Labels

    fwd = close.shift(-12) / close - 1     # 12 bars = 60 min
    y = +1 if fwd >  0.0015
        -1 if fwd < -0.0015
         0 otherwise

Design: thresholded labels (not raw up/down). Threshold = 0.15%, roughly the
round-trip cost. Forces the model to predict meaningful moves, not noise.

Class balance (approx): -1 ~ 25%, 0 ~ 50%, +1 ~ 25%.

---

## 6. Model

Algorithm: LightGBM multiclass (3-class softmax)

    PARAMS = dict(
        objective="multiclass", num_class=3,
        learning_rate=0.03, num_leaves=31,
        min_data_in_leaf=200, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1,
        verbose=-1,
    )
    num_boost_round = 400

Validation: 6-fold walk-forward with 12-bar embargo.

- Each fold: train on past, predict on future.
- Fold size: ~40k bars (~4.6 months).
- All predictions concatenated = OOS signal for the whole test period.

Why: random k-fold leaks future info in time series. Walk-forward with embargo
is the honest baseline.

---

## 7. Feature Importance (single-model fit)

| Feature      | % gain |
|--------------|--------|
| 4h_ret_1     | 19.64  |
| 1h_ret_1     |  9.22  |
| 5m_atr_14    |  7.55  |
| 4h_body      |  5.63  |
| 5m_ret_12    |  5.36  |
| 5m_ema_ratio |  4.67  |
| 15m_ret_12   |  4.53  |
| 5m_ret_6     |  4.41  |
| 4h_vol_z     |  3.38  |

Gain share by timeframe:

| TF  | %    |
|-----|------|
| 4h  | 36.5 |
| 5m  | 27.1 |
| 1h  | 23.6 |
| 15m | 12.8 |

Interpretation: the model is fundamentally a 4h + 1h regime follower, timed
by 5m features. Exactly what MTF stacking should produce.

---

## 8. Ablation — which timeframes matter?

| Feature set                 | Mean test accuracy |
|-----------------------------|--------------------|
| all (5m + 15m + 1h + 4h)    | 0.6147             |
| no 15m                      | 0.6126 (-0.002)    |
| no 5m                       | 0.6030 (-0.012)    |
| 4h only                     | 0.5266 (-0.088)    |

Conclusions:
- MTF stacking does real work (+8.8 pts over 4h alone).
- 15m features are redundant — drop them.
- 5m contributes modestly; 4h/1h carry the model.

---

## 9. Classification Performance (walk-forward)

| Fold | Accuracy |
|------|----------|
| 0    | 0.598    |
| 1    | 0.600    |
| 2    | 0.618    |
| 3    | 0.624    |
| 4    | 0.627    |
| 5    | 0.634    |

Trend: accuracy improves over time. Unusual and encouraging — model is stable
across regimes.

Per-class (fold 5):

| Class    | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| Short    | 0.643     | 0.566  | 0.602 |
| No-trade | 0.623     | 0.713  | 0.665 |
| Long     | 0.646     | 0.576  | 0.609 |

Interpretation: model is best at avoiding trades (no-trade recall 0.71).
Real edge on entries (precision ~0.64 for long/short).

---

## 10. Backtest iterations and bug fixes

### v1 (broken)
- Issue: every signal opened a separate trade, even if an earlier one was open.
- Result: Sharpe 6.46, equity $1.15M — impossible.
- Cause: overlapping positions compounded 12-bar returns many times over.

### v2 (fixed, over-leveraged)
- Fix: one position at a time.
- Result: Sharpe 5.18, equity 20.78, hit rate 0.745, DD -9.5%.
- Still wrong: size = RISK_FRAC / ATR clipped to 1.0 -> 100% equity per trade.

### v3 (final)
- Fix: deterministic size = RISK_FRAC / STOP_DIST_PCT = 0.5 (50% equity/trade).
- Result: equity ~4.5, DD ~5%, hit rate ~0.74.

Cost model:
    FEE  = 0.001     # 0.1% per side (Binance spot taker)
    SLIP = 0.0005    # 0.05% per side
    COST = 0.0015    # round trip 0.3%

Buy and Hold baseline over same period: equity 1.68.
Strategy: equity ~4.5, DD ~5%.
Strategy beats HODL on both return and risk.

---

## 11. Random-signal control (leak test)

Purpose: if the backtest has a mechanical bug, random signals would also
appear profitable. If random signals lose money, the pipeline is honest.

Result (matched trade count, same gate):

| Metric        | Value  |
|---------------|--------|
| final_equity  | 0.001  |
| sharpe_bar    | -17.08 |
| hit_rate      | 0.503  (chance) |
| profit_factor | 1.04   |
| trades        | 9,675  |

Interpretation: random noise + realistic costs = total ruin. Proves the
backtest mechanics are honest and the model's edge is real.

---

## 12. What is proven vs. unproven

PROVEN
- Data pipeline is clean (no lookahead in MTF alignment).
- MTF stacking adds real value (+8.8 pts over 4h alone).
- Classification edge is real (60%+ accuracy, 74% hit rate on trades).
- Backtest is honest (random signals fail).
- Strategy beats buy-and-hold.

UNPROVEN / CONCERNING
- 74% hit rate is unusually high for BTC 5m. Must compare to trivial
  baselines: always long in uptrend, 5m momentum, RSI > 50.
- All trades are LONG. Cannot short on spot. Requires futures for full test.
- Cost sensitivity: at 2x cost (0.3%/side), does the edge survive?
- Regime dependence: 2024-2026 was mostly bull. What about a bear market?
- Live execution: no slippage from order book depth, no partial fills, no latency.

---

## 13. Open questions / next steps

1. Baseline comparison (in progress):
   - Always long in uptrend
   - 5m momentum
   - RSI > 50
   - 1h momentum
   If model Sharpe >> all baselines, model is doing real work beyond
   trend-following.

2. Cost sensitivity: rerun with COST = 0.003 (double). If Sharpe drops to
   < 0.5, edge is fragile.

3. Yearly breakdown: split equity curve by year. If one year carries
   everything, it's regime-dependent.

4. Paper trading: deploy signal 2-4 weeks with small size. Compare live
   fills to backtested fills.

5. Label sweep: try horizon = 6, 24, 48 and cost = 0.003. Pick the
   combination with best OOS Sharpe.

6. Shorting: switch to Binance USDT-M futures to test long+short version
   with funding rates included.

---

## 14. Reproducibility

Files:
- src/fetch.py            download OHLCV from data.binance.vision
- src/features.py         compute per-TF features
- src/merge.py            align MTF features safely
- src/labels.py           create thresholded labels
- src/train.py            walk-forward training loop
- src/train_proba.py      train + save OOS probabilities
- src/backtest_final.py   honest backtest with proper sizing
- models/oos_proba.parquet  cached OOS probabilities (fast iteration)

Steps:
1. Mount Drive, install ccxt lightgbm pyarrow.
2. Run fetch.py to download (once).
3. Run train_proba.py (once, ~6 min).
4. Run backtest_final.py (5 sec per run, cached probabilities).

Speed note: Colab free tier throttles CPU. Fold training takes 20-90s each.
Cache OOS probabilities to avoid retraining.

---

## 15. Honest summary

We have a legitimate ML pipeline with real out-of-sample classification edge
(~62% accuracy, ~74% trade hit rate) on 32 months of BTC data. The backtest
is honest (random signals fail). The strategy roughly triples buy-and-hold
with 5% drawdown.

However, we have NOT yet proven:
- The edge exceeds trivial baselines (in progress).
- The strategy survives higher costs.
- The edge is not concentrated in the 2024-2026 bull regime.
- Live execution matches backtest.

Status: promising but not yet tradeable. Complete baseline comparison and
cost-sensitivity tests before drawing conclusions.

Rule for going live: do not deploy capital until (a) baselines are beaten
cleanly, (b) double-cost Sharpe > 1, and (c) at least 3 months of paper
trading match the backtest.

---

## 16. Corrected evaluation status

The original results must not be treated as valid after an audit found two
backtest defects:

1. Higher-timeframe features were shifted after forward-filling. This exposed
   the currently forming higher-timeframe candle on most lower-timeframe bars.
   Alignment now shifts the source timeframe before forward-filling.
2. Entry fees were not charged because turnover was multiplied by the previous
   bar's zero position size. Costs now use notional turnover on both entry and
   exit.

The cached `models/oos_proba.parquet` was produced before these corrections and
must be replaced. Run `python src/train_oos.py`, then
`python src/evaluate.py`. The evaluator compares the model with simple
long-only baselines at multiple cost levels and writes `evaluation.csv`.
Until the corrected OOS run shows a positive edge across reasonable costs and
regimes, the strategy is not validated or suitable for live trading.

The corrected retrain completed on 2026-09-20, but it currently loses money
even at the lowest tested cost. A 0.70 confidence threshold reduces turnover
and drawdown but remains unprofitable. This result invalidates the earlier
claims of a profitable edge; further feature/model work is required before
paper trading.

For repeatable diagnostics run `python src/research_report.py`. It writes
year-by-year threshold/cost results to `research_report.csv` and the latest
model output to `paper_signal.csv`. The latter is explicitly marked
`RESEARCH_ONLY` and is not an execution order.

## 17. Change log

- 2026-09-20: Initial pipeline built. Classification accuracy ~61-63%.
- 2026-09-20: Fixed backtest overlap bug (Sharpe 6.46 -> 5.18 -> 4.5).
- 2026-09-20: Random-signal control test passed. Pipeline is honest.
- 2026-09-20: MTF ablation complete. 15m features redundant.
- TODO: Baseline comparison.
- TODO: Cost sensitivity test.
- TODO: Yearly breakdown.
- TODO: Paper trading.
