# Implementation Steps

## Phase 1: Data Pipeline
- Scripted `fetch.py` to download historical OHLCV data from Binance using the `ccxt` library.
- Fetched data across 3 timeframes (15m, 1h, 4h) to give the model a "multi-timeframe" perspective (micro and macro trends).

## Phase 2: Feature Engineering
- Created `features.py` to calculate technical indicators.
- **Trend:** SMA, EMA, MACD, ADX
- **Momentum:** RSI, Stochastic
- **Volatility:** Bollinger Bands, ATR
- Handled the complex task of aligning 1h and 4h context data to the 15m base timeframe **without lookahead bias** (using `.shift(1)` before forward-filling).

## Phase 3: Model Training & Backtesting
- Implemented a Walk-Forward Validation strategy (6 folds) with an embargo to prevent data leakage (`train_xgb.py`).
- Trained an XGBoost Classifier for direction and an XGBoost Regressor for magnitude.
- Built a vectorised backtester (`backtest.py`) and an event-driven fractional backtester (`backtest_final.py`) to simulate real-world trading costs (0.15% slippage/fees).

## Phase 4: Live Bot Pipeline
- Created `live_data.py` to build the exact same feature dataframe in real-time.
- Built `decision.py` to translate model probabilities into BUY/SELL/HOLD actions.
- Built `position_manager.py` to track unrealized PnL, bars held, and enforce risk limits.
- Built `executor.py` to interact with the Binance API.
- Tied everything together in the main loop: `bot.py`.

## Phase 5: Cloud Deployment & Notifications
- Wrote `notifier.py` to push cycle reports and trade alerts directly to a Telegram bot.
- Wrapped the bot in a Flask web server (`server.py`) so it could bind to a port on Render.com.
- Attached UptimeRobot to ping the server every 5 minutes to bypass Render's free-tier sleep mechanism.
