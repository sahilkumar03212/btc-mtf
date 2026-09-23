# Project Overview: BTC/USDT Machine Learning Trading Bot

## Goal
The objective of this MSc Project was to build, backtest, and deploy a fully automated cryptocurrency trading bot. The bot uses Machine Learning (XGBoost) to predict the short-term price direction of Bitcoin (BTC) and automatically executes trades on the Binance exchange.

## Architecture
- **Asset:** BTC/USDT
- **Base Timeframe:** 15-minutes (with 1-hour and 4-hour context data)
- **Machine Learning:** 
  - XGBoost Classifier (Predicts probability of price going UP)
  - XGBoost Regressor (Predicts magnitude of return)
- **Live Execution:** Binance Spot Testnet (Paper Trading)
- **Hosting:** Render.com (Free Tier Web Service, 24/7 uptime)
- **Notifications:** Real-time Telegram alerts

## Core Logic
Every 15 minutes, the bot:
1. Wakes up and fetches the latest 15m, 1h, and 4h candles from Binance.
2. Calculates 72 technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.).
3. Feeds the data into the XGBoost models.
4. Makes a decision:
   - **BUY** if Probability(UP) > 0.52
   - **SELL** if Probability(UP) < 0.48
   - **HOLD** if in the dead zone, or if already in an open position.
5. Manages Risk (1% Stop Loss, 2% Take Profit).
6. Sends a comprehensive dashboard report to Telegram.
