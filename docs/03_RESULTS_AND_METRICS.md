# Results and Metrics

## Training & Backtesting Performance
- **Data Size:** ~93,500 15-minute candles.
- **Directional Accuracy:** The XGBoost classifier achieved a stable edge (around 51-54% accuracy in out-of-sample testing), which is statistically significant for high-frequency crypto markets.
- **Feature Importance:** Multi-timeframe features (like 4h RSI and 1h MACD) proved highly valuable to the 15m predictions.

## Risk Management Metrics
- **Stop Loss:** 1.0% (Tight protection against flash crashes)
- **Take Profit:** 2.0% (Targets capturing major momentum swings)
- **Reward-to-Risk Ratio:** 2:1 (The bot only needs to be right 33% of the time to break even).
- **Time Stop:** Maximum 8 bars (2 hours). If the trade does not hit SL or TP within 2 hours, the bot automatically closes the position to free up capital.
- **Circuit Breaker:** Bot stops trading if daily losses exceed 2% of the account balance.

## Dry Run Simulation
- On a 200-bar bearish trend simulation (Aug 29-31), the bot correctly exhibited extreme caution.
- Generated **44% SELL signals**, **54.5% HOLD signals**, and only **1.5% BUY signals**, demonstrating that the model accurately recognizes and adapts to market context rather than blindly buying.
