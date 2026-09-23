# Problems Encountered & Solutions

### 1. Binance US Geo-Blocking
**Problem:** When deployed to Render.com, the bot immediately crashed with `HTTP 451: Service unavailable from a restricted location`.
**Solution:** Render defaulted to an Oregon (US) server. Binance blocks all US IP addresses. We deleted the Oregon instance and redeployed the service to Frankfurt (EU) / Singapore, which instantly bypassed the restriction.

### 2. Render Free-Tier Sleeping
**Problem:** Render's free tier forces apps to sleep after 15 minutes of inactivity, which would kill our trading loop.
**Solution:** Wrapped the bot inside a Flask web server (`server.py`) and set up an external service (UptimeRobot) to send an HTTP ping to the server every 5 minutes, keeping it awake indefinitely.

### 3. Invalid Binance API Keys (Error -2015)
**Problem:** The bot threw `Invalid API-key, IP, or permissions` errors.
**Solution:** Identified a minor typo in the transcription of the API key from a screenshot (a `q` instead of a `p`). Regenerated fresh keys on Binance Testnet and passed them securely via environment variables to fix the authentication.

### 4. Absurdly Wide Stop Losses (e.g., $21,000 SL on a $84,000 asset)
**Problem:** The live bot attempted to set a Stop Loss 75% below the entry price. 
**Solution:** Discovered that the XGBoost Regressor model was outputting a constant `+50%` expected return because it was likely trained on the binary labels (`0` and `1`) instead of the continuous return labels. We decoupled the SL/TP logic from the regressor and hardcoded realistic professional risk metrics (1% SL, 2% TP) directly into `decision.py`.

### 5. Telegram Markdown Parsing Crashes
**Problem:** The bot successfully placed a trade but failed to send the Telegram alert, throwing a `400 Bad Request: can't parse entities` error.
**Solution:** Telegram's `Markdown` parser crashed when it encountered an unescaped underscore (`_`) in the variable `p_up`. Rewrote `notifier.py` to use `HTML` parsing instead, which securely allows underscores and complex formatting without crashing.

### 6. Silent Thread Crashes in the Cloud
**Problem:** The bot thread crashed once on Render, but the server appeared "online" to UptimeRobot because the web thread was still alive.
**Solution:** Implemented a robust `run_bot_with_recovery()` wrapper in `server.py`. It now catches fatal exceptions, logs the full traceback, sends a `⚠️ Bot Error` alert to Telegram, and automatically restarts the trading thread after a 30-second cooldown.

### 7. Duplicate Trades Executing Simultaneously
**Problem:** Received two identical Telegram `BUY` alerts within 2 minutes of each other.
**Solution:** Diagnosed as a side effect of Render's "Zero-Downtime Deployments". During a code update, Render keeps the old server alive until the new one is healthy. For 2 minutes, both servers were active and triggered a BUY simultaneously. Confirmed this is a harmless, transient deployment artifact, not a code bug.
