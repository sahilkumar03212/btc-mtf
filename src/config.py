"""
Central configuration for the BTC trading bot.
All settings in one place — switch between testnet and live here.
"""

# ── Exchange Mode ─────────────────────────────────────────────
MODE = "testnet"  # "testnet" or "live"

# Binance Testnet API keys
# Get yours at: https://testnet.binance.vision/
TESTNET_API_KEY = ""
TESTNET_API_SECRET = ""

# Real Binance API keys (only used when MODE = "live")
LIVE_API_KEY = ""
LIVE_API_SECRET = ""

# ── Telegram Notifications ────────────────────────────────────
# Follow setup instructions in src/notifier.py to get these
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# ── Symbol ────────────────────────────────────────────────────
SYMBOL = "BTC/USDT"
BASE_TIMEFRAME = "15m"
CONTEXT_TIMEFRAMES = ["1h", "4h"]

# ── Trading Parameters ────────────────────────────────────────
# Costs
FEE_PER_SIDE = 0.001       # 0.1% taker fee
SLIPPAGE = 0.0005           # 0.05% estimated slippage
ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE)  # 0.3%
SAFETY_MARGIN = 0.0001      # 0.01% buffer

# Decision thresholds
MIN_EXPECTED_RETURN = 0.0005  # 0.05% — let the model trade on subtle signals
P_UP_BUY_THRESHOLD = 0.52    # buy when P(up) > this
P_UP_SELL_THRESHOLD = 0.48   # sell when P(up) < this

# Position sizing
MAX_POSITION_PCT = 0.02     # max 2% of account per trade
POSITION_SCALE_FACTOR = 1.0 # multiply predicted magnitude for sizing

# ── Risk Management ───────────────────────────────────────────
STOP_LOSS_MULTIPLIER = 1.5  # SL = entry ± (predicted_magnitude × this)
TAKE_PROFIT_MULTIPLIER = 2.0  # TP = entry ± (predicted_magnitude × this)
MAX_HOLD_BARS = 8           # force close after 8 × 15m = 2 hours
MAX_DAILY_LOSS_PCT = 0.02   # stop trading if daily loss exceeds 2%
MAX_OPEN_POSITIONS = 1      # one position at a time

# ── Data Parameters ───────────────────────────────────────────
# How many historical bars to fetch for feature computation
# Need enough bars for the longest rolling window (48 bars)
LOOKBACK_BARS = {
    "15m": 100,
    "1h": 100,
    "4h": 100,
}

# ── Paths ─────────────────────────────────────────────────────
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

CLASSIFIER_PATH = MODEL_DIR / "xgb_classifier.json"
REGRESSOR_PATH = MODEL_DIR / "xgb_regressor.json"

# ── Loop Timing ───────────────────────────────────────────────
LOOP_INTERVAL_SECONDS = 60  # check every 60s, act on new 15m candle
