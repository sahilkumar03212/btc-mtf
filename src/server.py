"""
Web Server Wrapper for Render.com
Render's free tier requires a web service that binds to a port.
This script runs a lightweight Flask server and starts the trading bot in a background thread.
"""
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from flask import Flask, jsonify

# Add the src/ directory to the Python path so it can find bot.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bot  # Imports your existing bot.py

app = Flask(__name__)

# Track bot status
bot_status = {"state": "starting", "last_error": None, "started_at": None}


def run_bot_with_recovery():
    """Run the bot with automatic restart on crash."""
    import time
    from notifier import notify_error

    bot_status["started_at"] = datetime.now(timezone.utc).isoformat()

    while True:
        try:
            bot_status["state"] = "running"
            bot.main()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            bot_status["state"] = "crashed"
            bot_status["last_error"] = error_msg

            # Print to Render logs so we can SEE the error
            print(f"\n[FATAL] Bot thread crashed: {error_msg}")
            traceback.print_exc()

            # Send Telegram alert
            try:
                notify_error(f"Bot crashed! Restarting in 30s...\n{error_msg}")
            except Exception:
                pass

            # Wait 30 seconds then restart
            print("[RECOVERY] Restarting bot in 30 seconds...")
            time.sleep(30)
            print("[RECOVERY] Restarting bot now...")


# Start the bot in a background thread with crash recovery
bot_thread = threading.Thread(target=run_bot_with_recovery, daemon=True)
bot_thread.start()


@app.route('/')
def home():
    """Health check endpoint for Render and UptimeRobot"""
    return jsonify({
        "status": bot_status["state"],
        "started_at": bot_status["started_at"],
        "last_error": bot_status["last_error"],
        "message": "BTC Trading Bot is running in the background."
    })


if __name__ == '__main__':
    # Render assigns a port dynamically via the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
