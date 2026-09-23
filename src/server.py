"""
Web Server Wrapper for Render.com
Render's free tier requires a web service that binds to a port.
This script runs a lightweight Flask server and starts the trading bot in a background thread.
"""
import os
import threading
from flask import Flask, jsonify
import bot  # Imports your existing bot.py

app = Flask(__name__)

# Start the bot in a background thread so the web server isn't blocked
bot_thread = threading.Thread(target=bot.main, daemon=True)
bot_thread.start()

@app.route('/')
def home():
    """Health check endpoint for Render and UptimeRobot"""
    return jsonify({
        "status": "online",
        "message": "BTC Trading Bot is running in the background."
    })

if __name__ == '__main__':
    # Render assigns a port dynamically via the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
