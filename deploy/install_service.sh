#!/bin/bash
# Install the bot as a systemd service (auto-start, auto-restart)
# Usage: sudo bash deploy/install_service.sh

set -e

echo "Installing btc-bot service..."

# Copy service file
sudo cp deploy/btc-bot.service /etc/systemd/system/btc-bot.service

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable btc-bot

echo ""
echo "✓ Service installed!"
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start btc-bot"
echo "  Stop:    sudo systemctl stop btc-bot"
echo "  Status:  sudo systemctl status btc-bot"
echo "  Logs:    journalctl -u btc-bot -f"
echo "  Restart: sudo systemctl restart btc-bot"
