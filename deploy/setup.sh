#!/bin/bash
# ============================================================
# BTC Trading Bot — Server Setup Script
# Run this ONCE on your Oracle Cloud / any Linux VPS
# Usage: bash deploy/setup.sh
# ============================================================

set -e
echo "═══ BTC Trading Bot — Server Setup ═══"

# 1. Update system
echo "[1/5] Updating system..."
sudo apt-get update -y && sudo apt-get upgrade -y

# 2. Install Python
echo "[2/5] Installing Python..."
sudo apt-get install -y python3 python3-pip python3-venv git

# 3. Create virtual environment
echo "[3/5] Creating virtual environment..."
cd ~/btc-mtf
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
echo "[4/5] Installing Python packages..."
pip install --upgrade pip
pip install ccxt xgboost pandas numpy pyarrow scikit-learn

# 5. Test the bot loads correctly
echo "[5/5] Testing bot loads..."
cd ~/btc-mtf
python3 -c "
from src.predictor import Predictor
p = Predictor()
print('✓ Models loaded successfully')
"

echo ""
echo "═══ Setup Complete! ═══"
echo ""
echo "Next steps:"
echo "  1. Add your API keys:  nano src/config.py"
echo "  2. Install the service: sudo bash deploy/install_service.sh"
echo "  3. Start the bot:       sudo systemctl start btc-bot"
echo "  4. Check logs:          journalctl -u btc-bot -f"
