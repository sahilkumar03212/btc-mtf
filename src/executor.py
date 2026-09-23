"""
Order executor — places and manages orders on Binance (Testnet or Live).
"""
import time


class Executor:
    def __init__(self, exchange):
        self.exchange = exchange
        self.symbol = "BTC/USDT"

    def get_balance(self):
        """Get available USDT balance."""
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            free = float(usdt.get("free", 0))
            total = float(usdt.get("total", 0))
            return {"free": free, "total": total}
        except Exception as e:
            print(f"  [ERROR] get_balance failed: {e}")
            return {"free": 0, "total": 0}

    def get_btc_balance(self):
        """Get available BTC balance."""
        try:
            balance = self.exchange.fetch_balance()
            btc = balance.get("BTC", {})
            return float(btc.get("free", 0))
        except Exception as e:
            print(f"  [ERROR] get_btc_balance failed: {e}")
            return 0.0

    def get_current_price(self):
        """Get current BTC/USDT market price."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker["last"])
        except Exception as e:
            print(f"  [ERROR] get_current_price failed: {e}")
            return None

    def place_buy(self, size_usd):
        """Place a market buy order for a given USD amount."""
        try:
            price = self.get_current_price()
            if price is None or price <= 0:
                return None

            # Convert USD to BTC quantity
            quantity = size_usd / price
            # Binance minimum precision
            quantity = round(quantity, 5)

            if quantity <= 0:
                print(f"  [WARN] Buy quantity too small: {quantity}")
                return None

            print(f"  Placing BUY: {quantity:.5f} BTC (~${size_usd:.2f}) at ~${price:.2f}")
            order = self.exchange.create_market_buy_order(self.symbol, quantity)
            print(f"  ✓ BUY filled: order_id={order['id']}")
            return order

        except Exception as e:
            print(f"  [ERROR] place_buy failed: {e}")
            return None

    def place_sell(self, quantity=None):
        """Place a market sell order. If quantity is None, sells all BTC."""
        try:
            if quantity is None:
                quantity = self.get_btc_balance()

            quantity = round(quantity, 5)
            if quantity <= 0:
                print(f"  [WARN] Nothing to sell: {quantity} BTC")
                return None

            price = self.get_current_price()
            print(f"  Placing SELL: {quantity:.5f} BTC (~${quantity * price:.2f}) at ~${price:.2f}")
            order = self.exchange.create_market_sell_order(self.symbol, quantity)
            print(f"  ✓ SELL filled: order_id={order['id']}")
            return order

        except Exception as e:
            print(f"  [ERROR] place_sell failed: {e}")
            return None
