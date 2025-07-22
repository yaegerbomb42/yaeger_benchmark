import time
import random
import threading
from typing import Dict, List, Callable
from dataclasses import dataclass
from queue import Queue

@dataclass
class MarketUpdate:
    symbol: str
    price: float
    volume: int
    timestamp: float
    update_type: str  # "TRADE", "QUOTE", "NEWS"

class MarketData:
    """Real-time market data feed simulator."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.subscribers: List[Callable[[MarketUpdate], None]] = []
        self.running = False
        self.thread = None
        self.update_queue = Queue()
        
        # Initialize price tracking
        self.current_prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0,
            "MSFT": 300.0,
            "TSLA": 800.0,
            "AMZN": 3200.0
        }
        
        # Market volatility parameters
        self.volatilities = {
            "AAPL": 0.02,
            "GOOGL": 0.025,
            "MSFT": 0.018,
            "TSLA": 0.04,
            "AMZN": 0.022
        }
    
    def subscribe(self, callback: Callable[[MarketUpdate], None]):
        """Subscribe to market data updates."""
        self.subscribers.append(callback)
    
    def start(self):
        """Start the market data feed."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._generate_data)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop the market data feed."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _generate_data(self):
        """Generate realistic market data updates."""
        while self.running:
            # Generate updates for random symbols
            symbol = random.choice(self.symbols)
            update_type = random.choices(
                ["TRADE", "QUOTE", "NEWS"],
                weights=[0.7, 0.25, 0.05]
            )[0]
            
            if update_type == "TRADE":
                # Simulate a trade
                current_price = self.current_prices[symbol]
                volatility = self.volatilities[symbol]
                
                # Random walk with mean reversion
                price_change = random.gauss(0, volatility * current_price / 100)
                
                # Add mean reversion bias
                if abs(price_change) > volatility * current_price / 50:
                    price_change *= 0.5  # Reduce extreme moves
                
                new_price = max(0.01, current_price + price_change)
                self.current_prices[symbol] = new_price
                
                volume = random.randint(100, 10000)
                
                update = MarketUpdate(
                    symbol=symbol,
                    price=new_price,
                    volume=volume,
                    timestamp=time.time(),
                    update_type="TRADE"
                )
                
            elif update_type == "QUOTE":
                # Simulate quote update
                current_price = self.current_prices[symbol]
                spread = current_price * random.uniform(0.0005, 0.002)  # 0.05-0.2% spread
                
                update = MarketUpdate(
                    symbol=symbol,
                    price=current_price,
                    volume=random.randint(100, 5000),
                    timestamp=time.time(),
                    update_type="QUOTE"
                )
                
            else:  # NEWS
                # Simulate news impact
                current_price = self.current_prices[symbol]
                impact = random.uniform(-0.05, 0.05)  # ±5% news impact
                new_price = max(0.01, current_price * (1 + impact))
                self.current_prices[symbol] = new_price
                
                update = MarketUpdate(
                    symbol=symbol,
                    price=new_price,
                    volume=0,
                    timestamp=time.time(),
                    update_type="NEWS"
                )
            
            # Notify subscribers
            for callback in self.subscribers:
                try:
                    callback(update)
                except Exception as e:
                    print(f"Error in market data callback: {e}")
            
            # Control update frequency (100-1000 updates per second)
            time.sleep(random.uniform(0.001, 0.010))
    
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        return self.current_prices.get(symbol, 0.0)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices."""
        return self.current_prices.copy()
    
    def simulate_market_event(self, event_type: str = "volatility_spike"):
        """Simulate special market events for testing."""
        if event_type == "volatility_spike":
            # Increase volatility for all symbols
            for symbol in self.volatilities:
                self.volatilities[symbol] *= 2.0
                
            # Schedule volatility normalization
            def reset_volatility():
                time.sleep(30)  # High volatility for 30 seconds
                for symbol in self.volatilities:
                    self.volatilities[symbol] /= 2.0
            
            threading.Thread(target=reset_volatility, daemon=True).start()
            
        elif event_type == "market_crash":
            # Simulate sudden price drops
            for symbol in self.current_prices:
                crash_magnitude = random.uniform(0.05, 0.15)  # 5-15% drop
                self.current_prices[symbol] *= (1 - crash_magnitude)
                
        elif event_type == "circuit_breaker":
            # Simulate trading halt
            self.running = False
            time.sleep(10)  # 10 second halt
            self.running = True
