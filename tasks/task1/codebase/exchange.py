import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
from queue import Queue, Empty

@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: float

@dataclass
class Order:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    order_type: str  # "MARKET" or "LIMIT"
    order_id: Optional[str] = None

@dataclass
class OrderResult:
    order_id: str
    status: str  # "FILLED", "PARTIAL", "REJECTED", "PENDING"
    filled_quantity: int
    filled_price: float
    remaining_quantity: int
    commission: float
    timestamp: float

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float

class Exchange:
    """Simulated exchange with realistic latency and market dynamics."""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.quotes: Dict[str, Quote] = {}
        self.lock = threading.Lock()
        
        # Simulate initial market data
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        base_prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0, "TSLA": 800.0, "AMZN": 3200.0}
        
        for symbol in symbols:
            base_price = base_prices[symbol]
            spread = base_price * 0.001  # 0.1% spread
            self.quotes[symbol] = Quote(
                symbol=symbol,
                bid=base_price - spread/2,
                ask=base_price + spread/2,
                bid_size=random.randint(100, 1000),
                ask_size=random.randint(100, 1000),
                timestamp=time.time()
            )
            
            # Initialize empty positions
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
    
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current bid/ask for symbol with simulated latency."""
        # Simulate network latency (1-5ms)
        time.sleep(random.uniform(0.001, 0.005))
        
        with self.lock:
            quote = self.quotes.get(symbol)
            if quote:
                # Add some random price movement
                price_change = random.uniform(-0.002, 0.002)  # ±0.2%
                mid_price = (quote.bid + quote.ask) / 2
                new_mid = mid_price * (1 + price_change)
                spread = quote.ask - quote.bid
                
                quote.bid = new_mid - spread / 2
                quote.ask = new_mid + spread / 2
                quote.timestamp = time.time()
                quote.bid_size = random.randint(100, 1000)
                quote.ask_size = random.randint(100, 1000)
            
            return quote
    
    def place_order(self, order: Order) -> OrderResult:
        """Place buy/sell order with realistic execution simulation."""
        # Simulate order processing latency (5-20ms)
        time.sleep(random.uniform(0.005, 0.020))
        
        with self.lock:
            self.order_counter += 1
            order_id = f"ORD{self.order_counter:06d}"
            order.order_id = order_id
            
            quote = self.quotes.get(order.symbol)
            if not quote:
                return OrderResult(
                    order_id=order_id,
                    status="REJECTED",
                    filled_quantity=0,
                    filled_price=0.0,
                    remaining_quantity=order.quantity,
                    commission=0.0,
                    timestamp=time.time()
                )
            
            # Simulate market order execution
            if order.order_type == "MARKET":
                if order.side == "BUY":
                    execution_price = quote.ask
                    available_size = quote.ask_size
                else:
                    execution_price = quote.bid
                    available_size = quote.bid_size
                
                # Add slippage for large orders
                slippage = min(0.001, order.quantity / 10000)  # Up to 0.1% slippage
                if order.side == "BUY":
                    execution_price *= (1 + slippage)
                else:
                    execution_price *= (1 - slippage)
                
                # Determine fill quantity
                filled_quantity = min(order.quantity, available_size)
                
                # Update position
                position = self.positions.get(order.symbol)
                if position:
                    if order.side == "BUY":
                        new_quantity = position.quantity + filled_quantity
                        if position.quantity == 0:
                            position.avg_price = execution_price
                        else:
                            total_cost = (position.quantity * position.avg_price + 
                                        filled_quantity * execution_price)
                            position.avg_price = total_cost / new_quantity
                        position.quantity = new_quantity
                    else:  # SELL
                        position.quantity -= filled_quantity
                        if position.quantity <= 0:
                            position.avg_price = 0.0
                            position.quantity = 0
                
                commission = filled_quantity * execution_price * 0.0001  # 0.01% commission
                
                status = "FILLED" if filled_quantity == order.quantity else "PARTIAL"
                
                return OrderResult(
                    order_id=order_id,
                    status=status,
                    filled_quantity=filled_quantity,
                    filled_price=execution_price,
                    remaining_quantity=order.quantity - filled_quantity,
                    commission=commission,
                    timestamp=time.time()
                )
            
            else:  # LIMIT order
                self.orders[order_id] = order
                return OrderResult(
                    order_id=order_id,
                    status="PENDING",
                    filled_quantity=0,
                    filled_price=0.0,
                    remaining_quantity=order.quantity,
                    commission=0.0,
                    timestamp=time.time()
                )
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        with self.lock:
            # Update unrealized PnL
            for symbol, position in self.positions.items():
                if position.quantity != 0:
                    quote = self.quotes.get(symbol)
                    if quote:
                        current_price = (quote.bid + quote.ask) / 2
                        position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
            
            return self.positions.copy()
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        with self.lock:
            if order_id in self.orders:
                del self.orders[order_id]
                return True
            return False
    
    def get_market_value(self) -> float:
        """Get total market value of all positions."""
        total_value = 0.0
        with self.lock:
            for symbol, position in self.positions.items():
                if position.quantity != 0:
                    quote = self.quotes.get(symbol)
                    if quote:
                        current_price = (quote.bid + quote.ask) / 2
                        total_value += current_price * position.quantity
        return total_value
