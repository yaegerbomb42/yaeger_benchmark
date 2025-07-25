"""
Enhanced Exchange Implementation for EXTREME Portfolio Optimizer
Supports Byzantine fault tolerance, microsecond latency, and enterprise features
"""

import time
import random
import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import threading
from collections import deque
import secrets

try:
    import numpy as np
except ImportError:
    # Fallback for numpy functionality
    class np:
        @staticmethod
        def sqrt(x):
            return x ** 0.5
        
        @staticmethod
        def median(arr):
            sorted_arr = sorted(arr)
            n = len(sorted_arr)
            if n % 2 == 0:
                return (sorted_arr[n//2-1] + sorted_arr[n//2]) / 2
            else:
                return sorted_arr[n//2]
        
        @staticmethod
        def abs(arr):
            return [abs(x) for x in arr]
        
        @staticmethod
        def array(arr):
            return arr

class MarketRegimeType(Enum):
    BULL = "BULL"
    BEAR = "BEAR" 
    SIDEWAYS = "SIDEWAYS"
    CRISIS = "CRISIS"

class VolatilityRegime(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: float
    exchange_id: str = "MAIN"
    is_corrupted: bool = False
    confidence_score: float = 1.0
    
@dataclass
class OrderBook:
    symbol: str
    bids: List[tuple[float, int]]  # price, size
    asks: List[tuple[float, int]]  # price, size
    timestamp: float

@dataclass
class SignedOrder:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    order_type: str  # "MARKET", "LIMIT", "ICEBERG", "TWAP"
    signature: str
    venue_preferences: List[str] = field(default_factory=list)
    execution_strategy: str = "AGGRESSIVE"
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
    venue: str = "MAIN"

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: float = 0.0

@dataclass
class DarkPoolHint:
    symbol: str
    estimated_size: int
    price_improvement: float  # potential improvement over lit market
    confidence: float

@dataclass
class ExecutionMetrics:
    avg_fill_time: float  # microseconds
    slippage_bps: float   # basis points
    fill_rate: float      # percentage
    adverse_selection: float
    
@dataclass
class MarketRegime:
    regime_type: MarketRegimeType
    confidence: float
    transition_probability: Dict[str, float]
    expected_duration_days: int
    volatility_regime: VolatilityRegime

class ByzantineExchange:
    """
    Enhanced exchange with Byzantine fault tolerance and extreme performance
    """
    
    def __init__(self, corruption_rate: float = 0.0, enable_dark_pools: bool = True):
        self.corruption_rate = corruption_rate
        self.enable_dark_pools = enable_dark_pools
        
        # Core state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, SignedOrder] = {}
        self.quotes: Dict[str, Quote] = {}
        self.order_books: Dict[str, OrderBook] = {}
        
        # Performance tracking
        self.execution_metrics = ExecutionMetrics(0.0, 0.0, 0.0, 0.0)
        self.order_counter = 0
        
        # Byzantine fault tolerance
        self.exchange_feeds = ["EXCHANGE_A", "EXCHANGE_B", "EXCHANGE_C", "EXCHANGE_D"]
        self.corrupted_feeds: Set[str] = set()
        
        # Threading and performance
        self.lock = threading.RLock()
        self.price_history: Dict[str, deque] = {}
        
        # Market regime tracking
        self.current_regime = MarketRegime(
            regime_type=MarketRegimeType.SIDEWAYS,
            confidence=0.8,
            transition_probability={"BULL": 0.3, "BEAR": 0.2, "SIDEWAYS": 0.5},
            expected_duration_days=30,
            volatility_regime=VolatilityRegime.MEDIUM
        )
        
        # Initialize extreme market data (1000+ symbols)
        self._initialize_extreme_market_data()
        
    def _initialize_extreme_market_data(self):
        """Initialize market data for 1000+ assets"""
        # Major indices and stocks
        major_symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        sector_etfs = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]
        
        # Generate 1000 synthetic symbols
        symbols = major_symbols + sector_etfs
        for i in range(980):  # Fill to 1000 total
            symbols.append(f"SYM{i:04d}")
            
        # Initialize with realistic price data
        for symbol in symbols:
            base_price = random.uniform(10.0, 5000.0)
            spread_bps = random.uniform(1, 50)  # 1-50 basis points
            spread = base_price * (spread_bps / 10000)
            
            quote = Quote(
                symbol=symbol,
                bid=base_price - spread/2,
                ask=base_price + spread/2,
                bid_size=random.randint(100, 10000),
                ask_size=random.randint(100, 10000),
                timestamp=time.time(),
                exchange_id=random.choice(self.exchange_feeds),
                is_corrupted=False,
                confidence_score=1.0
            )
            
            self.quotes[symbol] = quote
            self.price_history[symbol] = deque(maxlen=1000)
            
            # Initialize position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                last_update=time.time()
            )
            
            # Initialize order book
            self.order_books[symbol] = self._generate_order_book(symbol, base_price)
    
    def _generate_order_book(self, symbol: str, mid_price: float) -> OrderBook:
        """Generate realistic order book depth"""
        bids = []
        asks = []
        
        # Generate 10 levels each side
        for i in range(10):
            bid_price = mid_price * (1 - (i + 1) * 0.0001)  # 1 bps per level
            ask_price = mid_price * (1 + (i + 1) * 0.0001)
            
            bid_size = random.randint(100, 1000) * (10 - i)  # Larger size closer to mid
            ask_size = random.randint(100, 1000) * (10 - i)
            
            bids.append((bid_price, bid_size))
            asks.append((ask_price, ask_size))
            
        return OrderBook(
            symbol=symbol,
            bids=sorted(bids, key=lambda x: x[0], reverse=True),
            asks=sorted(asks, key=lambda x: x[0]),
            timestamp=time.time()
        )
    
    def _inject_byzantine_corruption(self, quote: Quote) -> Quote:
        """Simulate Byzantine faults in data feeds"""
        if random.random() < self.corruption_rate:
            # Various types of corruption
            corruption_type = random.choice(["price_spike", "stale_data", "inverted_spread"])
            
            if corruption_type == "price_spike":
                multiplier = random.choice([0.1, 10.0])  # 90% drop or 10x spike
                quote.bid *= multiplier
                quote.ask *= multiplier
                quote.is_corrupted = True
                quote.confidence_score = 0.1
                
            elif corruption_type == "stale_data":
                quote.timestamp -= random.uniform(60, 3600)  # 1-60 minutes old
                quote.is_corrupted = True
                quote.confidence_score = 0.3
                
            elif corruption_type == "inverted_spread":
                quote.bid, quote.ask = quote.ask, quote.bid  # Negative spread
                quote.is_corrupted = True
                quote.confidence_score = 0.0
                
        return quote
    
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current bid/ask with Byzantine fault simulation"""
        start_time = time.perf_counter()
        
        # Simulate microsecond latency (1-10 μs)
        time.sleep(random.uniform(0.000001, 0.00001))
        
        with self.lock:
            quote = self.quotes.get(symbol)
            if not quote:
                return None
                
            # Update price with regime-dependent volatility
            self._update_price_with_regime(quote)
            
            # Apply Byzantine corruption
            quote = self._inject_byzantine_corruption(quote)
            
            # Track performance
            latency = (time.perf_counter() - start_time) * 1_000_000  # microseconds
            
            return quote
    
    def _update_price_with_regime(self, quote: Quote):
        """Update price based on current market regime"""
        base_volatility = {
            VolatilityRegime.LOW: 0.0005,      # 0.05%
            VolatilityRegime.MEDIUM: 0.002,    # 0.2%
            VolatilityRegime.HIGH: 0.01,       # 1%
            VolatilityRegime.EXTREME: 0.05     # 5%
        }
        
        vol = base_volatility[self.current_regime.volatility_regime]
        
        # Regime-dependent drift
        drift = {
            MarketRegimeType.BULL: 0.0001,
            MarketRegimeType.BEAR: -0.0001, 
            MarketRegimeType.SIDEWAYS: 0.0,
            MarketRegimeType.CRISIS: -0.005
        }
        
        price_change = random.normalvariate(
            drift[self.current_regime.regime_type], vol
        )
        
        mid_price = (quote.bid + quote.ask) / 2
        new_mid = mid_price * (1 + price_change)
        spread = quote.ask - quote.bid
        
        quote.bid = new_mid - spread / 2
        quote.ask = new_mid + spread / 2
        quote.timestamp = time.time()
        quote.bid_size = random.randint(100, 10000)
        quote.ask_size = random.randint(100, 10000)
        
        # Update price history
        self.price_history[quote.symbol].append(new_mid)
    
    def get_market_depth(self, symbol: str, levels: int = 10) -> Optional[OrderBook]:
        """Get full order book depth"""
        with self.lock:
            book = self.order_books.get(symbol)
            if book:
                # Update book with current pricing
                quote = self.quotes.get(symbol)
                if quote:
                    mid_price = (quote.bid + quote.ask) / 2
                    book = self._generate_order_book(symbol, mid_price)
                    self.order_books[symbol] = book
            return book
    
    def place_order(self, order: SignedOrder) -> OrderResult:
        """Place cryptographically signed order"""
        start_time = time.perf_counter()
        
        # Verify signature (simplified)
        if not self._verify_order_signature(order):
            return OrderResult(
                order_id="INVALID",
                status="REJECTED",
                filled_quantity=0,
                filled_price=0.0,
                remaining_quantity=order.quantity,
                commission=0.0,
                timestamp=time.time(),
                venue="NONE"
            )
        
        # Simulate ultra-low latency (10-50 μs)
        time.sleep(random.uniform(0.00001, 0.00005))
        
        with self.lock:
            self.order_counter += 1
            order_id = f"EXTORD{self.order_counter:08d}"
            order.order_id = order_id
            
            # Route to best venue
            venue = self._smart_order_routing(order)
            
            quote = self.quotes.get(order.symbol)
            if not quote or quote.is_corrupted:
                return OrderResult(
                    order_id=order_id,
                    status="REJECTED",
                    filled_quantity=0,
                    filled_price=0.0,
                    remaining_quantity=order.quantity,
                    commission=0.0,
                    timestamp=time.time(),
                    venue=venue
                )
            
            # Execute order with advanced logic
            result = self._execute_advanced_order(order, quote, venue)
            
            # Update execution metrics
            latency = (time.perf_counter() - start_time) * 1_000_000
            self._update_execution_metrics(latency, result)
            
            return result
    
    def _verify_order_signature(self, order: SignedOrder) -> bool:
        """Verify cryptographic signature (simplified implementation)"""
        # In production, would use proper HSM integration
        expected_sig = hashlib.sha256(
            f"{order.symbol}{order.side}{order.quantity}{order.price}".encode()
        ).hexdigest()[:16]
        return order.signature == expected_sig
    
    def _smart_order_routing(self, order: SignedOrder) -> str:
        """Route order to best venue"""
        if self.enable_dark_pools and order.execution_strategy == "HIDDEN":
            return random.choice(["DARK_POOL_1", "DARK_POOL_2"])
        return random.choice(["MAIN", "ARCA", "NASDAQ", "BATS"])
    
    def _execute_advanced_order(self, order: SignedOrder, quote: Quote, venue: str) -> OrderResult:
        """Execute order with advanced features"""
        if order.order_type == "MARKET":
            return self._execute_market_order(order, quote, venue)
        elif order.order_type == "ICEBERG":
            return self._execute_iceberg_order(order, quote, venue)
        elif order.order_type == "TWAP":
            return self._execute_twap_order(order, quote, venue)
        else:  # LIMIT
            return self._execute_limit_order(order, quote, venue)
    
    def _execute_market_order(self, order: SignedOrder, quote: Quote, venue: str) -> OrderResult:
        """Execute market order with realistic slippage"""
        if order.side == "BUY":
            execution_price = quote.ask
            available_size = quote.ask_size
        else:
            execution_price = quote.bid
            available_size = quote.bid_size
            
        # Market impact model
        impact = self._calculate_market_impact(order.quantity, available_size, quote.symbol)
        if order.side == "BUY":
            execution_price *= (1 + impact)
        else:
            execution_price *= (1 - impact)
            
        filled_quantity = min(order.quantity, available_size)
        
        # Update position
        self._update_position(order.symbol, order.side, filled_quantity, execution_price)
        
        commission = filled_quantity * execution_price * 0.00005  # 0.5 bps
        status = "FILLED" if filled_quantity == order.quantity else "PARTIAL"
        
        return OrderResult(
            order_id=order.order_id,
            status=status,
            filled_quantity=filled_quantity,
            filled_price=execution_price,
            remaining_quantity=order.quantity - filled_quantity,
            commission=commission,
            timestamp=time.time(),
            venue=venue
        )
    
    def _calculate_market_impact(self, order_size: int, available_size: int, symbol: str) -> float:
        """Calculate market impact using square-root law"""
        participation_rate = order_size / max(available_size, 1)
        
        # Square root market impact model
        impact = 0.1 * np.sqrt(participation_rate)  # 10 bps * sqrt(participation)
        
        # Regime adjustment
        regime_multiplier = {
            MarketRegimeType.BULL: 0.8,
            MarketRegimeType.BEAR: 1.2,
            MarketRegimeType.SIDEWAYS: 1.0,
            MarketRegimeType.CRISIS: 2.0
        }
        
        return impact * regime_multiplier[self.current_regime.regime_type]
    
    def _execute_iceberg_order(self, order: SignedOrder, quote: Quote, venue: str) -> OrderResult:
        """Execute iceberg order (simplified)"""
        # For now, execute as regular limit order
        return self._execute_limit_order(order, quote, venue)
    
    def _execute_twap_order(self, order: SignedOrder, quote: Quote, venue: str) -> OrderResult:
        """Execute TWAP order (simplified)"""
        # For now, execute as market order
        return self._execute_market_order(order, quote, venue)
    
    def _execute_limit_order(self, order: SignedOrder, quote: Quote, venue: str) -> OrderResult:
        """Execute limit order"""
        self.orders[order.order_id] = order
        return OrderResult(
            order_id=order.order_id,
            status="PENDING",
            filled_quantity=0,
            filled_price=0.0,
            remaining_quantity=order.quantity,
            commission=0.0,
            timestamp=time.time(),
            venue=venue
        )
    
    def _update_position(self, symbol: str, side: str, quantity: int, price: float):
        """Update position with new fill"""
        position = self.positions.get(symbol)
        if not position:
            return
            
        if side == "BUY":
            new_quantity = position.quantity + quantity
            if position.quantity == 0:
                position.avg_price = price
            else:
                total_cost = (position.quantity * position.avg_price + quantity * price)
                position.avg_price = total_cost / new_quantity
            position.quantity = new_quantity
        else:  # SELL
            position.quantity -= quantity
            if position.quantity <= 0:
                position.avg_price = 0.0
                position.quantity = 0
                
        position.last_update = time.time()
    
    def _update_execution_metrics(self, latency: float, result: OrderResult):
        """Update execution quality metrics"""
        # Simplified metrics update
        self.execution_metrics.avg_fill_time = (
            self.execution_metrics.avg_fill_time * 0.99 + latency * 0.01
        )
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        with self.lock:
            # Update unrealized PnL
            for symbol, position in self.positions.items():
                if position.quantity != 0:
                    quote = self.quotes.get(symbol)
                    if quote and not quote.is_corrupted:
                        current_price = (quote.bid + quote.ask) / 2
                        position.unrealized_pnl = (
                            (current_price - position.avg_price) * position.quantity
                        )
            return self.positions.copy()
    
    def get_dark_pool_indication(self, symbol: str) -> Optional[DarkPoolHint]:
        """Get dark pool liquidity hints"""
        if not self.enable_dark_pools:
            return None
            
        return DarkPoolHint(
            symbol=symbol,
            estimated_size=random.randint(1000, 50000),
            price_improvement=random.uniform(0.1, 2.0),  # 0.1-2 bps
            confidence=random.uniform(0.6, 0.95)
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        with self.lock:
            if order_id in self.orders:
                del self.orders[order_id]
                return True
            return False
    
    def get_execution_quality_metrics(self) -> ExecutionMetrics:
        """Get real-time execution quality metrics"""
        return self.execution_metrics
    
    def detect_byzantine_feeds(self) -> List[str]:
        """Identify corrupted/malicious data feeds"""
        corrupted = []
        for feed_id in self.exchange_feeds:
            if random.random() < self.corruption_rate:
                corrupted.append(feed_id)
        return corrupted
    
    def set_market_regime(self, regime: MarketRegime):
        """Update market regime (for testing/simulation)"""
        self.current_regime = regime

# Alias for backward compatibility
Exchange = ByzantineExchange
