"""
Enhanced Market Data Module for EXTREME Portfolio Optimizer
Supports regime detection, correlation modeling, and factor analysis
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import threading

try:
    import numpy as np
except ImportError:
    # Fallback for numpy functionality
    class np:
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def multivariate_normal(mean, cov, size=None):
                    # Simple fallback - return random values
                    if size:
                        return [[random.gauss(m, 0.1) for m in mean] for _ in range(size)]
                    else:
                        return [random.gauss(m, 0.1) for m in mean]
                
                @staticmethod
                def normal(loc=0, scale=1, size=None):
                    if size:
                        return [random.gauss(loc, scale) for _ in range(size)]
                    else:
                        return random.gauss(loc, scale)
            return Random()
        
        @staticmethod
        def array(arr):
            return arr
        
        @staticmethod
        def cov(arr):
            # Simple covariance fallback
            return [[0.01 if i==j else 0.001 for j in range(len(arr[0]))] for i in range(len(arr[0]))]
        
        @staticmethod
        def mean(arr, axis=None):
            if axis == 0:
                return [sum(row[i] for row in arr) / len(arr) for i in range(len(arr[0]))]
            else:
                flat = [x for row in arr for x in row] if hasattr(arr[0], '__iter__') else arr
                return sum(flat) / len(flat)
        
        @staticmethod
        def eye(n):
            return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
        
        @staticmethod
        def dot(a, b):
            if isinstance(a[0], (list, tuple)):
                return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
            else:
                return sum(a[i] * b[i] for i in range(len(a)))

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
class MarketUpdate:
    symbol: str
    price: float
    volume: int
    timestamp: float
    update_type: str  # "TRADE", "QUOTE", "NEWS"
    exchange_id: str = "MAIN"

@dataclass
class MarketRegime:
    regime_type: MarketRegimeType
    confidence: float
    transition_probability: Dict[str, float]
    expected_duration_days: int
    volatility_regime: VolatilityRegime
    
@dataclass
class FactorExposure:
    market_factor: float
    size_factor: float
    value_factor: float
    momentum_factor: float
    volatility_factor: float
    quality_factor: float
    residual_risk: float

@dataclass
class DataFeed:
    feed_id: str
    symbols: List[str]
    is_corrupted: bool = False
    corruption_type: str = ""
    reliability_score: float = 1.0

class ExtremeMarketData:
    """
    Enhanced market data system with regime detection and factor analysis
    """
    
    def __init__(self, num_assets: int = 1000):
        self.num_assets = num_assets
        
        # Core data storage
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.returns_history: Dict[str, deque] = {}
        
        # Current market state
        self.current_prices: Dict[str, float] = {}
        self.current_volumes: Dict[str, int] = {}
        
        # Regime detection
        self.current_regime = MarketRegime(
            regime_type=MarketRegimeType.SIDEWAYS,
            confidence=0.8,
            transition_probability={"BULL": 0.3, "BEAR": 0.2, "SIDEWAYS": 0.5},
            expected_duration_days=30,
            volatility_regime=VolatilityRegime.MEDIUM
        )
        
        # Factor model parameters
        self.factor_loadings: Dict[str, FactorExposure] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        
        # Data feeds and Byzantine fault tolerance
        self.data_feeds: List[DataFeed] = []
        self.corrupted_feeds: set = set()
        
        # Performance tracking
        self.lock = threading.RLock()
        self.update_count = 0
        self.last_regime_check = time.time()
        
        # Initialize data
        self._initialize_assets()
        self._initialize_factor_model()
        self._initialize_data_feeds()
    
    def _initialize_assets(self):
        """Initialize price and volume history for all assets - optimized for speed"""
        # Generate realistic symbol names  
        major_symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        sector_symbols = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]
        
        symbols = major_symbols + sector_symbols
        for i in range(self.num_assets - len(symbols)):
            symbols.append(f"SYM{i:04d}")
            
        self.symbols = symbols[:self.num_assets]
        
        # Initialize with minimal historical data for speed
        for symbol in self.symbols:
            # Generate initial price based on symbol type
            if symbol in major_symbols:
                base_price = random.uniform(100, 300)  # Reduced range for speed
            elif symbol in sector_symbols:
                base_price = random.uniform(50, 150)
            else:
                base_price = random.uniform(20, 100)
                
            # Generate only 50 days of historical data for speed
            prices = [base_price]
            volumes = []
            returns = []
            
            for day in range(50):  # Reduced from 252 to 50
                # Simple random walk
                daily_return = random.gauss(0.0005, 0.02)  # Use random.gauss instead of np
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
                returns.append(daily_return)
                
                # Simple volume generation
                volume = random.randint(10000, 100000)  # Simplified volume
                volumes.append(volume)
            
            # Store in deques for efficiency
            self.price_history[symbol] = deque(prices, maxlen=1000)
            self.volume_history[symbol] = deque(volumes, maxlen=1000)
            self.returns_history[symbol] = deque(returns, maxlen=1000)
            
            # Set current values
            self.current_prices[symbol] = prices[-1]
            self.current_volumes[symbol] = volumes[-1]
    
    def _initialize_factor_model(self):
        """Initialize factor exposures for all assets - simplified for speed"""
        for symbol in self.symbols:
            # Generate simple factor exposures
            market_beta = random.gauss(1.0, 0.3)  # Market beta around 1.0
            size_factor = random.gauss(0.0, 0.5)  # Size factor
            value_factor = random.gauss(0.0, 0.3)  # Value factor
            momentum_factor = random.gauss(0.0, 0.4)  # Momentum factor
            volatility_factor = random.gauss(0.0, 0.2)  # Vol factor
            quality_factor = random.gauss(0.0, 0.3)  # Quality factor
            residual_risk = random.uniform(0.1, 0.4)  # Idiosyncratic risk
            
            self.factor_loadings[symbol] = FactorExposure(
                market_factor=market_beta,
                size_factor=size_factor,
                value_factor=value_factor,
                momentum_factor=momentum_factor,
                volatility_factor=volatility_factor,
                quality_factor=quality_factor,
                residual_risk=residual_risk
            )
                size_factor=size_factor,
                value_factor=value_factor,
                momentum_factor=momentum_factor,
                volatility_factor=volatility_factor,
                quality_factor=quality_factor,
                residual_risk=residual_risk
            )
    
    def _initialize_data_feeds(self):
        """Initialize multiple data feeds for Byzantine fault tolerance"""
        feed_names = ["EXCHANGE_A", "EXCHANGE_B", "EXCHANGE_C", "EXCHANGE_D", "VENDOR_1", "VENDOR_2"]
        
        for feed_name in feed_names:
            # Each feed covers different subsets of assets
            num_symbols = random.randint(100, self.num_assets)
            covered_symbols = random.sample(self.symbols, num_symbols)
            
            feed = DataFeed(
                feed_id=feed_name,
                symbols=covered_symbols,
                is_corrupted=False,
                reliability_score=random.uniform(0.95, 0.999)
            )
            self.data_feeds.append(feed)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.current_prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices"""
        return self.current_prices.copy()
    
    def get_price_history(self, symbol: str, periods: int = 100) -> List[float]:
        """Get historical prices"""
        history = self.price_history.get(symbol, deque())
        return list(history)[-periods:]
    
    def get_returns_history(self, symbol: str, periods: int = 100) -> List[float]:
        """Get historical returns"""
        history = self.returns_history.get(symbol, deque())
        return list(history)[-periods:]
    
    def get_regime_classification(self) -> MarketRegime:
        """Current market regime detection"""
        # Check if it's time to update regime
        current_time = time.time()
        if current_time - self.last_regime_check > 60:  # Update every minute
            self._update_regime_classification()
            self.last_regime_check = current_time
            
        return self.current_regime
    
    def _update_regime_classification(self):
        """Update market regime using multiple indicators"""
        with self.lock:
            # Get market index (SPY) returns for regime detection
            spy_returns = self.get_returns_history("SPY", 50)
            if len(spy_returns) < 20:
                return
                
            # Calculate regime indicators
            recent_returns = spy_returns[-20:]  # Last 20 periods
            avg_return = np.mean(recent_returns)
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
            # Trend indicator (moving average comparison)
            short_ma = np.mean(spy_returns[-5:])
            long_ma = np.mean(spy_returns[-20:])
            trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
            
            # VIX-like volatility measure
            rolling_vol = [np.std(spy_returns[i:i+10]) for i in range(len(spy_returns)-10)]
            vol_of_vol = np.std(rolling_vol) if len(rolling_vol) > 1 else 0.01
            
            # Regime classification logic
            if volatility > 0.4:  # 40% annualized vol
                regime_type = MarketRegimeType.CRISIS
                vol_regime = VolatilityRegime.EXTREME
            elif avg_return > 0.001 and trend > 0.02:  # Positive returns and trend
                regime_type = MarketRegimeType.BULL
                vol_regime = VolatilityRegime.LOW if volatility < 0.15 else VolatilityRegime.MEDIUM
            elif avg_return < -0.001 and trend < -0.02:  # Negative returns and trend
                regime_type = MarketRegimeType.BEAR
                vol_regime = VolatilityRegime.HIGH if volatility > 0.25 else VolatilityRegime.MEDIUM
            else:
                regime_type = MarketRegimeType.SIDEWAYS
                vol_regime = VolatilityRegime.MEDIUM
            
            # Calculate confidence based on signal strength
            signal_strength = abs(trend) + (volatility - 0.15) / 0.15
            confidence = min(0.95, max(0.6, signal_strength))
            
            # Update regime
            self.current_regime = MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                transition_probability=self._calculate_transition_probabilities(regime_type),
                expected_duration_days=random.randint(10, 60),
                volatility_regime=vol_regime
            )
    
    def _calculate_transition_probabilities(self, current_regime: MarketRegimeType) -> Dict[str, float]:
        """Calculate regime transition probabilities"""
        # Simplified transition matrix
        if current_regime == MarketRegimeType.BULL:
            return {"BULL": 0.7, "SIDEWAYS": 0.25, "BEAR": 0.04, "CRISIS": 0.01}
        elif current_regime == MarketRegimeType.BEAR:
            return {"BULL": 0.1, "SIDEWAYS": 0.3, "BEAR": 0.58, "CRISIS": 0.02}
        elif current_regime == MarketRegimeType.CRISIS:
            return {"BULL": 0.05, "SIDEWAYS": 0.15, "BEAR": 0.3, "CRISIS": 0.5}
        else:  # SIDEWAYS
            return {"BULL": 0.3, "SIDEWAYS": 0.5, "BEAR": 0.19, "CRISIS": 0.01}
    
    def get_correlation_matrix(self, assets: List[str], lookback: int = 50) -> np.ndarray:
        """Calculate real-time correlation matrix"""
        if not assets:
            return np.array([])
            
        # Get returns for all assets
        returns_matrix = []
        for symbol in assets:
            returns = self.get_returns_history(symbol, lookback)
            if len(returns) >= lookback:
                returns_matrix.append(returns[-lookback:])
            else:
                # Fill with zeros if not enough history
                returns_matrix.append([0.0] * lookback)
        
        if not returns_matrix:
            return np.eye(len(assets))
            
        returns_array = np.array(returns_matrix)
        
        # Calculate correlation matrix
        try:
            correlation_matrix = np.corrcoef(returns_array)
            
            # Handle regime-dependent correlation changes
            if self.current_regime.regime_type == MarketRegimeType.CRISIS:
                # During crisis, correlations spike higher
                correlation_matrix = correlation_matrix * 0.3 + 0.7 * np.ones_like(correlation_matrix)
                np.fill_diagonal(correlation_matrix, 1.0)
                
            return correlation_matrix
        except:
            # Return identity matrix if calculation fails
            return np.eye(len(assets))
    
    def get_factor_loadings(self, assets: List[str]) -> Dict[str, FactorExposure]:
        """Get factor model exposures"""
        return {symbol: self.factor_loadings.get(symbol, FactorExposure(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3)) 
                for symbol in assets if symbol in self.factor_loadings}
    
    def detect_byzantine_feeds(self) -> List[str]:
        """Identify corrupted/malicious data feeds"""
        corrupted = []
        
        for feed in self.data_feeds:
            # Simple anomaly detection
            if feed.reliability_score < 0.9:
                corrupted.append(feed.feed_id)
                feed.is_corrupted = True
                
        return corrupted
    
    def update_market_data(self, symbol: str, price: float, volume: int):
        """Update market data (for simulation)"""
        with self.lock:
            if symbol in self.symbols:
                old_price = self.current_prices.get(symbol, price)
                
                # Calculate return
                daily_return = (price - old_price) / old_price if old_price > 0 else 0.0
                
                # Update storage
                self.price_history[symbol].append(price)
                self.volume_history[symbol].append(volume)
                self.returns_history[symbol].append(daily_return)
                
                self.current_prices[symbol] = price
                self.current_volumes[symbol] = volume
                
                self.update_count += 1
    
    def simulate_market_update(self):
        """Simulate real-time market data updates"""
        # Update a random subset of symbols
        num_updates = random.randint(50, 200)
        symbols_to_update = random.sample(self.symbols, min(num_updates, len(self.symbols)))
        
        updates = []
        for symbol in symbols_to_update:
            current_price = self.current_prices[symbol]
            
            # Regime-dependent price movement
            vol_multiplier = {
                VolatilityRegime.LOW: 0.5,
                VolatilityRegime.MEDIUM: 1.0,
                VolatilityRegime.HIGH: 2.0,
                VolatilityRegime.EXTREME: 5.0
            }
            
            base_vol = 0.002 * vol_multiplier[self.current_regime.volatility_regime]
            price_change = np.random.normal(0, base_vol)
            
            new_price = current_price * (1 + price_change)
            new_volume = random.randint(1000, 100000)
            
            self.update_market_data(symbol, new_price, new_volume)
            
            update = MarketUpdate(
                symbol=symbol,
                price=new_price,
                volume=new_volume,
                timestamp=time.time(),
                update_type="TRADE",
                exchange_id=random.choice([feed.feed_id for feed in self.data_feeds])
            )
            updates.append(update)
            
        return updates
    
    def get_volatility_surface(self, symbol: str) -> Dict[str, float]:
        """Get implied volatility surface (simplified)"""
        returns = self.get_returns_history(symbol, 30)
        if not returns:
            return {"30d": 0.2, "60d": 0.22, "90d": 0.25}
            
        vol_30d = np.std(returns) * np.sqrt(252)
        vol_60d = vol_30d * 1.1  # Term structure
        vol_90d = vol_30d * 1.15
        
        return {
            "30d": vol_30d,
            "60d": vol_60d, 
            "90d": vol_90d
        }
    
    def calculate_beta(self, symbol: str, benchmark: str = "SPY") -> float:
        """Calculate beta relative to benchmark"""
        symbol_returns = self.get_returns_history(symbol, 100)
        benchmark_returns = self.get_returns_history(benchmark, 100)
        
        if len(symbol_returns) < 20 or len(benchmark_returns) < 20:
            return 1.0
            
        # Align lengths
        min_len = min(len(symbol_returns), len(benchmark_returns))
        symbol_returns = symbol_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        try:
            covariance = np.cov(symbol_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            return max(0.1, min(3.0, beta))  # Clamp to reasonable range
        except:
            return 1.0

# Alias for backward compatibility
MarketData = ExtremeMarketData
