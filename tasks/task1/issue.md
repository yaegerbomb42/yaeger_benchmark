# Task 1: EXTREME Real-Time Multi-Asset Portfolio Optimizer

## Problem Description

You must implement an enterprise-grade, multi-asset portfolio optimization system that manages risk across 1000+ assets simultaneously while executing under sub-100μs latency constraints. The system must implement real-time regime detection, Byzantine fault tolerance, and regulatory compliance while maintaining <50MB memory usage.

## Background

You are building a mission-critical trading infrastructure for a tier-1 investment bank that requires:
1. Real-time portfolio optimization across 1000+ assets
2. Multi-regime strategy switching (bull/bear/sideways markets)
3. Byzantine fault tolerance for exchange feed failures
4. Regulatory compliance (MiFID II position reporting, Dodd-Frank risk limits)
5. Real-time backtesting with walk-forward analysis
6. Hardware security module integration for order authentication
7. Market microstructure analysis and impact modeling

## Requirements

### Functional Requirements (Must implement ALL)
- **Portfolio Optimizer**: Implement `optimize_portfolio(assets, constraints, risk_model)` 
- **Regime Detection**: Real-time market regime classification using multiple indicators
- **Risk Management**: Dynamic position sizing with real-time VaR calculation
- **Strategy Switching**: Automatic strategy selection based on market conditions
- **Backtesting Engine**: Walk-forward analysis with 252-day rolling window
- **Compliance Module**: Real-time regulatory reporting and limit monitoring
- **Order Router**: Smart order routing with dark pool integration
- **Fault Tolerance**: Handle Byzantine failures in data feeds

### EXTREME Performance Requirements
- **Ultra-Low Latency**: Each portfolio decision must complete in <100μs (microseconds!)
- **High Throughput**: Process 100,000+ market updates per second per asset
- **Memory Constraint**: Entire system must use <50MB RAM (not 512MB!)
- **Accuracy**: Achieve >95% profitable trades AND >2.5 Sharpe ratio
- **Uptime**: System must maintain 99.999% uptime (5.26 minutes downtime/year)
- **Scalability**: Handle 1000+ assets simultaneously without degradation

### EXTREME Technical Constraints
- **No Machine Learning Libraries**: Implement all ML algorithms from scratch
- **No External APIs**: All data must come through provided exchange feeds only
- **Byzantine Fault Tolerance**: System must function with up to 33% corrupted data feeds
- **Hardware Security**: All orders must be cryptographically signed
- **Memory Pool Management**: Implement custom memory allocator for zero-garbage collection
- **Lock-Free Programming**: All data structures must be lock-free for concurrency

## Starter Code

The codebase includes:
- `exchange.py`: Byzantine fault-tolerant exchange with realistic slippage and latency simulation
- `market_data.py`: Multi-regime market data generator with correlation matrices
- `portfolio.py`: Enterprise portfolio management with real-time risk analytics
- `risk_model.py`: Factor risk models with regime-dependent parameters
- `compliance.py`: Regulatory compliance monitoring (MiFID II, Dodd-Frank)
- `hardware_security.py`: HSM integration for order authentication
- `submission.py`: Your implementation goes here

## API Reference

### Exchange Class (Enhanced)
```python
class Exchange:
    def get_quote(self, symbol: str) -> Quote:
        """Get current bid/ask for symbol - may return corrupted data"""
        
    def get_market_depth(self, symbol: str, levels: int = 10) -> OrderBook:
        """Get full order book depth"""
        
    def place_order(self, order: SignedOrder) -> OrderResult:
        """Place cryptographically signed order"""
        
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions across all venues"""
        
    def get_dark_pool_indication(self, symbol: str) -> Optional[DarkPoolHint]:
        """Get dark pool liquidity hints"""
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
    def get_execution_quality_metrics(self) -> ExecutionMetrics:
        """Real-time execution quality analysis"""

### Portfolio Class (Enhanced)
```python
class Portfolio:
    def optimize_allocation(self, assets: List[str], 
                          risk_budget: float,
                          regime: MarketRegime) -> Dict[str, float]:
        """Real-time portfolio optimization"""
        
    def calculate_risk_metrics(self) -> RiskMetrics:
        """VaR, CVaR, maximum drawdown, etc."""
        
    def check_regulatory_limits(self) -> List[ComplianceViolation]:
        """Real-time compliance monitoring"""
        
    def get_attribution_analysis(self) -> AttributionReport:
        """Performance attribution by factor"""

### Market Data Class (Enhanced)  
```python
class MarketData:
    def get_regime_classification(self) -> MarketRegime:
        """Current market regime (bull/bear/sideways)"""
        
    def get_correlation_matrix(self, assets: List[str]) -> np.ndarray:
        """Real-time correlation matrix"""
        
    def get_factor_loadings(self, assets: List[str]) -> Dict[str, FactorExposure]:
        """Factor model exposures"""
        
    def detect_byzantine_feeds(self) -> List[str]:
        """Identify corrupted/malicious data feeds"""
        """Get current bid/ask for symbol"""
        
    def place_order(self, order: Order) -> OrderResult:
        """Place buy/sell order"""
        
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
```

### Key Data Structures (Enhanced)
```python
@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: float
    exchange_id: str
    is_corrupted: bool  # Byzantine fault detection
    confidence_score: float

@dataclass
class SignedOrder:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    order_type: str  # "MARKET", "LIMIT", "ICEBERG", "TWAP"
    signature: str  # Cryptographic signature
    venue_preferences: List[str]  # Dark pools, lit venues
    execution_strategy: str  # "AGGRESSIVE", "PASSIVE", "HIDDEN"

@dataclass 
class RiskMetrics:
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR 
    max_drawdown: float
    sharpe_ratio: float
    beta_market: float
    tracking_error: float
    leverage: float

@dataclass
class MarketRegime:
    regime_type: str  # "BULL", "BEAR", "SIDEWAYS", "CRISIS"
    confidence: float
    transition_probability: Dict[str, float]
    expected_duration_days: int
    volatility_regime: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"

@dataclass
class ComplianceViolation:
    rule_name: str  # "MIFID_POSITION_LIMIT", "DODD_FRANK_PROP_TRADING"
    severity: str  # "WARNING", "CRITICAL", "BREACH"
    current_value: float
    limit_value: float
    time_to_compliance: int  # seconds until must be resolved
    
@dataclass
class FactorExposure:
    market_factor: float
    size_factor: float
    value_factor: float
    momentum_factor: float
    volatility_factor: float
    quality_factor: float
    residual_risk: float
```

## Evaluation Criteria (EXTREME DIFFICULTY)

### Correctness (40 points)
- Portfolio optimization convergence (10 points)
- Regime detection accuracy >90% (10 points) 
- Byzantine fault tolerance (10 points)
- Regulatory compliance (10 points)

### Performance (30 points)
- Sub-100μs latency under load (15 points)
- Memory usage <50MB (10 points)
- 100K+ updates/sec processing (5 points)

### Scalability (20 points)
- 1000+ asset handling (10 points)
- Walk-forward backtesting (5 points)
- Real-time risk calculation (5 points)

### Innovation (10 points)
- Novel optimization algorithms (5 points)
- Advanced microstructure modeling (3 points)
- Creative risk management (2 points)

## Test Scenarios (EXTREME)

1. **Market Crash Simulation**: 2008-style crisis with 40% drop, correlation breakdown
2. **Flash Crash**: Microsecond-level price disruptions with liquidity evaporation  
3. **Byzantine Attack**: 30% of data feeds providing malicious/corrupted data
4. **Regulatory Stress**: Real-time compliance under changing position limits
5. **High Frequency Competition**: Compete against 100+ other HFT algorithms
6. **Multi-Venue Arbitrage**: Cross-venue opportunities with venue-specific quirks
7. **Correlation Breakdown**: During crisis, correlations spike to 0.8+ unexpectedly
8. **Memory Pressure**: System memory artificially constrained to 25MB mid-execution

## Sample Strategy Ideas

- **Mean Reversion**: Trade against short-term price movements
- **Momentum**: Follow strong directional moves
- **Arbitrage**: Exploit price differences between correlated assets
- **Market Making**: Provide liquidity for small profits

## Submission Format (EXTREME IMPLEMENTATION REQUIRED)

Implement your solution in `submission.py` - ALL functions must be implemented:

```python
class ExtremePortfolioOptimizer:
    """Enterprise-grade portfolio optimization system"""
    
    def __init__(self, max_assets: int = 1000, memory_limit_mb: int = 50):
        """Initialize with extreme constraints"""
        
    def optimize_portfolio(self, assets: List[str], market_data: MarketData, 
                          risk_model: RiskModel, regime: MarketRegime) -> Dict[str, float]:
        """
        Real-time portfolio optimization under extreme constraints.
        MUST complete in <100μs for 1000 assets.
        """
        
    def detect_market_regime(self, price_history: np.ndarray, 
                           volume_history: np.ndarray) -> MarketRegime:
        """
        Real-time regime classification using multiple indicators.
        Must achieve >90% accuracy on regime transitions.
        """
        
    def handle_byzantine_feeds(self, feeds: List[DataFeed]) -> List[DataFeed]:
        """
        Detect and filter corrupted/malicious data feeds.
        Must handle up to 33% Byzantine failures.
        """
        
    def calculate_real_time_risk(self, positions: Dict[str, float], 
                               market_data: MarketData) -> RiskMetrics:
        """
        Calculate VaR, CVaR, and other risk metrics in real-time.
        Must update every microsecond without blocking trading.
        """
        
    def execute_compliance_monitoring(self, positions: Dict[str, float]) -> List[ComplianceViolation]:
        """
        Real-time regulatory compliance monitoring.
        Must check MiFID II and Dodd-Frank requirements.
        """
        
    def run_walk_forward_backtest(self, strategy: TradingStrategy, 
                                 start_date: datetime, end_date: datetime) -> BacktestResults:
        """
        Walk-forward analysis with 252-day rolling window.
        Must complete full backtest in <10 seconds.
        """

def trading_algorithm(exchange: Exchange, market_data: MarketData, 
                     portfolio: Portfolio, optimizer: ExtremePortfolioOptimizer) -> None:
    """
    Main trading algorithm - orchestrates all components.
    
    EXTREME REQUIREMENTS:
    - <100μs latency per decision
    - Handle 1000+ assets simultaneously  
    - Maintain >95% profitability
    - Achieve >2.5 Sharpe ratio
    - Use <50MB memory total
    - 99.999% uptime
    """
    # Your implementation here
    pass
```

## Tips for EXTREME Success

1. **Microsecond Optimization**: Use memory pools, SIMD instructions, lock-free data structures
2. **Regime Detection**: Implement HMM or neural networks for market state classification  
3. **Portfolio Theory**: Implement Black-Litterman, risk parity, or mean-variance optimization
4. **Byzantine Tolerance**: Use consensus algorithms or statistical outlier detection
5. **Regulatory Compliance**: Implement real-time position monitoring and automatic limits
6. **Backtesting Framework**: Implement vectorized operations for speed
7. **Memory Management**: Custom allocators, object pooling, zero-copy operations
8. **Concurrency**: Lock-free queues, atomic operations, NUMA-aware algorithms

## Common Pitfalls (EXTREME EDITION)

- Underestimating memory constraints (50MB is VERY tight for 1000 assets)
- Not implementing proper Byzantine fault detection
- Ignoring regulatory requirements until it's too late
- Using standard libraries instead of custom optimized implementations
- Not profiling for microsecond-level performance
- Failing to handle regime transitions gracefully
- Memory allocations in tight loops (causes GC pressure)
- Not implementing proper error recovery mechanisms
- Underestimating the difficulty of walk-forward backtesting
- Not considering market microstructure effects

## Advanced Requirements

### Performance Benchmarks
- Single asset optimization: <1μs
- 1000 asset optimization: <100μs  
- Regime detection: <10μs
- Risk calculation: <5μs
- Compliance check: <2μs

### Memory Allocation Breakdown
- Portfolio weights: <5MB
- Market data buffers: <15MB
- Risk model parameters: <10MB
- Historical data: <10MB
- Working memory: <10MB

### Algorithmic Requirements
- Custom implementation of matrix operations (no NumPy/SciPy)
- Lock-free data structures for all shared state
- SIMD vectorization for mathematical operations
- Custom memory allocator with object pooling
- Statistical models implemented from scratch
