# Task 1: Real-Time Trading Optimizer

## Problem Description

You must implement a high-frequency trading algorithm for a simulated stock exchange that maximizes profit while operating under strict latency constraints. The algorithm must process trading opportunities and execute orders within 100ms per trade decision.

## Background

You are building a trading system for a proprietary trading firm that needs to:
1. Process incoming market data in real-time
2. Identify profitable trading opportunities
3. Execute trades with minimal latency
4. Handle race conditions and market volatility
5. Maintain risk management controls

## Requirements

### Functional Requirements
- Implement `trading_algorithm(exchange, market_data, portfolio)` function
- Process market data updates and execute profitable trades
- Maintain position limits (max 1000 shares per symbol)
- Implement stop-loss protection (max 5% loss per trade)
- Handle multiple symbols simultaneously

### Performance Requirements
- **Latency**: Each trade decision must complete in <100ms
- **Throughput**: Handle 1000+ market updates per second
- **Memory**: Use <512MB RAM during execution
- **Accuracy**: Achieve >80% profitable trades in backtesting

### Technical Constraints
- Use only the provided Exchange API
- No external market data sources
- No pre-computed lookup tables
- Must handle network failures gracefully

## Starter Code

The codebase includes:
- `exchange.py`: Simulated exchange with realistic latency and slippage
- `market_data.py`: Market data generator with realistic price movements
- `portfolio.py`: Portfolio management utilities
- `submission.py`: Your implementation goes here

## API Reference

### Exchange Class
```python
class Exchange:
    def get_quote(self, symbol: str) -> Quote:
        """Get current bid/ask for symbol"""
        
    def place_order(self, order: Order) -> OrderResult:
        """Place buy/sell order"""
        
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
```

### Key Data Structures
```python
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
```

## Evaluation Criteria

### Correctness (70 points)
- Pass all unit tests (30 points)
- Maintain risk limits (20 points)
- Handle edge cases (20 points)

### Performance (20 points)
- Average latency <100ms (10 points)
- Memory usage <512MB (5 points)
- Handle 1000+ updates/sec (5 points)

### Security (10 points)
- No buffer overflows
- Proper error handling
- Input validation

## Test Scenarios

1. **Normal Market Conditions**: Steady price movements, low volatility
2. **High Volatility**: Rapid price changes, large spreads
3. **Market Stress**: Network delays, partial fills, order rejections
4. **Edge Cases**: Market close, halted symbols, extreme prices

## Sample Strategy Ideas

- **Mean Reversion**: Trade against short-term price movements
- **Momentum**: Follow strong directional moves
- **Arbitrage**: Exploit price differences between correlated assets
- **Market Making**: Provide liquidity for small profits

## Submission Format

Implement your solution in `submission.py`:

```python
def trading_algorithm(exchange: Exchange, market_data: MarketData, portfolio: Portfolio) -> None:
    """
    Main trading algorithm entry point.
    
    Args:
        exchange: Exchange API for trading operations
        market_data: Live market data feed
        portfolio: Current portfolio state
    """
    # Your implementation here
    pass
```

## Tips for Success

1. **Start Simple**: Implement basic buy-low-sell-high logic first
2. **Optimize Gradually**: Profile your code to find bottlenecks
3. **Test Thoroughly**: Use the provided backtesting framework
4. **Handle Failures**: Implement robust error handling
5. **Monitor Risk**: Always check position limits and stop-losses

## Common Pitfalls

- Ignoring latency in decision logic
- Not handling partial fills
- Over-optimizing on historical data
- Insufficient risk management
- Memory leaks in tight loops
