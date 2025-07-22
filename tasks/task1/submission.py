"""
Trading Algorithm Implementation

Implement your high-frequency trading algorithm here.
Your solution will be evaluated on:
1. Profitability (primary metric)
2. Latency (<100ms per decision)
3. Risk management
4. Code quality and robustness

API Usage:
- exchange.get_quote(symbol) -> Quote
- exchange.place_order(order) -> OrderResult
- exchange.get_positions() -> Dict[str, Position]
- market_data.get_latest_price(symbol) -> float
- portfolio.can_buy/can_sell() -> bool
"""

from codebase.exchange import Exchange, Order
from codebase.market_data import MarketData, MarketUpdate
from codebase.portfolio import Portfolio
import time
from typing import Dict, List

def trading_algorithm(exchange: Exchange, market_data: MarketData, portfolio: Portfolio) -> None:
    """
    Main trading algorithm entry point.
    
    This function should implement your trading strategy. It will be called
    continuously during the simulation and must make trading decisions based
    on real-time market data.
    
    Args:
        exchange: Exchange API for trading operations
        market_data: Live market data feed
        portfolio: Current portfolio state
    
    Performance Requirements:
    - Each call must complete in <100ms
    - Handle 1000+ market updates per second
    - Maintain profitability >80% of trades
    - Use <512MB memory
    """
    
    # TODO: Implement your trading algorithm here
    
    # Example basic implementation (replace with your strategy):
    
    # Get current market prices
    current_prices = market_data.get_all_prices()
    
    # Check risk limits
    risk_violations = portfolio.check_risk_limits(current_prices)
    if risk_violations:
        print(f"Risk violations detected: {risk_violations}")
        return
    
    # Simple mean reversion strategy example
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    for symbol in symbols:
        quote = exchange.get_quote(symbol)
        if not quote:
            continue
            
        current_position = portfolio.get_position(symbol)
        
        # Simple logic: buy if price dropped, sell if price increased
        # (This is just an example - implement your own strategy!)
        
        mid_price = (quote.bid + quote.ask) / 2
        
        # You should implement sophisticated logic here, such as:
        # - Technical indicators (moving averages, RSI, etc.)
        # - Statistical arbitrage
        # - Market microstructure analysis
        # - Machine learning models
        # - Risk-adjusted position sizing
        
        pass  # Remove this and implement your strategy

def on_market_update(update: MarketUpdate, exchange: Exchange, portfolio: Portfolio) -> None:
    """
    Callback function for real-time market data updates.
    
    This function is called for every market data update and can be used
    for event-driven trading strategies.
    
    Args:
        update: Market data update (trade, quote, or news)
        exchange: Exchange API for trading operations
        portfolio: Current portfolio state
    """
    
    # TODO: Implement real-time market data processing
    
    # Example: React to large price movements
    if update.update_type == "TRADE" and update.volume > 5000:
        # Large trade detected - you might want to react
        pass
    
    elif update.update_type == "NEWS":
        # News event - might cause volatility
        pass

# Advanced Strategy Ideas to Consider:
#
# 1. Mean Reversion:
#    - Calculate rolling mean and standard deviation
#    - Buy when price is below mean - 2*std
#    - Sell when price is above mean + 2*std
#
# 2. Momentum Trading:
#    - Detect breakouts from trading ranges
#    - Follow strong directional moves
#    - Use stop-losses for risk management
#
# 3. Pairs Trading:
#    - Find correlated assets
#    - Trade the spread when it deviates from mean
#    - Market-neutral strategy
#
# 4. Market Making:
#    - Provide liquidity by placing limit orders
#    - Profit from bid-ask spread
#    - Manage inventory risk
#
# 5. Statistical Arbitrage:
#    - Use statistical models to predict short-term prices
#    - Trade based on probability distributions
#    - High-frequency, low-risk trades
#
# Remember to:
# - Always check portfolio.can_buy() and portfolio.can_sell()
# - Monitor latency and optimize for speed
# - Implement proper error handling
# - Test with different market conditions
