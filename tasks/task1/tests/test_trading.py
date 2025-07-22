import pytest
import time
import threading
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codebase.exchange import Exchange, Order, Quote
from codebase.market_data import MarketData, MarketUpdate
from codebase.portfolio import Portfolio
from submission import trading_algorithm, on_market_update

class TestTradingAlgorithm:
    """Test suite for the trading algorithm implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exchange = Exchange()
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        self.market_data = MarketData(self.symbols)
        self.portfolio = Portfolio(initial_cash=100000.0)
        
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.market_data, 'stop'):
            self.market_data.stop()
    
    def test_algorithm_basic_functionality(self):
        """Test that the algorithm runs without errors."""
        # Start market data feed
        self.market_data.start()
        
        try:
            # Run the algorithm
            start_time = time.time()
            trading_algorithm(self.exchange, self.market_data, self.portfolio)
            execution_time = time.time() - start_time
            
            # Check latency requirement (<100ms)
            assert execution_time < 0.1, f"Algorithm took {execution_time:.3f}s, exceeds 100ms limit"
            
        finally:
            self.market_data.stop()
    
    def test_algorithm_handles_market_updates(self):
        """Test that the algorithm properly handles market data updates."""
        update_count = 0
        
        def mock_callback(update):
            nonlocal update_count
            update_count += 1
            on_market_update(update, self.exchange, self.portfolio)
        
        self.market_data.subscribe(mock_callback)
        self.market_data.start()
        
        # Let it run for a short time
        time.sleep(0.5)
        self.market_data.stop()
        
        # Should have received multiple updates
        assert update_count > 10, f"Only received {update_count} market updates"
    
    def test_risk_management(self):
        """Test that the algorithm respects risk limits."""
        # Set up a scenario that should trigger risk management
        self.portfolio.max_daily_loss = 1000.0
        
        # Simulate some losing trades
        self.portfolio.execute_trade("AAPL", 100, 150.0, "BUY")
        self.portfolio.execute_trade("AAPL", 100, 140.0, "SELL")  # $1000 loss
        
        current_prices = {"AAPL": 140.0}
        violations = self.portfolio.check_risk_limits(current_prices)
        
        # Should detect the loss limit violation
        assert len(violations) > 0, "Risk management should detect loss limit violation"
    
    def test_position_limits(self):
        """Test that position limits are respected."""
        # Try to exceed position limit
        large_quantity = self.portfolio.max_position_size + 100
        current_price = 150.0
        
        can_buy = self.portfolio.can_buy("AAPL", large_quantity, current_price)
        assert not can_buy, "Should not allow buying beyond position limits"
    
    def test_cash_management(self):
        """Test that cash limits are respected."""
        # Try to buy more than available cash
        expensive_quantity = 1000
        expensive_price = 1000.0  # $1M total cost
        
        can_buy = self.portfolio.can_buy("AAPL", expensive_quantity, expensive_price)
        assert not can_buy, "Should not allow buying beyond available cash"
    
    def test_order_execution(self):
        """Test that orders are executed correctly."""
        # Place a market buy order
        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=0.0,  # Market order
            order_type="MARKET"
        )
        
        result = self.exchange.place_order(order)
        
        assert result.status in ["FILLED", "PARTIAL"], f"Order status: {result.status}"
        assert result.filled_quantity > 0, "Order should have some fill"
        assert result.order_id is not None, "Order should have an ID"
    
    def test_quote_retrieval(self):
        """Test that market quotes can be retrieved."""
        for symbol in self.symbols:
            quote = self.exchange.get_quote(symbol)
            
            assert quote is not None, f"Should get quote for {symbol}"
            assert quote.symbol == symbol, f"Quote symbol mismatch"
            assert quote.bid > 0, f"Bid price should be positive for {symbol}"
            assert quote.ask > quote.bid, f"Ask should be higher than bid for {symbol}"
    
    def test_portfolio_tracking(self):
        """Test that portfolio is tracked correctly."""
        initial_cash = self.portfolio.cash
        
        # Execute a trade
        self.portfolio.execute_trade("AAPL", 100, 150.0, "BUY")
        
        # Check portfolio state
        assert self.portfolio.cash < initial_cash, "Cash should decrease after purchase"
        assert self.portfolio.get_position("AAPL") == 100, "Position should be updated"
        
        # Sell the position
        self.portfolio.execute_trade("AAPL", 100, 155.0, "SELL")
        
        assert self.portfolio.get_position("AAPL") == 0, "Position should be closed"
        assert self.portfolio.cash > initial_cash, "Should have profit"
    
    def test_performance_under_load(self):
        """Test algorithm performance under high load."""
        self.market_data.start()
        
        # Run algorithm multiple times to simulate high frequency
        execution_times = []
        
        for _ in range(100):
            start_time = time.time()
            trading_algorithm(self.exchange, self.market_data, self.portfolio)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
        
        self.market_data.stop()
        
        # Check that all executions meet latency requirements
        max_time = max(execution_times)
        avg_time = sum(execution_times) / len(execution_times)
        
        assert max_time < 0.1, f"Maximum execution time {max_time:.3f}s exceeds 100ms"
        assert avg_time < 0.05, f"Average execution time {avg_time:.3f}s too high"
    
    def test_error_handling(self):
        """Test that the algorithm handles errors gracefully."""
        # Test with invalid symbol
        invalid_quote = self.exchange.get_quote("INVALID")
        assert invalid_quote is None, "Should return None for invalid symbol"
        
        # Test selling without position
        can_sell = self.portfolio.can_sell("AAPL", 100)
        assert not can_sell, "Should not allow selling non-existent position"
    
    def test_market_volatility_handling(self):
        """Test algorithm behavior during high volatility."""
        self.market_data.start()
        
        # Simulate volatility spike
        self.market_data.simulate_market_event("volatility_spike")
        
        # Run algorithm during volatility
        start_time = time.time()
        trading_algorithm(self.exchange, self.market_data, self.portfolio)
        execution_time = time.time() - start_time
        
        self.market_data.stop()
        
        # Algorithm should still meet latency requirements
        assert execution_time < 0.1, f"Algorithm should handle volatility within latency limits"
    
    @pytest.mark.timeout(30)
    def test_memory_usage(self):
        """Test that memory usage stays within limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.market_data.start()
        
        # Run algorithm many times to test for memory leaks
        for _ in range(1000):
            trading_algorithm(self.exchange, self.market_data, self.portfolio)
        
        self.market_data.stop()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 256, f"Memory usage increased by {memory_increase:.1f}MB, exceeds 256MB limit"
    
    def test_profitability_simulation(self):
        """Test algorithm profitability over a simulation period."""
        self.market_data.start()
        
        initial_value = self.portfolio.get_total_value(self.market_data.get_all_prices())
        
        # Run simulation for a period
        for _ in range(100):
            trading_algorithm(self.exchange, self.market_data, self.portfolio)
            time.sleep(0.01)  # 10ms between decisions
        
        self.market_data.stop()
        
        final_value = self.portfolio.get_total_value(self.market_data.get_all_prices())
        pnl = self.portfolio.calculate_pnl(self.market_data.get_all_prices())
        
        # Algorithm should at least preserve capital (not lose more than 5%)
        assert final_value > initial_value * 0.95, f"Algorithm lost too much money: {pnl['total_pnl']:.2f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
