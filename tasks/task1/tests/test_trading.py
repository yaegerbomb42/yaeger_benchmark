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
    
    def test_order_execution_latency(self):
        """Test order execution under various latency conditions."""
        # Test normal latency
        order = Order("AAPL", "BUY", 100, 150.0, "MARKET")
        start_time = time.time()
        result = self.exchange.place_order(order)
        latency = time.time() - start_time
        
        assert latency < 0.05, f"Order execution too slow: {latency:.3f}s"
        assert result.status in ["FILLED", "PARTIAL"], "Order should be executed"
    
    def test_high_frequency_trading(self):
        """Test algorithm performance under high-frequency conditions."""
        self.market_data.start()
        
        # Simulate rapid-fire trading decisions
        start_time = time.time()
        decisions = 0
        
        try:
            while time.time() - start_time < 1.0:  # 1 second test
                trading_algorithm(self.exchange, self.market_data, self.portfolio)
                decisions += 1
        finally:
            self.market_data.stop()
        
        # Should handle at least 100 decisions per second
        assert decisions >= 100, f"Algorithm too slow: {decisions} decisions/second"
    
    def test_market_volatility_handling(self):
        """Test algorithm behavior during high market volatility."""
        # Inject high volatility scenario
        volatile_updates = [
            MarketUpdate("AAPL", 150.0, time.time(), volume=10000),
            MarketUpdate("AAPL", 145.0, time.time(), volume=15000),  # -3.3%
            MarketUpdate("AAPL", 155.0, time.time(), volume=20000),  # +6.9%
            MarketUpdate("AAPL", 140.0, time.time(), volume=25000),  # -9.7%
        ]
        
        for update in volatile_updates:
            on_market_update(update, self.exchange, self.portfolio)
        
        # Check that algorithm didn't make excessive risky trades
        total_exposure = sum(abs(pos.quantity * pos.avg_price) 
                           for pos in self.portfolio.positions.values())
        assert total_exposure < self.portfolio.cash * 2, "Excessive exposure during volatility"
    
    def test_network_failure_recovery(self):
        """Test algorithm behavior during network/exchange failures."""
        # Mock exchange failures
        original_get_quote = self.exchange.get_quote
        failure_count = 0
        
        def failing_get_quote(symbol):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise ConnectionError("Network timeout")
            return original_get_quote(symbol)
        
        self.exchange.get_quote = failing_get_quote
        
        # Algorithm should handle failures gracefully
        try:
            trading_algorithm(self.exchange, self.market_data, self.portfolio)
        except ConnectionError:
            pytest.fail("Algorithm should handle network failures gracefully")
    
    def test_memory_usage_limits(self):
        """Test that algorithm stays within memory limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.market_data.start()
        
        # Run algorithm for extended period
        start_time = time.time()
        try:
            while time.time() - start_time < 2.0:
                trading_algorithm(self.exchange, self.market_data, self.portfolio)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Should not exceed 512MB increase
                assert memory_increase < 512, f"Memory usage too high: {memory_increase:.1f}MB"
        finally:
            self.market_data.stop()
    
    def test_concurrent_market_updates(self):
        """Test algorithm behavior with concurrent market data streams."""
        update_counts = {symbol: 0 for symbol in self.symbols}
        
        def update_counter(update):
            update_counts[update.symbol] += 1
            on_market_update(update, self.exchange, self.portfolio)
        
        # Start multiple concurrent data streams
        for symbol in self.symbols:
            market_feed = MarketData([symbol])
            market_feed.subscribe(update_counter)
            market_feed.start()
        
        time.sleep(1.0)
        
        # Stop all feeds
        for symbol in self.symbols:
            if hasattr(self.market_data, 'stop'):
                self.market_data.stop()
        
        # Should handle updates from all symbols
        for symbol, count in update_counts.items():
            assert count > 0, f"No updates received for {symbol}"
    
    def test_extreme_price_movements(self):
        """Test algorithm response to extreme price movements."""
        extreme_scenarios = [
            ("FLASH_CRASH", 100.0, 50.0),    # 50% drop
            ("SHORT_SQUEEZE", 100.0, 200.0), # 100% spike
            ("CIRCUIT_BREAKER", 100.0, 85.0), # 15% drop
        ]
        
        for scenario_name, start_price, end_price in extreme_scenarios:
            # Reset portfolio for each scenario
            self.portfolio = Portfolio(initial_cash=100000.0)
            
            # Simulate extreme price movement
            update1 = MarketUpdate("TEST", start_price, time.time(), volume=1000)
            update2 = MarketUpdate("TEST", end_price, time.time(), volume=50000)
            
            on_market_update(update1, self.exchange, self.portfolio)
            on_market_update(update2, self.exchange, self.portfolio)
            
            # Check that algorithm maintained risk controls
            max_position_value = max((abs(pos.quantity * pos.avg_price) 
                                    for pos in self.portfolio.positions.values()), 
                                   default=0)
            assert max_position_value < self.portfolio.cash, f"Excessive risk in {scenario_name}"
    
    def test_order_queue_management(self):
        """Test algorithm behavior with order queue congestion."""
        # Fill up order queue
        orders = []
        for i in range(100):
            order = Order(f"STOCK_{i%5}", "BUY", 10, 100.0 + i, "LIMIT")
            orders.append(order)
            self.exchange.place_order(order)
        
        # Algorithm should still execute efficiently
        start_time = time.time()
        trading_algorithm(self.exchange, self.market_data, self.portfolio)
        execution_time = time.time() - start_time
        
        assert execution_time < 0.1, f"Algorithm slowed by order queue: {execution_time:.3f}s"
    
    def test_multi_symbol_trading(self):
        """Test algorithm performance across multiple symbols simultaneously."""
        self.market_data.start()
        
        # Track trades per symbol
        trades_per_symbol = {symbol: 0 for symbol in self.symbols}
        
        try:
            for _ in range(50):  # 50 algorithm iterations
                trading_algorithm(self.exchange, self.market_data, self.portfolio)
                
                # Count current positions as proxy for trading activity
                for symbol in self.symbols:
                    if symbol in self.portfolio.positions:
                        trades_per_symbol[symbol] += 1
        finally:
            self.market_data.stop()
        
        # Should have some activity across multiple symbols
        active_symbols = sum(1 for count in trades_per_symbol.values() if count > 0)
        assert active_symbols >= 2, f"Algorithm only traded {active_symbols} symbols"
    
    def test_algorithmic_bias_detection(self):
        """Test for potential algorithmic biases in trading decisions."""
        # Test symbol bias
        symbol_trades = {symbol: 0 for symbol in self.symbols}
        
        self.market_data.start()
        
        try:
            for _ in range(100):
                trading_algorithm(self.exchange, self.market_data, self.portfolio)
                
                # Record trading activity
                for symbol, position in self.portfolio.positions.items():
                    if position.quantity != 0:
                        symbol_trades[symbol] += 1
        finally:
            self.market_data.stop()
        
        # Check for excessive bias toward any single symbol
        total_trades = sum(symbol_trades.values())
        if total_trades > 0:
            max_symbol_ratio = max(symbol_trades.values()) / total_trades
            assert max_symbol_ratio < 0.8, f"Excessive bias toward single symbol: {max_symbol_ratio:.2f}"
    
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
