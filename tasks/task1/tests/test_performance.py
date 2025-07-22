import pytest
import time
import threading
from unittest.mock import Mock
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codebase.exchange import Exchange, Order
from codebase.market_data import MarketData
from codebase.portfolio import Portfolio
from submission import trading_algorithm

class TestPerformance:
    """Performance and stress tests for the trading algorithm."""
    
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
    
    @pytest.mark.timeout(60)
    def test_sustained_high_frequency(self):
        """Test algorithm under sustained high-frequency conditions."""
        self.market_data.start()
        
        execution_times = []
        error_count = 0
        
        # Run for 30 seconds at high frequency
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < 30.0:
            iteration += 1
            
            try:
                algo_start = time.time()
                trading_algorithm(self.exchange, self.market_data, self.portfolio)
                algo_time = time.time() - algo_start
                execution_times.append(algo_time)
                
            except Exception as e:
                error_count += 1
                print(f"Error in iteration {iteration}: {e}")
            
            # Target 100 Hz (10ms between calls)
            time.sleep(0.01)
        
        self.market_data.stop()
        
        # Analyze results
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            p95_time = sorted(execution_times)[int(len(execution_times) * 0.95)]
            
            print(f"Iterations: {iteration}")
            print(f"Errors: {error_count}")
            print(f"Average execution time: {avg_time*1000:.2f}ms")
            print(f"Max execution time: {max_time*1000:.2f}ms") 
            print(f"95th percentile: {p95_time*1000:.2f}ms")
            
            # Performance assertions
            assert avg_time < 0.050, f"Average execution time {avg_time*1000:.2f}ms too high"
            assert max_time < 0.100, f"Maximum execution time {max_time*1000:.2f}ms exceeds limit"
            assert p95_time < 0.080, f"95th percentile {p95_time*1000:.2f}ms too high"
            assert error_count < iteration * 0.01, f"Error rate {error_count/iteration*100:.2f}% too high"
    
    def test_concurrent_market_updates(self):
        """Test algorithm with concurrent market data updates."""
        update_count = 0
        callback_times = []
        
        def fast_callback(update):
            nonlocal update_count
            start_time = time.time()
            update_count += 1
            # Simulate some processing
            trading_algorithm(self.exchange, self.market_data, self.portfolio)
            callback_times.append(time.time() - start_time)
        
        self.market_data.subscribe(fast_callback)
        self.market_data.start()
        
        # Let it run for 10 seconds
        time.sleep(10.0)
        self.market_data.stop()
        
        print(f"Processed {update_count} market updates")
        
        if callback_times:
            avg_callback_time = sum(callback_times) / len(callback_times)
            max_callback_time = max(callback_times)
            
            print(f"Average callback time: {avg_callback_time*1000:.2f}ms")
            print(f"Max callback time: {max_callback_time*1000:.2f}ms")
            
            assert update_count > 500, f"Should process many updates, got {update_count}"
            assert avg_callback_time < 0.050, f"Callback processing too slow: {avg_callback_time*1000:.2f}ms"
            assert max_callback_time < 0.100, f"Max callback time too slow: {max_callback_time*1000:.2f}ms"
    
    def test_memory_stability(self):
        """Test for memory leaks over extended operation."""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Take baseline measurement
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.market_data.start()
        
        # Run for extended period
        for i in range(5000):
            trading_algorithm(self.exchange, self.market_data, self.portfolio)
            
            # Check memory every 1000 iterations
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - baseline_memory
                print(f"Iteration {i}: Memory usage {current_memory:.1f}MB (+{memory_growth:.1f}MB)")
                
                # Should not grow more than 100MB
                assert memory_growth < 100, f"Memory growth {memory_growth:.1f}MB exceeds limit"
        
        self.market_data.stop()
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - baseline_memory
        
        print(f"Total memory growth: {total_growth:.1f}MB")
        assert total_growth < 200, f"Total memory growth {total_growth:.1f}MB too high"
    
    def test_order_throughput(self):
        """Test order processing throughput."""
        order_count = 0
        order_times = []
        
        # Place many orders rapidly
        for i in range(1000):
            order = Order(
                symbol="AAPL",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=10,
                price=0.0,
                order_type="MARKET"
            )
            
            start_time = time.time()
            result = self.exchange.place_order(order)
            order_time = time.time() - start_time
            
            if result.status in ["FILLED", "PARTIAL"]:
                order_count += 1
                order_times.append(order_time)
        
        if order_times:
            avg_order_time = sum(order_times) / len(order_times)
            max_order_time = max(order_times)
            
            print(f"Processed {order_count} orders")
            print(f"Average order time: {avg_order_time*1000:.2f}ms")
            print(f"Max order time: {max_order_time*1000:.2f}ms")
            
            assert order_count > 500, f"Should process many orders, got {order_count}"
            assert avg_order_time < 0.020, f"Order processing too slow: {avg_order_time*1000:.2f}ms"
    
    def test_market_stress_conditions(self):
        """Test algorithm under various market stress conditions."""
        self.market_data.start()
        
        stress_tests = [
            ("normal", lambda: None),
            ("volatility_spike", lambda: self.market_data.simulate_market_event("volatility_spike")),
            ("market_crash", lambda: self.market_data.simulate_market_event("market_crash")),
        ]
        
        for condition_name, setup_func in stress_tests:
            print(f"Testing {condition_name} conditions...")
            
            # Reset portfolio for each test
            self.portfolio = Portfolio(initial_cash=100000.0)
            
            # Set up the market condition
            setup_func()
            
            execution_times = []
            error_count = 0
            
            # Run for 10 seconds under this condition
            start_time = time.time()
            iteration = 0
            
            while time.time() - start_time < 10.0:
                iteration += 1
                
                try:
                    algo_start = time.time()
                    trading_algorithm(self.exchange, self.market_data, self.portfolio)
                    algo_time = time.time() - algo_start
                    execution_times.append(algo_time)
                    
                except Exception as e:
                    error_count += 1
                
                time.sleep(0.01)  # 100 Hz
            
            # Analyze results for this condition
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                max_time = max(execution_times)
                
                print(f"  {condition_name}: {iteration} iterations, {error_count} errors")
                print(f"  Average time: {avg_time*1000:.2f}ms, Max time: {max_time*1000:.2f}ms")
                
                # Should maintain performance under all conditions
                assert avg_time < 0.100, f"Performance degraded under {condition_name}: {avg_time*1000:.2f}ms"
                assert error_count < iteration * 0.05, f"Too many errors under {condition_name}: {error_count}"
        
        self.market_data.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
