"""
Comprehensive test suite for Task 3: Dynamic Data Pipeline
Tests stream processing with high throughput requirements.
"""
import pytest
import asyncio
import time
import threading
import random
import json
import sys
import os
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
import queue

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from submission import DataPipeline, StreamProcessor, DataTransformer
    from codebase.stream_source import StreamSource, DataRecord
    from codebase.stream_sink import StreamSink
    from codebase.pipeline_metrics import PipelineMetrics
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestDataPipelinePerformance:
    """Test suite focusing on performance requirements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline()
        self.test_records = self._generate_test_records(1000)
        
    def _generate_test_records(self, count):
        """Generate test data records."""
        records = []
        for i in range(count):
            record = DataRecord(
                id=f"record_{i}",
                timestamp=time.time(),
                data={
                    "user_id": random.randint(1, 10000),
                    "event_type": random.choice(["click", "view", "purchase", "signup"]),
                    "value": random.uniform(0, 1000),
                    "metadata": {"source": "test", "batch": i // 100}
                }
            )
            records.append(record)
        return records
    
    def test_throughput_requirement_1m_per_second(self):
        """Test that pipeline can handle 1M records per second."""
        # Test with smaller batch to avoid timeout, but measure rate
        batch_size = 10000
        test_records = self._generate_test_records(batch_size)
        
        start_time = time.time()
        processed_count = self.pipeline.process_batch(test_records)
        end_time = time.time()
        
        processing_time = end_time - start_time
        records_per_second = processed_count / processing_time
        
        # Should process at least 100K records/second (scaled down for testing)
        assert records_per_second >= 100000, f"Throughput too low: {records_per_second:.0f} records/sec"
        assert processed_count == batch_size, f"Lost records: {batch_size - processed_count}"
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset in chunks
        large_dataset = self._generate_test_records(100000)
        
        for chunk in self._chunk_data(large_dataset, 10000):
            self.pipeline.process_batch(chunk)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            # Memory should not grow beyond 1GB
            assert memory_increase < 1024, f"Memory usage too high: {memory_increase:.1f}MB"
    
    def _chunk_data(self, data, chunk_size):
        """Split data into chunks."""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    def test_concurrent_stream_processing(self):
        """Test concurrent processing of multiple streams."""
        stream_count = 5
        records_per_stream = 1000
        
        def process_stream(stream_id):
            records = self._generate_test_records(records_per_stream)
            start_time = time.time()
            processed = self.pipeline.process_stream(stream_id, records)
            return processed, time.time() - start_time
        
        # Process multiple streams concurrently
        with ThreadPoolExecutor(max_workers=stream_count) as executor:
            futures = [executor.submit(process_stream, f"stream_{i}") 
                      for i in range(stream_count)]
            
            results = [future.result(timeout=30) for future in futures]
        
        # All streams should complete successfully
        for processed, duration in results:
            assert processed == records_per_stream, "Stream processing incomplete"
            assert duration < 10.0, f"Stream took too long: {duration:.2f}s"
    
    def test_backpressure_handling(self):
        """Test pipeline behavior under backpressure."""
        # Create a slow sink to simulate backpressure
        slow_sink = Mock()
        slow_sink.write.side_effect = lambda x: time.sleep(0.01)  # 10ms delay per record
        
        self.pipeline.add_sink(slow_sink)
        
        # Send records faster than sink can handle
        fast_records = self._generate_test_records(100)
        
        start_time = time.time()
        result = self.pipeline.process_batch(fast_records)
        duration = time.time() - start_time
        
        # Pipeline should handle backpressure gracefully
        assert result > 0, "Pipeline should process some records despite backpressure"
        # Should implement queuing or buffering, not just fail
        assert duration < 30.0, "Pipeline should not hang indefinitely"


class TestDataTransformation:
    """Test suite for data transformation logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = DataTransformer()
    
    def test_data_validation_and_cleaning(self):
        """Test data validation and cleaning operations."""
        dirty_records = [
            {"user_id": None, "value": "invalid"},           # Null/invalid data
            {"user_id": "123", "value": -50, "email": ""},   # Negative value, empty email
            {"user_id": 456, "value": 999999999},            # Extremely large value
            {"user_id": 789, "value": 50.5, "extra": "ok"}, # Valid record with extra field
            {},                                              # Empty record
            {"user_id": "abc", "timestamp": "not-a-date"},   # Invalid timestamp
        ]
        
        cleaned = self.transformer.clean_batch(dirty_records)
        
        # Should filter out invalid records and clean valid ones
        assert len(cleaned) <= len(dirty_records), "Cleaning should not increase record count"
        
        # All cleaned records should be valid
        for record in cleaned:
            assert "user_id" in record, "Missing required field"
            assert isinstance(record.get("user_id"), (int, str)), "Invalid user_id type"
            if "value" in record:
                assert isinstance(record["value"], (int, float)), "Invalid value type"
                assert record["value"] >= 0, "Value should be non-negative"
    
    def test_schema_evolution_handling(self):
        """Test handling of schema changes over time."""
        # Simulate old schema records
        old_schema_records = [
            {"id": 1, "name": "John", "age": 25},
            {"id": 2, "name": "Jane", "age": 30}
        ]
        
        # Simulate new schema records
        new_schema_records = [
            {"user_id": 3, "first_name": "Bob", "last_name": "Smith", "birth_year": 1990},
            {"user_id": 4, "first_name": "Alice", "last_name": "Johnson", "birth_year": 1985}
        ]
        
        # Transformer should handle both schemas
        old_result = self.transformer.transform_batch(old_schema_records)
        new_result = self.transformer.transform_batch(new_schema_records)
        
        assert len(old_result) == len(old_schema_records), "Should process old schema"
        assert len(new_result) == len(new_schema_records), "Should process new schema"
    
    def test_data_enrichment(self):
        """Test data enrichment capabilities."""
        base_records = [
            {"user_id": 123, "event": "purchase", "amount": 50.0},
            {"user_id": 456, "event": "view", "product_id": "ABC123"}
        ]
        
        enriched = self.transformer.enrich_batch(base_records)
        
        # Should add derived fields
        for i, record in enumerate(enriched):
            assert "timestamp" in record, "Should add timestamp"
            assert "processed_at" in record, "Should add processing metadata"
            
            # Should preserve original data
            for key, value in base_records[i].items():
                assert record[key] == value, f"Original data modified: {key}"
    
    def test_aggregation_operations(self):
        """Test aggregation and windowing operations."""
        time_series_data = []
        base_time = time.time()
        
        # Generate 1 hour of data (1 minute intervals)
        for i in range(60):
            record = {
                "timestamp": base_time + (i * 60),  # Every minute
                "user_id": random.randint(1, 10),
                "value": random.uniform(10, 100)
            }
            time_series_data.append(record)
        
        # Test 5-minute windowed aggregation
        windows = self.transformer.window_aggregate(time_series_data, window_size=300)  # 5 minutes
        
        assert len(windows) <= 12, "Should create ~12 windows for 1 hour of data"
        
        for window in windows:
            assert "start_time" in window, "Window should have start time"
            assert "end_time" in window, "Window should have end time"
            assert "count" in window, "Window should have record count"
            assert "sum" in window or "avg" in window, "Window should have aggregated values"
    
    def test_duplicate_detection_and_deduplication(self):
        """Test duplicate record detection and handling."""
        records_with_duplicates = [
            {"id": "A", "value": 100, "timestamp": 1000},
            {"id": "B", "value": 200, "timestamp": 2000},
            {"id": "A", "value": 100, "timestamp": 1000},  # Exact duplicate
            {"id": "A", "value": 150, "timestamp": 1500},  # Same ID, different data
            {"id": "C", "value": 300, "timestamp": 3000},
        ]
        
        deduplicated = self.transformer.deduplicate(records_with_duplicates)
        
        # Should remove exact duplicates but keep updated records
        unique_ids = set(record["id"] for record in deduplicated)
        assert len(unique_ids) == 3, "Should have 3 unique IDs (A, B, C)"
        
        # Should keep the most recent version of each ID
        id_a_records = [r for r in deduplicated if r["id"] == "A"]
        if len(id_a_records) == 1:
            assert id_a_records[0]["timestamp"] == 1500, "Should keep most recent version"


class TestStreamingErrorHandling:
    """Test suite for error handling and recovery."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline()
    
    def test_malformed_data_handling(self):
        """Test handling of malformed or corrupted data."""
        malformed_data = [
            b"corrupted binary data",
            '{"invalid": json syntax',
            {"nested": {"too": {"deep": {"structure": {"for": {"processing": True}}}}}},
            None,
            42,  # Wrong data type
            "",  # Empty string
            {"huge_field": "x" * 1000000},  # Extremely large field
        ]
        
        # Pipeline should not crash on malformed data
        try:
            result = self.pipeline.process_batch(malformed_data)
            # Should either process what it can or return 0
            assert isinstance(result, int), "Should return processing count"
            assert result >= 0, "Processing count should be non-negative"
        except Exception as e:
            pytest.fail(f"Pipeline crashed on malformed data: {e}")
    
    def test_network_failure_simulation(self):
        """Test pipeline behavior during network failures."""
        # Mock network sink that fails intermittently
        failing_sink = Mock()
        failure_count = 0
        
        def simulate_network_failure(*args):
            nonlocal failure_count
            failure_count += 1
            if failure_count % 3 == 0:  # Fail every 3rd call
                raise ConnectionError("Network timeout")
            return True
        
        failing_sink.write.side_effect = simulate_network_failure
        self.pipeline.add_sink(failing_sink)
        
        test_records = [{"id": i, "data": f"record_{i}"} for i in range(10)]
        
        # Pipeline should handle network failures gracefully
        result = self.pipeline.process_batch(test_records, retry_on_failure=True)
        
        # Should process most records despite some failures
        assert result >= len(test_records) * 0.5, "Should handle network failures with retries"
    
    def test_memory_pressure_handling(self):
        """Test pipeline behavior under memory pressure."""
        # Simulate memory pressure by processing very large records
        large_records = []
        for i in range(100):
            large_record = {
                "id": i,
                "large_data": "x" * 100000,  # 100KB per record
                "metadata": {"size": "large", "index": i}
            }
            large_records.append(large_record)
        
        # Pipeline should handle large records without OOM
        try:
            result = self.pipeline.process_batch(large_records)
            assert result >= 0, "Should process large records successfully"
        except MemoryError:
            pytest.fail("Pipeline should handle memory pressure gracefully")
    
    def test_downstream_service_failures(self):
        """Test handling of downstream service failures."""
        # Mock multiple downstream services with different failure modes
        services = {
            "database": Mock(),
            "cache": Mock(),
            "analytics": Mock()
        }
        
        # Configure different failure scenarios
        services["database"].write.side_effect = Exception("Database connection lost")
        services["cache"].write.return_value = False  # Soft failure
        services["analytics"].write.return_value = True  # Success
        
        for name, service in services.items():
            self.pipeline.add_sink(service, name=name)
        
        test_records = [{"id": i, "value": i * 10} for i in range(5)]
        
        # Pipeline should continue processing despite partial failures
        result = self.pipeline.process_batch(test_records, fail_on_error=False)
        
        # Should succeed for at least the working services
        assert result > 0, "Should process records to working services"
    
    def test_poison_message_handling(self):
        """Test handling of poison messages that cause processing failures."""
        poison_messages = [
            {"id": "poison_1", "action": "infinite_loop"},
            {"id": "poison_2", "action": "memory_bomb"},
            {"id": "poison_3", "action": "cpu_intensive"},
        ]
        
        normal_messages = [
            {"id": "normal_1", "value": 100},
            {"id": "normal_2", "value": 200},
        ]
        
        mixed_batch = poison_messages + normal_messages
        
        # Pipeline should isolate poison messages and process normal ones
        result = self.pipeline.process_batch(mixed_batch, 
                                           isolate_failures=True,
                                           timeout_per_record=1.0)
        
        # Should process at least the normal messages
        assert result >= len(normal_messages), "Should process normal messages despite poison ones"


class TestPipelineMetricsAndMonitoring:
    """Test suite for metrics and monitoring capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline()
        self.metrics = PipelineMetrics()
    
    def test_throughput_metrics(self):
        """Test throughput measurement and reporting."""
        test_records = [{"id": i} for i in range(1000)]
        
        start_time = time.time()
        self.pipeline.process_batch(test_records)
        end_time = time.time()
        
        # Get metrics
        metrics = self.metrics.get_current_metrics()
        
        assert "throughput" in metrics, "Should report throughput metrics"
        assert "records_processed" in metrics, "Should count processed records"
        assert "processing_time" in metrics, "Should measure processing time"
        
        # Verify throughput calculation
        expected_throughput = len(test_records) / (end_time - start_time)
        reported_throughput = metrics["throughput"]
        
        # Allow 10% variance in throughput measurement
        assert abs(reported_throughput - expected_throughput) / expected_throughput < 0.1
    
    def test_error_rate_tracking(self):
        """Test error rate monitoring."""
        # Mix of successful and failing records
        test_batch = [
            {"id": 1, "status": "success"},
            {"id": 2, "status": "fail"},
            {"id": 3, "status": "success"},
            {"id": 4, "status": "fail"},
            {"id": 5, "status": "success"},
        ]
        
        self.pipeline.process_batch(test_batch)
        metrics = self.metrics.get_current_metrics()
        
        assert "error_rate" in metrics, "Should track error rate"
        assert "total_errors" in metrics, "Should count total errors"
        
        # Error rate should be calculated correctly
        expected_error_rate = 2 / 5  # 2 failures out of 5 records
        actual_error_rate = metrics["error_rate"]
        
        assert abs(actual_error_rate - expected_error_rate) < 0.01, "Error rate calculation incorrect"
    
    def test_latency_percentile_tracking(self):
        """Test latency percentile monitoring."""
        # Process multiple batches to build latency distribution
        for batch_size in [100, 500, 1000, 200, 800]:
            test_records = [{"id": i} for i in range(batch_size)]
            self.pipeline.process_batch(test_records)
        
        metrics = self.metrics.get_latency_percentiles()
        
        required_percentiles = ["p50", "p95", "p99"]
        for percentile in required_percentiles:
            assert percentile in metrics, f"Should report {percentile} latency"
            assert metrics[percentile] > 0, f"{percentile} latency should be positive"
        
        # p99 should be >= p95 should be >= p50
        assert metrics["p99"] >= metrics["p95"] >= metrics["p50"]
    
    def test_resource_utilization_monitoring(self):
        """Test CPU and memory utilization monitoring."""
        # Process workload while monitoring resources
        large_batch = [{"id": i, "data": "x" * 1000} for i in range(10000)]
        
        self.pipeline.process_batch(large_batch)
        metrics = self.metrics.get_resource_metrics()
        
        assert "cpu_usage" in metrics, "Should monitor CPU usage"
        assert "memory_usage" in metrics, "Should monitor memory usage"
        
        # Resource usage should be reasonable
        assert 0 <= metrics["cpu_usage"] <= 100, "CPU usage should be 0-100%"
        assert metrics["memory_usage"] > 0, "Memory usage should be positive"
    
    def test_alerts_and_thresholds(self):
        """Test alerting when metrics exceed thresholds."""
        # Configure thresholds
        self.metrics.set_threshold("error_rate", 0.1)  # 10% max error rate
        self.metrics.set_threshold("latency_p99", 1000)  # 1 second max p99 latency
        
        # Create scenario that should trigger alerts
        high_error_batch = [{"id": i, "force_error": True} for i in range(100)]
        
        self.pipeline.process_batch(high_error_batch)
        alerts = self.metrics.get_active_alerts()
        
        # Should generate alerts for high error rate
        error_rate_alerts = [a for a in alerts if a["metric"] == "error_rate"]
        assert len(error_rate_alerts) > 0, "Should alert on high error rate"
