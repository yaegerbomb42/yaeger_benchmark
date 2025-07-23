"""
Comprehensive test suite for Task 4: Distributed Cache System
Tests distributed caching with consistency, performance, and fault tolerance.
"""
import pytest
import time
import threading
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import random

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from submission import DistributedCache, CacheCluster
    from codebase.cache_node import CacheNode
    from codebase.consistent_hash import ConsistentHashRing, LoadBalancer
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestDistributedCacheBasics:
    """Test basic distributed cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nodes = ["node1:8001", "node2:8002", "node3:8003"]
        self.cache = DistributedCache(self.nodes, replication_factor=2)
    
    def test_basic_get_set_operations(self):
        """Test basic cache operations."""
        # Test set operation
        assert self.cache.set("key1", "value1") == True
        assert self.cache.set("key2", {"data": "complex_value"}) == True
        
        # Test get operation
        assert self.cache.get("key1") == "value1"
        assert self.cache.get("key2") == {"data": "complex_value"}
        assert self.cache.get("nonexistent") is None
    
    def test_delete_operations(self):
        """Test cache deletion."""
        # Set and then delete
        self.cache.set("temp_key", "temp_value")
        assert self.cache.get("temp_key") == "temp_value"
        
        assert self.cache.delete("temp_key") == True
        assert self.cache.get("temp_key") is None
        
        # Delete non-existent key
        assert self.cache.delete("nonexistent") == False
    
    def test_ttl_functionality(self):
        """Test TTL (Time To Live) functionality."""
        # Set with TTL
        self.cache.set("expiring_key", "expiring_value", ttl=2)
        
        # Should be available immediately
        assert self.cache.get("expiring_key") == "expiring_value"
        
        # Should still be available after 1 second
        time.sleep(1)
        assert self.cache.get("expiring_key") == "expiring_value"
        
        # Should expire after TTL
        time.sleep(2)
        assert self.cache.get("expiring_key") is None
    
    def test_batch_operations(self):
        """Test batch get and set operations."""
        # Batch set
        items = {
            "batch_key1": "batch_value1",
            "batch_key2": "batch_value2",
            "batch_key3": "batch_value3"
        }
        assert self.cache.batch_set(items) == True
        
        # Batch get
        keys = ["batch_key1", "batch_key2", "batch_key3", "nonexistent"]
        results = self.cache.batch_get(keys)
        
        assert results["batch_key1"] == "batch_value1"
        assert results["batch_key2"] == "batch_value2"
        assert results["batch_key3"] == "batch_value3"
        assert "nonexistent" not in results


class TestDistributedCacheConsistency:
    """Test consistency guarantees across nodes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nodes = ["node1:8001", "node2:8002", "node3:8003", "node4:8004"]
        self.cache = DistributedCache(self.nodes, replication_factor=3)
    
    def test_write_consistency(self):
        """Test that writes are consistent across replicas."""
        key = "consistency_test"
        value = "consistent_value"
        
        # Write to cache
        assert self.cache.set(key, value) == True
        
        # Read multiple times to potentially hit different nodes
        for _ in range(10):
            assert self.cache.get(key) == value
    
    def test_concurrent_writes(self):
        """Test concurrent writes to the same key."""
        key = "concurrent_key"
        
        def write_worker(worker_id):
            for i in range(10):
                value = f"worker_{worker_id}_value_{i}"
                self.cache.set(f"{key}_{worker_id}_{i}", value)
        
        # Start multiple writers
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=write_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all writes succeeded
        for worker_id in range(5):
            for i in range(10):
                expected_value = f"worker_{worker_id}_value_{i}"
                actual_value = self.cache.get(f"{key}_{worker_id}_{i}")
                assert actual_value == expected_value, f"Consistency violation for worker {worker_id}, iteration {i}"
    
    def test_read_write_consistency(self):
        """Test consistency between concurrent reads and writes."""
        key = "read_write_test"
        
        # Initial value
        self.cache.set(key, "initial_value")
        
        results = []
        
        def reader():
            for _ in range(20):
                value = self.cache.get(key)
                results.append(("read", value))
                time.sleep(0.01)
        
        def writer():
            for i in range(10):
                value = f"updated_value_{i}"
                self.cache.set(key, value)
                results.append(("write", value))
                time.sleep(0.02)
        
        # Start reader and writer threads
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        
        reader_thread.start()
        writer_thread.start()
        
        reader_thread.join()
        writer_thread.join()
        
        # Verify no stale reads after writes
        writes = [(i, result[1]) for i, result in enumerate(results) if result[0] == "write"]
        reads = [(i, result[1]) for i, result in enumerate(results) if result[0] == "read"]
        
        # Check that reads after writes don't return older values
        for write_idx, write_value in writes:
            for read_idx, read_value in reads:
                if read_idx > write_idx and read_value is not None:
                    # This read happened after the write, so it should not be older
                    write_num = int(write_value.split("_")[-1])
                    if read_value.startswith("updated_value_"):
                        read_num = int(read_value.split("_")[-1])
                        assert read_num >= write_num, f"Stale read detected: {read_value} after {write_value}"


class TestDistributedCacheFaultTolerance:
    """Test fault tolerance and node failure handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nodes = ["node1:8001", "node2:8002", "node3:8003", "node4:8004", "node5:8005"]
        self.cache = DistributedCache(self.nodes, replication_factor=3)
    
    def test_single_node_failure(self):
        """Test cache operation with single node failure."""
        # Store data
        test_data = {
            "key1": "value1",
            "key2": "value2", 
            "key3": "value3"
        }
        
        for key, value in test_data.items():
            assert self.cache.set(key, value) == True
        
        # Simulate node failure
        failed_node = self.nodes[0]
        self.cache.simulate_node_failure(failed_node)
        
        # Verify data is still accessible
        for key, expected_value in test_data.items():
            actual_value = self.cache.get(key)
            assert actual_value == expected_value, f"Data lost after node failure: {key}"
        
        # Verify writes still work
        assert self.cache.set("post_failure_key", "post_failure_value") == True
        assert self.cache.get("post_failure_key") == "post_failure_value"
    
    def test_multiple_node_failures(self):
        """Test cache operation with multiple node failures."""
        # Store critical data
        critical_data = {"critical_key": "critical_value"}
        assert self.cache.set("critical_key", "critical_value") == True
        
        # Fail multiple nodes (but less than replication factor)
        failed_nodes = self.nodes[:2]  # Fail 2 out of 5 nodes
        for node in failed_nodes:
            self.cache.simulate_node_failure(node)
        
        # Data should still be accessible
        assert self.cache.get("critical_key") == "critical_value"
        
        # New writes should still work
        assert self.cache.set("post_multi_failure", "value") == True
        assert self.cache.get("post_multi_failure") == "value"
    
    def test_node_recovery(self):
        """Test node recovery after failure."""
        # Store data
        self.cache.set("recovery_test", "recovery_value")
        
        # Fail and recover node
        failed_node = self.nodes[0]
        self.cache.simulate_node_failure(failed_node)
        
        # Verify data accessible during failure
        assert self.cache.get("recovery_test") == "recovery_value"
        
        # Recover node
        self.cache.recover_node(failed_node)
        
        # Verify data still accessible after recovery
        assert self.cache.get("recovery_test") == "recovery_value"
        
        # Verify new operations work
        assert self.cache.set("post_recovery", "post_recovery_value") == True
        assert self.cache.get("post_recovery") == "post_recovery_value"
    
    def test_cluster_expansion(self):
        """Test adding nodes to the cluster."""
        # Store initial data
        initial_data = {f"key_{i}": f"value_{i}" for i in range(10)}
        for key, value in initial_data.items():
            self.cache.set(key, value)
        
        # Add new node
        new_node = "node6:8006"
        assert self.cache.add_node(new_node) == True
        
        # Verify existing data still accessible
        for key, expected_value in initial_data.items():
            assert self.cache.get(key) == expected_value
        
        # Verify new writes work with expanded cluster
        assert self.cache.set("new_cluster_key", "new_cluster_value") == True
        assert self.cache.get("new_cluster_key") == "new_cluster_value"
    
    def test_cluster_shrinking(self):
        """Test removing nodes from the cluster."""
        # Store data across cluster
        test_data = {f"shrink_key_{i}": f"shrink_value_{i}" for i in range(20)}
        for key, value in test_data.items():
            self.cache.set(key, value)
        
        # Remove a node
        node_to_remove = self.nodes[-1]
        assert self.cache.remove_node(node_to_remove) == True
        
        # Verify most data still accessible (some might be lost depending on replication)
        accessible_count = 0
        for key, expected_value in test_data.items():
            actual_value = self.cache.get(key)
            if actual_value == expected_value:
                accessible_count += 1
        
        # Should maintain majority of data
        assert accessible_count >= len(test_data) * 0.7, f"Too much data lost: {accessible_count}/{len(test_data)}"


class TestDistributedCachePerformance:
    """Test performance characteristics of the distributed cache."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nodes = ["node1:8001", "node2:8002", "node3:8003"]
        self.cache = DistributedCache(self.nodes, replication_factor=2)
    
    def test_throughput_performance(self):
        """Test cache throughput under load."""
        num_operations = 1000
        
        # Write performance test
        start_time = time.time()
        for i in range(num_operations):
            key = f"perf_key_{i}"
            value = f"perf_value_{i}"
            self.cache.set(key, value)
        write_time = time.time() - start_time
        
        write_throughput = num_operations / write_time
        print(f"Write throughput: {write_throughput:.0f} ops/sec")
        
        # Read performance test
        start_time = time.time()
        for i in range(num_operations):
            key = f"perf_key_{i}"
            value = self.cache.get(key)
            assert value == f"perf_value_{i}"
        read_time = time.time() - start_time
        
        read_throughput = num_operations / read_time
        print(f"Read throughput: {read_throughput:.0f} ops/sec")
        
        # Performance requirements (adjusted for test environment)
        assert write_throughput >= 1000, f"Write throughput too low: {write_throughput}"
        assert read_throughput >= 2000, f"Read throughput too low: {read_throughput}"
    
    def test_latency_performance(self):
        """Test cache operation latency."""
        # Warm up the cache
        for i in range(100):
            self.cache.set(f"latency_key_{i}", f"latency_value_{i}")
        
        # Measure read latency
        latencies = []
        for i in range(100):
            start_time = time.time()
            value = self.cache.get(f"latency_key_{i}")
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            assert value == f"latency_value_{i}"
        
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        print(f"Latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
        
        # Latency requirements (relaxed for test environment)
        assert p50 < 50, f"P50 latency too high: {p50}ms"
        assert p95 < 100, f"P95 latency too high: {p95}ms"
        assert p99 < 200, f"P99 latency too high: {p99}ms"
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        num_threads = 10
        operations_per_thread = 100
        
        def worker(thread_id):
            for i in range(operations_per_thread):
                key = f"concurrent_{thread_id}_{i}"
                value = f"value_{thread_id}_{i}"
                
                # Mix of reads and writes
                if i % 2 == 0:
                    self.cache.set(key, value)
                else:
                    self.cache.get(key)
        
        # Run concurrent operations
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in futures:
                future.result()
        
        total_time = time.time() - start_time
        total_operations = num_threads * operations_per_thread
        concurrent_throughput = total_operations / total_time
        
        print(f"Concurrent throughput: {concurrent_throughput:.0f} ops/sec")
        
        # Should handle concurrent load efficiently
        assert concurrent_throughput >= 500, f"Concurrent throughput too low: {concurrent_throughput}"
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        # Store large amount of data
        large_dataset_size = 1000
        
        for i in range(large_dataset_size):
            key = f"memory_test_{i}"
            value = f"memory_value_{i}" * 10  # Larger values
            self.cache.set(key, value)
        
        # Get memory statistics
        stats = self.cache.get_stats()
        total_memory_mb = stats["resource_stats"]["total_memory_usage_mb"]
        
        print(f"Memory usage for {large_dataset_size} items: {total_memory_mb:.2f} MB")
        
        # Memory should be reasonable (allowing for overhead)
        expected_max_memory = 100  # MB - rough estimate
        assert total_memory_mb < expected_max_memory, f"Memory usage too high: {total_memory_mb}MB"


class TestDistributedCacheAdvanced:
    """Test advanced distributed cache features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cluster = CacheCluster(cluster_size=3)
        self.cache = self.cluster.get_cache()
    
    def test_consistent_hashing_distribution(self):
        """Test that consistent hashing distributes keys evenly."""
        # Store many keys
        num_keys = 1000
        for i in range(num_keys):
            key = f"distribution_test_{i}"
            value = f"value_{i}"
            self.cache.set(key, value)
        
        # Get statistics to check distribution
        stats = self.cache.get_stats()
        load_balancer_stats = stats["load_balancer_stats"]
        
        # Check that requests are distributed (not perfectly even due to test environment)
        node_requests = []
        for node_stats in load_balancer_stats["node_stats"].values():
            node_requests.append(node_stats["requests"])
        
        if node_requests and max(node_requests) > 0:
            # Calculate distribution variance
            avg_requests = sum(node_requests) / len(node_requests)
            variance = sum((req - avg_requests) ** 2 for req in node_requests) / len(node_requests)
            
            print(f"Request distribution - Average: {avg_requests:.1f}, Variance: {variance:.1f}")
            
            # Distribution should not be extremely skewed
            assert variance < avg_requests * 2, "Request distribution too skewed"
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        # Perform various operations
        self.cache.set("stats_key1", "value1")
        self.cache.set("stats_key2", "value2")
        self.cache.get("stats_key1")
        self.cache.get("stats_key1")  # Another hit
        self.cache.get("nonexistent")  # Miss
        self.cache.delete("stats_key2")
        
        # Get statistics
        stats = self.cache.get_stats()
        
        # Verify statistics structure
        assert "cluster_stats" in stats
        assert "performance_stats" in stats
        assert "resource_stats" in stats
        assert "node_health" in stats
        
        # Verify some basic counters
        perf_stats = stats["performance_stats"]
        assert perf_stats["total_requests"] > 0
        assert perf_stats["successful_requests"] > 0
        assert perf_stats["cache_hit_rate"] >= 0
        
        cluster_stats = stats["cluster_stats"]
        assert cluster_stats["total_nodes"] == 3
        assert cluster_stats["healthy_nodes"] >= 0
    
    def test_cluster_management(self):
        """Test dynamic cluster management."""
        initial_nodes = len(self.cluster.nodes)
        
        # Expand cluster
        new_nodes = self.cluster.expand_cluster(2)
        assert len(new_nodes) == 2
        assert len(self.cluster.nodes) == initial_nodes + 2
        
        # Test operations with expanded cluster
        self.cache.set("expand_test", "expand_value")
        assert self.cache.get("expand_test") == "expand_value"
        
        # Shrink cluster
        removed_nodes = self.cluster.shrink_cluster(1)
        assert len(removed_nodes) == 1
        assert len(self.cluster.nodes) == initial_nodes + 1
        
        # Test operations with shrunk cluster
        assert self.cache.get("expand_test") == "expand_value"  # Should still exist
        self.cache.set("shrink_test", "shrink_value")
        assert self.cache.get("shrink_test") == "shrink_value"
