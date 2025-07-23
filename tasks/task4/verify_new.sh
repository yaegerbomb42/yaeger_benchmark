#!/bin/bash

# Task 4: Distributed Cache System Verification
# Tests distributed caching with fault tolerance and consistency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Task 4: Distributed Cache System Verification ==="

# Initialize scores
CORRECTNESS_SCORE=0
PERFORMANCE_SCORE=0  
SECURITY_SCORE=0
MAX_CORRECTNESS=70
MAX_PERFORMANCE=20
MAX_SECURITY=10

# Test basic functionality
echo "Testing distributed cache operations..."

python3 - << 'EOF'
import sys
sys.path.append('.')

try:
    from submission import DistributedCache, CacheCluster
    import time
    import concurrent.futures
    import random
    
    print("✓ Successfully imported distributed cache classes")
    
    # Test 1: Basic Operations
    print("\n1. Testing basic cache operations...")
    cluster = CacheCluster(cluster_size=3)
    cache = cluster.get_cache()
    
    # Test set/get
    success = cache.set("test_key", "test_value")
    if success:
        print("✓ Set operation successful")
    else:
        print("✗ Set operation failed")
        sys.exit(1)
    
    value = cache.get("test_key")
    if value == "test_value":
        print("✓ Get operation successful")
    else:
        print("✗ Get operation failed - expected 'test_value', got:", value)
        sys.exit(1)
    
    # Test 2: Complex Data Types
    print("\n2. Testing complex data types...")
    complex_data = {
        "user_id": 12345,
        "profile": {
            "name": "Alice Smith",
            "age": 30,
            "preferences": ["coding", "reading", "hiking"]
        },
        "metadata": {
            "created": 1234567890,
            "last_login": 1234567891
        }
    }
    
    cache.set("user:12345", complex_data)
    retrieved = cache.get("user:12345")
    if retrieved == complex_data:
        print("✓ Complex data storage/retrieval successful")
    else:
        print("✗ Complex data test failed")
        sys.exit(1)
    
    # Test 3: Batch Operations
    print("\n3. Testing batch operations...")
    batch_data = {
        f"key_{i}": f"value_{i}" for i in range(10)
    }
    
    if cache.batch_set(batch_data):
        print("✓ Batch set successful")
    else:
        print("✗ Batch set failed")
        sys.exit(1)
    
    retrieved_batch = cache.batch_get(list(batch_data.keys()))
    if len(retrieved_batch) == len(batch_data):
        print("✓ Batch get successful")
    else:
        print(f"✗ Batch get incomplete - got {len(retrieved_batch)}/{len(batch_data)} items")
        sys.exit(1)
    
    # Test 4: Delete Operations
    print("\n4. Testing delete operations...")
    cache.set("temp_key", "temp_value")
    if cache.delete("temp_key"):
        print("✓ Delete operation successful")
        if cache.get("temp_key") is None:
            print("✓ Key properly deleted")
        else:
            print("✗ Key still exists after deletion")
            sys.exit(1)
    else:
        print("✗ Delete operation failed")
        sys.exit(1)
    
    # Test 5: TTL Support
    print("\n5. Testing TTL functionality...")
    cache.set("ttl_key", "ttl_value", ttl=1)  # 1 second TTL
    
    immediate_value = cache.get("ttl_key")
    if immediate_value == "ttl_value":
        print("✓ TTL key retrieved immediately")
    else:
        print("✗ TTL key not found immediately")
        sys.exit(1)
    
    # Wait for TTL expiration
    time.sleep(1.5)
    expired_value = cache.get("ttl_key")
    if expired_value is None:
        print("✓ TTL expiration working")
    else:
        print("✗ TTL key did not expire")
        sys.exit(1)
    
    print("\n✓ All basic functionality tests passed!")
    
except Exception as e:
    print(f"✗ Basic functionality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 30))
    echo "✓ Basic functionality: 30/30 points"
else
    echo "✗ Basic functionality test failed"
    exit 1
fi

# Test distributed operations and fault tolerance
echo -e "\nTesting fault tolerance and distributed operations..."

python3 - << 'EOF'
import sys
sys.path.append('.')

try:
    from submission import DistributedCache, CacheCluster
    import time
    
    print("1. Testing cluster expansion...")
    cluster = CacheCluster(cluster_size=3)
    cache = cluster.get_cache()
    
    # Add some data
    cache.set("persistent_key", "persistent_value")
    
    # Expand cluster
    new_nodes = cluster.expand_cluster(2)
    if len(new_nodes) == 2:
        print("✓ Cluster expansion successful")
    else:
        print("✗ Cluster expansion failed")
        sys.exit(1)
    
    # Verify data still accessible
    value = cache.get("persistent_key")
    if value == "persistent_value":
        print("✓ Data accessible after cluster expansion")
    else:
        print("✗ Data lost after cluster expansion")
        sys.exit(1)
    
    print("\n2. Testing node failure simulation...")
    # Simulate node failure
    cache.simulate_node_failure("node0:8000")
    
    # Operations should still work
    cache.set("failure_test", "still_working")
    value = cache.get("failure_test")
    if value == "still_working":
        print("✓ Operations work during node failure")
    else:
        print("✗ Operations failed during node failure")
        sys.exit(1)
    
    # Recover node
    cache.recover_node("node0:8000")
    
    # Verify recovery
    recovery_value = cache.get("failure_test")
    if recovery_value == "still_working":
        print("✓ Node recovery successful")
    else:
        print("✗ Node recovery failed")
        sys.exit(1)
    
    print("\n3. Testing cluster shrinking...")
    # Test cluster shrinking
    removed_nodes = cluster.shrink_cluster(1)
    if len(removed_nodes) == 1:
        print("✓ Cluster shrinking successful")
    else:
        print("✗ Cluster shrinking failed")
        sys.exit(1)
    
    # Verify data still accessible
    value = cache.get("persistent_key")
    if value == "persistent_value":
        print("✓ Data accessible after cluster shrinking")
    else:
        print("✗ Data lost after cluster shrinking")
        sys.exit(1)
    
    print("\n✓ All fault tolerance tests passed!")
    
except Exception as e:
    print(f"✗ Fault tolerance test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 25))
    echo "✓ Fault tolerance: 25/25 points"
else
    echo "✗ Fault tolerance test failed"
    exit 1
fi

# Test consistency and replication
echo -e "\nTesting consistency and replication..."

python3 - << 'EOF'
import sys
sys.path.append('.')

try:
    from submission import DistributedCache, CacheCluster
    import time
    import threading
    
    print("1. Testing data consistency across nodes...")
    cluster = CacheCluster(cluster_size=5, base_port=9000)
    cache = cluster.get_cache()
    
    # Write data to cache
    test_data = {"message": "consistency_test", "timestamp": time.time()}
    cache.set("consistency_key", test_data)
    
    # Verify data is retrievable (should be replicated)
    retrieved = cache.get("consistency_key")
    if retrieved == test_data:
        print("✓ Data consistency maintained")
    else:
        print("✗ Data consistency failed")
        sys.exit(1)
    
    print("\n2. Testing concurrent operations...")
    results = []
    errors = []
    
    def concurrent_writer(thread_id):
        try:
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                success = cache.set(key, value)
                if success:
                    retrieved = cache.get(key)
                    if retrieved == value:
                        results.append(f"thread_{thread_id}_success_{i}")
                    else:
                        errors.append(f"thread_{thread_id}_mismatch_{i}")
                else:
                    errors.append(f"thread_{thread_id}_set_failed_{i}")
        except Exception as e:
            errors.append(f"thread_{thread_id}_exception: {e}")
    
    threads = []
    for i in range(5):
        thread = threading.Thread(target=concurrent_writer, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    if len(errors) == 0 and len(results) == 50:  # 5 threads * 10 operations
        print("✓ Concurrent operations successful")
    else:
        print(f"✗ Concurrent operations failed - {len(errors)} errors, {len(results)} successes")
        sys.exit(1)
    
    print("\n3. Testing statistics and monitoring...")
    stats = cache.get_stats()
    required_fields = [
        'cluster_stats', 'performance_stats', 'resource_stats', 'node_health'
    ]
    
    for field in required_fields:
        if field not in stats:
            print(f"✗ Missing statistics field: {field}")
            sys.exit(1)
    
    if stats['cluster_stats']['total_nodes'] > 0:
        print("✓ Statistics collection working")
    else:
        print("✗ Invalid statistics data")
        sys.exit(1)
    
    print("\n✓ All consistency tests passed!")
    
except Exception as e:
    print(f"✗ Consistency test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
    echo "✓ Consistency & replication: 15/15 points"
else
    echo "✗ Consistency test failed"
    exit 1
fi

# Performance testing
echo -e "\nTesting performance..."

python3 - << 'EOF'
import sys
sys.path.append('.')

try:
    from submission import DistributedCache, CacheCluster
    import time
    import threading
    import concurrent.futures
    
    print("1. Testing throughput performance...")
    cluster = CacheCluster(cluster_size=3)
    cache = cluster.get_cache()
    
    # Warmup
    for i in range(100):
        cache.set(f"warmup_{i}", f"value_{i}")
    
    # Throughput test - measure operations per second
    start_time = time.time()
    operations = 1000
    
    for i in range(operations):
        key = f"perf_key_{i}"
        value = f"perf_value_{i}"
        cache.set(key, value)
        cache.get(key)
    
    end_time = time.time()
    duration = end_time - start_time
    ops_per_second = (operations * 2) / duration  # 2 ops per iteration (set + get)
    
    print(f"Throughput: {ops_per_second:.0f} ops/second")
    
    if ops_per_second >= 1000:  # Expect at least 1000 ops/sec
        print("✓ Throughput test passed")
        throughput_score = 10
    elif ops_per_second >= 500:
        print("✓ Adequate throughput")
        throughput_score = 7
    else:
        print("✗ Low throughput")
        throughput_score = 3
    
    print("\n2. Testing latency performance...")
    latencies = []
    
    for i in range(100):
        start = time.time()
        cache.get(f"perf_key_{i}")
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to milliseconds
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Max latency: {max_latency:.2f}ms")
    
    if avg_latency <= 10:  # Target <10ms average
        print("✓ Latency test passed")
        latency_score = 10
    elif avg_latency <= 50:
        print("✓ Acceptable latency")
        latency_score = 7
    else:
        print("✗ High latency")
        latency_score = 3
    
    performance_score = throughput_score + latency_score
    print(f"\nPerformance score: {performance_score}/20")
    
    with open("/tmp/task4_performance_score", "w") as f:
        f.write(str(performance_score))
    
except Exception as e:
    print(f"✗ Performance test failed: {e}")
    import traceback
    traceback.print_exc()
    with open("/tmp/task4_performance_score", "w") as f:
        f.write("5")  # Partial credit
EOF

if [ -f "/tmp/task4_performance_score" ]; then
    PERFORMANCE_SCORE=$(cat /tmp/task4_performance_score)
    echo "✓ Performance testing completed: $PERFORMANCE_SCORE/$MAX_PERFORMANCE points"
    rm -f /tmp/task4_performance_score
else
    PERFORMANCE_SCORE=0
    echo "✗ Performance test failed completely"
fi

# Security testing
echo -e "\nTesting security aspects..."

python3 - << 'EOF'
import sys
sys.path.append('.')

try:
    from submission import DistributedCache, CacheCluster
    import time
    
    print("1. Testing input validation...")
    cluster = CacheCluster(cluster_size=3)
    cache = cluster.get_cache()
    
    # Test with various input types
    test_cases = [
        ("", "empty_key"),
        ("normal_key", ""),
        ("special@#$%key", "special_value"),
        ("long_key" * 100, "long_key_value"),
        ("unicode_key_🔑", "unicode_value_🎯"),
        (123, "numeric_key"),  # Should handle gracefully
        ("key", None),  # Should handle gracefully
    ]
    
    safe_operations = 0
    for key, value in test_cases:
        try:
            result = cache.set(key, value)
            if isinstance(result, bool):  # Should return boolean
                safe_operations += 1
        except Exception as e:
            # Graceful error handling is acceptable
            safe_operations += 1
    
    if safe_operations >= len(test_cases) - 2:  # Allow some edge cases to fail
        print("✓ Input validation working")
        validation_score = 5
    else:
        print("✗ Input validation insufficient")
        validation_score = 2
    
    print("\n2. Testing memory safety...")
    # Test large data handling
    large_data = "x" * (1024 * 1024)  # 1MB string
    
    try:
        cache.set("large_key", large_data)
        retrieved = cache.get("large_key")
        if retrieved == large_data:
            print("✓ Large data handled safely")
            memory_score = 5
        else:
            print("✗ Large data corruption")
            memory_score = 2
    except Exception as e:
        # Graceful handling of large data is acceptable
        print("✓ Large data limits enforced")
        memory_score = 3
    
    security_score = validation_score + memory_score
    print(f"\nSecurity score: {security_score}/10")
    
    with open("/tmp/task4_security_score", "w") as f:
        f.write(str(security_score))
    
except Exception as e:
    print(f"✗ Security test failed: {e}")
    import traceback
    traceback.print_exc()
    with open("/tmp/task4_security_score", "w") as f:
        f.write("3")  # Partial credit
EOF

if [ -f "/tmp/task4_security_score" ]; then
    SECURITY_SCORE=$(cat /tmp/task4_security_score)
    echo "✓ Security testing completed: $SECURITY_SCORE/$MAX_SECURITY points"
    rm -f /tmp/task4_security_score
else
    SECURITY_SCORE=0
    echo "✗ Security test failed completely"
fi

# Calculate final score
TOTAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SECURITY_SCORE))
MAX_TOTAL=$((MAX_CORRECTNESS + MAX_PERFORMANCE + MAX_SECURITY))

echo -e "\n=== Task 4 Results ==="
echo "Correctness: $CORRECTNESS_SCORE/$MAX_CORRECTNESS"
echo "Performance: $PERFORMANCE_SCORE/$MAX_PERFORMANCE" 
echo "Security: $SECURITY_SCORE/$MAX_SECURITY"
echo "TOTAL: $TOTAL_SCORE/$MAX_TOTAL"

# Performance percentage for external verification
PERCENTAGE=$(( (TOTAL_SCORE * 100) / MAX_TOTAL ))
echo "Percentage: $PERCENTAGE%"

if [ $TOTAL_SCORE -ge 85 ]; then
    echo "✓ EXCELLENT - Production ready distributed cache"
elif [ $TOTAL_SCORE -ge 70 ]; then
    echo "✓ GOOD - Solid distributed cache implementation"
elif [ $TOTAL_SCORE -ge 50 ]; then
    echo "⚠ FAIR - Basic distributed cache with room for improvement"
else
    echo "✗ NEEDS WORK - Significant issues with distributed cache implementation"
fi

exit 0
