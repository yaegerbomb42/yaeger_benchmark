#!/bin/bash
# Task 4 Verification Script - Distributed Cache System
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Yaeger Benchmark - Task 4: Distributed Cache System"
echo "================================================="

if [ ! -f "$SCRIPT_DIR/submission.py" ]; then
    echo "Error: submission.py not found"
    echo "Score: 0"
    exit 1
fi

#!/bin/bash

# Task 4 Verification Script - Distributed Cache System
# Tests distributed caching, consistency, and fault tolerance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Yaeger Benchmark - Task 4: Distributed Cache System${NC}"
echo "=================================================="

# Check if submission exists
if [ ! -f "$TASK_DIR/submission.py" ]; then
    echo -e "${RED}Error: submission.py not found${NC}"
    echo "Score: 0"
    exit 1
fi

# Initialize scores
CORRECTNESS_SCORE=0
PERFORMANCE_SCORE=0
SECURITY_SCORE=10  # Default full security score

cd "$TASK_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q pytest

echo ""
echo "Running tests..."

# Test 1: Basic Cache Operations (15 points)
echo -n "Testing basic cache operations... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002', 'node3:8003'])
assert cache.set('test_key', 'test_value') == True
assert cache.get('test_key') == 'test_value'
assert cache.delete('test_key') == True
assert cache.get('test_key') is None
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 2: Distributed Operations (15 points)
echo -n "Testing distributed operations... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002', 'node3:8003'], replication_factor=2)
# Test batch operations
items = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
assert cache.batch_set(items) == True
results = cache.batch_get(['key1', 'key2', 'key3'])
assert len(results) == 3
assert results['key1'] == 'value1'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 3: TTL Functionality (10 points)
echo -n "Testing TTL functionality... "
if timeout 15s python3 -c "
import sys, time
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002'])
cache.set('ttl_key', 'ttl_value', ttl=2)
assert cache.get('ttl_key') == 'ttl_value'
time.sleep(3)
assert cache.get('ttl_key') is None
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 10))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 4: Node Failure Handling (15 points)
echo -n "Testing node failure handling... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002', 'node3:8003'], replication_factor=2)
cache.set('critical_data', 'important_value')
assert cache.get('critical_data') == 'important_value'

# Simulate node failure
cache.simulate_node_failure('node1:8001')
# Data should still be accessible
assert cache.get('critical_data') == 'important_value'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 5: Consistency (15 points)
echo -n "Testing data consistency... "
if timeout 30s python3 -c "
import sys, threading
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002', 'node3:8003'])

# Test concurrent writes
def writer(worker_id):
    for i in range(10):
        cache.set(f'worker_{worker_id}_key_{i}', f'value_{i}')

threads = []
for i in range(3):
    thread = threading.Thread(target=writer, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

# Verify all writes succeeded
for i in range(3):
    for j in range(10):
        assert cache.get(f'worker_{i}_key_{j}') == f'value_{j}'

print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Performance Tests
echo ""
echo "Running performance tests..."

# Performance Test 1: Throughput (15 points)
echo -n "Testing throughput performance... "
if timeout 60s python3 -c "
import sys, time
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002', 'node3:8003'])

# Write performance test
num_ops = 1000
start_time = time.time()
for i in range(num_ops):
    cache.set(f'perf_key_{i}', f'perf_value_{i}')
write_time = time.time() - start_time
write_throughput = num_ops / write_time

# Read performance test
start_time = time.time()
for i in range(num_ops):
    value = cache.get(f'perf_key_{i}')
read_time = time.time() - start_time
read_throughput = num_ops / read_time

print(f'Write: {write_throughput:.0f} ops/sec, Read: {read_throughput:.0f} ops/sec')

# Performance requirements (scaled for test environment)
assert write_throughput >= 500, f'Write throughput too low: {write_throughput}'
assert read_throughput >= 1000, f'Read throughput too low: {read_throughput}'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Performance Test 2: Memory Efficiency (5 points)
echo -n "Testing memory efficiency... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002'])

# Store data and check memory usage
for i in range(500):
    cache.set(f'memory_key_{i}', f'memory_value_{i}' * 10)

stats = cache.get_stats()
memory_usage = stats['resource_stats']['total_memory_usage_mb']
print(f'Memory usage: {memory_usage:.1f} MB')

# Should be reasonable for the amount of data
assert memory_usage < 100, f'Memory usage too high: {memory_usage}MB'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 5))
else
    echo -e "${RED}FAIL${NC}"
fi

# Security Tests
echo ""
echo "Running security tests..."

echo -n "Testing input validation... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DistributedCache

cache = DistributedCache(['node1:8001', 'node2:8002'])

# Test with various input types
test_inputs = [
    ('normal_key', 'normal_value'),
    ('', 'empty_key'),
    ('key_with_special_chars!@#', 'value_with_special_chars!@#'),
    ('very_long_key' * 100, 'very_long_value' * 100)
]

for key, value in test_inputs:
    try:
        cache.set(key, value)
        cache.get(key)
    except Exception as e:
        # Should handle gracefully
        pass

print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    SECURITY_SCORE=$((SECURITY_SCORE - 5))
fi

# Calculate final score
FINAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SECURITY_SCORE))

echo ""
echo "===================="
echo "Verification Results"
echo "===================="
echo -e "Correctness: ${CORRECTNESS_SCORE}/70"
echo -e "Performance: ${PERFORMANCE_SCORE}/20"
echo -e "Security: ${SECURITY_SCORE}/10"
echo ""
echo -e "${BLUE}Final Score: ${FINAL_SCORE}/100${NC}"

if [ $FINAL_SCORE -ge 70 ]; then
    echo -e "${GREEN}✓ Task 4 completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Task 4 needs improvement${NC}"
fi

echo "Score: $FINAL_SCORE"