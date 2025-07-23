#!/bin/bash

# Task 3 Verification Script - Dynamic Data Pipeline
# Tests stream processing performance and correctness

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Yaeger Benchmark - Task 3: Dynamic Data Pipeline${NC}"
echo "================================================="

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
pip install -q psutil pytest

echo ""
echo "Running tests..."

# Test 1: Basic Functionality (20 points)
echo -n "Testing basic pipeline functionality... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DataPipeline, DataTransformer, StreamProcessor
from codebase.stream_source import DataRecord

pipeline = DataPipeline()
records = [{'id': i, 'value': i*10} for i in range(100)]
result = pipeline.process_batch(records)
assert result > 0, 'Pipeline should process records'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 20))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 2: Data Transformation (15 points)  
echo -n "Testing data transformation... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DataTransformer

transformer = DataTransformer()
test_records = [
    {'id': 1, 'value': 100, 'name': 'test'},
    {'id': 2, 'value': -50},  # Should be filtered
    None,  # Should be filtered
    {'id': 3, 'value': 200}
]
cleaned = transformer.clean_batch(test_records)
assert len(cleaned) >= 2, 'Should clean invalid records'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 3: Stream Processing (15 points)
echo -n "Testing stream processing... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DataPipeline

pipeline = DataPipeline()
for stream_id in ['stream1', 'stream2']:
    records = [{'id': f'{stream_id}_{i}', 'value': i} for i in range(50)]
    result = pipeline.process_stream(stream_id, records)
    assert result > 0, f'Should process {stream_id}'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 4: Error Handling (10 points)
echo -n "Testing error handling... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DataPipeline

pipeline = DataPipeline()
malformed_data = [
    'invalid string',
    {'corrupted': 'data'},
    None,
    42,
    {'id': 'valid', 'value': 100}
]
try:
    result = pipeline.process_batch(malformed_data)
    assert isinstance(result, int), 'Should return int count'
    print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 10))
else
    echo -e "${RED}FAIL${NC}"
fi

# Test 5: Memory Efficiency (10 points)
echo -n "Testing memory efficiency... "
if timeout 60s python3 -c "
import sys
sys.path.append('.')
from submission import DataPipeline
import psutil
import os

pipeline = DataPipeline()
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

# Process large dataset
large_records = [{'id': i, 'data': 'x' * 1000} for i in range(10000)]
result = pipeline.process_batch(large_records)

final_memory = process.memory_info().rss / 1024 / 1024
memory_increase = final_memory - initial_memory

assert memory_increase < 500, f'Memory usage too high: {memory_increase}MB'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 10))
else
    echo -e "${RED}FAIL${NC}"
fi

# Performance Tests
echo ""
echo "Running performance tests..."

# Performance Test 1: Throughput (15 points)
echo -n "Testing throughput performance... "
if timeout 60s python3 -c "
import sys
sys.path.append('.')
from submission import DataPipeline
import time

pipeline = DataPipeline()
test_records = [{'id': i, 'value': i} for i in range(10000)]

start_time = time.time()
result = pipeline.process_batch(test_records)
end_time = time.time()

processing_time = end_time - start_time
throughput = result / processing_time

# Should process at least 50K records/second for this test
assert throughput >= 50000, f'Throughput too low: {throughput:.0f} rec/sec'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 15))
else
    echo -e "${RED}FAIL${NC}"
fi

# Performance Test 2: Latency (5 points)
echo -n "Testing latency performance... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DataPipeline
import time

pipeline = DataPipeline()
small_batch = [{'id': i, 'value': i} for i in range(100)]

start_time = time.time()
result = pipeline.process_batch(small_batch)
end_time = time.time()

latency = end_time - start_time
avg_latency_per_record = latency / len(small_batch) * 1000  # ms

# Should process each record in <1ms on average
assert avg_latency_per_record < 1.0, f'Latency too high: {avg_latency_per_record:.2f}ms/record'
print('PASS')
" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 5))
else
    echo -e "${RED}FAIL${NC}"
fi

# Security Tests (basic validation)
echo ""
echo "Running security tests..."

echo -n "Testing input validation... "
if timeout 30s python3 -c "
import sys
sys.path.append('.')
from submission import DataPipeline

pipeline = DataPipeline()
# Test with potentially malicious inputs
malicious_inputs = [
    {'id': '<script>alert(\"xss\")</script>', 'value': 100},
    {'id': \"'; DROP TABLE users; --\", 'value': 200},
    {'id': 'normal', 'value': 300}
]

try:
    result = pipeline.process_batch(malicious_inputs)
    # Should handle gracefully without crashing
    print('PASS')
except Exception as e:
    print(f'FAIL: Should handle malicious input gracefully')
    exit(1)
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
    echo -e "${GREEN}✓ Task 3 completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Task 3 needs improvement${NC}"
fi

echo "Score: $FINAL_SCORE"
