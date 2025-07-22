#!/bin/bash

# Task 1 Verification Script
# Tests the Real-Time Trading Optimizer implementation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$SCRIPT_DIR"
TEST_DIR="$TASK_DIR/tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Yaeger Benchmark - Task 1: Real-Time Trading Optimizer${NC}"
echo "======================================================="

# Check if submission exists
if [ ! -f "$TASK_DIR/submission.py" ]; then
    echo -e "${RED}Error: submission.py not found${NC}"
    echo "Score: 0"
    exit 1
fi

# Initialize scores
CORRECTNESS_SCORE=0
PERFORMANCE_SCORE=0
SECURITY_SCORE=0
TOTAL_TESTS=0
PASSED_TESTS=0

# Function to run tests and capture results
run_test_suite() {
    local test_file=$1
    local test_name=$2
    local max_score=$3
    
    echo -e "\n${YELLOW}Running $test_name...${NC}"
    
    if python -m pytest "$test_file" -v --tb=short --timeout=300 > test_output.tmp 2>&1; then
        local passed=$(grep -c "PASSED" test_output.tmp || echo "0")
        local failed=$(grep -c "FAILED" test_output.tmp || echo "0")
        local total=$((passed + failed))
        
        if [ $total -gt 0 ]; then
            local score=$((passed * max_score / total))
            echo -e "${GREEN}$test_name: $passed/$total tests passed (Score: $score/$max_score)${NC}"
            
            TOTAL_TESTS=$((TOTAL_TESTS + total))
            PASSED_TESTS=$((PASSED_TESTS + passed))
            
            return $score
        else
            echo -e "${RED}$test_name: No tests found${NC}"
            return 0
        fi
    else
        echo -e "${RED}$test_name: Test suite failed to run${NC}"
        cat test_output.tmp
        return 0
    fi
}

# Change to task directory
cd "$TASK_DIR"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install pytest pytest-timeout psutil > /dev/null 2>&1
else
    source venv/bin/activate
fi

# Add current directory to Python path
export PYTHONPATH="$TASK_DIR:$PYTHONPATH"

# Run correctness tests (70% of score)
echo -e "\n${BLUE}=== CORRECTNESS TESTS (70%) ===${NC}"
run_test_suite "$TEST_DIR/test_trading.py" "Core Trading Tests" 70
CORRECTNESS_SCORE=$?

# Run performance tests (20% of score)
echo -e "\n${BLUE}=== PERFORMANCE TESTS (20%) ===${NC}"
run_test_suite "$TEST_DIR/test_performance.py" "Performance Tests" 20
PERFORMANCE_SCORE=$?

# Run security tests (10% of score)
echo -e "\n${BLUE}=== SECURITY TESTS (10%) ===${NC}"

# Check for basic security issues
SECURITY_ISSUES=0

# Check for eval/exec usage
if grep -q "eval\|exec" "$TASK_DIR/submission.py"; then
    echo -e "${RED}Security issue: Found eval/exec usage${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for import of dangerous modules
if grep -q "import os\|import subprocess\|import sys" "$TASK_DIR/submission.py"; then
    echo -e "${YELLOW}Warning: Potentially dangerous imports found${NC}"
fi

# Check for infinite loops (basic check)
if grep -q "while True:" "$TASK_DIR/submission.py"; then
    echo -e "${YELLOW}Warning: Infinite loop detected${NC}"
fi

# Run static analysis with bandit if available
if command -v bandit &> /dev/null; then
    echo "Running security analysis with bandit..."
    if bandit -r "$TASK_DIR/submission.py" -f txt > bandit_output.tmp 2>&1; then
        SECURITY_ISSUES=$(grep -c "Issue:" bandit_output.tmp || echo "0")
    fi
fi

# Calculate security score
if [ $SECURITY_ISSUES -eq 0 ]; then
    SECURITY_SCORE=10
    echo -e "${GREEN}Security: No issues found (Score: 10/10)${NC}"
elif [ $SECURITY_ISSUES -lt 3 ]; then
    SECURITY_SCORE=5
    echo -e "${YELLOW}Security: Minor issues found (Score: 5/10)${NC}"
else
    SECURITY_SCORE=0
    echo -e "${RED}Security: Major issues found (Score: 0/10)${NC}"
fi

# Measure runtime performance
echo -e "\n${BLUE}=== RUNTIME MEASUREMENT ===${NC}"
START_TIME=$(python3 -c "import time; print(time.time())")

# Run a quick performance test
python3 -c "
import sys
sys.path.append('$TASK_DIR')
from codebase.exchange import Exchange
from codebase.market_data import MarketData
from codebase.portfolio import Portfolio
from submission import trading_algorithm
import time

exchange = Exchange()
market_data = MarketData(['AAPL', 'GOOGL', 'MSFT'])
portfolio = Portfolio()

start = time.time()
for _ in range(10):
    trading_algorithm(exchange, market_data, portfolio)
end = time.time()

avg_time = (end - start) / 10 * 1000  # Convert to ms
print(f'Average execution time: {avg_time:.2f}ms')
" > runtime_output.tmp 2>&1

RUNTIME=$(grep "Average execution time:" runtime_output.tmp | awk '{print $4}' | sed 's/ms//' || echo "999")

# Check memory usage
echo "Measuring memory usage..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
import psutil
import os
from codebase.exchange import Exchange
from codebase.market_data import MarketData
from codebase.portfolio import Portfolio
from submission import trading_algorithm

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

exchange = Exchange()
market_data = MarketData(['AAPL', 'GOOGL', 'MSFT'])
portfolio = Portfolio()

for _ in range(100):
    trading_algorithm(exchange, market_data, portfolio)

final_memory = process.memory_info().rss / 1024 / 1024
memory_used = final_memory - initial_memory

print(f'Memory usage: {final_memory:.1f}MB')
print(f'Memory increase: {memory_used:.1f}MB')
" > memory_output.tmp 2>&1

MEMORY=$(grep "Memory usage:" memory_output.tmp | awk '{print $3}' | sed 's/MB//' || echo "999")

# Calculate final score
TOTAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SECURITY_SCORE))

# Generate summary
echo -e "\n${BLUE}=== RESULTS SUMMARY ===${NC}"
echo "========================================"
echo "Tests: $PASSED_TESTS/$TOTAL_TESTS passed"
echo "Runtime: ${RUNTIME}ms"
echo "Memory: ${MEMORY}MB"
echo "Vulns: $SECURITY_ISSUES"
echo ""
echo "Correctness Score: $CORRECTNESS_SCORE/70"
echo "Performance Score: $PERFORMANCE_SCORE/20"
echo "Security Score: $SECURITY_SCORE/10"
echo "========================================"
echo "Score: $TOTAL_SCORE"

# Performance feedback
if (( $(echo "$RUNTIME > 100" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${RED}Warning: Runtime exceeds 100ms limit${NC}"
fi

if (( $(echo "$MEMORY > 512" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${RED}Warning: Memory usage exceeds 512MB limit${NC}"
fi

# Clean up
rm -f test_output.tmp runtime_output.tmp memory_output.tmp bandit_output.tmp

# Deactivate virtual environment
deactivate

# Exit with success if score > 70
if [ $TOTAL_SCORE -ge 70 ]; then
    echo -e "\n${GREEN}✓ Task 1 verification passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Task 1 verification failed (score: $TOTAL_SCORE/100)${NC}"
    exit 1
fi
