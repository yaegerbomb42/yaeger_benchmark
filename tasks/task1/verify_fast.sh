#!/bin/bash

# Fast Task 1 Verification Script
# Optimized for speed while maintaining accuracy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Yaeger Benchmark - Task 1: Real-Time Trading Optimizer (Fast)${NC}"
echo "=============================================================="

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

cd "$TASK_DIR"

# Quick Python check
echo "Checking Python syntax..."
if python3 -m py_compile submission.py 2>/dev/null; then
    echo -e "${GREEN}✓ Python syntax valid${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 20))
else
    echo -e "${RED}✗ Python syntax errors${NC}"
fi

# Quick import test
echo "Testing imports..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    import submission
    if hasattr(submission, 'trading_algorithm'):
        print('✓ trading_algorithm function found')
        exit(0)
    else:
        print('✗ trading_algorithm function missing')
        exit(1)
except Exception as e:
    print(f'✗ Import failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Function imports correctly${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 30))
else
    echo -e "${RED}✗ Import test failed${NC}"
fi

# Quick execution test
echo "Testing basic execution..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    from codebase.exchange import Exchange
    from codebase.market_data import MarketData
    from codebase.portfolio import Portfolio
    from submission import trading_algorithm
    
    exchange = Exchange()
    market_data = MarketData(['AAPL'])
    portfolio = Portfolio()
    
    trading_algorithm(exchange, market_data, portfolio)
    print('✓ Algorithm executed successfully')
    exit(0)
except Exception as e:
    print(f'✗ Execution failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Algorithm executes without errors${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 20))
else
    echo -e "${RED}✗ Execution test failed${NC}"
fi

# Performance test (simplified)
echo "Testing performance..."
RUNTIME=$(python3 -c "
import sys, time
sys.path.append('$TASK_DIR')
try:
    from codebase.exchange import Exchange
    from codebase.market_data import MarketData
    from codebase.portfolio import Portfolio
    from submission import trading_algorithm
    
    exchange = Exchange()
    market_data = MarketData(['AAPL'])
    portfolio = Portfolio()
    
    start = time.time()
    trading_algorithm(exchange, market_data, portfolio)
    end = time.time()
    
    runtime_ms = (end - start) * 1000
    print(f'{runtime_ms:.1f}')
except:
    print('999')
" 2>/dev/null)

if (( $(echo "$RUNTIME < 100" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${GREEN}✓ Performance acceptable (${RUNTIME}ms)${NC}"
    PERFORMANCE_SCORE=20
else
    echo -e "${YELLOW}⚠ Performance could be better (${RUNTIME}ms)${NC}"
    PERFORMANCE_SCORE=10
fi

# Security test (improved)
echo "Running security checks..."
SECURITY_ISSUES=0

# Check for dangerous patterns
if grep -q "eval\|exec\|subprocess\|__import__" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${RED}✗ Dangerous code patterns detected${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if grep -q "open(\|file(\|with open" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${RED}✗ File system operations detected${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if grep -q "socket\|urllib\|requests\|http" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${RED}✗ Network operations detected${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if [ $SECURITY_ISSUES -eq 0 ]; then
    SECURITY_SCORE=10
    echo -e "${GREEN}✓ Security: No issues found${NC}"
elif [ $SECURITY_ISSUES -eq 1 ]; then
    SECURITY_SCORE=7
    echo -e "${YELLOW}⚠ Security: Minor issues found${NC}"
else
    SECURITY_SCORE=3
    echo -e "${RED}✗ Security: Multiple issues found${NC}"
fi

# Calculate final score
TOTAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SECURITY_SCORE))

# Generate summary
echo -e "\n${BLUE}=== RESULTS SUMMARY ===${NC}"
echo "========================================"
echo "Runtime: ${RUNTIME}ms"
echo "Memory: N/A (fast mode)"
echo "Vulns: $SECURITY_ISSUES"
echo ""
echo "Correctness Score: $CORRECTNESS_SCORE/70"
echo "Performance Score: $PERFORMANCE_SCORE/20"
echo "Security Score: $SECURITY_SCORE/10"
echo "========================================"
echo "Score: $TOTAL_SCORE"

# Exit with appropriate code
if [ $TOTAL_SCORE -ge 70 ]; then
    echo -e "\n${GREEN}✓ Task 1 verification passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Task 1 verification failed (score: $TOTAL_SCORE/100)${NC}"
    exit 1
fi
