#!/bin/bash

# EXTREME Task 1 Verification Script
# Tests microsecond latency, 1000+ assets, Byzantine fault tolerance, and compliance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}Yaeger Benchmark - Task 1: EXTREME Portfolio Optimizer${NC}"
echo "================================================================="

# Check if submission exists
if [ ! -f "$TASK_DIR/submission.py" ]; then
    echo -e "${RED}Error: submission.py not found${NC}"
    echo "Score: 0"
    exit 1
fi

# Initialize scores
CORRECTNESS_SCORE=0
PERFORMANCE_SCORE=0
SCALABILITY_SCORE=0
INNOVATION_SCORE=0

cd "$TASK_DIR"

echo -e "${PURPLE}Phase 1: Syntax and Import Validation${NC}"
echo "======================================"

# Check Python syntax
echo "Checking Python syntax..."
if python3 -m py_compile submission.py 2>/dev/null; then
    echo -e "${GREEN}✓ Python syntax valid${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 5))
else
    echo -e "${RED}✗ Python syntax errors${NC}"
    echo "Score: 0"
    exit 1
fi

# Check enhanced imports
echo "Testing enhanced imports..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    import submission
    
    # Check for extreme functions
    required_functions = [
        'ExtremePortfolioOptimizer',
        'trading_algorithm'
    ]
    
    missing_functions = []
    for func in required_functions:
        if not hasattr(submission, func):
            missing_functions.append(func)
    
    if missing_functions:
        print(f'✗ Missing functions: {missing_functions}')
        exit(1)
    else:
        print('✓ All required functions found')
        exit(0)
        
except Exception as e:
    print(f'✗ Import failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enhanced imports successful${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 5))
else
    echo -e "${RED}✗ Enhanced import test failed${NC}"
fi

echo -e "\n${PURPLE}Phase 2: Correctness Testing${NC}"
echo "============================"

# Test basic functionality
echo "Testing basic algorithm execution..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer, trading_algorithm
    from codebase.enhanced_exchange import ByzantineExchange
    from codebase.enhanced_market_data import ExtremeMarketData
    from codebase.enhanced_portfolio import ExtremePortfolio
    
    # Create instances
    exchange = ByzantineExchange(corruption_rate=0.1)
    market_data = ExtremeMarketData(num_assets=100)
    portfolio = ExtremePortfolio()
    optimizer = ExtremePortfolioOptimizer()
    
    # Test basic execution
    trading_algorithm(exchange, market_data, portfolio, optimizer)
    print('✓ Basic algorithm execution successful')
    exit(0)
    
except Exception as e:
    print(f'✗ Basic execution failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Basic execution successful${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 10))
else
    echo -e "${RED}✗ Basic execution failed${NC}"
fi

# Test portfolio optimization
echo "Testing portfolio optimization..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer
    from codebase.enhanced_market_data import ExtremeMarketData, MarketRegime, MarketRegimeType, VolatilityRegime
    from codebase.risk_model import RiskModel
    import numpy as np
    
    optimizer = ExtremePortfolioOptimizer()
    market_data = ExtremeMarketData(num_assets=50)
    risk_model = RiskModel()
    
    regime = MarketRegime(
        regime_type=MarketRegimeType.BULL,
        confidence=0.9,
        transition_probability={'BULL': 0.8, 'SIDEWAYS': 0.2},
        expected_duration_days=30,
        volatility_regime=VolatilityRegime.MEDIUM
    )
    
    assets = market_data.symbols[:20]
    weights = optimizer.optimize_portfolio(assets, market_data, risk_model, regime)
    
    if isinstance(weights, dict) and len(weights) > 0:
        print('✓ Portfolio optimization successful')
        exit(0)
    else:
        print('✗ Portfolio optimization returned invalid result')
        exit(1)
        
except Exception as e:
    print(f'✗ Portfolio optimization failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Portfolio optimization successful${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 10))
else
    echo -e "${RED}✗ Portfolio optimization failed${NC}"
fi

# Test regime detection
echo "Testing regime detection..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer
    import numpy as np
    
    optimizer = ExtremePortfolioOptimizer()
    
    # Create test data
    price_history = np.array([100 + i + np.random.normal(0, 2) for i in range(50)])
    volume_history = np.array([1000 + np.random.randint(-200, 200) for _ in range(50)])
    
    regime = optimizer.detect_market_regime(price_history, volume_history)
    
    if hasattr(regime, 'regime_type') and hasattr(regime, 'confidence'):
        print('✓ Regime detection successful')
        exit(0)
    else:
        print('✗ Regime detection returned invalid result')
        exit(1)
        
except Exception as e:
    print(f'✗ Regime detection failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Regime detection successful${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 10))
else
    echo -e "${RED}✗ Regime detection failed${NC}"
fi

echo -e "\n${PURPLE}Phase 3: EXTREME Performance Testing${NC}"
echo "====================================="

# Test microsecond latency
echo "Testing microsecond latency (target: <100μs)..."
LATENCY_RESULT=$(python3 -c "
import sys, time
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer
    from codebase.enhanced_market_data import ExtremeMarketData, MarketRegime, MarketRegimeType, VolatilityRegime
    from codebase.risk_model import RiskModel
    
    optimizer = ExtremePortfolioOptimizer()
    market_data = ExtremeMarketData(num_assets=100)
    risk_model = RiskModel()
    
    regime = MarketRegime(
        regime_type=MarketRegimeType.BULL,
        confidence=0.9,
        transition_probability={'BULL': 0.8, 'SIDEWAYS': 0.2},
        expected_duration_days=30,
        volatility_regime=VolatilityRegime.MEDIUM
    )
    
    assets = market_data.symbols[:50]
    
    # Warm up
    optimizer.optimize_portfolio(assets, market_data, risk_model, regime)
    
    # Measure latency
    start = time.perf_counter()
    optimizer.optimize_portfolio(assets, market_data, risk_model, regime)
    end = time.perf_counter()
    
    latency_us = (end - start) * 1_000_000
    print(f'{latency_us:.1f}')
    
except Exception as e:
    print('999999')  # Failure value
" 2>/dev/null)

LATENCY_INT=${LATENCY_RESULT%.*}

if [ "$LATENCY_INT" -lt 100 ]; then
    echo -e "${GREEN}✓ EXTREME latency achieved: ${LATENCY_RESULT}μs${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 15))
elif [ "$LATENCY_INT" -lt 500 ]; then
    echo -e "${YELLOW}⚠ Good latency: ${LATENCY_RESULT}μs (target: <100μs)${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 10))
else
    echo -e "${RED}✗ Latency too high: ${LATENCY_RESULT}μs (target: <100μs)${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 5))
fi

# Test memory usage
echo "Testing memory constraint (<50MB)..."
MEMORY_RESULT=$(python3 -c "
import sys, psutil, os
sys.path.append('$TASK_DIR')
try:
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    from submission import ExtremePortfolioOptimizer
    from codebase.enhanced_market_data import ExtremeMarketData
    
    optimizer = ExtremePortfolioOptimizer(max_assets=1000, memory_limit_mb=50)
    market_data = ExtremeMarketData(num_assets=1000)
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = mem_after - mem_before
    
    print(f'{memory_used:.1f}')
    
except Exception as e:
    print('999')  # Failure value
" 2>/dev/null)

MEMORY_INT=${MEMORY_RESULT%.*}

if [ "$MEMORY_INT" -lt 50 ]; then
    echo -e "${GREEN}✓ Memory usage excellent: ${MEMORY_RESULT}MB${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 10))
elif [ "$MEMORY_INT" -lt 100 ]; then
    echo -e "${YELLOW}⚠ Memory usage acceptable: ${MEMORY_RESULT}MB (target: <50MB)${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 5))
else
    echo -e "${RED}✗ Memory usage too high: ${MEMORY_RESULT}MB (target: <50MB)${NC}"
fi

# Test high throughput
echo "Testing high throughput (100K+ updates/sec)..."
THROUGHPUT_RESULT=$(python3 -c "
import sys, time
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer
    from codebase.enhanced_market_data import ExtremeMarketData
    
    optimizer = ExtremePortfolioOptimizer()
    market_data = ExtremeMarketData(num_assets=100)
    
    # Simulate high-frequency updates
    start_time = time.time()
    update_count = 0
    
    for _ in range(1000):  # Simulate 1000 market updates
        market_data.simulate_market_update()
        update_count += 1
        
        # Break if too slow
        if time.time() - start_time > 0.1:  # 100ms limit
            break
    
    elapsed = time.time() - start_time
    throughput = update_count / elapsed if elapsed > 0 else 0
    
    print(f'{throughput:.0f}')
    
except Exception as e:
    print('0')  # Failure value
" 2>/dev/null)

if [ "$THROUGHPUT_RESULT" -gt 100000 ]; then
    echo -e "${GREEN}✓ Throughput excellent: ${THROUGHPUT_RESULT} updates/sec${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 5))
elif [ "$THROUGHPUT_RESULT" -gt 50000 ]; then
    echo -e "${YELLOW}⚠ Throughput good: ${THROUGHPUT_RESULT} updates/sec${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 3))
else
    echo -e "${RED}✗ Throughput insufficient: ${THROUGHPUT_RESULT} updates/sec${NC}"
fi

echo -e "\n${PURPLE}Phase 4: Scalability Testing${NC}"
echo "============================"

# Test 1000+ assets
echo "Testing 1000+ asset handling..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer
    from codebase.enhanced_market_data import ExtremeMarketData, MarketRegime, MarketRegimeType, VolatilityRegime
    from codebase.risk_model import RiskModel
    
    optimizer = ExtremePortfolioOptimizer(max_assets=1000)
    market_data = ExtremeMarketData(num_assets=1000)
    risk_model = RiskModel()
    
    regime = MarketRegime(
        regime_type=MarketRegimeType.SIDEWAYS,
        confidence=0.8,
        transition_probability={'BULL': 0.3, 'SIDEWAYS': 0.5, 'BEAR': 0.2},
        expected_duration_days=30,
        volatility_regime=VolatilityRegime.MEDIUM
    )
    
    assets = market_data.symbols  # All 1000 assets
    weights = optimizer.optimize_portfolio(assets, market_data, risk_model, regime)
    
    if len(weights) >= 1000:
        print('✓ 1000+ asset handling successful')
        exit(0)
    else:
        print(f'✗ Only handled {len(weights)} assets (target: 1000+)')
        exit(1)
        
except Exception as e:
    print(f'✗ 1000+ asset test failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 1000+ asset handling successful${NC}"
    SCALABILITY_SCORE=$((SCALABILITY_SCORE + 10))
else
    echo -e "${RED}✗ 1000+ asset handling failed${NC}"
fi

# Test Byzantine fault tolerance
echo "Testing Byzantine fault tolerance..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer
    from codebase.enhanced_market_data import ExtremeMarketData
    
    optimizer = ExtremePortfolioOptimizer()
    market_data = ExtremeMarketData(num_assets=100)
    
    # Create corrupted feeds
    feeds = market_data.simulate_market_update()
    
    # Corrupt 30% of feeds (Byzantine threshold)
    corrupted_feeds = []
    for i, feed in enumerate(feeds):
        if i % 3 == 0:  # Corrupt every 3rd feed
            feed.price *= 10  # Price spike corruption
        corrupted_feeds.append(feed)
    
    clean_feeds = optimizer.handle_byzantine_feeds(corrupted_feeds)
    
    if len(clean_feeds) < len(corrupted_feeds):
        print('✓ Byzantine fault tolerance working')
        exit(0)
    else:
        print('✗ Byzantine fault tolerance failed')
        exit(1)
        
except Exception as e:
    print(f'✗ Byzantine fault tolerance test failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Byzantine fault tolerance successful${NC}"
    SCALABILITY_SCORE=$((SCALABILITY_SCORE + 5))
else
    echo -e "${RED}✗ Byzantine fault tolerance failed${NC}"
fi

# Test compliance monitoring
echo "Testing regulatory compliance..."
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    from submission import ExtremePortfolioOptimizer
    
    optimizer = ExtremePortfolioOptimizer()
    
    # Test positions that should trigger compliance violations
    test_positions = {
        'AAPL': 0.08,  # 8% position - should trigger MiFID II warning
        'GOOGL': 0.03,
        'MSFT': 0.02
    }
    
    violations = optimizer.execute_compliance_monitoring(test_positions)
    
    # Should detect at least one violation for oversized position
    if violations and len(violations) > 0:
        print('✓ Compliance monitoring working')
        exit(0)
    else:
        print('✗ Compliance monitoring not detecting violations')
        exit(1)
        
except Exception as e:
    print(f'✗ Compliance monitoring test failed: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compliance monitoring successful${NC}"
    SCALABILITY_SCORE=$((SCALABILITY_SCORE + 5))
else
    echo -e "${RED}✗ Compliance monitoring failed${NC}"
fi

echo -e "\n${PURPLE}Phase 5: Innovation Assessment${NC}"
echo "=============================="

# Check for advanced algorithms
echo "Checking for advanced implementations..."

INNOVATION_FEATURES=0

# Check for custom memory management
if grep -q "MemoryPool\|memory_pool" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ Custom memory management detected${NC}"
    INNOVATION_FEATURES=$((INNOVATION_FEATURES + 1))
fi

# Check for lock-free structures
if grep -q "lock.*free\|atomic\|lockfree" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ Lock-free programming detected${NC}"
    INNOVATION_FEATURES=$((INNOVATION_FEATURES + 1))
fi

# Check for regime detection
if grep -q "regime\|MarketRegime" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ Market regime detection implemented${NC}"
    INNOVATION_FEATURES=$((INNOVATION_FEATURES + 1))
fi

# Check for HSM integration
if grep -q "hsm\|hardware_security\|sign.*order" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ Hardware security module integration${NC}"
    INNOVATION_FEATURES=$((INNOVATION_FEATURES + 1))
fi

# Check for factor models
if grep -q "factor.*model\|risk.*model" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ Advanced risk/factor models${NC}"
    INNOVATION_FEATURES=$((INNOVATION_FEATURES + 1))
fi

INNOVATION_SCORE=$((INNOVATION_FEATURES * 2))
echo "Innovation features detected: $INNOVATION_FEATURES/5"

# Calculate final score
TOTAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SCALABILITY_SCORE + INNOVATION_SCORE))

# Generate summary
echo -e "\n${BLUE}=== EXTREME VERIFICATION RESULTS ===${NC}"
echo "====================================================="
echo "Latency: ${LATENCY_RESULT}μs (target: <100μs)"
echo "Memory: ${MEMORY_RESULT}MB (target: <50MB)"
echo "Throughput: ${THROUGHPUT_RESULT} updates/sec (target: >100K)"
echo "Assets: 1000+ handling capability"
echo "Innovation Features: $INNOVATION_FEATURES/5"
echo ""
echo "Correctness Score: $CORRECTNESS_SCORE/40"
echo "Performance Score: $PERFORMANCE_SCORE/30"
echo "Scalability Score: $SCALABILITY_SCORE/20"
echo "Innovation Score: $INNOVATION_SCORE/10"
echo "====================================================="
echo "Score: $TOTAL_SCORE"

# Exit with appropriate code
if [ $TOTAL_SCORE -ge 80 ]; then
    echo -e "\n${GREEN}🔥 EXTREME Task 1 verification PASSED! 🔥${NC}"
    echo -e "${GREEN}This is enterprise-grade implementation!${NC}"
    exit 0
elif [ $TOTAL_SCORE -ge 60 ]; then
    echo -e "\n${YELLOW}⚡ Task 1 verification passed (Good)${NC}"
    exit 0
else
    echo -e "\n${RED}✗ EXTREME Task 1 verification FAILED${NC}"
    echo -e "${RED}Score $TOTAL_SCORE/100 - Needs significant improvement${NC}"
    exit 1
fi
