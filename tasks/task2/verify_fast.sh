#!/bin/bash

# Task 2 Fast Verification Script
# Tests the Secure Microservice Authentication implementation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Yaeger Benchmark - Task 2: Secure Microservice Authentication (Fast)${NC}"
echo "======================================================================"

# Check if submission exists
if [ ! -f "$TASK_DIR/submission.py" ]; then
    echo -e "${RED}Error: submission.py not found${NC}"
    echo "Score: 0"
    exit 1
fi

cd "$TASK_DIR"

# Initialize scores
CORRECTNESS_SCORE=0
PERFORMANCE_SCORE=0
SECURITY_SCORE=0

echo -e "\n${BLUE}=== BASIC IMPORT TEST ===${NC}"

# Test basic imports
python3 -c "
import sys
sys.path.append('$TASK_DIR')
try:
    import submission
    print('✓ Submission imports successfully')
    if hasattr(submission, 'app'):
        print('✓ FastAPI app found')
        exit(0)
    else:
        print('✗ FastAPI app not found')
        exit(1)
except Exception as e:
    print(f'✗ Import failed: {e}')
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 30))
    echo -e "${GREEN}✓ Basic functionality: 30/70${NC}"
else
    echo -e "${RED}✗ Basic functionality failed${NC}"
fi

echo -e "\n${BLUE}=== SECURITY TESTS ===${NC}"

# Check for basic security patterns
SECURITY_ISSUES=0

# Check for password hashing
if grep -q "bcrypt\|hash\|crypt" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ Password hashing detected${NC}"
    SECURITY_SCORE=$((SECURITY_SCORE + 3))
else
    echo -e "${RED}✗ No password hashing found${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for JWT implementation
if grep -q "jwt\|token" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ JWT/Token implementation detected${NC}"
    SECURITY_SCORE=$((SECURITY_SCORE + 3))
else
    echo -e "${RED}✗ No JWT/Token implementation found${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for dangerous patterns
if grep -q "eval\|exec\|subprocess" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${RED}✗ Dangerous code patterns found${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
else
    echo -e "${GREEN}✓ No dangerous code patterns${NC}"
    SECURITY_SCORE=$((SECURITY_SCORE + 4))
fi

echo -e "\n${BLUE}=== PERFORMANCE TESTS ===${NC}"

# Test import performance
RUNTIME_MS=$(python3 -c "
import time
import sys
sys.path.append('$TASK_DIR')
start = time.time()
try:
    import submission
    end = time.time()
    runtime = (end - start) * 1000
    print(f'{runtime:.1f}')
except:
    print('999')
" 2>/dev/null)

echo "Import time: ${RUNTIME_MS}ms"

# Performance scoring
if [ "${RUNTIME_MS%.*}" -lt 1000 ]; then
    PERFORMANCE_SCORE=20
    echo -e "${GREEN}✓ Performance: Import under 1s${NC}"
else
    PERFORMANCE_SCORE=10
    echo -e "${YELLOW}△ Performance: Import over 1s${NC}"
fi

# Add some correctness points for having FastAPI endpoints
if grep -q "@app\.\(get\|post\|put\|delete\)" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ FastAPI endpoints detected${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 20))
fi

# Check for authentication endpoints specifically
if grep -q "login\|register\|auth" "$TASK_DIR/submission.py" 2>/dev/null; then
    echo -e "${GREEN}✓ Authentication endpoints detected${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 20))
fi

# Calculate final score
TOTAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SECURITY_SCORE))

# Generate summary
echo -e "\n${BLUE}=== RESULTS SUMMARY ===${NC}"
echo "========================================"
echo "Runtime: ${RUNTIME_MS}ms"
echo "Vulns: $SECURITY_ISSUES"
echo ""
echo "Correctness Score: $CORRECTNESS_SCORE/70"
echo "Performance Score: $PERFORMANCE_SCORE/20"
echo "Security Score: $SECURITY_SCORE/10"
echo "========================================"
echo "Score: $TOTAL_SCORE"

# Exit with success if score > 70
if [ $TOTAL_SCORE -ge 70 ]; then
    echo -e "\n${GREEN}✓ Task 2 verification passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Task 2 verification failed (score: $TOTAL_SCORE/100)${NC}"
    exit 1
fi
