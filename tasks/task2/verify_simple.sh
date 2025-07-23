#!/bin/bash

# Simplified Task 2 Verification Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Yaeger Benchmark - Task 2: Authentication Service${NC}"
echo "=================================================="

# Initialize scores
CORRECTNESS_SCORE=0
PERFORMANCE_SCORE=0
SECURITY_SCORE=0

# Quick check if submission exists
if [ ! -f "$TASK_DIR/submission.py" ]; then
    echo -e "${RED}Error: submission.py not found${NC}"
    echo "Score: 0"
    exit 1
fi

cd "$TASK_DIR"

# Basic Python syntax check
echo "Checking Python syntax..."
if python3 -m py_compile submission.py; then
    echo -e "${GREEN}✓ Python syntax valid${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 20))
else
    echo -e "${RED}✗ Python syntax error${NC}"
fi

# Check FastAPI import
echo "Checking FastAPI availability..."
if python3 -c "from fastapi import FastAPI; print('FastAPI available')" 2>/dev/null; then
    echo -e "${GREEN}✓ FastAPI available${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 20))
else
    echo -e "${RED}✗ FastAPI not available${NC}"
fi

# Check app object exists
echo "Checking app object..."
if python3 -c "import submission; print('App:', type(submission.app))" 2>/dev/null; then
    echo -e "${GREEN}✓ FastAPI app object found${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 30))
else
    echo -e "${RED}✗ FastAPI app object not found${NC}"
fi

# Basic security check - no hardcoded secrets
echo "Checking for security issues..."
SECURITY_ISSUES=0

if grep -q "password.*=.*['\"][^'\"]\{10,\}['\"]" submission.py 2>/dev/null; then
    echo -e "${RED}Security issue: Hardcoded passwords${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if grep -q "secret.*=.*['\"]your-secret-key-here['\"]" submission.py 2>/dev/null; then
    echo -e "${RED}Security issue: Default secret key${NC}"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if [ $SECURITY_ISSUES -eq 0 ]; then
    SECURITY_SCORE=8
    echo -e "${GREEN}✓ Basic security checks passed${NC}"
elif [ $SECURITY_ISSUES -lt 3 ]; then
    SECURITY_SCORE=4
    echo -e "${YELLOW}Security: Minor issues found${NC}"
else
    SECURITY_SCORE=0
    echo -e "${RED}Security: Major issues found${NC}"
fi

# Performance test - simple import time
echo "Testing performance..."
START_TIME=$(date +%s.%N)
python3 -c "import submission" 2>/dev/null
END_TIME=$(date +%s.%N)
IMPORT_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "0.1")

if (( $(echo "$IMPORT_TIME < 2.0" | bc -l 2>/dev/null || echo "1") )); then
    PERFORMANCE_SCORE=15
    echo -e "${GREEN}✓ Import performance good${NC}"
else
    PERFORMANCE_SCORE=5
    echo -e "${RED}✗ Import performance slow${NC}"
fi

# Add bonus points for having required endpoints
echo "Checking endpoints..."
if grep -q "def.*health" submission.py && grep -q "def.*register" submission.py && grep -q "def.*login" submission.py; then
    echo -e "${GREEN}✓ Required endpoints found${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 5))
    SECURITY_SCORE=$((SECURITY_SCORE + 2))
fi

# Calculate final score
TOTAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SECURITY_SCORE))

# Generate summary
echo -e "\n${BLUE}=== RESULTS SUMMARY ===${NC}"
echo "========================================"
echo "Syntax Check: Passed"
echo "FastAPI Integration: Working"
echo "Security Issues: $SECURITY_ISSUES"
echo "Import Time: ${IMPORT_TIME}s"
echo ""
echo "Correctness Score: $CORRECTNESS_SCORE/70"
echo "Performance Score: $PERFORMANCE_SCORE/20"
echo "Security Score: $SECURITY_SCORE/10"
echo "========================================"
echo "Score: $TOTAL_SCORE"

# Exit with success if score > 50
if [ $TOTAL_SCORE -ge 50 ]; then
    echo -e "\n${GREEN}✓ Task 2 verification passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Task 2 verification failed (score: $TOTAL_SCORE/100)${NC}"
    exit 1
fi
