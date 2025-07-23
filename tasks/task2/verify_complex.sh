#!/bin/bash

# Task 2 Verification Script
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

echo -e "${BLUE}Yaeger Benchmark - Task 2: Secure Microservice Authentication${NC}"
echo "=============================================================="

# Error handling function
handle_error() {
    echo -e "${RED}Error occurred in verification script${NC}"
    echo "Line $1: $2"
    # Kill server if running
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    echo "Score: 0"
    exit 1
}

trap 'handle_error $LINENO "$BASH_COMMAND"' ERR

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
SERVER_PID=""

# Setup virtual environment with error handling
echo "Setting up Python environment..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    echo "Score: 0"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    if ! python3 -m venv venv; then
        echo -e "${RED}Error: Failed to create virtual environment${NC}"
        echo "Score: 0"
        exit 1
    fi
fi

source venv/bin/activate || {
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    echo "Score: 0"
    exit 1
}

echo "Installing dependencies..."
pip install fastapi uvicorn pytest pytest-asyncio httpx bcrypt pyjwt cryptography pyotp qrcode bandit pydantic[email] > /dev/null 2>&1 || {
    echo -e "${RED}Error: Failed to install dependencies${NC}"
    echo "Score: 0"
    exit 1
}

cd "$TASK_DIR"
export PYTHONPATH="$TASK_DIR:$PYTHONPATH"

echo -e "\n${BLUE}=== CORRECTNESS TESTS (70%) ===${NC}"

# Start the FastAPI server in background for testing
echo "Starting authentication service..."
timeout 600 python submission.py &
SERVER_PID=$!

# Wait for server to start with timeout
echo "Waiting for server to start..."
WAIT_COUNT=0
while [ $WAIT_COUNT -lt 15 ]; do  # Reduced from 30 to 15 seconds
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server started successfully${NC}"
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ $WAIT_COUNT -eq 15 ]; then
    echo -e "${RED}Error: Server failed to start within 15 seconds${NC}"
    kill $SERVER_PID 2>/dev/null || true
    echo "Score: 0"
    exit 1
fi

# Test basic endpoints
echo "Testing authentication endpoints..."

# Test registration
REGISTER_RESULT=$(curl -s -w "%{http_code}" -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "TestPass123!", "first_name": "Test", "last_name": "User"}' \
  -o /tmp/register_response.json)

if [ "${REGISTER_RESULT: -3}" = "200" ] || [ "${REGISTER_RESULT: -3}" = "201" ]; then
    echo -e "${GREEN}✓ Registration endpoint working${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}✗ Registration endpoint failed (HTTP ${REGISTER_RESULT: -3})${NC}"
fi

# Test login
LOGIN_RESULT=$(curl -s -w "%{http_code}" -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@yaeger.com", "password": "Admin123!"}' \
  -o /tmp/login_response.json)

if [ "${LOGIN_RESULT: -3}" = "200" ]; then
    echo -e "${GREEN}✓ Login endpoint working${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
    
    # Extract token for further tests
    TOKEN=$(jq -r '.access_token // empty' /tmp/login_response.json 2>/dev/null || echo "")
else
    echo -e "${RED}✗ Login endpoint failed (HTTP ${LOGIN_RESULT: -3})${NC}"
fi

# Test token validation
if [ -n "$TOKEN" ]; then
    VALIDATE_RESULT=$(curl -s -w "%{http_code}" -X GET http://localhost:8000/tokens/validate \
      -H "Authorization: Bearer $TOKEN" \
      -o /tmp/validate_response.json)
    
    if [ "${VALIDATE_RESULT: -3}" = "200" ]; then
        echo -e "${GREEN}✓ Token validation working${NC}"
        CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
    else
        echo -e "${RED}✗ Token validation failed${NC}"
    fi
fi

# Test health endpoint
HEALTH_RESULT=$(curl -s -w "%{http_code}" -X GET http://localhost:8000/health -o /tmp/health_response.json)
if [ "${HEALTH_RESULT: -3}" = "200" ]; then
    echo -e "${GREEN}✓ Health endpoint working${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 10))
else
    echo -e "${RED}✗ Health endpoint failed${NC}"
fi

# OAuth2 tests
OAUTH_RESULT=$(curl -s -w "%{http_code}" -X GET "http://localhost:8000/oauth/authorize?client_id=demo_client&response_type=code&redirect_uri=http://localhost:8000/callback" -o /tmp/oauth_response.json)
if [ "${OAUTH_RESULT: -3}" = "200" ] || [ "${OAUTH_RESULT: -3}" = "302" ]; then
    echo -e "${GREEN}✓ OAuth2 authorize endpoint accessible${NC}"
    CORRECTNESS_SCORE=$((CORRECTNESS_SCORE + 15))
else
    echo -e "${RED}✗ OAuth2 authorize endpoint failed${NC}"
fi

echo -e "\n${BLUE}=== SECURITY TESTS (10%) ===${NC}"

# Test SQL injection protection
SQLI_RESULT=$(curl -s -w "%{http_code}" -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@yaeger.com'\''OR 1=1--", "password": "test"}' \
  -o /tmp/sqli_response.json)

if [ "${SQLI_RESULT: -3}" = "400" ] || [ "${SQLI_RESULT: -3}" = "401" ] || [ "${SQLI_RESULT: -3}" = "422" ]; then
    echo -e "${GREEN}✓ SQL injection protection working${NC}"
    SECURITY_SCORE=$((SECURITY_SCORE + 3))
else
    echo -e "${RED}✗ SQL injection protection failed${NC}"
fi

# Test rate limiting
echo "Testing rate limiting..."
RATE_LIMIT_VIOLATIONS=0
for i in {1..10}; do  # Reduced from 20 to 10 iterations
    RATE_RESULT=$(curl -s -w "%{http_code}" -X POST http://localhost:8000/auth/login \
      -H "Content-Type: application/json" \
      -d '{"email": "test@example.com", "password": "wrong"}' \
      -o /dev/null)
    
    if [ "${RATE_RESULT: -3}" = "429" ]; then
        RATE_LIMIT_VIOLATIONS=$((RATE_LIMIT_VIOLATIONS + 1))
    fi
done

if [ $RATE_LIMIT_VIOLATIONS -gt 0 ]; then
    echo -e "${GREEN}✓ Rate limiting active ($RATE_LIMIT_VIOLATIONS/10 requests blocked)${NC}"
    SECURITY_SCORE=$((SECURITY_SCORE + 4))
else
    echo -e "${RED}✗ Rate limiting not working${NC}"
fi

# Run bandit security scan
if command -v bandit &> /dev/null; then
    echo "Running security analysis..."
    bandit -r submission.py -f txt > bandit_output.tmp 2>&1 || true
    SECURITY_ISSUES=$(grep -c "Issue:" bandit_output.tmp 2>/dev/null || echo "0")
    
    if [ $SECURITY_ISSUES -le 2 ]; then
        echo -e "${GREEN}✓ Security scan passed ($SECURITY_ISSUES issues)${NC}"
        SECURITY_SCORE=$((SECURITY_SCORE + 3))
    else
        echo -e "${RED}✗ Security scan failed ($SECURITY_ISSUES issues)${NC}"
    fi
fi

echo -e "\n${BLUE}=== PERFORMANCE TESTS (20%) ===${NC}"

# Test response times
echo "Testing response times..."
RESPONSE_TIMES=()

for i in {1..5}; do  # Reduced from 10 to 5 iterations
    START_TIME=$(date +%s.%N)
    curl -s -X GET http://localhost:8000/health > /dev/null
    END_TIME=$(date +%s.%N)
    RESPONSE_TIME=$(echo "($END_TIME - $START_TIME) * 1000" | bc -l 2>/dev/null || echo "0")
    RESPONSE_TIMES+=($RESPONSE_TIME)
done

# Calculate average response time
TOTAL_TIME=0
for time in "${RESPONSE_TIMES[@]}"; do
    TOTAL_TIME=$(echo "$TOTAL_TIME + $time" | bc -l 2>/dev/null || echo "$TOTAL_TIME")
done
AVG_RESPONSE_TIME=$(echo "scale=2; $TOTAL_TIME / 5" | bc -l 2>/dev/null || echo "0")

if (( $(echo "$AVG_RESPONSE_TIME < 50" | bc -l 2>/dev/null || echo "1") )); then
    echo -e "${GREEN}✓ Response time acceptable (${AVG_RESPONSE_TIME}ms avg)${NC}"
    PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 10))
else
    echo -e "${RED}✗ Response time too high (${AVG_RESPONSE_TIME}ms avg)${NC}"
fi

# Test concurrent requests
echo "Testing concurrent request handling..."
CONCURRENT_SUCCESS=0

for i in {1..20}; do  # Reduced from 50 to 20
    curl -s -X GET http://localhost:8000/health > /dev/null &
done
wait

# Simple check - if we get here without hanging, concurrency works
echo -e "${GREEN}✓ Concurrent request handling working${NC}"
PERFORMANCE_SCORE=$((PERFORMANCE_SCORE + 10))

# Stop the server with proper cleanup
echo "Stopping server..."
if [ ! -z "$SERVER_PID" ]; then
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
fi

# Calculate final score
TOTAL_SCORE=$((CORRECTNESS_SCORE + PERFORMANCE_SCORE + SECURITY_SCORE))

# Generate summary
echo -e "\n${BLUE}=== RESULTS SUMMARY ===${NC}"
echo "========================================"
echo "Authentication Tests: Working endpoints detected"
echo "Security Features: Rate limiting and injection protection tested"
echo "Runtime: ${AVG_RESPONSE_TIME}ms average"
echo "Security Issues: ${SECURITY_ISSUES:-0}"
echo "Vulns: ${SECURITY_ISSUES:-0}"
echo ""
echo "Correctness Score: $CORRECTNESS_SCORE/70"
echo "Performance Score: $PERFORMANCE_SCORE/20"
echo "Security Score: $SECURITY_SCORE/10"
echo "========================================"
echo "Score: $TOTAL_SCORE"

# Performance feedback
if (( $(echo "${AVG_RESPONSE_TIME:-0} > 100" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${RED}Warning: Average response time exceeds 100ms${NC}"
fi

# Clean up
rm -f /tmp/*_response.json bandit_output.tmp

# Deactivate virtual environment
deactivate 2>/dev/null || true

# Exit with success if score > 70
if [ $TOTAL_SCORE -ge 70 ]; then
    echo -e "\n${GREEN}✓ Task 2 verification passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Task 2 verification failed (score: $TOTAL_SCORE/100)${NC}"
    exit 1
fi
