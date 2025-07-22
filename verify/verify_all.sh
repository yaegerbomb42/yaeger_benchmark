#!/bin/bash

# Yaeger Benchmark - Global Verification Script
# Runs all task verifications and generates a summary report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TASKS_DIR="$BASE_DIR/tasks"
RESULTS_DIR="$BASE_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Yaeger Benchmark - Global Verification${NC}"
echo "======================================"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Initialize summary
TOTAL_TASKS=0
PASSED_TASKS=0
TOTAL_SCORE=0

# Summary file
SUMMARY_FILE="$RESULTS_DIR/verification_summary.txt"
echo "Yaeger Benchmark Verification Summary" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"

# Verify each task
for i in {1..10}; do
    TASK_ID="task$i"
    TASK_DIR="$TASKS_DIR/$TASK_ID"
    
    echo -e "\n${YELLOW}Verifying $TASK_ID...${NC}"
    
    if [ ! -d "$TASK_DIR" ]; then
        echo -e "${RED}Task directory not found: $TASK_DIR${NC}"
        echo "$TASK_ID: MISSING" >> "$SUMMARY_FILE"
        continue
    fi
    
    VERIFY_SCRIPT="$TASK_DIR/verify.sh"
    if [ ! -f "$VERIFY_SCRIPT" ]; then
        echo -e "${RED}Verification script not found: $VERIFY_SCRIPT${NC}"
        echo "$TASK_ID: NO_VERIFY_SCRIPT" >> "$SUMMARY_FILE"
        continue
    fi
    
    # Make script executable
    chmod +x "$VERIFY_SCRIPT"
    
    TOTAL_TASKS=$((TOTAL_TASKS + 1))
    
    # Run verification
    if cd "$TASK_DIR" && ./verify.sh > "$RESULTS_DIR/${TASK_ID}_result.txt" 2>&1; then
        SCORE=$(grep "Score:" "$RESULTS_DIR/${TASK_ID}_result.txt" | awk '{print $2}' || echo "0")
        TESTS_PASSED=$(grep "Tests:" "$RESULTS_DIR/${TASK_ID}_result.txt" | head -1 || echo "Tests: 0/0")
        RUNTIME=$(grep "Runtime:" "$RESULTS_DIR/${TASK_ID}_result.txt" | awk '{print $2}' || echo "0ms")
        VULNS=$(grep "Vulns:" "$RESULTS_DIR/${TASK_ID}_result.txt" | awk '{print $2}' || echo "0")
        
        echo -e "${GREEN}✓ $TASK_ID completed - Score: $SCORE${NC}"
        echo "$TASK_ID: PASS (Score: $SCORE, $TESTS_PASSED, Runtime: $RUNTIME, Vulns: $VULNS)" >> "$SUMMARY_FILE"
        
        PASSED_TASKS=$((PASSED_TASKS + 1))
        TOTAL_SCORE=$(echo "$TOTAL_SCORE + $SCORE" | bc -l 2>/dev/null || echo "$TOTAL_SCORE")
    else
        echo -e "${RED}✗ $TASK_ID failed${NC}"
        echo "$TASK_ID: FAIL" >> "$SUMMARY_FILE"
        
        # Show error details
        if [ -f "$RESULTS_DIR/${TASK_ID}_result.txt" ]; then
            echo "Error details:"
            tail -n 5 "$RESULTS_DIR/${TASK_ID}_result.txt"
        fi
    fi
done

# Generate final summary
echo -e "\n${BLUE}Verification Summary${NC}"
echo "===================="
echo -e "Tasks verified: ${YELLOW}$TOTAL_TASKS${NC}"
echo -e "Tasks passed: ${GREEN}$PASSED_TASKS${NC}"
echo -e "Tasks failed: ${RED}$((TOTAL_TASKS - PASSED_TASKS))${NC}"

if [ "$TOTAL_TASKS" -gt 0 ]; then
    PASS_RATE=$(echo "scale=2; $PASSED_TASKS * 100 / $TOTAL_TASKS" | bc -l 2>/dev/null || echo "0")
    echo -e "Pass rate: ${YELLOW}$PASS_RATE%${NC}"
fi

if command -v bc &> /dev/null; then
    AVG_SCORE=$(echo "scale=2; $TOTAL_SCORE / $TOTAL_TASKS" | bc -l 2>/dev/null || echo "0")
    echo -e "Average score: ${YELLOW}$AVG_SCORE${NC}"
    echo -e "Total score: ${YELLOW}$TOTAL_SCORE${NC}"
fi

# Add summary to file
echo "" >> "$SUMMARY_FILE"
echo "SUMMARY:" >> "$SUMMARY_FILE"
echo "Tasks verified: $TOTAL_TASKS" >> "$SUMMARY_FILE"
echo "Tasks passed: $PASSED_TASKS" >> "$SUMMARY_FILE"
echo "Tasks failed: $((TOTAL_TASKS - PASSED_TASKS))" >> "$SUMMARY_FILE"

if [ "$TOTAL_TASKS" -gt 0 ] && command -v bc &> /dev/null; then
    PASS_RATE=$(echo "scale=2; $PASSED_TASKS * 100 / $TOTAL_TASKS" | bc -l)
    AVG_SCORE=$(echo "scale=2; $TOTAL_SCORE / $TOTAL_TASKS" | bc -l)
    echo "Pass rate: $PASS_RATE%" >> "$SUMMARY_FILE"
    echo "Average score: $AVG_SCORE" >> "$SUMMARY_FILE"
    echo "Total score: $TOTAL_SCORE" >> "$SUMMARY_FILE"
fi

echo -e "\n${BLUE}Results saved to: $SUMMARY_FILE${NC}"

# Exit with error if any tasks failed
if [ "$PASSED_TASKS" -ne "$TOTAL_TASKS" ]; then
    exit 1
fi

echo -e "\n${GREEN}All tasks verified successfully!${NC}"
