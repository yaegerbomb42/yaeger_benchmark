# Yaeger Benchmark

The Yaeger Benchmark is a comprehensive evaluation suite for AI coding agents, featuring 10 challenging, real-world programming tasks that require advanced problem-solving skills, multi-step reasoning, and interdisciplinary knowledge.

## Overview

The benchmark focuses on practical, grounded problems that are:
- **Extremely challenging**: Requiring 2-4 hours for expert human developers
- **Instantly verifiable**: REST API for immediate feedback
- **Novel**: No existing solutions on Stack Overflow or common datasets
- **Comprehensive**: Testing correctness, efficiency, and security

## Quick Start

### Submit via API
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"code": "def solve(): ..."}' \
  https://yaeger-benchmark.vercel.app/submit/task1
```

### Submit via Pull Request
1. Fork this repository
2. Create a branch named after the task (e.g., `task1`)
3. Implement your solution in the appropriate task directory
4. Open a pull request with the title matching the task ID

## Tasks Overview

1. **Real-Time Trading Optimizer** - High-frequency trading algorithm with latency constraints
2. **Secure Microservice Authentication** - JWT/OAuth2 service with security hardening
3. **Dynamic Data Pipeline** - Stream processing with 1M records/second throughput
4. **Distributed Cache System** - Fault-tolerant caching with consistency guarantees
5. **ML Model Serving API** - Production ML inference with A/B testing
6. **Blockchain Transaction Validator** - Consensus algorithm implementation
7. **Edge Computing Load Balancer** - Geographic traffic distribution
8. **Real-Time Analytics Engine** - Low-latency metric aggregation
9. **Secure File Storage System** - Encrypted storage with access controls
10. **Network Protocol Optimizer** - Custom protocol for IoT devices

## Scoring

Each task is scored 0-100 based on:
- **Correctness (70%)**: Unit test pass rate
- **Efficiency (20%)**: Runtime and memory performance
- **Security (10%)**: Vulnerability scanning results

## Local Development

### Prerequisites
- Python 3.9+
- Docker
- Node.js 16+ (for some tasks)

### Setup
```bash
git clone https://github.com/yaegerbomb42/yaeger-benchmark.git
cd yaeger-benchmark
pip install -r requirements.txt
```

### Run All Tests
```bash
./verify/verify_all.sh
```

### Start API Server
```bash
cd submit
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Reference

### Submit Solution
```
POST /submit/{task_id}
```

**Request Body:**
```json
{
  "code": "string",
  "language": "python|javascript|go"
}
```

**Response:**
```json
{
  "task_id": "task1",
  "score": 85.0,
  "output": "Tests: 85/100 passed\nRuntime: 90ms\nVulns: 0",
  "details": {
    "tests_passed": 85,
    "runtime": 90.0,
    "vulnerabilities": 0,
    "memory_usage": 45.2
  }
}
```

### Get Leaderboard
```
GET /leaderboard
```

### Get Task Details
```
GET /task/{task_id}
```

## Contributing

We welcome contributions to improve the benchmark! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.