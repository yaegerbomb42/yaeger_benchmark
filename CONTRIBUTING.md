# Contributing to Yaeger Benchmark

Thank you for your interest in contributing to the Yaeger Benchmark! This document outlines how to contribute effectively.

## How to Submit Solutions

### Via Pull Request (Recommended)

1. **Fork the repository**
   ```bash
   git clone https://github.com/yaegerbomb42/yaeger-benchmark.git
   cd yaeger-benchmark
   ```

2. **Create a task branch**
   ```bash
   git checkout -b task1  # Replace with your task number
   ```

3. **Implement your solution**
   - Edit the `submission.py` file in the appropriate task directory
   - Follow the requirements outlined in `issue.md`
   - Test your implementation locally

4. **Submit your solution**
   ```bash
   git add tasks/task1/submission.py
   git commit -m "Implement Task 1: Real-Time Trading Optimizer"
   git push origin task1
   ```

5. **Open a Pull Request**
   - Title must match the task ID (e.g., "task1")
   - Include a description of your approach
   - Wait for automated verification results

### Via API

```bash
curl -X POST https://yaeger-benchmark.vercel.app/submit/task1 \
  -H "Content-Type: application/json" \
  -d @solution.json
```

## Submission Guidelines

### Code Quality
- Write clean, readable code with proper documentation
- Follow language-specific best practices
- Include error handling and edge case management
- Optimize for the specified performance requirements

### Testing
- Test your solution against the provided test cases
- Verify performance requirements are met
- Ensure security standards are followed
- Test edge cases and failure scenarios

### Documentation
- Document your approach and algorithms used
- Explain any trade-offs or assumptions made
- Include setup instructions if dependencies are required

## Evaluation Criteria

Each task is evaluated on three dimensions:

### Correctness (70%)
- **Functional Requirements**: Does the solution meet all specified requirements?
- **Test Coverage**: Does it pass all unit tests and integration tests?
- **Edge Cases**: How well does it handle boundary conditions and errors?
- **API Compliance**: Does it correctly implement the required interfaces?

### Performance (20%)
- **Speed**: Does it meet latency and throughput requirements?
- **Scalability**: How does it perform under increasing load?
- **Resource Usage**: Does it stay within memory and CPU limits?
- **Efficiency**: Are algorithms and data structures optimal?

### Security (10%)
- **Input Validation**: Proper sanitization and validation of inputs
- **Vulnerability Scanning**: No common security vulnerabilities
- **Access Control**: Appropriate authentication and authorization
- **Data Protection**: Secure handling of sensitive information

## Scoring System

- **Score Range**: 0-100 points per task
- **Passing Score**: 70+ points required to pass
- **Leaderboard**: Ranked by total score across all attempted tasks
- **Tiebreaker**: Average score per completed task

## Common Issues and Solutions

### Performance Problems
- **Memory Usage**: Use efficient data structures, avoid memory leaks
- **CPU Usage**: Optimize algorithms, use appropriate concurrency
- **I/O Bottlenecks**: Implement proper caching and batching

### Security Issues
- **Input Validation**: Always validate and sanitize user inputs
- **SQL Injection**: Use parameterized queries or ORMs
- **Authentication**: Implement proper session management
- **Encryption**: Use established cryptographic libraries

### Reliability Problems
- **Error Handling**: Implement comprehensive error handling
- **Graceful Degradation**: Handle failures gracefully
- **Resource Cleanup**: Properly manage resources and connections
- **Testing**: Include unit tests and integration tests

## Task-Specific Tips

### Task 1: Trading Algorithm
- Focus on latency optimization and risk management
- Implement proper backtesting and performance metrics
- Consider market microstructure and realistic constraints

### Task 2: Authentication Service
- Prioritize security best practices and OWASP compliance
- Implement proper rate limiting and audit logging
- Use established cryptographic libraries

### Task 3: Data Pipeline
- Optimize for throughput using async processing
- Implement efficient windowing and aggregation algorithms
- Consider memory usage for large data volumes

## Getting Help

- **Documentation**: Check the README and task-specific documentation
- **Issues**: Open a GitHub issue for bug reports or questions
- **Discussions**: Use GitHub Discussions for general questions
- **API Documentation**: Available at `/docs` endpoint when API is running

## Code of Conduct

- Be respectful and professional in all interactions
- Focus on constructive feedback and learning
- Help others learn and improve their solutions
- Follow GitHub's community guidelines

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
