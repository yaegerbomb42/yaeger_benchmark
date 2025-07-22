# Task 10: Network Protocol Optimizer

## Problem Description

Design a custom network protocol optimized for IoT devices that minimizes bandwidth usage, provides reliable delivery, and handles intermittent connectivity in resource-constrained environments.

## Requirements

### Functional Requirements
- Custom binary protocol with minimal overhead
- Reliable delivery with automatic retry
- Message compression and batching
- Device authentication and encryption
- Offline mode with message queuing

### Performance Requirements
- **Bandwidth**: <50% overhead compared to raw payload
- **Latency**: <100ms message delivery in good conditions
- **Reliability**: 99.9% message delivery rate
- **Battery**: Optimize for low-power devices

## Key Features

1. **Protocol**: Compact binary format with optional fields
2. **Reliability**: Acknowledgments and retransmission
3. **Security**: Device certificates and message encryption
4. **Efficiency**: Delta compression and message batching

## Evaluation Criteria

- **Correctness (70%)**: Reliable delivery, protocol compliance
- **Performance (20%)**: Bandwidth efficiency, latency
- **Security (10%)**: Encryption, authentication mechanisms
