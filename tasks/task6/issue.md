# Task 6: Blockchain Transaction Validator

## Problem Description

Implement a blockchain consensus mechanism with transaction validation, smart contract execution, and Byzantine fault tolerance for a cryptocurrency network.

## Requirements

### Functional Requirements
- Transaction validation and digital signatures
- Proof-of-Stake consensus algorithm
- Smart contract virtual machine
- Block creation and validation
- Network synchronization

### Performance Requirements
- **Throughput**: 1,000+ transactions/second
- **Latency**: <10 seconds for block confirmation
- **Fault Tolerance**: Handle up to 33% malicious nodes
- **Scalability**: Support 100+ validator nodes

## Key Features

1. **Consensus**: Proof-of-Stake with slashing
2. **VM**: Execute smart contracts safely
3. **Networking**: P2P communication protocol
4. **Cryptography**: Ed25519 signatures, Merkle trees

## Evaluation Criteria

- **Correctness (70%)**: Valid consensus, transaction integrity
- **Performance (20%)**: Transaction throughput, block time
- **Security (10%)**: Cryptographic security, fault tolerance
