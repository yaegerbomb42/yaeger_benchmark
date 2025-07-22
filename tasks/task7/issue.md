# Task 7: Edge Computing Load Balancer

## Problem Description

Design a globally distributed load balancer that routes traffic to the nearest edge locations with intelligent failover, geographic routing, and real-time performance optimization.

## Requirements

### Functional Requirements
- Geographic traffic routing (GeoDNS)
- Health checking and automatic failover
- Load balancing algorithms (round-robin, least-connections, latency-based)
- SSL termination and certificate management
- Rate limiting and DDoS protection

### Performance Requirements
- **Latency**: <50ms routing decisions
- **Throughput**: 100,000+ requests/second globally  
- **Availability**: 99.99% uptime across all regions
- **Scalability**: Support 50+ edge locations

## Key Components

1. **Traffic Manager**: Global routing decisions
2. **Health Monitor**: Real-time endpoint checking
3. **Analytics**: Traffic patterns and performance metrics
4. **Security**: DDoS mitigation and access control

## Evaluation Criteria

- **Correctness (70%)**: Accurate routing, failover logic
- **Performance (20%)**: Routing latency, throughput
- **Security (10%)**: DDoS protection, secure communications
