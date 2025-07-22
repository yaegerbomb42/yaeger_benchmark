# Task 3: Dynamic Data Pipeline

## Problem Description

Build a high-performance data pipeline that processes streaming JSON data at 1 million records per second, performs real-time aggregations, and maintains sub-second latency for query responses.

## Requirements

### Functional Requirements
- Process 1M+ JSON records per second
- Real-time metric calculations (sum, average, count, percentiles)
- Time-window aggregations (1min, 5min, 1hour)
- Configurable data retention policies
- Support for data filtering and transformations

### Performance Requirements
- **Throughput**: 1,000,000 records/second sustained
- **Latency**: <1 second for aggregation queries
- **Memory**: <2GB memory usage
- **Reliability**: Handle failures gracefully

### Data Format
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "user_id": "user123",
  "event_type": "purchase",
  "amount": 99.99,
  "metadata": {
    "product_id": "prod456",
    "category": "electronics"
  }
}
```

## Implementation Guide

Your solution should implement:

1. **Stream Processor**: Ingest and process JSON streams
2. **Aggregation Engine**: Real-time metric calculations  
3. **Query Interface**: Fast data retrieval
4. **Storage Layer**: Efficient data storage and retrieval

## Evaluation Criteria

- **Correctness (70%)**: Accurate aggregations, proper error handling
- **Performance (20%)**: Throughput and latency requirements
- **Security (10%)**: Input validation, resource limits
