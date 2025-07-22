# Task 2: Secure Microservice Authentication

## Problem Description

Design and implement a production-ready authentication service for a microservice architecture that supports JWT tokens, OAuth2 flows, and advanced security features including rate limiting, intrusion detection, and comprehensive audit logging.

## Background

You are building the central authentication service for a high-traffic e-commerce platform with the following requirements:
- Handle 10,000+ authentication requests per second
- Support multiple authentication methods (JWT, OAuth2, API keys)
- Integrate with external identity providers (Google, GitHub, etc.)
- Provide comprehensive security features against common attacks
- Maintain detailed audit logs for compliance

## Requirements

### Functional Requirements
- Implement JWT token generation and validation
- Support OAuth2 authorization code flow
- API key management for service-to-service authentication
- User registration and login with secure password handling
- Multi-factor authentication (MFA) support
- Session management with configurable expiration

### Security Requirements
- **Input Validation**: Prevent SQL injection, XSS, CSRF attacks
- **Rate Limiting**: Configurable per-IP and per-user limits
- **Password Security**: Bcrypt hashing, complexity requirements, breach detection
- **Token Security**: Secure token generation, rotation, blacklisting
- **Audit Logging**: Comprehensive security event logging
- **OWASP Compliance**: Pass OWASP ZAP security scan

### Performance Requirements
- **Throughput**: Handle 10,000 requests/second
- **Latency**: <50ms response time for token validation
- **Scalability**: Support horizontal scaling
- **Availability**: 99.9% uptime with graceful degradation

## Starter Code

The service is built with FastAPI and includes:
- `auth_service.py`: Main authentication endpoints
- `token_manager.py`: JWT and token management
- `oauth_handler.py`: OAuth2 flow implementation
- `security_middleware.py`: Security and rate limiting
- `models.py`: Data models and schemas
- `submission.py`: Your implementation goes here

## API Endpoints

### Authentication
```
POST /auth/register
POST /auth/login
POST /auth/logout
POST /auth/refresh
POST /auth/mfa/enable
POST /auth/mfa/verify
```

### OAuth2
```
GET /oauth/authorize
POST /oauth/token
GET /oauth/userinfo
```

### Token Management
```
POST /tokens/api-key
DELETE /tokens/{token_id}
GET /tokens/validate
```

### Admin
```
GET /admin/users
GET /admin/audit-logs
POST /admin/rate-limits
```

## Security Features to Implement

### 1. Input Validation and Sanitization
```python
from pydantic import BaseModel, validator
from typing import Optional
import re

class LoginRequest(BaseModel):
    email: str
    password: str
    
    @validator('email')
    def validate_email(cls, v):
        # Implement comprehensive email validation
        pass
    
    @validator('password')
    def validate_password(cls, v):
        # Implement password strength validation
        pass
```

### 2. Rate Limiting
```python
from functools import wraps
from collections import defaultdict
import time

def rate_limit(max_requests: int, window_seconds: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Implement sliding window rate limiting
            pass
        return wrapper
    return decorator
```

### 3. JWT Security
```python
import jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization

class TokenManager:
    def generate_jwt(self, payload: dict) -> str:
        # Implement secure JWT generation
        pass
    
    def validate_jwt(self, token: str) -> dict:
        # Implement JWT validation with security checks
        pass
```

### 4. OAuth2 Implementation
```python
from authlib.integrations.fastapi_oauth2 import OAuth2

class OAuth2Handler:
    def __init__(self):
        # Initialize OAuth2 providers
        pass
    
    async def authorize(self, provider: str, redirect_uri: str):
        # Implement OAuth2 authorization flow
        pass
```

## Evaluation Criteria

### Correctness (70 points)
- **Authentication Flow** (25 points): Login, registration, logout work correctly
- **JWT Implementation** (20 points): Secure token generation and validation
- **OAuth2 Flow** (15 points): Complete OAuth2 implementation
- **API Security** (10 points): Input validation, error handling

### Performance (20 points)
- **Throughput** (10 points): Handle 10,000+ requests/second
- **Latency** (5 points): <50ms response time
- **Memory Efficiency** (5 points): <1GB memory usage under load

### Security (10 points)
- **OWASP Compliance** (5 points): Pass security scan
- **Rate Limiting** (3 points): Effective protection against abuse
- **Audit Logging** (2 points): Comprehensive security logging

## Test Scenarios

### 1. Basic Authentication Flow
```bash
# User registration
curl -X POST /auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!"}'

# User login
curl -X POST /auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!"}'
```

### 2. Security Testing
```bash
# SQL injection attempt
curl -X POST /auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com'\''OR 1=1--", "password": "test"}'

# Rate limiting test
for i in {1..100}; do
  curl -X POST /auth/login \
    -H "Content-Type: application/json" \
    -d '{"email": "test@example.com", "password": "wrong"}' &
done
```

### 3. OAuth2 Flow
```bash
# Start OAuth2 flow
curl -X GET "/oauth/authorize?client_id=test&response_type=code&redirect_uri=http://localhost:8000/callback"

# Exchange code for token
curl -X POST /oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code&code=AUTH_CODE&client_id=test&client_secret=secret"
```

## Common Security Vulnerabilities to Prevent

### 1. Authentication Bypass
- Weak password policies
- Insufficient session management
- JWT algorithm confusion attacks

### 2. Injection Attacks
- SQL injection in login forms
- NoSQL injection in user queries
- LDAP injection in directory lookups

### 3. Broken Access Control
- Insecure direct object references
- Missing authorization checks
- Privilege escalation vulnerabilities

### 4. Security Misconfigurations
- Default credentials
- Unnecessary features enabled
- Verbose error messages

## Implementation Tips

### 1. Password Security
```python
import bcrypt
import hashlib

def hash_password(password: str) -> str:
    # Use bcrypt with appropriate rounds
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
```

### 2. Secure Token Storage
```python
import secrets
from cryptography.fernet import Fernet

class SecureTokenStore:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_token(self, token: str) -> str:
        return self.cipher.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        return self.cipher.decrypt(encrypted_token.encode()).decode()
```

### 3. Comprehensive Logging
```python
import logging
import json
from datetime import datetime

class SecurityLogger:
    def __init__(self):
        self.logger = logging.getLogger('security')
    
    def log_auth_attempt(self, email: str, success: bool, ip: str):
        self.logger.info(json.dumps({
            'event': 'auth_attempt',
            'email': email,
            'success': success,
            'ip': ip,
            'timestamp': datetime.utcnow().isoformat()
        }))
```

## Submission Format

Implement your solution in `submission.py`:

```python
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio

app = FastAPI(title="Secure Authentication Service")

@app.post("/auth/register")
async def register(request: RegisterRequest):
    """User registration endpoint"""
    # Your implementation here
    pass

@app.post("/auth/login")
async def login(request: LoginRequest):
    """User login endpoint"""
    # Your implementation here
    pass

# Additional endpoints...
```

## Performance Testing

The verification script includes load testing:
- 10,000 concurrent authentication requests
- Rate limiting effectiveness testing
- Memory and CPU usage monitoring
- Response time percentile analysis

## Security Scanning

Automated security tests include:
- OWASP ZAP dynamic application security testing
- Static code analysis for common vulnerabilities
- Dependency vulnerability scanning
- Penetration testing simulation
