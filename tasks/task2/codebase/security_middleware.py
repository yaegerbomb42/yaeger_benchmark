from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import json
import hashlib
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from .models import RateLimitConfig, AuditLog

# Configure security logger
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
security_logger.addHandler(handler)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with sliding window and adaptive limits."""
    
    def __init__(self, app, config: RateLimitConfig = None, **kwargs):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        
        # Storage for rate limit counters
        # Format: {key: deque([timestamp1, timestamp2, ...])}
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Adaptive rate limiting - track failed attempts
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        
        # Cleanup task
        self.cleanup_task = None
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting."""
        
        # Get client identifier
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent', 'unknown')
        
        # Check if IP is temporarily blocked
        if await self._is_ip_blocked(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "error_description": "IP temporarily blocked due to excessive requests",
                    "retry_after": 300
                }
            )
        
        # Apply endpoint-specific rate limits
        endpoint = f"{request.method}:{request.url.path}"
        rate_limit_key = f"{client_ip}:{endpoint}"
        
        # Get rate limit for this endpoint
        limit, window = self._get_rate_limit_for_endpoint(endpoint)
        
        # Check rate limit
        if not await self._check_rate_limit(rate_limit_key, limit, window):
            # Log rate limit violation
            await self._log_security_event(
                "rate_limit_exceeded",
                f"Rate limit exceeded for {endpoint}",
                client_ip,
                user_agent,
                {"endpoint": endpoint, "limit": limit, "window": window}
            )
            
            # Track failed attempts for adaptive limiting
            self.failed_attempts[client_ip] += 1
            
            # Block IP if too many violations
            if self.failed_attempts[client_ip] >= 10:
                self.blocked_ips[client_ip] = datetime.utcnow() + timedelta(minutes=5)
                await self._log_security_event(
                    "ip_blocked",
                    f"IP blocked due to excessive rate limit violations",
                    client_ip,
                    user_agent
                )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "error_description": f"Rate limit exceeded: {limit} requests per {window} seconds",
                    "retry_after": window
                }
            )
        
        # Process request
        start_time = time.time()
        try:
            response = await call_next(request)
            
            # Add rate limit headers
            remaining = await self._get_remaining_requests(rate_limit_key, limit, window)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + window))
            
            # Reset failed attempts on successful request
            if response.status_code < 400:
                self.failed_attempts[client_ip] = 0
            
            return response
            
        except Exception as e:
            # Log security events for errors
            await self._log_security_event(
                "request_error",
                f"Request processing error: {str(e)}",
                client_ip,
                user_agent,
                {"endpoint": endpoint, "error": str(e)}
            )
            raise
        
        finally:
            # Log request timing
            duration = time.time() - start_time
            if duration > 5.0:  # Log slow requests
                await self._log_security_event(
                    "slow_request",
                    f"Slow request detected",
                    client_ip,
                    user_agent,
                    {"endpoint": endpoint, "duration": duration}
                )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to client host
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return 'unknown'
    
    def _get_rate_limit_for_endpoint(self, endpoint: str) -> Tuple[int, int]:
        """Get rate limit configuration for specific endpoint."""
        
        # Authentication endpoints - stricter limits
        if '/auth/login' in endpoint:
            return self.config.login_attempts_per_minute, 60
        elif '/auth/register' in endpoint:
            return self.config.registration_attempts_per_hour, 3600
        elif '/auth/password-reset' in endpoint:
            return self.config.password_reset_attempts_per_hour, 3600
        
        # API endpoints - general limits
        elif endpoint.startswith('POST:') or endpoint.startswith('PUT:') or endpoint.startswith('DELETE:'):
            return self.config.api_requests_per_minute, 60
        
        # Default for GET requests
        else:
            return self.config.api_requests_per_minute * 2, 60
    
    async def _check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit using sliding window."""
        now = time.time()
        cutoff = now - window
        
        # Clean old entries
        request_times = self.request_counts[key]
        while request_times and request_times[0] < cutoff:
            request_times.popleft()
        
        # Check if within limit
        if len(request_times) >= limit:
            return False
        
        # Add current request
        request_times.append(now)
        return True
    
    async def _get_remaining_requests(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests for rate limit key."""
        now = time.time()
        cutoff = now - window
        
        request_times = self.request_counts[key]
        
        # Count valid requests
        valid_requests = sum(1 for t in request_times if t >= cutoff)
        return max(0, limit - valid_requests)
    
    async def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is temporarily blocked."""
        if ip in self.blocked_ips:
            if datetime.utcnow() < self.blocked_ips[ip]:
                return True
            else:
                # Unblock expired blocks
                del self.blocked_ips[ip]
                self.failed_attempts[ip] = 0
        return False
    
    async def _log_security_event(self, event_type: str, description: str, 
                                 ip: str, user_agent: str, additional_data: dict = None):
        """Log security events."""
        event_data = {
            'event_type': event_type,
            'description': description,
            'ip_address': ip,
            'user_agent': user_agent,
            'timestamp': datetime.utcnow().isoformat(),
            'additional_data': additional_data or {}
        }
        
        security_logger.warning(json.dumps(event_data))

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize incoming requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Check request size
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return JSONResponse(
                status_code=413,
                content={"error": "payload_too_large", "error_description": "Request payload too large"}
            )
        
        # Check for suspicious patterns in URL
        suspicious_patterns = ['../', '<script', 'javascript:', 'data:', 'vbscript:']
        url_path = str(request.url.path).lower()
        
        for pattern in suspicious_patterns:
            if pattern in url_path:
                client_ip = request.headers.get('x-forwarded-for', '').split(',')[0].strip()
                if not client_ip:
                    client_ip = getattr(request.client, 'host', 'unknown')
                
                security_logger.warning(json.dumps({
                    'event_type': 'suspicious_request',
                    'description': f'Suspicious pattern detected in URL: {pattern}',
                    'ip_address': client_ip,
                    'url': str(request.url),
                    'timestamp': datetime.utcnow().isoformat()
                }))
                
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid_request", "error_description": "Invalid request format"}
                )
        
        return await call_next(request)

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive audit logging for security events."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.audit_logs: list = []
    
    async def dispatch(self, request: Request, call_next):
        # Capture request details
        client_ip = request.headers.get('x-forwarded-for', '').split(',')[0].strip()
        if not client_ip:
            client_ip = getattr(request.client, 'host', 'unknown')
        
        user_agent = request.headers.get('user-agent', 'unknown')
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log based on response status and endpoint
            await self._log_request(
                request, response, client_ip, user_agent, 
                time.time() - start_time
            )
            
            return response
            
        except Exception as e:
            # Log errors
            await self._log_error(request, client_ip, user_agent, str(e))
            raise
    
    async def _log_request(self, request: Request, response: Response, 
                          ip: str, user_agent: str, duration: float):
        """Log request details based on sensitivity."""
        
        endpoint = f"{request.method} {request.url.path}"
        
        # Always log authentication endpoints
        auth_endpoints = ['/auth/', '/oauth/', '/admin/']
        should_log = any(auth_ep in endpoint for auth_ep in auth_endpoints)
        
        # Log failed requests
        if response.status_code >= 400:
            should_log = True
        
        # Log slow requests
        if duration > 2.0:
            should_log = True
        
        if should_log:
            log_entry = AuditLog(
                id=hashlib.md5(f"{time.time()}{ip}{endpoint}".encode()).hexdigest(),
                user_id=None,  # Will be filled by auth layer if available
                event_type="api_request",
                description=f"{endpoint} - {response.status_code}",
                ip_address=ip,
                user_agent=user_agent,
                created_at=datetime.utcnow(),
                additional_data={
                    "method": request.method,
                    "path": str(request.url.path),
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "query_params": dict(request.query_params) if request.query_params else {}
                }
            )
            
            self.audit_logs.append(log_entry)
            
            # Also log to security logger for immediate monitoring
            security_logger.info(json.dumps({
                "event": "api_request",
                "endpoint": endpoint,
                "status": response.status_code,
                "ip": ip,
                "duration_ms": round(duration * 1000, 2),
                "timestamp": datetime.utcnow().isoformat()
            }))
    
    async def _log_error(self, request: Request, ip: str, user_agent: str, error: str):
        """Log request errors."""
        log_entry = AuditLog(
            id=hashlib.md5(f"{time.time()}{ip}{error}".encode()).hexdigest(),
            user_id=None,
            event_type="api_error",
            description=f"Request error: {error}",
            ip_address=ip,
            user_agent=user_agent,
            created_at=datetime.utcnow(),
            additional_data={
                "method": request.method,
                "path": str(request.url.path),
                "error": error
            }
        )
        
        self.audit_logs.append(log_entry)
        
        security_logger.error(json.dumps({
            "event": "api_error",
            "endpoint": f"{request.method} {request.url.path}",
            "error": error,
            "ip": ip,
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    def get_audit_logs(self, limit: int = 100, event_type: str = None) -> list:
        """Retrieve audit logs with optional filtering."""
        logs = self.audit_logs
        
        if event_type:
            logs = [log for log in logs if log.event_type == event_type]
        
        # Sort by timestamp descending and limit
        logs.sort(key=lambda x: x.created_at, reverse=True)
        return logs[:limit]
