"""
Security middleware for authentication service.
Implements rate limiting, request validation, and audit logging.
"""
import time
import hashlib
import json
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse."""
    
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 3600, **kwargs):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        # Check rate limit
        now = time.time()
        client_requests = self.requests[client_ip]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < now - self.window_seconds:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # Add current request
        client_requests.append(now)
        
        # Continue to next middleware/app
        response = await call_next(request)
        return response

class SecurityValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for security validation and input sanitization."""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "subprocess",
            "os.system",
            "'; DROP",
            "' OR '1'='1",
            "UNION SELECT",
            "<script",
            "javascript:",
            "vbscript:",
            "onload=",
            "onerror="
        ]
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Check request body for malicious patterns
        if request.method == "POST":
            try:
                body = await request.body()
                body_str = body.decode("utf-8").lower()
                
                for pattern in self.dangerous_patterns:
                    if pattern.lower() in body_str:
                        logger.warning(f"Blocked malicious request with pattern: {pattern}")
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Invalid request content"}
                        )
            except (UnicodeDecodeError, Exception):
                pass
        
        response = await call_next(request)
        return response

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging and monitoring."""
    
    def __init__(self, app, log_sensitive_data: bool = False, **kwargs):
        super().__init__(app)
        self.log_sensitive_data = log_sensitive_data
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        method = request.method
        path = str(request.url.path)
        
        logger.info(f"Request: {method} {path} from {client_ip}")
        
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(f"Response: {response.status_code} for {method} {path} ({duration:.3f}s)")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request {method} {path}: {str(e)}")
            raise

class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware."""
    
    def __init__(self, app, secret_key: str = 'default-csrf-secret', **kwargs):
        super().__init__(app)
        self.secret_key = secret_key
    
    async def dispatch(self, request: Request, call_next: Callable):
        method = request.method
        path = str(request.url.path)
        
        # Skip CSRF for safe methods and auth endpoints
        if method in ["GET", "HEAD", "OPTIONS"] or path.startswith("/auth"):
            response = await call_next(request)
            return response
        
        # Check for CSRF token in headers
        csrf_token = request.headers.get("x-csrf-token")
        
        if not csrf_token:
            return JSONResponse(
                status_code=403,
                content={"error": "CSRF token required"}
            )
        
        # In a real implementation, you would validate the token
        # For this demo, we'll accept any non-empty token
        if len(csrf_token) < 8:
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid CSRF token"}
            )
        
        response = await call_next(request)
        return response

class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware for cross-origin requests."""
    
    def __init__(self, app, allowed_origins=None, allowed_methods=None, allowed_headers=None, **kwargs):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ['*']
        self.allowed_methods = allowed_methods or ['GET', 'POST', 'PUT', 'DELETE']
        self.allowed_headers = allowed_headers or ['*']
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Handle preflight requests
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "86400",
                }
            )
        
        response = await call_next(request)
        
        # Add CORS headers to response
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

# Utility functions for middleware configuration
def configure_security_middleware(app):
    """Configure all security middleware for the application."""
    
    # Add middleware in the correct order
    app.add_middleware(CORSMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=1000, window_seconds=3600)
    app.add_middleware(SecurityValidationMiddleware)
    app.add_middleware(CSRFProtectionMiddleware)
    app.add_middleware(AuditLoggingMiddleware, log_sensitive_data=False)
    
    return app

def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for logging."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]
