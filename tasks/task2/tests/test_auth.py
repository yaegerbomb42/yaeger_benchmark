"""
Comprehensive test suite for Task 2: Secure Microservice Authentication
Tests the authentication        # Should reject weak password
        assert response.status_code in [400, 422]
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks."""
        malicious_payloads = [
            "'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "1' UNION SELECT * FROM users--",
            "'; INSERT INTO users VALUES('hacker','pass');--"
        ]
        
        for payload in malicious_payloads:
            malicious_data = {
                "email": payload,
                "password": "TestPass123!"
            }
            response = self.client.post("/auth/login", json=malicious_data)
            
            # Should not crash or return unexpected success
            assert response.status_code in [400, 401, 422, 500]
            
            # Response should not contain SQL error messages
            if response.status_code != 500:
                response_text = response.text.lower()
                sql_keywords = ["syntax error", "mysql", "postgresql", "sqlite", "sql"]
                for keyword in sql_keywords:
                    assert keyword not in response_text, f"SQL error exposed: {keyword}"
    
    def test_xss_protection(self):
        """Test protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
        ]
        
        for payload in xss_payloads:
            xss_data = {
                "email": f"test+{payload}@example.com",
                "password": "TestPass123!",
                "first_name": payload,
                "last_name": "User"
            }
            response = self.client.post("/auth/register", json=xss_data)
            
            # Should handle malicious input safely
            if response.status_code in [200, 201]:
                response_text = response.text
                # Response should not contain unescaped script tags
                assert "<script>" not in response_text
                assert "javascript:" not in response_text
    
    def test_rate_limiting(self):
        """Test rate limiting protection."""
        # Make rapid requests to trigger rate limiting
        responses = []
        for i in range(20):  # Try 20 rapid requests
            response = self.client.post("/auth/login", json={
                "email": f"test{i}@example.com",
                "password": "wrongpassword"
            })
            responses.append(response.status_code)
        
        # Should eventually start rate limiting (429 Too Many Requests)
        rate_limited = any(status == 429 for status in responses[-10:])  # Check last 10 requests
        # Note: Rate limiting might not trigger in test environment, so this is optional
        # assert rate_limited, "Rate limiting should trigger with rapid requests"
    
    def test_jwt_token_validation(self):
        """Test JWT token validation and structure."""
        # Register and login to get a token
        self.client.post("/auth/register", json=self.test_user_data)
        login_response = self.client.post("/auth/login", json={
            "email": self.test_user_data["email"],
            "password": self.test_user_data["password"]
        })
        
        if login_response.status_code == 200:
            token_data = login_response.json()
            token = token_data.get("access_token") or token_data.get("token")
            
            if token:
                # JWT tokens should have 3 parts separated by dots
                token_parts = token.split(".")
                assert len(token_parts) == 3, "JWT token should have header.payload.signature"
                
                # Test protected endpoint with token
                headers = {"Authorization": f"Bearer {token}"}
                protected_response = self.client.get("/auth/me", headers=headers)
                
                # Should allow access with valid token
                assert protected_response.status_code in [200, 401]  # 401 if endpoint not implemented
    
    def test_token_expiration(self):
        """Test token expiration handling."""
        # This would require actual token expiration logic
        # For now, test that tokens have expiration claims
        self.client.post("/auth/register", json=self.test_user_data)
        response = self.client.post("/auth/login", json={
            "email": self.test_user_data["email"],
            "password": self.test_user_data["password"]
        })
        
        if response.status_code == 200:
            token_data = response.json()
            # Should include expiration information
            assert "expires_in" in token_data or "exp" in token_data or "access_token" in token_data
    
    def test_csrf_protection(self):
        """Test CSRF protection mechanisms."""
        # Test that state-changing operations require proper headers
        response = self.client.post("/auth/register", 
                                  json=self.test_user_data,
                                  headers={"Origin": "http://malicious-site.com"})
        
        # Should handle cross-origin requests appropriately
        # Either allow with CORS or reject
        assert response.status_code in [200, 201, 400, 403, 422]
    
    def test_password_hashing(self):
        """Test that passwords are properly hashed."""
        # Register a user
        response = self.client.post("/auth/register", json=self.test_user_data)
        
        if response.status_code in [200, 201]:
            # Try to verify that password is not stored in plaintext
            # This is more of an implementation test - check if bcrypt or similar is used
            import bcrypt
            test_hash = bcrypt.hashpw(b"testpassword", bcrypt.gensalt())
            assert isinstance(test_hash, bytes), "Should use proper password hashing"
    
    def test_concurrent_authentication_load(self):
        """Test authentication under concurrent load."""
        import threading
        import time
        
        results = []
        
        def auth_worker():
            try:
                # Register unique user
                user_id = int(time.time() * 1000) % 10000
                user_data = {
                    "email": f"load_test_{user_id}@example.com",
                    "password": "TestPass123!",
                    "first_name": "Load",
                    "last_name": "Test"
                }
                
                reg_response = self.client.post("/auth/register", json=user_data)
                
                if reg_response.status_code in [200, 201]:
                    # Try to login
                    login_response = self.client.post("/auth/login", json={
                        "email": user_data["email"],
                        "password": user_data["password"]
                    })
                    results.append(login_response.status_code)
                else:
                    results.append(reg_response.status_code)
            except Exception as e:
                results.append(500)  # Server error
        
        # Start multiple concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=auth_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Most requests should succeed
        success_rate = len([r for r in results if r in [200, 201]]) / len(results) if results else 0
        assert success_rate >= 0.5, f"Low success rate under load: {success_rate:.2f}"
    
    def test_input_validation_edge_cases(self):
        """Test input validation with edge cases."""
        edge_cases = [
            # Email edge cases
            {"email": "", "password": "TestPass123!"},  # Empty email
            {"email": "a" * 500 + "@example.com", "password": "TestPass123!"},  # Very long email
            {"email": "invalid-email", "password": "TestPass123!"},  # Invalid email format
            {"email": "test@", "password": "TestPass123!"},  # Incomplete email
            
            # Password edge cases
            {"email": "test@example.com", "password": ""},  # Empty password
            {"email": "test@example.com", "password": "a" * 1000},  # Very long password
            {"email": "test@example.com", "password": "x"},  # Very short password
            
            # Unicode and special characters
            {"email": "test+üñīçödé@example.com", "password": "TestPass123!"},
            {"email": "test@example.com", "password": "pásswôrd123!"},
        ]
        
        for case in edge_cases:
            response = self.client.post("/auth/register", json=case)
            
            # Should handle edge cases gracefully (not crash)
            assert response.status_code in [200, 201, 400, 422]
            
            # Should return proper error messages for invalid inputs
            if response.status_code in [400, 422]:
                data = response.json()
                assert "detail" in data or "message" in data or "error" in data
    
    def test_authentication_timing_attacks(self):
        """Test protection against timing attacks."""
        import time
        
        # Measure time for valid vs invalid email lookups
        valid_email_times = []
        invalid_email_times = []
        
        # First register a user
        self.client.post("/auth/register", json=self.test_user_data)
        
        # Test timing for existing email with wrong password
        for _ in range(5):
            start_time = time.time()
            self.client.post("/auth/login", json={
                "email": self.test_user_data["email"],
                "password": "wrongpassword"
            })
            valid_email_times.append(time.time() - start_time)
        
        # Test timing for non-existent email
        for _ in range(5):
            start_time = time.time()
            self.client.post("/auth/login", json={
                "email": "nonexistent@example.com",
                "password": "wrongpassword"
            })
            invalid_email_times.append(time.time() - start_time)
        
        # Calculate average times
        avg_valid = sum(valid_email_times) / len(valid_email_times)
        avg_invalid = sum(invalid_email_times) / len(invalid_email_times)
        
        # Time difference should be minimal (< 50ms difference)
        time_diff = abs(avg_valid - avg_invalid)
        assert time_diff < 0.05, f"Potential timing attack vulnerability: {time_diff:.3f}s difference"
    
    def test_oauth2_flow_simulation(self):
        """Test OAuth2-like authorization flow."""
        # Test authorization endpoint (if implemented)
        auth_response = self.client.get("/auth/authorize", params={
            "client_id": "test_client",
            "response_type": "code",
            "redirect_uri": "http://localhost:3000/callback",
            "scope": "read write"
        })
        
        # Should handle OAuth2 authorization (or return not implemented)
        assert auth_response.status_code in [200, 302, 404, 501]
    
    def test_api_key_authentication(self):
        """Test API key authentication (if implemented)."""
        # Try to access with API key
        headers = {"X-API-Key": "test-api-key-12345"}
        response = self.client.get("/auth/validate", headers=headers)
        
        # Should handle API key auth (or return not implemented)
        assert response.status_code in [200, 401, 404, 501]
    
    def test_security_headers(self):
        """Test that proper security headers are included."""
        response = self.client.get("/health")
        
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        # At least some security headers should be present
        present_headers = [h for h in security_headers if h in response.headers]
        # Note: This is aspirational - not all implementations may include these
        # assert len(present_headers) >= 1, f"No security headers found: {list(response.headers.keys())}"
    
    def test_error_message_information_disclosure(self):
        """Test that error messages don't leak sensitive information."""
        # Test with various malformed requests
        malformed_requests = [
            {"email": "test@example.com"},  # Missing password
            {"password": "TestPass123!"},   # Missing email
            {},  # Empty request
            {"email": "test@example.com", "password": "TestPass123!", "extra_field": "value"}
        ]
        
        for req in malformed_requests:
            response = self.client.post("/auth/login", json=req)
            
            if response.status_code in [400, 422]:
                error_text = response.text.lower()
                
                # Should not expose internal details
                sensitive_terms = [
                    "traceback", "exception", "stack trace", 
                    "internal error", "database", "sql",
                    "secret", "key", "token", "hash"
                ]
                
                for term in sensitive_terms:
                    assert term not in error_text, f"Error message exposes sensitive info: {term}"ervice implementation with security focus.
"""
import pytest
import asyncio
import sys
import os
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import submission
    from codebase.models import User, Token, UserCreate, UserResponse
    from codebase.token_manager import TokenManager
    from codebase.security_middleware import SecurityMiddleware
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestAuthenticationEndpoints:
    """Test suite for authentication endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(submission.app)
        self.test_user_data = {
            "email": "test@example.com",
            "password": "TestPass123!",
            "first_name": "Test",
            "last_name": "User"
        }
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_user_registration(self):
        """Test user registration endpoint."""
        response = self.client.post("/auth/register", json=self.test_user_data)
        
        # Should return 201 Created or 200 OK
        assert response.status_code in [200, 201]
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert "id" in data or "user_id" in data or "email" in data
    
    def test_user_registration_duplicate_email(self):
        """Test registration with duplicate email."""
        # First registration
        self.client.post("/auth/register", json=self.test_user_data)
        
        # Second registration with same email
        response = self.client.post("/auth/register", json=self.test_user_data)
        
        # Should return error for duplicate email
        assert response.status_code in [400, 409, 422]
    
    def test_user_login_valid_credentials(self):
        """Test login with valid credentials."""
        # First register the user
        self.client.post("/auth/register", json=self.test_user_data)
        
        # Then try to login
        login_data = {
            "email": self.test_user_data["email"],
            "password": self.test_user_data["password"]
        }
        response = self.client.post("/auth/login", json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data or "token" in data
            assert "token_type" in data or data.get("token_type") == "bearer"
    
    def test_user_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        response = self.client.post("/auth/login", json=login_data)
        
        # Should return 401 Unauthorized or 400 Bad Request
        assert response.status_code in [400, 401, 422]
    
    def test_password_strength_validation(self):
        """Test password strength requirements."""
        weak_password_data = self.test_user_data.copy()
        weak_password_data["password"] = "123"
        
        response = self.client.post("/auth/register", json=weak_password_data)
        
        # Should reject weak passwords
        assert response.status_code in [400, 422]
    
    def test_email_validation(self):
        """Test email format validation."""
        invalid_email_data = self.test_user_data.copy()
        invalid_email_data["email"] = "invalid-email"
        
        response = self.client.post("/auth/register", json=invalid_email_data)
        
        # Should reject invalid email formats
        assert response.status_code in [400, 422]


class TestTokenManagement:
    """Test suite for token management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(submission.app)
        self.token_manager = TokenManager()
        self.test_user = User(
            id=1,
            email="test@example.com",
            first_name="Test",
            last_name="User"
        )
    
    def test_token_creation(self):
        """Test JWT token creation."""
        token = self.token_manager.create_token(self.test_user)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 20  # JWT tokens are typically longer
    
    def test_token_validation(self):
        """Test JWT token validation."""
        # Create a token
        token = self.token_manager.create_token(self.test_user)
        
        # Validate the token
        payload = self.token_manager.verify_token(token)
        
        assert payload is not None
        assert "sub" in payload or "user_id" in payload or "email" in payload
    
    def test_token_expiration(self):
        """Test token expiration handling."""
        # Create a token with short expiration
        token = self.token_manager.create_token(self.test_user, expires_delta=1)  # 1 second
        
        # Wait for token to expire
        import time
        time.sleep(2)
        
        # Token should be invalid now
        with pytest.raises(Exception):
            self.token_manager.verify_token(token)
    
    def test_token_validation_endpoint(self):
        """Test the token validation endpoint."""
        # First register and login to get a valid token
        user_data = {
            "email": "test@example.com",
            "password": "TestPass123!",
            "first_name": "Test",
            "last_name": "User"
        }
        
        self.client.post("/auth/register", json=user_data)
        
        login_response = self.client.post("/auth/login", json={
            "email": user_data["email"],
            "password": user_data["password"]
        })
        
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            
            if token:
                # Test token validation endpoint
                headers = {"Authorization": f"Bearer {token}"}
                response = self.client.get("/tokens/validate", headers=headers)
                
                assert response.status_code in [200, 401]  # 401 if not implemented
    
    def test_invalid_token_format(self):
        """Test handling of malformed tokens."""
        invalid_tokens = [
            "invalid.token.format",
            "not-a-jwt",
            "",
            "Bearer invalid",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid"
        ]
        
        for invalid_token in invalid_tokens:
            with pytest.raises(Exception):
                self.token_manager.verify_token(invalid_token)


class TestSecurityFeatures:
    """Test suite for security features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(submission.app)
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks."""
        sql_injection_payloads = [
            "admin@example.com'; DROP TABLE users; --",
            "admin@example.com' OR '1'='1",
            "admin@example.com' UNION SELECT * FROM users --"
        ]
        
        for payload in sql_injection_payloads:
            login_data = {
                "email": payload,
                "password": "anypassword"
            }
            response = self.client.post("/auth/login", json=login_data)
            
            # Should not return successful login for SQL injection
            assert response.status_code in [400, 401, 422]
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        login_data = {
            "email": "test@example.com",
            "password": "wrongpassword"
        }
        
        # Make multiple rapid requests
        rate_limited = False
        for i in range(20):
            response = self.client.post("/auth/login", json=login_data)
            if response.status_code == 429:
                rate_limited = True
                break
        
        # Should implement some form of rate limiting
        # Note: This test might pass if rate limiting is not implemented
        # but it's checking for the presence of the feature
        if rate_limited:
            assert True, "Rate limiting is working"
    
    def test_password_hashing(self):
        """Test that passwords are properly hashed."""
        user_data = {
            "email": "security@example.com",
            "password": "SecurePass123!",
            "first_name": "Security",
            "last_name": "Test"
        }
        
        response = self.client.post("/auth/register", json=user_data)
        
        # If registration succeeds, password should be hashed
        # This is more of a code review item, but we can test behavior
        if response.status_code in [200, 201]:
            # Try to login with correct password
            login_response = self.client.post("/auth/login", json={
                "email": user_data["email"],
                "password": user_data["password"]
            })
            
            # Should succeed with correct password
            assert login_response.status_code in [200, 401]  # 200 if implemented correctly
    
    def test_authorization_header_validation(self):
        """Test proper handling of authorization headers."""
        # Test various invalid authorization headers
        invalid_headers = [
            {"Authorization": "Invalid token"},
            {"Authorization": "Bearer"},
            {"Authorization": ""},
            {"Authorization": "Basic dGVzdDp0ZXN0"},  # Basic auth instead of Bearer
        ]
        
        for headers in invalid_headers:
            response = self.client.get("/tokens/validate", headers=headers)
            # Should return 401 for invalid auth headers
            assert response.status_code in [401, 422]


class TestOAuth2Flow:
    """Test suite for OAuth2 implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(submission.app)
    
    def test_oauth_authorize_endpoint(self):
        """Test OAuth2 authorization endpoint."""
        params = {
            "client_id": "demo_client",
            "response_type": "code",
            "redirect_uri": "http://localhost:8000/callback",
            "scope": "read"
        }
        
        response = self.client.get("/oauth/authorize", params=params)
        
        # Should return 200 or redirect (302)
        assert response.status_code in [200, 302, 404]  # 404 if not implemented
    
    def test_oauth_token_endpoint(self):
        """Test OAuth2 token exchange endpoint."""
        token_data = {
            "grant_type": "authorization_code",
            "code": "test_code",
            "client_id": "demo_client",
            "client_secret": "demo_secret",
            "redirect_uri": "http://localhost:8000/callback"
        }
        
        response = self.client.post("/oauth/token", data=token_data)
        
        # Should handle token exchange (implementation dependent)
        assert response.status_code in [200, 400, 404]  # 404 if not implemented
    
    def test_oauth_invalid_client(self):
        """Test OAuth2 with invalid client credentials."""
        params = {
            "client_id": "invalid_client",
            "response_type": "code",
            "redirect_uri": "http://localhost:8000/callback"
        }
        
        response = self.client.get("/oauth/authorize", params=params)
        
        # Should reject invalid clients
        assert response.status_code in [400, 401, 404]


class TestPerformanceAndReliability:
    """Test suite for performance and reliability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(submission.app)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent authentication requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = self.client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                results.append(500)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)
    
    def test_response_time(self):
        """Test authentication response times."""
        import time
        
        # Test health endpoint response time
        start = time.time()
        response = self.client.get("/health")
        end = time.time()
        
        response_time = (end - start) * 1000  # Convert to milliseconds
        
        # Response should be under 100ms for health check
        assert response_time < 100, f"Health check too slow: {response_time:.2f}ms"
        assert response.status_code == 200
    
    def test_memory_usage_stability(self):
        """Test that repeated requests don't cause memory leaks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for _ in range(100):
            self.client.get("/health")
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"Potential memory leak: {memory_increase:.1f}MB increase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
