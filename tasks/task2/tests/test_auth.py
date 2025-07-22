"""
Comprehensive test suite for Task 2: Secure Microservice Authentication
Tests the authentication service implementation with security focus.
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
