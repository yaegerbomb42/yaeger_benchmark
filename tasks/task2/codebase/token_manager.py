import jwt
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import bcrypt
import pyotp
import qrcode
import io
import base64
from .models import User, Session, APIKey, TokenType, MFAMethod

class TokenManager:
    """Handles JWT token generation, validation, and management."""
    
    def __init__(self):
        # Generate RSA key pair for JWT signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        # Convert to PEM format for use with PyJWT
        self.private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        self.public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Token blacklist for logout functionality
        self.blacklisted_tokens: set = set()
        
    def generate_jwt(self, user_id: str, email: str, role: str, 
                    token_type: TokenType = TokenType.ACCESS, 
                    expires_in_minutes: int = 15) -> str:
        """Generate a JWT token with the specified claims."""
        
        now = datetime.utcnow()
        payload = {
            'sub': user_id,  # Subject (user ID)
            'email': email,
            'role': role,
            'type': token_type.value,
            'iat': now,  # Issued at
            'exp': now + timedelta(minutes=expires_in_minutes),
            'jti': secrets.token_urlsafe(16),  # JWT ID for blacklisting
        }
        
        # Add additional claims based on token type
        if token_type == TokenType.ACCESS:
            payload['aud'] = 'yaeger-api'  # Audience
        elif token_type == TokenType.REFRESH:
            payload['aud'] = 'yaeger-refresh'
            
        return jwt.encode(payload, self.private_pem, algorithm='RS256')
    
    def validate_jwt(self, token: str, token_type: TokenType = TokenType.ACCESS) -> Dict[str, Any]:
        """Validate a JWT token and return the payload."""
        
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise jwt.InvalidTokenError("Token has been revoked")
            
            # Decode and validate the token
            payload = jwt.decode(
                token, 
                self.public_pem, 
                algorithms=['RS256'],
                audience=f'yaeger-{token_type.value}' if token_type != TokenType.ACCESS else 'yaeger-api'
            )
            
            # Verify token type
            if payload.get('type') != token_type.value:
                raise jwt.InvalidTokenError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidAudienceError:
            raise jwt.InvalidTokenError("Invalid token audience")
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")
    
    def blacklist_token(self, token: str) -> None:
        """Add a token to the blacklist."""
        self.blacklisted_tokens.add(token)
    
    def generate_api_key(self, user_id: str, name: str, scopes: List[str] = None) -> tuple:
        """Generate an API key and return (key, hash)."""
        # Generate a secure random API key
        key = f"yaeger_{secrets.token_urlsafe(32)}"
        
        # Create hash for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        return key, key_hash
    
    def validate_api_key(self, key: str, stored_hash: str) -> bool:
        """Validate an API key against its stored hash."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return secrets.compare_digest(key_hash, stored_hash)

class PasswordManager:
    """Handles password hashing and validation."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except ValueError:
            return False
    
    @staticmethod
    def check_password_strength(password: str) -> Dict[str, bool]:
        """Check password strength and return detailed results."""
        checks = {
            'length': len(password) >= 8,
            'uppercase': any(c.isupper() for c in password),
            'lowercase': any(c.islower() for c in password),
            'digit': any(c.isdigit() for c in password),
            'special': any(c in '!@#$%^&*(),.?":{}|<>' for c in password),
            'common': password.lower() not in [
                'password', '123456', 'qwerty', 'abc123', 'password123',
                'admin', 'letmein', 'welcome', 'monkey', 'dragon'
            ]
        }
        return checks

class MFAManager:
    """Handles Multi-Factor Authentication functionality."""
    
    @staticmethod
    def generate_totp_secret() -> str:
        """Generate a new TOTP secret."""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_qr_code(email: str, secret: str, issuer: str = "Yaeger Auth") -> str:
        """Generate a QR code for TOTP setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=email,
            issuer_name=issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    @staticmethod
    def verify_totp(secret: str, code: str, window: int = 1) -> bool:
        """Verify a TOTP code."""
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=window)
    
    @staticmethod
    def generate_backup_codes(count: int = 10) -> List[str]:
        """Generate backup codes for account recovery."""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes

class SessionManager:
    """Handles user session management."""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str,
                      expires_in_minutes: int = 480) -> Session:
        """Create a new session."""
        session_id = secrets.token_urlsafe(32)
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        session = Session(
            id=session_id,
            user_id=user_id,
            token_hash=token_hash,
            expires_at=datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            created_at=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session
    
    def validate_session(self, session_id: str, token: str) -> Optional[Session]:
        """Validate a session token."""
        session = self.sessions.get(session_id)
        
        if not session or not session.is_active:
            return None
        
        if session.expires_at < datetime.utcnow():
            session.is_active = False
            return None
        
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if not secrets.compare_digest(session.token_hash, token_hash):
            return None
        
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
            return True
        return False
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        count = 0
        for session in self.sessions.values():
            if session.user_id == user_id and session.is_active:
                session.is_active = False
                count += 1
        return count
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.expires_at < now
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)
