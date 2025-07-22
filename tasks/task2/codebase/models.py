from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import re

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    SERVICE = "service"

class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"

class MFAMethod(str, Enum):
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"

# Request Models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('first_name', 'last_name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z\s-\']+$', v):
            raise ValueError('Name contains invalid characters')
        if len(v.strip()) < 1:
            raise ValueError('Name cannot be empty')
        return v.strip()

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    mfa_code: Optional[str] = None
    remember_me: bool = False

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        # Reuse password validation from RegisterRequest
        return RegisterRequest.validate_password(v)

class MFAEnableRequest(BaseModel):
    method: MFAMethod
    phone_number: Optional[str] = None
    
    @validator('phone_number')
    def validate_phone(cls, v, values):
        if values.get('method') == MFAMethod.SMS and not v:
            raise ValueError('Phone number required for SMS MFA')
        if v and not re.match(r'^\+[1-9]\d{1,14}$', v):
            raise ValueError('Invalid phone number format')
        return v

class MFAVerifyRequest(BaseModel):
    code: str
    method: MFAMethod

class OAuth2AuthorizeRequest(BaseModel):
    client_id: str
    response_type: str = "code"
    redirect_uri: str
    scope: Optional[str] = "openid email profile"
    state: Optional[str] = None

class OAuth2TokenRequest(BaseModel):
    grant_type: str
    code: Optional[str] = None
    refresh_token: Optional[str] = None
    client_id: str
    client_secret: str
    redirect_uri: Optional[str] = None

class APIKeyCreateRequest(BaseModel):
    name: str
    expires_in_days: Optional[int] = 365
    scopes: List[str] = []

# Response Models
class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    is_verified: bool
    mfa_enabled: bool
    created_at: datetime
    last_login: Optional[datetime]

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    scope: Optional[str] = None

class MFASetupResponse(BaseModel):
    secret: Optional[str] = None  # For TOTP
    qr_code: Optional[str] = None  # Base64 encoded QR code
    backup_codes: List[str] = []

class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: str  # Only returned on creation
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    scopes: List[str]

class APIKeyListResponse(BaseModel):
    id: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    scopes: List[str]
    # Note: key is not included in list response

class AuditLogResponse(BaseModel):
    id: str
    user_id: Optional[str]
    event_type: str
    description: str
    ip_address: str
    user_agent: str
    created_at: datetime
    additional_data: Dict[str, Any] = {}

class RateLimitResponse(BaseModel):
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None

# Internal Models
class User(BaseModel):
    id: str
    email: str
    password_hash: str
    first_name: str
    last_name: str
    role: UserRole = UserRole.USER
    is_verified: bool = False
    is_active: bool = True
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    mfa_method: Optional[MFAMethod] = None
    phone_number: Optional[str] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

class Session(BaseModel):
    id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

class APIKey(BaseModel):
    id: str
    user_id: str
    name: str
    key_hash: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool = True

class AuditLog(BaseModel):
    id: str
    user_id: Optional[str]
    session_id: Optional[str]
    event_type: str
    description: str
    ip_address: str
    user_agent: str
    created_at: datetime
    additional_data: Dict[str, Any] = {}

class OAuth2Client(BaseModel):
    client_id: str
    client_secret_hash: str
    name: str
    redirect_uris: List[str]
    scopes: List[str]
    is_active: bool = True
    created_at: datetime

class OAuth2AuthorizationCode(BaseModel):
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scope: str
    expires_at: datetime
    created_at: datetime
    is_used: bool = False

# Error Models
class ErrorResponse(BaseModel):
    error: str
    error_description: Optional[str] = None
    error_code: Optional[str] = None

# Configuration Models
class SecurityConfig(BaseModel):
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 480  # 8 hours
    jwt_expiry_minutes: int = 15
    refresh_token_expiry_days: int = 30
    
class RateLimitConfig(BaseModel):
    login_attempts_per_minute: int = 5
    registration_attempts_per_hour: int = 3
    api_requests_per_minute: int = 100
    password_reset_attempts_per_hour: int = 3
    
class OAuth2Config(BaseModel):
    authorization_code_expiry_minutes: int = 10
    access_token_expiry_minutes: int = 60
    refresh_token_expiry_days: int = 30
