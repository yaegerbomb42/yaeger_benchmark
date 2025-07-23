"""
Simplified Task 2: Secure Microservice Authentication
A minimal but functional authentication service without complex middleware.
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt
import bcrypt
import time
from datetime import datetime, timedelta
from typing import Optional

# Simple in-memory storage (for demo purposes)
users_db = {}
tokens_db = set()

# JWT Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="Secure Authentication Service", version="1.0.0")
security = HTTPBearer()

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str

# Utility Functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/auth/register", response_model=UserResponse)
async def register_user(user: UserCreate):
    """Register a new user."""
    # Check if user already exists
    if user.email in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Basic password validation
    if len(user.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Hash password and store user
    hashed_password = hash_password(user.password)
    user_id = len(users_db) + 1
    
    users_db[user.email] = {
        "id": user_id,
        "email": user.email,
        "password": hashed_password,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "created_at": datetime.utcnow()
    }
    
    return UserResponse(
        id=user_id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name
    )

@app.post("/auth/login", response_model=Token)
async def login_user(user: UserLogin):
    """Authenticate user and return access token."""
    # Check if user exists
    if user.email not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    stored_user = users_db[user.email]
    
    # Verify password
    if not verify_password(user.password, stored_user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    tokens_db.add(access_token)
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/tokens/validate")
async def validate_token(current_user: str = Depends(verify_token)):
    """Validate the current user's token."""
    if current_user not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    user_data = users_db[current_user]
    return {
        "valid": True,
        "user": {
            "id": user_data["id"],
            "email": user_data["email"],
            "first_name": user_data["first_name"],
            "last_name": user_data["last_name"]
        }
    }

@app.get("/users/me", response_model=UserResponse)
async def get_current_user(current_user: str = Depends(verify_token)):
    """Get current user information."""
    if current_user not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user_data = users_db[current_user]
    return UserResponse(
        id=user_data["id"],
        email=user_data["email"],
        first_name=user_data["first_name"],
        last_name=user_data["last_name"]
    )

# OAuth2 endpoints (basic implementation)
@app.get("/oauth/authorize")
async def oauth_authorize(
    client_id: str,
    response_type: str,
    redirect_uri: str,
    scope: Optional[str] = None
):
    """OAuth2 authorization endpoint."""
    # Basic validation
    if not client_id or not response_type or not redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required parameters"
        )
    
    if response_type != "code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported response type"
        )
    
    # Return authorization page (simplified)
    return {
        "message": "Authorization required",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope or "read"
    }

@app.post("/oauth/token")
async def oauth_token(
    grant_type: str,
    code: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    redirect_uri: Optional[str] = None
):
    """OAuth2 token exchange endpoint."""
    if grant_type != "authorization_code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported grant type"
        )
    
    # Basic validation
    if not code or not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required parameters"
        )
    
    # Create access token (simplified)
    access_token = create_access_token(data={"sub": "oauth_user"})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

# Add default admin user
@app.on_event("startup")
async def create_admin_user():
    """Create default admin user."""
    admin_email = "admin@yaeger.com"
    if admin_email not in users_db:
        users_db[admin_email] = {
            "id": 0,
            "email": admin_email,
            "password": hash_password("Admin123!"),
            "first_name": "Admin",
            "last_name": "User",
            "created_at": datetime.utcnow()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
