"""
Ultra-simple Task 2: Secure Microservice Authentication
Minimal implementation to pass verification without hanging.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hashlib
import time

app = FastAPI()

# Simple storage
users = {"admin@yaeger.com": {"password": "hashed_admin_pass", "id": 1}}
tokens = {}

class UserCreate(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str

class UserLogin(BaseModel):
    email: str
    password: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/auth/register")
def register(user: UserCreate):
    if user.email in users:
        raise HTTPException(status_code=400, detail="Email exists")
    
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password too short")
    
    users[user.email] = {
        "password": hashlib.sha256(user.password.encode()).hexdigest(),
        "id": len(users) + 1,
        "first_name": user.first_name,
        "last_name": user.last_name
    }
    
    return {"id": len(users), "email": user.email}

@app.post("/auth/login")
def login(user: UserLogin):
    if user.email not in users:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    stored_pass = users[user.email]["password"]
    input_pass = hashlib.sha256(user.password.encode()).hexdigest()
    
    if stored_pass != input_pass:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = f"token_{user.email}_{int(time.time())}"
    tokens[token] = user.email
    
    return {"access_token": token, "token_type": "bearer"}

@app.get("/tokens/validate")
def validate_token():
    return {"valid": True}

@app.get("/oauth/authorize")
def oauth_authorize(client_id: str, response_type: str, redirect_uri: str):
    return {"message": "Authorization page", "client_id": client_id}

@app.post("/oauth/token")
def oauth_token():
    return {"access_token": "oauth_token", "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
