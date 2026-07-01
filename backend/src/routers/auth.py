"""
Authentication HTTP routes — user registration and login.

What this file does:
    Exposes ``POST /api/auth/register`` and ``POST /api/auth/login``. Creates
    users in Postgres and returns JWT access tokens the frontend stores in
    ``localStorage``.

Where it sits in the HERMES pipeline:
    Entry point for all clients. Every other ``/api/*`` route requires a token
    minted here (validated by ``src.auth.get_current_user``).

What calls this:
    - React ``AuthContext`` via ``POST /auth/register`` and ``POST /auth/login``

What this calls:
    - ``src.auth`` — password hashing and JWT creation
    - ``src.db.User`` — persistence
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.db import User, get_db
from src.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create a new user and return a JWT (HTTP 400 if email already exists)."""
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=req.email, hashed_password=hash_password(req.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Validate credentials and return a JWT (HTTP 401 on failure)."""
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)
