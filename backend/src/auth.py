"""
JWT authentication — password hashing, token creation, and request guards.

What this file does:
    Provides the cryptographic and FastAPI dependency plumbing for user
    authentication: bcrypt password hashing, JWT access tokens, and a
    ``get_current_user`` dependency that protected routes use to identify
    the caller.

Where it sits in the HERMES pipeline:
    Sits between HTTP routers and the database. Every authenticated endpoint
    (ingest, research, eval) depends on ``get_current_user`` before doing work.

What calls this:
    - ``src/routers/auth.py`` — register and login endpoints
    - ``src/routers/ingest.py``, ``research.py``, ``eval.py`` — via
      ``Depends(get_current_user)``

What this calls:
    - ``src.db.User`` and ``get_db`` — loads the user row that matches the
      email embedded in the JWT ``sub`` claim
    - ``jose`` / ``passlib`` — JWT encode/decode and bcrypt hashing
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.db import User, get_db

_DEFAULT_SECRET = "hermes_super_secret_change_this_32ch"
SECRET_KEY = os.getenv("SECRET_KEY", _DEFAULT_SECRET)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Refuse to boot with the default signing key in production.
if os.getenv("ENV", "development").lower() == "production" and SECRET_KEY == _DEFAULT_SECRET:
    raise RuntimeError(
        "SECRET_KEY must be set to a unique value when ENV=production "
        "(the default key is insecure). See backend/.env.example."
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# HTTPBearer extracts "Authorization: Bearer <token>" from incoming requests.
bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    """
    Hash a plaintext password for storage in the database.

    Parameters:
        password: Raw password from the registration form.

    Returns:
        Bcrypt hash string safe to persist in ``User.hashed_password``.
    """
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Check whether a plaintext password matches a stored bcrypt hash.

    Parameters:
        plain: Password supplied at login time.
        hashed: Value previously returned by ``hash_password``.

    Returns:
        True if the password is correct, False otherwise.
    """
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Mint a signed JWT access token.

    The caller typically passes ``{"sub": user.email}`` so ``get_current_user``
    can recover the account email from the token.

    Parameters:
        data: Claims to embed in the JWT payload (must include ``sub``).
        expires_delta: Optional custom lifetime; defaults to 24 hours.

    Returns:
        Encoded JWT string sent to the client as ``access_token``.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency — resolve the authenticated user from a Bearer token.

    Decodes the JWT, reads the ``sub`` claim (email), and loads the matching
    ``User`` row. Raises HTTP 401 if the token is invalid, expired, or refers
    to a user that no longer exists.

    Parameters:
        credentials: Bearer token extracted by ``HTTPBearer``.
        db: Async SQLAlchemy session injected by ``get_db``.

    Returns:
        The authenticated ``User`` ORM object.

    Raises:
        HTTPException: 401 when the token or user lookup fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode and validate signature + expiration in one step.
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise credentials_exception
    return user
