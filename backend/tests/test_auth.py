"""
Unit tests for the authentication API (`/api/auth/*`).

Covers:
    - User registration (success and duplicate-email rejection)
    - Login (correct and incorrect password)
    - JWT enforcement on protected routes (401 without token)

Fixtures used (from conftest.py):
    - `client` — async ASGI test client; no auth header by default.
    - No mocks needed — auth routes hit the real database (truncated per test).
"""

import pytest


@pytest.mark.asyncio
async def test_register_success(client):
    """A new email/password pair should return 201 and an access_token."""
    resp = await client.post("/api/auth/register", json={
        "email": "new@hermes.dev", "password": "pass1234"
    })
    assert resp.status_code == 201
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    """Registering the same email twice must be rejected with 400."""
    payload = {"email": "dup@hermes.dev", "password": "pass1234"}
    await client.post("/api/auth/register", json=payload)
    resp = await client.post("/api/auth/register", json=payload)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_login_success(client):
    """After registration, login with the same credentials returns 200 + token."""
    await client.post("/api/auth/register", json={
        "email": "login@hermes.dev", "password": "pass1234"
    })
    resp = await client.post("/api/auth/login", json={
        "email": "login@hermes.dev", "password": "pass1234"
    })
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    """Login with an incorrect password must return 401 Unauthorized."""
    await client.post("/api/auth/register", json={
        "email": "wrong@hermes.dev", "password": "correct"
    })
    resp = await client.post("/api/auth/login", json={
        "email": "wrong@hermes.dev", "password": "incorrect"
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_without_token(client):
    """POST /api/research without a Bearer token must be rejected with 401."""
    resp = await client.post("/api/research", json={"query": "test"})
    assert resp.status_code == 401
