import pytest


@pytest.mark.asyncio
async def test_register_success(client):
    resp = await client.post("/api/auth/register", json={
        "email": "new@hermes.dev", "password": "pass1234"
    })
    assert resp.status_code == 201
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    payload = {"email": "dup@hermes.dev", "password": "pass1234"}
    await client.post("/api/auth/register", json=payload)
    resp = await client.post("/api/auth/register", json=payload)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_login_success(client):
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
    await client.post("/api/auth/register", json={
        "email": "wrong@hermes.dev", "password": "correct"
    })
    resp = await client.post("/api/auth/login", json={
        "email": "wrong@hermes.dev", "password": "incorrect"
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_without_token(client):
    resp = await client.post("/api/research", json={"query": "test"})
    assert resp.status_code == 401
