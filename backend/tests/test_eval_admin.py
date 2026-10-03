"""Unit tests for EVAL_ADMIN_EMAILS guard and dashboard scoping on /api/eval."""

import os
from unittest.mock import patch

import pytest
from sqlalchemy import select

from src.db import AsyncSessionLocal, QueryLog, User


@pytest.mark.asyncio
async def test_eval_run_denied_when_allowlist_empty(client, auth_token, monkeypatch):
    """Unset allow-list means nobody is an admin — the guard must fail closed."""
    monkeypatch.delenv("EVAL_ADMIN_EMAILS", raising=False)
    resp = await client.post(
        "/api/eval/run",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_eval_run_denied_when_allowlist_blank(client, auth_token, monkeypatch):
    monkeypatch.setenv("EVAL_ADMIN_EMAILS", "  ,  ")
    resp = await client.post(
        "/api/eval/run",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_eval_run_allowed_for_allowlisted_admin(client, auth_token, monkeypatch):
    monkeypatch.setenv("EVAL_ADMIN_EMAILS", "ci@hermes.dev")
    with patch("src.evaluation.ragas_eval.run_evaluation"):
        resp = await client.post(
            "/api/eval/run",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] in ("started", "already_running")


@pytest.mark.asyncio
async def test_eval_run_forbidden_for_non_admin(client, auth_token, monkeypatch):
    monkeypatch.setenv("EVAL_ADMIN_EMAILS", "admin@hermes.dev")
    # auth_token is ci@hermes.dev — not admin
    resp = await client.post(
        "/api/eval/run",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_dashboard_counts_only_current_user_queries(client, auth_token):
    """Dashboard QueryLog aggregates must not leak another user's activity."""
    other = await client.post("/api/auth/register", json={
        "email": "other@hermes.dev", "password": "otherpass123"
    })
    assert other.status_code == 201

    async with AsyncSessionLocal() as db:
        me = (await db.execute(select(User).where(User.email == "ci@hermes.dev"))).scalar_one()
        them = (await db.execute(select(User).where(User.email == "other@hermes.dev"))).scalar_one()
        db.add_all([
            QueryLog(user_id=me.id, query="a", answer="a", cache_hit=True),
            QueryLog(user_id=me.id, query="b", answer="b", cache_hit=False),
            QueryLog(user_id=them.id, query="c", answer="c", cache_hit=True),
            QueryLog(user_id=them.id, query="d", answer="d", cache_hit=True),
        ])
        await db.commit()

    resp = await client.get(
        "/api/eval/dashboard",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_queries"] == 2
    assert body["cache_hit_rate"] == 0.5
