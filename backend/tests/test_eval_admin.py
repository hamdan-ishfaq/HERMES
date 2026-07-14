"""Unit tests for EVAL_ADMIN_EMAILS guard on POST /api/eval/run."""

import os
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_eval_run_allowed_when_allowlist_empty(client, auth_token, monkeypatch):
    monkeypatch.delenv("EVAL_ADMIN_EMAILS", raising=False)
    with patch("src.routers.eval.BackgroundTasks.add_task"):
        # Patch at module level after import — BackgroundTasks is injected by FastAPI
        pass
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
