from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db, User
from src.auth import get_current_user

router = APIRouter(prefix="/api/e2e", tags=["e2e"])

BACKEND = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND / "scripts" / "e2e_smoke.py"


def _require_admin(user: User) -> None:
    raw = os.getenv("EVAL_ADMIN_EMAILS", "").strip()
    allowed = {e.strip().lower() for e in raw.split(",") if e.strip()}
    if user.email.lower() not in allowed:
        raise HTTPException(status_code=403, detail="E2E runs are restricted to admins.")


@router.post("/run")
async def run_e2e(
    skip_quality: bool = False,
    keep: bool = False,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    if not SCRIPT.exists():
        raise HTTPException(404, "e2e script not found")
    cmd = ["python3", str(SCRIPT)]
    if skip_quality:
        cmd.append("--skip-quality")
    if keep:
        cmd.append("--keep")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(BACKEND),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3600 * 2)
    except asyncio.TimeoutError:
        raise HTTPException(504, "e2e run timed out")
    return_code = proc.returncode or 0
    output = (stdout or b"").decode("utf-8", errors="replace")
    return {"return_code": return_code, "output": output}
