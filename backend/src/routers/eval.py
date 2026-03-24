import json, os
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.db import QueryLog, get_db
from src.auth import get_current_user
from src.db import User

router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.get("/dashboard")
async def eval_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total = await db.execute(select(func.count()).select_from(QueryLog))
    cache_hits = await db.execute(
        select(func.count()).select_from(QueryLog).where(QueryLog.cache_hit == True)
    )
    total_count = total.scalar() or 0
    hits_count = cache_hits.scalar() or 0

    report = {}
    report_path = os.path.join(os.path.dirname(__file__), "../../eval_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

    return {
        "total_queries": total_count,
        "cache_hit_rate": round(hits_count / total_count, 4) if total_count else 0.0,
        "ragas": report,
    }


@router.get("/golden")
async def golden_report(current_user: User = Depends(get_current_user)):
    report_path = os.path.join(os.path.dirname(__file__), "../../eval_report.json")
    if not os.path.exists(report_path):
        return {"error": "No eval report found. Run ragas_eval.py first."}
    with open(report_path) as f:
        return json.load(f)
