import json, os
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.db import QueryLog, get_db
from src.auth import get_current_user
from src.db import User

router = APIRouter(prefix="/api/eval", tags=["eval"])

# Guards against overlapping RAGAS runs (each run is heavy: minutes long).
_eval_running = False


def _require_eval_admin(user: User) -> None:
    """
    Restrict eval runs to admins when EVAL_ADMIN_EMAILS is configured.

    EVAL_ADMIN_EMAILS is a comma-separated allow-list. If it is unset/empty,
    any authenticated user may trigger a run (convenient for local/dev use).
    """
    raw = os.getenv("EVAL_ADMIN_EMAILS", "").strip()
    if not raw:
        return
    allowed = {e.strip().lower() for e in raw.split(",") if e.strip()}
    if user.email.lower() not in allowed:
        raise HTTPException(status_code=403, detail="Eval runs are restricted to admins.")


@router.post("/run")
async def run_eval(
    background: BackgroundTasks,
    n_questions: int = 10,
    current_user: User = Depends(get_current_user),
):
    """
    Trigger a RAGAS evaluation run in the background.

    The run ingests the golden knowledge base through the shared retriever and
    scores faithfulness/relevancy/precision/recall via local Ollama. It writes
    eval_report.json, which GET /api/eval/dashboard then surfaces. This is a
    long-running job (minutes); the endpoint returns immediately.

    Restricted to admins when EVAL_ADMIN_EMAILS is set (see .env.example).
    """
    _require_eval_admin(current_user)

    global _eval_running
    if _eval_running:
        return {"status": "already_running"}

    def _job(n: int):
        global _eval_running
        _eval_running = True
        try:
            from src.evaluation.ragas_eval import run_evaluation
            run_evaluation(n_questions=n)
        finally:
            _eval_running = False

    background.add_task(_job, n_questions)
    return {"status": "started", "n_questions": n_questions}


@router.get("/status")
async def eval_status(current_user: User = Depends(get_current_user)):
    return {"running": _eval_running}


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
