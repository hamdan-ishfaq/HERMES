from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import QueryLog, get_db
from src.auth import get_current_user
from src.db import User
from src.agents.graph import run_research

router = APIRouter(prefix="/api", tags=["research"])


from typing import Optional

class ResearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


@router.post("/research")
async def research(
    req: ResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    import uuid
    session_id = req.session_id or str(uuid.uuid4())
    state = run_research(req.query, session_id=session_id)

    answer = state.get("final_answer") or state.get("draft_answer") or ""
    citations = state.get("citations", [])

    is_cached = state.get("cache_hit", False)

    log = QueryLog(
        user_id=current_user.id,
        query=req.query,
        answer=answer,
        model_used=state.get("model_used", ""),
        cache_hit=is_cached,
    )
    db.add(log)
    await db.commit()

    return {
        "answer": answer,
        "citations": citations,
        "model_used": state.get("model_used", ""),
        "cache_hit": is_cached,
        "session_id": session_id,
    }
