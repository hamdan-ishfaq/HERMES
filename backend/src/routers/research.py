import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import QueryLog, get_db
from src.auth import get_current_user
from src.db import User
from src.agents.graph import run_research

router = APIRouter(prefix="/api", tags=["research"])


class ResearchRequest(BaseModel):
    query: str


@router.post("/research")
async def research(
    req: ResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    state = run_research(req.query)

    answer = state.get("final_answer") or state.get("draft_answer") or ""
    citations = state.get("citations", [])

    log = QueryLog(
        user_id=current_user.id,
        query=req.query,
        answer=answer,
        model_used=state.get("model_used", ""),
        cache_hit=state.get("cache_hit", False),
    )
    db.add(log)
    await db.commit()

    return {
        "answer": answer,
        "citations": citations,
        "model_used": state.get("model_used", ""),
        "cache_hit": state.get("cache_hit", False),
    }


@router.get("/stream/research")
async def stream_research(
    query: str,
    current_user: User = Depends(get_current_user),
):
    async def event_generator():
        steps = [
            {"step": "supervisor", "status": "Classifying query..."},
            {"step": "retrieval",  "status": "Retrieving contexts..."},
            {"step": "generation", "status": "Generating answer..."},
        ]
        for s in steps:
            yield f"data: {json.dumps(s)}\n\n"

        state = run_research(query)
        answer = state.get("final_answer") or state.get("draft_answer") or ""
        yield f"data: {json.dumps({'step': 'done', 'answer': answer, 'citations': state.get('citations', [])})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
