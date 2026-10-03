"""
Research HTTP route — main RAG Q&A endpoint.

Loads prior conversation turns for session_id, runs the LangGraph pipeline,
persists new turns, and returns answer + citations (+ tool_trace).

Also exposes POST /api/research/stream for real token SSE (HE-5).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.agents.graph import run_research
from src.agents.stream_research import iter_research_events
from src.auth import get_current_user
from src.db import AsyncSessionLocal, ConversationTurn, QueryLog, User, get_db

router = APIRouter(prefix="/api", tags=["research"])

MEMORY_TURNS = int(os.getenv("HERMES_MEMORY_TURNS", "4"))

def _llm_http_error(exc: Exception) -> HTTPException:
    """
    Translate a provider failure into an honest HTTP status.

    A hosted LLM that rate-limits (429) or goes down (5xx) is not an internal
    server fault: without this the FastAPI error handler reported every one of
    them as 500, which told the caller nothing and hid the cause in the logs.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "rate limit" in lowered or "429" in lowered:
        return HTTPException(
            status_code=429,
            detail="The language model provider is rate limiting this account. Retry shortly.",
            headers={"Retry-After": "5"},
        )
    if any(code in lowered for code in ("401", "403", "invalid api key", "unauthorized")):
        return HTTPException(status_code=502, detail="The language model provider rejected our credentials.")
    if any(code in lowered for code in ("timeout", "timed out", "connect", "502", "503", "504", "service unavailable")):
        return HTTPException(
            status_code=503,
            detail="The language model provider is unavailable.",
            headers={"Retry-After": "5"},
        )
    if any(code in lowered for code in ("500", "internal server")):
        return HTTPException(status_code=502, detail="The language model provider returned an error.")
    return HTTPException(status_code=502, detail="The language model provider could not be reached.")




class ResearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


async def _load_messages(db: AsyncSession, user_id: int, session_id: str) -> list[dict]:
    result = await db.execute(
        select(ConversationTurn)
        .where(
            ConversationTurn.user_id == user_id,
            ConversationTurn.session_id == session_id,
        )
        .order_by(ConversationTurn.created_at.desc())
        .limit(MEMORY_TURNS * 2)
    )
    turns = list(reversed(result.scalars().all()))
    return [{"role": t.role, "content": t.content} for t in turns]


async def _persist_turn(
    db: AsyncSession,
    *,
    user_id: int,
    session_id: str,
    query: str,
    answer: str,
    model_used: str,
    cache_hit: bool,
) -> None:
    db.add(
        QueryLog(
            user_id=user_id,
            query=query,
            answer=answer,
            model_used=model_used,
            cache_hit=cache_hit,
        )
    )
    db.add(
        ConversationTurn(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=query,
        )
    )
    db.add(
        ConversationTurn(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=answer,
        )
    )
    await db.commit()


@router.post("/research")
async def research(
    req: ResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run cache_check → supervisor → query_rewrite → research → (synthesis).

    Persists last turns so follow-ups rewrite retrieval using prior context.
    """
    session_id = req.session_id or str(uuid.uuid4())
    messages = await _load_messages(db, current_user.id, session_id)

    try:
        state = run_research(
            req.query,
            session_id=session_id,
            messages=messages,
            user_id=str(current_user.id),
        )
    except HTTPException:
        raise
    except Exception as exc:  # provider/network failure
        raise _llm_http_error(exc) from exc

    answer = state.get("final_answer") or state.get("draft_answer") or ""
    citations = state.get("citations", [])
    is_cached = state.get("cache_hit", False)
    tool_trace = state.get("tool_trace") or []

    await _persist_turn(
        db,
        user_id=current_user.id,
        session_id=session_id,
        query=req.query,
        answer=answer,
        model_used=state.get("model_used", ""),
        cache_hit=is_cached,
    )

    return {
        "answer": answer,
        "citations": citations,
        "model_used": state.get("model_used", ""),
        "cache_hit": is_cached,
        "session_id": session_id,
        "rewritten_query": state.get("rewritten_query") or req.query,
        "tool_trace": tool_trace,
    }


@router.post("/research/stream")
async def research_stream(
    req: ResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Real token SSE: status → token* → done (or error).

    Cache hits emit one token with the full cached answer, then done.
    """
    session_id = req.session_id or str(uuid.uuid4())
    messages = await _load_messages(db, current_user.id, session_id)
    user_id_str = str(current_user.id)
    uid = current_user.id

    async def event_generator():
        final = None
        # The upstream generator raises lazily, so it is advanced one event at a
        # time rather than materialised into a list, which would defeat SSE by
        # buffering the whole answer before the first token is sent. Once the
        # response has started the status line is already committed, so a
        # mid-stream failure is reported as an "error" event instead.
        events = iter_research_events(
            req.query,
            messages=messages,
            user_id=user_id_str,
            session_id=session_id,
        )
        while True:
            try:
                event, data = next(events)
            except StopIteration:
                break
            except Exception as exc:  # provider/network failure
                err = _llm_http_error(exc)
                yield {"event": "error", "data": json.dumps(
                    {"detail": err.detail, "status_code": err.status_code})}
                return
            if event == "done":
                final = data
                data = {**data, "session_id": session_id}
            yield {"event": event, "data": json.dumps(data)}
        if final is not None:
            async with AsyncSessionLocal() as session:
                await _persist_turn(
                    session,
                    user_id=uid,
                    session_id=session_id,
                    query=req.query,
                    answer=final.get("answer") or "",
                    model_used=final.get("model_used") or "",
                    cache_hit=bool(final.get("cache_hit")),
                )

    return EventSourceResponse(event_generator())
