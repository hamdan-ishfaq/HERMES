from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import tempfile, os

from src.db import IngestionJob, get_db
from src.auth import get_current_user
from src.db import User
from src.rag.retriever import HermesRetriever
from src.ingestion.url_loader import ingest_url
from src.ingestion.youtube_loader import ingest_youtube

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# Shared retriever — initialised once
_retriever = None
def get_retriever() -> HermesRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HermesRetriever(use_cache=True, use_reranker=True)
    return _retriever


class URLRequest(BaseModel):
    url: str


class YouTubeRequest(BaseModel):
    url: str


@router.post("/url")
async def ingest_url_endpoint(
    req: URLRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = IngestionJob(
        user_id=current_user.id,
        source_type="url",
        source_ref=req.url,
        status="pending",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        result = ingest_url(req.url, get_retriever())
        job.status = "ok"
        job.chunks_stored = result.get("total_children", 0)
    except Exception as e:
        job.status = "error"
        job.error_msg = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    await db.commit()
    return {"job_id": job.id, "status": job.status, "chunks_stored": job.chunks_stored}


@router.post("/youtube")
async def ingest_youtube_endpoint(
    req: YouTubeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = IngestionJob(
        user_id=current_user.id,
        source_type="youtube",
        source_ref=req.url,
        status="pending",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        result = ingest_youtube(req.url, get_retriever())
        job.status = "ok"
        job.chunks_stored = result.get("total_children", 0)
    except Exception as e:
        job.status = "error"
        job.error_msg = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    await db.commit()
    return {"job_id": job.id, "status": job.status, "chunks_stored": job.chunks_stored}


@router.post("/pdf")
async def ingest_pdf_endpoint(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    job = IngestionJob(
        user_id=current_user.id,
        source_type="pdf",
        source_ref=file.filename,
        status="pending",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        from src.ingestion.pdf_loader import ingest_pdf
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = ingest_pdf(tmp_path, get_retriever())
        os.unlink(tmp_path)

        job.status = "ok"
        job.chunks_stored = result.get("total_children", 0)
    except Exception as e:
        job.status = "error"
        job.error_msg = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    await db.commit()
    return {"job_id": job.id, "status": job.status, "chunks_stored": job.chunks_stored}
