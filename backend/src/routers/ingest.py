"""
Ingestion HTTP routes — load documents into the vector store.

What this file does:
    Accepts PDF uploads, web URLs, and YouTube links. Each request creates an
    ``IngestionJob`` row, runs the appropriate loader, and stores chunks in Qdrant
    via the shared ``get_retriever()`` singleton.

Where it sits in the HERMES pipeline:
    Write path into the knowledge base. Without ingestion, ``research_node`` has
    nothing to retrieve.

What calls this:
    - React ``KnowledgeBaseView`` — ``POST /ingest/url``, ``/youtube``, ``/pdf``

What this calls:
    - ``src.ingestion.*_loader`` — extract text + metadata
    - ``src.rag.factory.get_retriever`` — chunk, embed, upsert to Qdrant
    - ``src.db.IngestionJob`` — audit trail in Postgres
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import tempfile, os

from src.db import IngestionJob, get_db
from src.auth import get_current_user
from src.db import User
from src.rag.factory import get_retriever
from src.ingestion.url_loader import ingest_url
from src.ingestion.youtube_loader import ingest_youtube

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


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
    """
    Scrape a web page and ingest into Qdrant.

    Returns job metadata on success; HTTP 422 if trafilatura cannot extract text.
    """
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
        if result.get("status") == "failed":
            raise HTTPException(
                status_code=422,
                detail=result.get("error", f"Could not extract content from {req.url}"),
            )
        job.status = "ok"
        job.chunks_stored = result.get("total_children", 0)
    except HTTPException as e:
        job.status = "error"
        job.error_msg = str(e.detail)
        await db.commit()
        raise
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
        if result.get("status") == "failed":
            raise HTTPException(
                status_code=422,
                detail=result.get("error", f"Could not fetch transcript for {req.url}"),
            )
        job.status = "ok"
        job.chunks_stored = result.get("total_children", 0)
    except HTTPException as e:
        job.status = "error"
        job.error_msg = str(e.detail)
        await db.commit()
        raise
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
