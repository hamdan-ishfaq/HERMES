"""
Hermes FastAPI application — Phase 6a
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db import create_tables
from src.routers import auth, ingest, research, eval


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    print("✅ Database tables ready")
    yield


app = FastAPI(
    title="Hermes Research API",
    description="Autonomous multi-source RAG research agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(research.router)
app.include_router(eval.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "hermes"}
