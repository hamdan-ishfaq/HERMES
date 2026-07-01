"""
Hermes FastAPI application — HTTP entry point for the backend.

What this file does:
    Creates and configures the FastAPI app that exposes REST endpoints for
    authentication, document ingestion, research queries, and evaluation.

Where it sits in the HERMES pipeline:
    This is the outermost layer. The React frontend (or any HTTP client) talks
    to this server. Requests are routed to modules under ``src/routers/``,
    which in turn call ingestion loaders, the LangGraph agent pipeline, or
    the RAGAS evaluation harness.

What calls this:
    Uvicorn (or another ASGI server) imports ``app`` from this module when you
    run something like ``uvicorn src.main:app``.

What this calls:
    - ``src.db.create_tables`` — ensures Postgres tables exist on startup
    - ``src.routers.auth`` — register / login / JWT
    - ``src.routers.ingest`` — PDF, URL, YouTube ingestion
    - ``src.routers.research`` — runs the multi-agent research graph
    - ``src.routers.eval`` — triggers RAGAS evaluation and serves dashboards
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db import create_tables
from src.routers import auth, ingest, research, eval


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
  Startup/shutdown hook for the FastAPI application.

  On startup, creates database tables if they do not exist yet. The ``yield``
  separates startup from shutdown — nothing runs after ``yield`` today, but
  this is where you would close connections on shutdown.

  Parameters:
      app: The FastAPI instance (unused here, but required by the lifespan API).

  Yields:
      Control back to FastAPI while the server is running.
    """
    await create_tables()
    print("✅ Database tables ready")
    yield


app = FastAPI(
    title="Hermes Research API",
    description="Autonomous multi-source RAG research agent",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the Vite dev server (5173) and a generic React port (3000) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount each router at its own URL prefix (defined inside each router module).
app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(research.router)
app.include_router(eval.router)


@app.get("/health")
async def health():
    """
    Lightweight liveness probe.

    Returns a small JSON payload so load balancers or developers can confirm
    the API process is up without hitting auth or the database.

    Returns:
        dict with ``status`` and ``service`` keys.
    """
    return {"status": "ok", "service": "hermes"}
