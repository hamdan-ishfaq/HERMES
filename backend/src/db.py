"""
Database models and async session factory.

What this file does:
    Defines SQLAlchemy ORM models for users, ingestion jobs, and query logs,
    and exposes helpers to obtain async database sessions and create tables.

Where it sits in the HERMES pipeline:
    Persistence layer beneath the FastAPI routers. Ingestion endpoints write
    ``IngestionJob`` rows; the research endpoint writes ``QueryLog`` rows;
    auth flows read and write ``User`` rows.

What calls this:
    - ``src.main`` — ``create_tables`` on application startup
    - ``src.auth.get_current_user`` — loads users by email
    - All routers under ``src/routers/`` — via ``Depends(get_db)``

What this calls:
    - PostgreSQL through ``asyncpg`` (connection URL from ``DATABASE_URL``)

Tables:
    users, ingestion_jobs, query_logs
"""

import os
from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://hermes:hermes_pass@localhost:5432/hermes_db",
)

engine = create_async_engine(DATABASE_URL, echo=False)
# expire_on_commit=False keeps ORM attributes usable after commit (handy in routers).
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base for all Hermes ORM models."""
    pass


class User(Base):
    """
    Registered account that can ingest documents and run research queries.

    Relationships:
        ingestion_jobs — documents this user has uploaded or linked
        query_logs — research questions this user has asked
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="user")
    query_logs: Mapped[list["QueryLog"]] = relationship(back_populates="user")
    conversation_turns: Mapped[list["ConversationTurn"]] = relationship(back_populates="user")


class IngestionJob(Base):
    """
    Audit record for a single ingest attempt (PDF, URL, or YouTube).

    Routers create a row with ``status="pending"``, run the loader, then update
    ``status``, ``chunks_stored``, and optionally ``error_msg``.
    """
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)   # pdf | url | youtube
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)           # URL or filename
    status: Mapped[str] = mapped_column(String(20), default="pending")      # pending | ok | error
    chunks_stored: Mapped[int] = mapped_column(Integer, default=0)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="ingestion_jobs")


class QueryLog(Base):
    """
    History entry for one research question answered by the agent pipeline.

    Populated by ``routers/research.py`` after ``run_research`` completes.
    The eval dashboard aggregates ``cache_hit`` to report cache effectiveness.
    """
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="query_logs")


class ConversationTurn(Base):
    """
    One user or assistant message in a research session (multi-turn memory).

    Used to rewrite follow-up queries for retrieval — not long-term profiles.
    """
    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="conversation_turns")


async def get_db():
    """
    FastAPI dependency — yields an async SQLAlchemy session per request.

    The session is opened before the route handler runs and closed afterward,
    which is the standard pattern for request-scoped database access.

    Yields:
        AsyncSession bound to the shared engine.
    """
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    """
    Create all ORM tables in Postgres if they do not already exist.

    Called once during application startup from ``main.lifespan``.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
