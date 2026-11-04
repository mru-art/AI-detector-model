"""Optional database. Detection history is saved only when DATABASE_URL is set.

  DATABASE_URL=sqlite+aiosqlite:///./scram.db                          (simplest, no server)
  DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/forensics
"""
import os
from collections.abc import AsyncIterator

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = os.getenv("DATABASE_URL") or None

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

AsyncSessionFactory = (
    async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    if engine is not None
    else None
)


def database_enabled() -> bool:
    return AsyncSessionFactory is not None


async def get_session() -> AsyncIterator[AsyncSession]:
    """For endpoints that need the database (503 when it isn't configured)."""
    if AsyncSessionFactory is None:
        raise HTTPException(status_code=503, detail="No database is configured (set DATABASE_URL).")
    async with AsyncSessionFactory() as session:
        yield session


async def get_optional_session() -> AsyncIterator[AsyncSession | None]:
    """For endpoints that work without a database and just skip saving."""
    if AsyncSessionFactory is None:
        yield None
        return
    async with AsyncSessionFactory() as session:
        yield session


async def create_tables() -> None:
    if engine is None:
        return
    from .models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_database() -> None:
    if engine is not None:
        await engine.dispose()
