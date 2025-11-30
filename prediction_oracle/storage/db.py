"""Database session management."""

import logging
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings
from .models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine = None
_async_session_factory = None


def init_db():
    """Initialize database engine and create tables."""
    global _engine, _async_session_factory
    
    _engine = create_async_engine(
        settings.database_url,
        echo=False,
        future=True,
    )
    
    _async_session_factory = sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    logger.info(f"Database initialized: {settings.database_url}")


async def create_tables():
    """Create all tables in the database."""
    global _engine
    
    if _engine is None:
        init_db()
    
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created")


@asynccontextmanager
async def get_session():
    """Get an async database session."""
    global _async_session_factory
    
    if _async_session_factory is None:
        init_db()
    
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
