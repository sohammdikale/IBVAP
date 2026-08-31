"""
Database engine and session management.

SQLite is used for local development (zero setup). Because we go through
SQLAlchemy's engine/URL abstraction, switching to PostgreSQL later is just
changing DATABASE_URL — no model or query code needs to change.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_settings

settings = get_settings()

# check_same_thread=False is required for SQLite when the same connection
# pool is shared across FastAPI's async request handlers / background threads.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def init_db() -> None:
    """Create all tables. Called once at application startup."""
    # Import models here so they're registered on Base.metadata before create_all.
    from backend.models import alert, camera, event  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and guarantees it closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
