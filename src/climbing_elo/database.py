from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from climbing_elo.models import Base

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "climbing_elo.db"


def _database_url() -> str | None:
    """Return DATABASE_URL from environment, or None if unset/empty."""
    return os.environ.get("DATABASE_URL") or None


def get_engine(db_path: Path | str = DEFAULT_DB_PATH) -> Engine:
    url = _database_url()
    if url:
        # Postgres / Supabase path
        return create_engine(url, echo=False, pool_pre_ping=True)
    # SQLite fallback (local dev and tests)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"timeout": 60},  # wait up to 60s for locks
    )


def get_session_factory(db_path: Path | str = DEFAULT_DB_PATH) -> sessionmaker[Session]:
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> sessionmaker[Session]:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
