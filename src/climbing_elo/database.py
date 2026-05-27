from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from climbing_elo.models import Base


def _database_url() -> str | None:
    """Return DATABASE_URL from environment, or None if unset/empty."""
    return os.environ.get("DATABASE_URL") or None


def get_engine(db_path: Path | str | None = None) -> Engine:
    """Return a SQLAlchemy engine.

    - If ``db_path`` is provided, use a SQLite engine at that path. This is the
      explicit local/test/backtest path — callers pass a Path or string and
      take responsibility for the file's lifecycle.
    - Otherwise read ``DATABASE_URL`` from the environment and build a
      Postgres engine. The legacy silent fallback to a project-local SQLite
      file has been removed (Issue #82): if neither ``db_path`` nor
      ``DATABASE_URL`` is provided, this function raises ``RuntimeError``.
    """
    if db_path is not None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            f"sqlite:///{path}",
            echo=False,
            connect_args={"timeout": 60},
        )

    url = _database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL must be set. Point it at the Supabase session pooler "
            "for local dev (see CLAUDE.md → 'Connection strings'). The local "
            "SQLite fallback was removed in Issue #82."
        )

    # SQLite URLs (mostly used by the test suite) — no SSL / pool tuning.
    if url.startswith("sqlite"):
        return create_engine(url, echo=False, connect_args={"timeout": 60})

    # Postgres / Supabase — require SSL for external connections.
    return create_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"},
    )


def get_session_factory(db_path: Path | str | None = None) -> sessionmaker[Session]:
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)


def init_db(db_path: Path | str | None = None) -> sessionmaker[Session]:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
