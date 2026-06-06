from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from climbing_elo.models import Base


def _database_url() -> str | None:
    """Return DATABASE_URL from environment, or None if unset/empty."""
    return os.environ.get("DATABASE_URL") or None


def _is_transaction_pooler(url: str) -> bool:
    """Detect Supabase transaction pooler URLs (pgBouncer on port 6543).

    The transaction pooler recycles backend connections per transaction, so a
    client-side SQLAlchemy pool can hold a connection pgBouncer has already
    detached — leading to stale-connection errors on serverless cold starts.
    NullPool is the matching contract: open+close per checkout.
    """
    if not (url.startswith("postgresql://") or url.startswith("postgresql+")):
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return host.endswith("pooler.supabase.com") and parsed.port == 6543


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
    # Supabase transaction pooler (port 6543) needs NullPool so SQLAlchemy
    # doesn't fight pgBouncer's per-transaction connection recycling. The
    # session pooler (5432) and direct connections keep the default pool.
    #
    # Escape hatch: set CLIMBING_ELO_DB_NULLPOOL=1 to force NullPool on
    # the session pooler too. Useful for long-running bulk operations
    # (catch-up re-imports, multi-hour backfills) where the session
    # pooler appears to drop long-held connections — each statement
    # gets a fresh connection, no lifetime concerns. Trades ~50ms extra
    # latency per query for connection resilience. The daily cron does
    # NOT set this; only invoke when you need it.
    force_nullpool = os.environ.get("CLIMBING_ELO_DB_NULLPOOL") == "1"
    if _is_transaction_pooler(url) or force_nullpool:
        return create_engine(
            url,
            echo=False,
            poolclass=NullPool,
            connect_args={"sslmode": "require"},
        )

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
