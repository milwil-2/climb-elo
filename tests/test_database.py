"""Tests for SQLAlchemy engine construction in ``climbing_elo.database``."""

from __future__ import annotations

import pytest
from sqlalchemy.pool import NullPool

from climbing_elo.database import _is_transaction_pooler, get_engine


def test_repeated_sessions_share_engine_but_not_transactions(monkeypatch, tmp_path):
    from sqlalchemy import text
    from climbing_elo.database import get_session_factory

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'shared.db'}")
    first = get_session_factory()()
    second = get_session_factory()()
    try:
        assert first.bind is second.bind
        with first.bind.begin() as connection:
            connection.execute(text("CREATE TABLE entries (value INTEGER)"))
        first.execute(text("INSERT INTO entries VALUES (1)"))
        first.rollback()
        assert second.execute(text("SELECT count(*) FROM entries")).scalar_one() == 0
    finally:
        first.close()
        second.close()


def test_engine_changes_when_database_url_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'first.db'}")
    first = get_engine()
    assert get_engine() is first
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'second.db'}")
    assert get_engine() is not first


def test_in_memory_databases_stay_isolated(monkeypatch):
    from sqlalchemy import inspect, text

    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    first = get_engine()
    second = get_engine()
    with first.begin() as connection:
        connection.execute(text("CREATE TABLE isolated (value INTEGER)"))
    assert not inspect(second).has_table("isolated")


@pytest.mark.parametrize(
    "url, expected",
    [
        # Transaction pooler — match.
        (
            "postgresql://u:p@aws-1-us-west-2.pooler.supabase.com:6543/postgres",
            True,
        ),
        (
            "postgresql+psycopg2://u:p@aws-0-us-east-1.pooler.supabase.com:6543/postgres",
            True,
        ),
        # Session pooler (port 5432) — keep default pool.
        (
            "postgresql://u:p@aws-1-us-west-2.pooler.supabase.com:5432/postgres",
            False,
        ),
        # Direct Supabase URL — keep default pool.
        ("postgresql://u:p@db.abc.supabase.co:5432/postgres", False),
        # Non-Postgres and other shapes.
        ("sqlite:///:memory:", False),
        ("", False),
    ],
)
def test_is_transaction_pooler_shape_matching(url: str, expected: bool) -> None:
    assert _is_transaction_pooler(url) is expected


def test_get_engine_uses_nullpool_for_transaction_pooler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transaction-pooler URLs (port 6543) must use NullPool.

    Doesn't actually connect — just verifies the engine's pool class.
    """
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@aws-1-us-west-2.pooler.supabase.com:6543/postgres",
    )
    engine = get_engine()
    assert isinstance(engine.pool, NullPool)


def test_get_engine_uses_default_pool_for_session_pooler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session-pooler URLs (port 5432) keep the default (non-NullPool) pool."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@aws-1-us-west-2.pooler.supabase.com:5432/postgres",
    )
    monkeypatch.delenv("CLIMBING_ELO_DB_NULLPOOL", raising=False)
    engine = get_engine()
    assert not isinstance(engine.pool, NullPool)


def test_get_engine_force_nullpool_via_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting CLIMBING_ELO_DB_NULLPOOL=1 forces NullPool on session-pooler URLs.

    Used for long-running bulk operations (catch-up re-imports) where the
    session pooler appears to drop long-held connections.
    """
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@aws-1-us-west-2.pooler.supabase.com:5432/postgres",
    )
    monkeypatch.setenv("CLIMBING_ELO_DB_NULLPOOL", "1")
    engine = get_engine()
    assert isinstance(engine.pool, NullPool)


def test_get_engine_nullpool_env_var_only_when_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLIMBING_ELO_DB_NULLPOOL only triggers on the literal value "1"."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@aws-1-us-west-2.pooler.supabase.com:5432/postgres",
    )
    for falsy in ("0", "true", "yes", ""):
        monkeypatch.setenv("CLIMBING_ELO_DB_NULLPOOL", falsy)
        engine = get_engine()
        assert not isinstance(engine.pool, NullPool), (
            f"Expected default pool for CLIMBING_ELO_DB_NULLPOOL={falsy!r}"
        )
