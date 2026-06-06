"""Tests for SQLAlchemy engine construction in ``climbing_elo.database``."""

from __future__ import annotations

import pytest
from sqlalchemy.pool import NullPool

from climbing_elo.database import _is_transaction_pooler, get_engine


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
