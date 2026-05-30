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
    engine = get_engine()
    assert not isinstance(engine.pool, NullPool)
