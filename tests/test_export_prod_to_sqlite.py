"""Tests for the prod -> SQLite exporter's column selection (Issue #216).

The streaming read itself needs a live Postgres source, so it is exercised
manually against the session pooler. What is unit-testable — and what actually
regressed — is which columns the export pulls: a Core ``select(table)`` does
not inherit the ORM deferral on ``rating_history.contributing_pairs``, so the
heavy blob has to be dropped explicitly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from climbing_elo.models import Base, RatingHistory

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    """Import a script as a module (scripts/ is not on sys.path)."""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(name: str):
    return Base.metadata.tables[name]


def test_main_is_callable() -> None:
    assert callable(_load_script("export_prod_to_sqlite").main)


def test_contributing_pairs_dropped_by_default() -> None:
    module = _load_script("export_prod_to_sqlite")
    cols = module.export_columns(_table("rating_history"), include_heavy=False)
    names = {c.name for c in cols}
    assert "contributing_pairs" not in names
    # Everything else must survive — the export is still a faithful copy.
    assert "mu_after" in names and "kind" in names and "id" in names


def test_contributing_pairs_kept_when_requested() -> None:
    module = _load_script("export_prod_to_sqlite")
    cols = module.export_columns(_table("rating_history"), include_heavy=True)
    assert {c.name for c in cols} == {c.name for c in _table("rating_history").columns}


def test_other_tables_are_never_trimmed() -> None:
    """Only rating_history has a heavy column; every other table copies whole."""
    module = _load_script("export_prod_to_sqlite")
    for table in Base.metadata.sorted_tables:
        if table.name == "rating_history":
            continue
        cols = module.export_columns(table, include_heavy=False)
        assert [c.name for c in cols] == [c.name for c in table.columns], table.name


def test_dropped_column_is_nullable() -> None:
    """A skipped column must be nullable or the SQLite insert would fail."""
    assert RatingHistory.__table__.c.contributing_pairs.nullable
