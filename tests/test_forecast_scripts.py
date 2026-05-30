"""Smoke tests for the forecast CLI scripts.

The deep behaviour is covered by ``test_forecasting.py`` /
``test_forecast_scoring.py``. Here we only verify each script's ``main`` is
importable + callable and that it exits cleanly when nothing matches the
default filters (empty DB).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    """Import a script as a module (scripts/ is not on sys.path)."""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script_name",
    ["snapshot_forecasts", "score_forecasts", "backfill_forecasts"],
)
def test_main_is_callable(script_name: str) -> None:
    module = _load_script(script_name)
    assert callable(module.main)


def test_snapshot_forecasts_empty_db_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["snapshot_forecasts.py", "--within-days", "7"])
    module = _load_script("snapshot_forecasts")
    # conftest.py sets DATABASE_URL to sqlite:///:memory:, which gives us an
    # empty schema. main() should walk the empty window and return 0.
    assert module.main() == 0


def test_score_forecasts_empty_db_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["score_forecasts.py", "--all-unscored"])
    module = _load_script("score_forecasts")
    assert module.main() == 0
