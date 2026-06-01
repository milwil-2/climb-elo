"""Tests for ``scripts/backtest_monitor.py`` (Issue #120).

The script is loaded by absolute path so it doesn't need to live on
sys.path as a package — same pattern used by ``test_health_check.py``.

The Postgres → SQLite snapshot path is exercised indirectly via the
``_resolve_source_db`` precedence logic; we don't spin up a real Postgres
fixture here (that's the integration job in the workflow itself).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "backtest_monitor.py"


def _load_monitor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backtest_monitor", _SCRIPT_PATH)
    assert spec is not None, f"Could not load {_SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture()
def monitor() -> ModuleType:
    return _load_monitor()


# ---------------------------------------------------------------------------
# extract_metrics — happy path + NaN / missing handling
# ---------------------------------------------------------------------------


def _write_report(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "report.json"
    p.write_text(json.dumps(payload))
    return p


def test_extract_metrics_happy_path(monitor: ModuleType, tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        {
            "generated_at": "2026-05-31T05:00:00+00:00",
            "variant": "current",
            "oos_mode": "holdout(n_seasons=2)",
            "rng_seed": 42,
            "n_simulations": 10000,
            "disciplines": ["lead", "boulder"],
            "splits": [{"label": "holdout-2024"}, {"label": "holdout-2025"}],
            "aggregate": {
                "n_rounds": 120,
                "log_loss_win": 1.234,
                "log_loss_podium": 0.456,
                "log_loss_top8": 0.222,
                "hit_rate_top1": 0.45,
                "hit_rate_top3": 0.72,
                "hit_rate_top8": 0.91,
            },
        },
    )
    m = monitor.extract_metrics(report)
    assert m["variant"] == "current"
    assert m["n_rounds"] == 120
    assert m["log_loss_podium"] == pytest.approx(0.456)
    assert m["hit_rate_top3"] == pytest.approx(0.72)
    assert m["n_splits"] == 2
    assert m["disciplines"] == ["lead", "boulder"]


def test_extract_metrics_handles_nan_and_missing(
    monitor: ModuleType, tmp_path: Path
) -> None:
    # The harness emits the string "NaN" for un-computable metrics; we
    # coerce to None so downstream code (Discord embed, regression check)
    # treats missing-vs-zero correctly.
    report = _write_report(
        tmp_path,
        {
            "generated_at": "2026-05-31T05:00:00+00:00",
            "variant": "current",
            "oos_mode": "holdout(n_seasons=2)",
            "rng_seed": 42,
            "n_simulations": 10000,
            "disciplines": ["lead"],
            "splits": [],
            "aggregate": {
                "n_rounds": 0,
                "log_loss_win": "NaN",
                # log_loss_podium intentionally missing
                "hit_rate_top3": "NaN",
            },
        },
    )
    m = monitor.extract_metrics(report)
    assert m["log_loss_win"] is None
    assert m["log_loss_podium"] is None
    assert m["hit_rate_top3"] is None
    assert m["n_rounds"] == 0.0
    assert m["n_splits"] == 0


def test_extract_metrics_rejects_bools(monitor: ModuleType, tmp_path: Path) -> None:
    # Defensive: bool is a subclass of int. Make sure a stray True/False
    # in an aggregate slot doesn't get coerced to 1.0/0.0.
    report = _write_report(
        tmp_path,
        {
            "aggregate": {"log_loss_win": True, "n_rounds": 5},
        },
    )
    m = monitor.extract_metrics(report)
    assert m["log_loss_win"] is None
    assert m["n_rounds"] == 5.0


# ---------------------------------------------------------------------------
# detect_regression — absolute + relative thresholds, edge cases
# ---------------------------------------------------------------------------


def test_detect_regression_no_trailing_is_first_run(monitor: ModuleType) -> None:
    is_reg, reason = monitor.detect_regression(
        today_log_loss=0.5,
        trailing=[],
        absolute_threshold=0.02,
        relative_threshold=0.15,
    )
    assert is_reg is False
    assert "first run" in reason


def test_detect_regression_within_tolerance(monitor: ModuleType) -> None:
    # Today is +0.01 over the trailing mean — under both thresholds.
    is_reg, reason = monitor.detect_regression(
        today_log_loss=0.51,
        trailing=[0.50, 0.50, 0.50],
        absolute_threshold=0.02,
        relative_threshold=0.15,
    )
    assert is_reg is False
    assert "within tolerance" in reason


def test_detect_regression_absolute_breach(monitor: ModuleType) -> None:
    # Today is +0.03 over trailing mean — breaches absolute (0.02) but not
    # relative (0.06 < 0.15).
    is_reg, reason = monitor.detect_regression(
        today_log_loss=0.53,
        trailing=[0.50, 0.50, 0.50],
        absolute_threshold=0.02,
        relative_threshold=0.15,
    )
    assert is_reg is True
    assert "absolute" in reason


def test_detect_regression_relative_breach(monitor: ModuleType) -> None:
    # Today is +20% over trailing mean — breaches relative (0.15) but not
    # absolute (0.10 / 0.50 = +0.10 < absolute threshold here = 0.5).
    is_reg, reason = monitor.detect_regression(
        today_log_loss=0.60,
        trailing=[0.50, 0.50, 0.50],
        absolute_threshold=0.5,
        relative_threshold=0.15,
    )
    assert is_reg is True
    assert "%" in reason


def test_detect_regression_missing_today_skips(monitor: ModuleType) -> None:
    is_reg, reason = monitor.detect_regression(
        today_log_loss=None,
        trailing=[0.5, 0.5],
        absolute_threshold=0.02,
        relative_threshold=0.15,
    )
    assert is_reg is False
    assert "missing" in reason


def test_detect_regression_non_positive_mean(monitor: ModuleType) -> None:
    # Pathological case — should never happen in practice but the function
    # must not divide by zero.
    is_reg, reason = monitor.detect_regression(
        today_log_loss=0.5,
        trailing=[0.0, 0.0],
        absolute_threshold=0.02,
        relative_threshold=0.15,
    )
    assert is_reg is False
    assert "non-positive" in reason


# ---------------------------------------------------------------------------
# _resolve_source_db — precedence rules
# ---------------------------------------------------------------------------


def test_resolve_source_db_explicit_path_must_exist(
    monitor: ModuleType, tmp_path: Path
) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(FileNotFoundError):
        monitor._resolve_source_db(str(missing), tmp_path)


def test_resolve_source_db_explicit_path_returned(
    monitor: ModuleType, tmp_path: Path
) -> None:
    real = tmp_path / "real.db"
    real.touch()
    out = monitor._resolve_source_db(str(real), tmp_path)
    assert out == real


def test_resolve_source_db_sqlite_url(
    monitor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "from_url.db"
    target.touch()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target}")
    out = monitor._resolve_source_db(None, tmp_path)
    assert out == target


def test_resolve_source_db_missing_env_errors(
    monitor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="No source DB available"):
        monitor._resolve_source_db(None, tmp_path)


def test_resolve_source_db_in_memory_sqlite_rejected(
    monitor: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    with pytest.raises(RuntimeError, match="In-memory"):
        monitor._resolve_source_db(None, tmp_path)
