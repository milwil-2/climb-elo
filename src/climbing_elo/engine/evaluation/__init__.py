"""Backtest evaluation harness — probabilistic metrics, stratifications, variant-pluggable.

This package is the foundation for R0 (the comprehensive backtesting work described
in ``docs/RATING_SYSTEM_RESEARCH.md`` §5 R0). It replaces the legacy single-metric
``scripts/run_backtest.py``.

Package layout (Issue #83, Target 1)
------------------------------------

- :mod:`.metrics` — scoring primitives (log-loss, Brier, calibration, Spearman),
  bucket helpers, and the per-round :class:`RoundPrediction` record.
- :mod:`.runner` — :class:`BacktestRunner`, :class:`BacktestDataset`,
  :class:`BacktestReport`, engine variants (:class:`CurrentEloEngine`,
  ``BACKTEST_VARIANTS`` registry) and OOS modes (:class:`HoldoutMode`,
  ``OOS_MODES`` registry).
- :mod:`.reports` — markdown rendering + :func:`make_default_output_dir`.

All public symbols are re-exported here so the historical
``from climbing_elo.engine.evaluation import X`` form continues to work
unchanged.

Probabilistic source
--------------------

All probabilistic metrics consume the output of
:func:`compute_podium_probabilities` from ``engine.projections``. That
function is the canonical Monte Carlo source — we never reinvent
probability generation here.

Output
------

A single JSON report + a human-readable markdown summary, written to
``data/backtests/<UTC-timestamp>/`` (or a caller-supplied directory).
Reproducibility is enforced: rng seed is part of the report header, and
two runs with the same inputs produce byte-identical JSON.
"""

from __future__ import annotations

from .metrics import (
    EPSILON,
    FIELD_SIZE_BUCKETS,
    TENURE_BUCKETS,
    RoundPrediction,
    _aggregate_metrics,
    _brier,
    _calibration_buckets,
    _clip_prob,
    _field_size_bucket,
    _log_loss,
    _spearman,
    _stratify,
    _stratify_athlete_rounds,
    _tenure_bucket,
    score_round,
    score_split_events,
)
from .reports import (
    _PROJECT_ROOT,
    BacktestReport,
    _fmt,
    _json_default,
    make_default_output_dir,
    render_markdown,
)
from .runner import (
    BACKTEST_VARIANTS,
    OOS_MODES,
    BacktestDataset,
    BacktestRunner,
    CurrentEloEngine,
    EngineFactory,
    HoldoutMode,
    OOSMode,
    RatingEngine,
    RatingForecast,
    TrainEvalSplit,
    _build_engine,
    register_oos_mode,
    register_variant,
)

__all__ = [
    # Metric primitives
    "EPSILON",
    "_clip_prob",
    "_log_loss",
    "_brier",
    "_calibration_buckets",
    "_spearman",
    # Stratification helpers
    "TENURE_BUCKETS",
    "FIELD_SIZE_BUCKETS",
    "_tenure_bucket",
    "_field_size_bucket",
    "_aggregate_metrics",
    "_stratify",
    "_stratify_athlete_rounds",
    # DB-backed scoring helpers
    "score_round",
    "score_split_events",
    # Per-round record
    "RoundPrediction",
    # Engine variants / registry
    "RatingForecast",
    "RatingEngine",
    "CurrentEloEngine",
    "EngineFactory",
    "BACKTEST_VARIANTS",
    "register_variant",
    # OOS modes / registry
    "TrainEvalSplit",
    "OOSMode",
    "HoldoutMode",
    "OOS_MODES",
    "register_oos_mode",
    # Backtest entrypoints
    "BacktestDataset",
    "BacktestReport",
    "BacktestRunner",
    "_build_engine",
    "_json_default",
    # Reports
    "render_markdown",
    "_fmt",
    "make_default_output_dir",
    "_PROJECT_ROOT",
]


# ---------------------------------------------------------------------------
# Plug-in registration via import side-effects
# ---------------------------------------------------------------------------
# OOS mode registration — Issue #39 plug-ins.
# Importing this registers WalkForwardMode / LeaveOneEventOutMode /
# LeaveOneAthleteOutMode into ``OOS_MODES``.
from climbing_elo.engine import oos_modes  # noqa: E402,F401

# Variant registration — Issue #38 baselines.
# Importing baselines fires the ``register_variant`` calls in that module,
# adding ``random``, ``persistence``, ``ifsc_official``, and ``stripped_elo``
# to ``BACKTEST_VARIANTS``. Keep this import at the bottom of the package
# so the registry / protocols are fully defined first (avoids circular import).
from climbing_elo.engine import baselines as _baselines  # noqa: F401, E402

# G-Elo bucketed-MOV variant — Issue #84 (Szczecinski 2022 benchmark).
# Same import-side-effect registration pattern as the baselines above.
from climbing_elo.engine import gelo as _gelo  # noqa: F401, E402

# g2pl challenger — canonical Glicko-2 over Plackett-Luce pairs
# (docs/PLAN_CHALLENGER_G2PL.md, issues #174-#190). Registers "g2pl".
from climbing_elo.engine import g2pl as _g2pl  # noqa: F401, E402
