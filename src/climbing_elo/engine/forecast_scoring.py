"""Post-event forecast scoring (joyful-swinging-map plan).

Given a set of :class:`EventForecast` rows for an (event, gender, is_backfill)
triple and the actual finished results stored in the DB, compute:

* Per-stage Brier scores: ``mean((p − y)²)`` across athletes for
  ``prob_reach_semi``, ``prob_reach_final``, ``prob_podium``, ``prob_win``.
* Per-stage log-loss: ``-mean(y·log(p) + (1−y)·log(1−p))`` with probability
  clipping at ε=1e-9.
* Top-K intersection counts (top-3 and top-8) between predicted ordering
  (descending ``prob_podium`` / ``expected_rank``) and actual final ordering.
* Spearman rank correlation between predicted ``expected_rank`` and actual
  finishing rank.

Returns ``None`` when there are no forecast rows or no final-round results
yet (event still in progress / not scraped). Otherwise upserts exactly one
:class:`EventForecastScore` row.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.models import (
    EventForecast,
    EventForecastScore,
    Gender,
    Result,
    Round,
    RoundType,
)

log = logging.getLogger(__name__)


_LOGLOSS_EPSILON = 1e-9


def _is_postgres(session: Session) -> bool:
    try:
        return session.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def _load_actual_results(
    session: Session, event_id: int, gender: Gender
) -> tuple[dict[int, int], set[int], set[int], set[int]]:
    """Pull per-athlete actual outcomes from the DB.

    Returns:
        final_rank_by_athlete:
            athlete_id → rank in the highest-priority round
            (final > semi > qualification). Excludes DNS rows.
        qualified_athletes:
            athletes with any non-DNS Result in any round (==> y_qualify=1).
        semi_athletes:
            athletes who appeared in a SEMI or FINAL round (==> y_semi=1).
        final_athletes:
            athletes who appeared in a FINAL round (==> y_final=1).
    """
    rows = session.execute(
        select(
            Result.athlete_id,
            Round.round_type,
            Result.rank,
            Result.dns,
        )
        .join(Round, Result.round_id == Round.id)
        .where(Round.event_id == event_id, Round.gender == gender)
    ).all()

    qualified: set[int] = set()
    semi: set[int] = set()
    finalists: set[int] = set()
    # priority: FINAL=2, SEMI=1, QUAL=0 — store the highest-priority rank.
    priority_by_athlete: dict[int, tuple[int, int]] = {}
    priority_map = {
        RoundType.FINAL: 2,
        RoundType.SEMI: 1,
        RoundType.QUALIFICATION: 0,
    }
    for aid, round_type, rank, dns in rows:
        if dns:
            continue
        qualified.add(aid)
        if round_type == RoundType.SEMI or round_type == RoundType.FINAL:
            semi.add(aid)
        if round_type == RoundType.FINAL:
            finalists.add(aid)
        if rank is None:
            continue
        prio = priority_map.get(round_type, -1)
        prev = priority_by_athlete.get(aid)
        if prev is None or prio > prev[0]:
            priority_by_athlete[aid] = (prio, int(rank))

    final_rank_by_athlete = {
        aid: rank for aid, (_, rank) in priority_by_athlete.items()
    }
    return final_rank_by_athlete, qualified, semi, finalists


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p_clipped = np.clip(p, _LOGLOSS_EPSILON, 1.0 - _LOGLOSS_EPSILON)
    return float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))


def _spearman(predicted: list[float], actual: list[float]) -> float | None:
    """Spearman rank correlation; returns ``None`` for degenerate input.

    Uses scipy when available (already a runtime dep per CLAUDE.md). Falls
    back to a manual Pearson-on-ranks implementation if scipy is missing —
    keeping the module importable in stripped-down environments.
    """
    if len(predicted) < 2:
        return None
    if len(set(predicted)) == 1 or len(set(actual)) == 1:
        # Zero-variance side ⇒ correlation undefined.
        return None
    try:
        from scipy.stats import spearmanr

        result = spearmanr(predicted, actual)
        rho = float(result.statistic)
        if np.isnan(rho):
            return None
        return rho
    except ImportError:  # pragma: no cover — scipy is a runtime dep
        # Manual Pearson-on-ranks fallback.
        p_ranks = _rankdata(predicted)
        a_ranks = _rankdata(actual)
        p_arr = np.array(p_ranks, dtype=np.float64)
        a_arr = np.array(a_ranks, dtype=np.float64)
        cov = np.mean((p_arr - p_arr.mean()) * (a_arr - a_arr.mean()))
        std_p = p_arr.std()
        std_a = a_arr.std()
        if std_p == 0 or std_a == 0:
            return None
        return float(cov / (std_p * std_a))


def _rankdata(values):
    """Average-rank tie-breaking, mimicking ``scipy.stats.rankdata``."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _upsert_score_row(
    session: Session,
    *,
    event_id: int,
    gender: Gender,
    is_backfill: bool,
    values: dict,
) -> EventForecastScore:
    # ``engine_version`` is part of the unique key (#131) — it lives in
    # ``values`` already, but the upsert conflict-target / lookup filter
    # needs to reference it explicitly.
    engine_version = values["engine_version"]
    if _is_postgres(session):
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(EventForecastScore).values(
            event_id=event_id,
            gender=gender,
            is_backfill=is_backfill,
            **values,
        )
        update_cols = {key: stmt.excluded[key] for key in values.keys()}
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "event_id",
                "gender",
                "is_backfill",
                "engine_version",
            ],
            set_=update_cols,
        )
        session.execute(stmt)
        row = session.execute(
            select(EventForecastScore).where(
                EventForecastScore.event_id == event_id,
                EventForecastScore.gender == gender,
                EventForecastScore.is_backfill == is_backfill,
                EventForecastScore.engine_version == engine_version,
            )
        ).scalar_one()
        return row

    # SQLite emulation.
    existing = session.execute(
        select(EventForecastScore).where(
            EventForecastScore.event_id == event_id,
            EventForecastScore.gender == gender,
            EventForecastScore.is_backfill == is_backfill,
            EventForecastScore.engine_version == engine_version,
        )
    ).scalar_one_or_none()
    if existing is not None:
        for key, val in values.items():
            setattr(existing, key, val)
        session.flush()
        return existing
    row = EventForecastScore(
        event_id=event_id,
        gender=gender,
        is_backfill=is_backfill,
        **values,
    )
    session.add(row)
    session.flush()
    return row


def score_forecast(
    session: Session,
    event_id: int,
    gender: Gender,
    *,
    is_backfill: bool = False,
) -> EventForecastScore | None:
    """Score a frozen forecast against actual results.

    Returns ``None`` when:

    * No :class:`EventForecast` rows exist for ``(event_id, gender, is_backfill)``.
    * The event has no final-round Results yet (the canonical "is this scoreable?"
      signal — finals decide podium/win, which are the highest-signal outcomes).

    Otherwise upserts exactly one :class:`EventForecastScore` row keyed on
    ``(event_id, gender, is_backfill)`` and returns it.
    """
    forecast_rows = (
        session.execute(
            select(EventForecast).where(
                EventForecast.event_id == event_id,
                EventForecast.gender == gender,
                EventForecast.is_backfill == is_backfill,
            )
        )
        .scalars()
        .all()
    )
    if not forecast_rows:
        return None

    # No final-round results ⇒ unscoreable. Check via the round, not the
    # result list, so we treat "final round exists but is empty" the same as
    # "no final round yet".
    has_final = (
        session.execute(
            select(Result.id)
            .join(Round, Result.round_id == Round.id)
            .where(
                Round.event_id == event_id,
                Round.gender == gender,
                Round.round_type == RoundType.FINAL,
                Result.dns.is_(False),
            )
            .limit(1)
        ).first()
        is not None
    )
    if not has_final:
        return None

    actual_rank, qualified_set, semi_set, final_set = _load_actual_results(
        session, event_id, gender
    )

    # Predicted vectors (athlete-aligned).
    aids = [fr.athlete_id for fr in forecast_rows]
    p_semi = np.array([fr.prob_reach_semi for fr in forecast_rows], dtype=np.float64)
    p_final = np.array([fr.prob_reach_final for fr in forecast_rows], dtype=np.float64)
    p_podium = np.array([fr.prob_podium for fr in forecast_rows], dtype=np.float64)
    p_win = np.array([fr.prob_win for fr in forecast_rows], dtype=np.float64)
    pred_expected_rank = np.array(
        [fr.expected_rank for fr in forecast_rows], dtype=np.float64
    )

    # Actual outcome vectors.
    y_semi = np.array([1.0 if aid in semi_set else 0.0 for aid in aids])
    y_final = np.array([1.0 if aid in final_set else 0.0 for aid in aids])
    y_podium = np.array(
        [1.0 if aid in actual_rank and actual_rank[aid] <= 3 else 0.0 for aid in aids]
    )
    y_win = np.array(
        [1.0 if aid in actual_rank and actual_rank[aid] == 1 else 0.0 for aid in aids]
    )

    # Top-K intersection. Use a stable predicted ordering: highest
    # ``prob_podium`` first, ties broken by lowest ``expected_rank``.
    predicted_order = sorted(
        range(len(aids)),
        key=lambda i: (-float(p_podium[i]), float(pred_expected_rank[i])),
    )
    predicted_top3 = {aids[i] for i in predicted_order[:3]}
    predicted_top8 = {aids[i] for i in predicted_order[:8]}

    actual_top3 = {aid for aid, r in actual_rank.items() if r <= 3}
    actual_top8 = {aid for aid, r in actual_rank.items() if r <= 8}

    top3_intersection = min(3, len(predicted_top3 & actual_top3))
    top8_intersection = min(8, len(predicted_top8 & actual_top8))

    # Spearman over athletes for whom we have an actual rank.
    scored_pairs = [
        (float(pred_expected_rank[i]), float(actual_rank[aids[i]]))
        for i in range(len(aids))
        if aids[i] in actual_rank
    ]
    if scored_pairs:
        spearman = _spearman(
            [p for p, _ in scored_pairs],
            [a for _, a in scored_pairs],
        )
    else:
        spearman = None

    # n_simulations is recorded on each forecast row; pick the max in case a
    # mid-run scrape mixed two snapshots (shouldn't happen — upsert overwrites
    # — but cheap defence).
    n_simulations = max(int(fr.n_simulations) for fr in forecast_rows)
    engine_version = forecast_rows[0].engine_version

    values = {
        "engine_version": engine_version,
        "n_athletes": len(aids),
        "n_simulations": n_simulations,
        "brier_semi": _brier(p_semi, y_semi),
        "brier_final": _brier(p_final, y_final),
        "brier_podium": _brier(p_podium, y_podium),
        "brier_win": _brier(p_win, y_win),
        "logloss_semi": _logloss(p_semi, y_semi),
        "logloss_final": _logloss(p_final, y_final),
        "logloss_podium": _logloss(p_podium, y_podium),
        "logloss_win": _logloss(p_win, y_win),
        "top3_intersection": int(top3_intersection),
        "top8_intersection": int(top8_intersection),
        "spearman_rank": spearman,
        "computed_at": datetime.now(timezone.utc),
    }

    return _upsert_score_row(
        session,
        event_id=event_id,
        gender=gender,
        is_backfill=is_backfill,
        values=values,
    )
