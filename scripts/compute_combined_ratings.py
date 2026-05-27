#!/usr/bin/env python3
"""Compute Boulder+Lead (BL) combined ratings for the Olympic combined format.

Default algorithm: geometric mean of the individual mu ratings.
  mu_combined = sqrt(mu_boulder * mu_lead)
  (equivalent to mu_lead**0.5 * mu_boulder**0.5)

This is the generalised geometric-mean form with equal exponents. If a fitted
weights file is present at ``data/learned_combined_weights.json`` (Issue #54),
the script switches to the **learned** weighted form:

  mu_combined = mu_lead**w_lead * mu_boulder**w_boulder

with ``w_lead + w_boulder = 1``. The learned weights are fitted by
``scripts/fit_combined_weights.py`` against held-out combined-format World
Championship outcomes and are only shipped when they beat the geometric-mean
baseline on log-loss without regressing rank correlation by more than 5%.

Rationale (geometric-mean baseline): The Olympic Boulder+Lead format rewards
all-around excellence and severely penalises athletes who are weak in either
discipline. The geometric mean naturally reflects this — a climber rated 2000
in Boulder but only 1000 in Lead gets a combined rating of sqrt(2000*1000) ≈
1414, much less than the arithmetic mean (1500). This mirrors how the Olympic
scoring works: a specialist who tanks one discipline falls far down the
combined ranking.

Only athletes with n_events >= 3 in BOTH Boulder AND Lead are included, to
ensure statistical reliability (the same provisional threshold used by the ELO
engine).

Sigma combination: by default RMS of the two sigmas, i.e.
``sqrt((sigma_b**2 + sigma_l**2) / 2)`` — the natural pooled uncertainty when
combining two independent estimates. When the learned-weights JSON also
contains ``w_sigma_lead`` / ``w_sigma_boulder`` keys (Issue #78), the formula
generalises to ``sqrt((w_σL * sigma_l**2 + w_σB * sigma_b**2) / (w_σL + w_σB))``
— which collapses to the RMS formula at (0.5, 0.5). The σ weights are
optimised independently of the μ weights against a Brier-score target.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

# Ensure src/ is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climbing_elo.database import init_db
from climbing_elo.models import Athlete, Discipline, Gender, Rating

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Minimum events in each individual discipline to be included.
MIN_EVENTS = 3

# Default = unweighted geometric mean (w_lead = w_boulder = 0.5).
DEFAULT_W_LEAD = 0.5
DEFAULT_W_BOULDER = 0.5

# Default σ weights (v_lead = v_boulder = 0.5 collapses to the classical
# RMS formula sqrt((σ_b² + σ_l²) / 2)).
DEFAULT_W_SIGMA_LEAD = 0.5
DEFAULT_W_SIGMA_BOULDER = 0.5

# Where the fitter writes its output. Read at startup; if missing or malformed
# the script falls back to the geometric mean (DEFAULT_W_*).
LEARNED_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "learned_combined_weights.json"
)


@dataclass(frozen=True)
class CombinedWeights:
    """Exponents used in ``mu_lead**w_lead * mu_boulder**w_boulder``.

    The ``source`` field tags where the weights came from for runtime logging:
    ``"geometric_mean"`` for the default (0.5, 0.5) baseline or ``"learned"``
    when they were loaded from ``data/learned_combined_weights.json``.

    ``w_sigma_lead`` / ``w_sigma_boulder`` are the σ-combination weights
    introduced by Issue #78. They default to (0.5, 0.5) which collapses the
    weighted σ formula back to the classical RMS used historically. A v1
    learned-weights JSON without the new keys therefore loads with
    ``sigma_source="rms"`` — fully backward compatible.
    """

    w_lead: float
    w_boulder: float
    source: str  # "geometric_mean" or "learned"
    w_sigma_lead: float = DEFAULT_W_SIGMA_LEAD
    w_sigma_boulder: float = DEFAULT_W_SIGMA_BOULDER
    sigma_source: str = "rms"  # "rms" or "learned"


def load_combined_weights(
    path: Path = LEARNED_WEIGHTS_PATH,
) -> CombinedWeights:
    """Return the active weights, falling back to the geometric mean.

    The JSON schema is::

        {
          "w_lead": 0.6,
          "w_boulder": 0.4,
          "log_loss": ...,
          "rank_corr": ...,
          "baseline_log_loss": ...,
          "baseline_rank_corr": ...,
          "fit_date": "..."
        }

    Any of:

    - file missing,
    - file empty / not valid JSON,
    - missing ``w_lead`` or ``w_boulder`` keys,
    - non-finite weights,
    - weights that don't sum to ~1.0 (tolerance 1e-3),

    triggers a fallback to the geometric-mean baseline. We log the reason so
    a misconfigured production deploy is debuggable from a single line.
    """
    if not path.exists():
        return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")
    try:
        raw = path.read_text()
        if not raw.strip():
            log.warning(
                "learned_combined_weights.json is empty; falling back to geometric mean"
            )
            return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "Failed to read learned_combined_weights.json (%s); "
            "falling back to geometric mean",
            exc,
        )
        return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")

    if not isinstance(payload, dict):
        log.warning(
            "learned_combined_weights.json is not a JSON object; "
            "falling back to geometric mean"
        )
        return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")

    try:
        w_lead = float(payload["w_lead"])
        w_boulder = float(payload["w_boulder"])
    except (KeyError, TypeError, ValueError):
        log.warning(
            "learned_combined_weights.json missing/invalid w_lead/w_boulder; "
            "falling back to geometric mean"
        )
        return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")

    if not (math.isfinite(w_lead) and math.isfinite(w_boulder)):
        log.warning(
            "learned weights are non-finite (w_lead=%s, w_boulder=%s); "
            "falling back to geometric mean",
            w_lead,
            w_boulder,
        )
        return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")

    if abs((w_lead + w_boulder) - 1.0) > 1e-3:
        log.warning(
            "learned weights don't sum to ~1.0 (w_lead=%s, w_boulder=%s, sum=%.4f); "
            "falling back to geometric mean",
            w_lead,
            w_boulder,
            w_lead + w_boulder,
        )
        return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")

    if w_lead < 0 or w_boulder < 0:
        log.warning(
            "learned weights are negative (w_lead=%s, w_boulder=%s); "
            "falling back to geometric mean",
            w_lead,
            w_boulder,
        )
        return CombinedWeights(DEFAULT_W_LEAD, DEFAULT_W_BOULDER, "geometric_mean")

    # Optional σ weights (#78). Backward-compatible: a v1 payload without
    # these keys keeps the historical RMS formula by leaving the defaults
    # in place.
    sigma_source = "rms"
    w_sigma_lead = DEFAULT_W_SIGMA_LEAD
    w_sigma_boulder = DEFAULT_W_SIGMA_BOULDER
    if "w_sigma_lead" in payload or "w_sigma_boulder" in payload:
        try:
            w_sigma_lead_raw = float(payload["w_sigma_lead"])
            w_sigma_boulder_raw = float(payload["w_sigma_boulder"])
        except (KeyError, TypeError, ValueError):
            log.warning("learned σ weights missing/invalid; falling back to RMS")
        else:
            if not (
                math.isfinite(w_sigma_lead_raw) and math.isfinite(w_sigma_boulder_raw)
            ):
                log.warning(
                    "learned σ weights non-finite (w_sigma_lead=%s, w_sigma_boulder=%s); "
                    "falling back to RMS",
                    w_sigma_lead_raw,
                    w_sigma_boulder_raw,
                )
            elif w_sigma_lead_raw < 0 or w_sigma_boulder_raw < 0:
                log.warning(
                    "learned σ weights are negative (w_sigma_lead=%s, w_sigma_boulder=%s); "
                    "falling back to RMS",
                    w_sigma_lead_raw,
                    w_sigma_boulder_raw,
                )
            elif w_sigma_lead_raw + w_sigma_boulder_raw <= 0:
                log.warning(
                    "learned σ weights sum to ≤0 (w_sigma_lead=%s, w_sigma_boulder=%s); "
                    "falling back to RMS",
                    w_sigma_lead_raw,
                    w_sigma_boulder_raw,
                )
            else:
                w_sigma_lead = w_sigma_lead_raw
                w_sigma_boulder = w_sigma_boulder_raw
                sigma_source = "learned"

    return CombinedWeights(
        w_lead=w_lead,
        w_boulder=w_boulder,
        source="learned",
        w_sigma_lead=w_sigma_lead,
        w_sigma_boulder=w_sigma_boulder,
        sigma_source=sigma_source,
    )


def compute_combined_mu(
    mu_boulder: float,
    mu_lead: float,
    weights: CombinedWeights | None = None,
) -> float:
    """Weighted geometric mean of Boulder and Lead ratings.

    With the default ``weights=None`` (or equal weights), this collapses to
    ``sqrt(mu_boulder * mu_lead)`` — the historical geometric-mean baseline.

    Defensive guard: ELO ratings should always be positive (starts at 1500,
    can drift down but practically floors well above zero). Reject non-positive
    inputs explicitly rather than letting ``**`` return 0 or raise opaquely.
    """
    if mu_boulder <= 0 or mu_lead <= 0:
        raise ValueError(
            f"compute_combined_mu requires positive ratings; got mu_boulder={mu_boulder}, mu_lead={mu_lead}"
        )
    if weights is None:
        return math.sqrt(mu_boulder * mu_lead)
    return (mu_lead**weights.w_lead) * (mu_boulder**weights.w_boulder)


def compute_combined_sigma(
    sigma_boulder: float,
    sigma_lead: float,
    weights: CombinedWeights | None = None,
) -> float:
    """Weighted RMS of Boulder and Lead sigmas (pooled uncertainty).

    With ``weights=None`` or with the default σ weights
    ``(w_sigma_lead = w_sigma_boulder = 0.5)``, this collapses to the
    classical RMS formula ``sqrt((σ_b² + σ_l²) / 2)`` and matches the
    historical behaviour exactly. When the learned-weights JSON includes
    σ-specific weights (Issue #78), the formula becomes

        sqrt((w_σL * σ_l² + w_σB * σ_b²) / (w_σL + w_σB))

    which lets one discipline's uncertainty dominate the combined-format
    σ if calibration favours it. The σ weights are independent of the μ
    weights — they are optimised against a Brier-score target on top-3
    finish probability, while the μ weights are optimised on log-loss.
    """
    if weights is None or weights.sigma_source == "rms":
        return math.sqrt((sigma_boulder**2 + sigma_lead**2) / 2.0)
    denom = weights.w_sigma_lead + weights.w_sigma_boulder
    if denom <= 0:
        # Defensive: load_combined_weights already rejects this case, but
        # treat it as RMS rather than crash on a malformed in-memory value.
        return math.sqrt((sigma_boulder**2 + sigma_lead**2) / 2.0)
    return math.sqrt(
        (
            weights.w_sigma_lead * sigma_lead**2
            + weights.w_sigma_boulder * sigma_boulder**2
        )
        / denom
    )


def main() -> None:
    weights = load_combined_weights()
    if weights.source == "learned":
        log.info(
            "Using LEARNED Boulder+Lead μ weights: w_lead=%.4f, w_boulder=%.4f "
            "(loaded from %s)",
            weights.w_lead,
            weights.w_boulder,
            LEARNED_WEIGHTS_PATH,
        )
        if weights.sigma_source == "learned":
            log.info(
                "Using LEARNED σ-combination weights: w_σ_lead=%.4f, w_σ_boulder=%.4f",
                weights.w_sigma_lead,
                weights.w_sigma_boulder,
            )
        else:
            log.info("Using RMS σ combination (no learned σ weights in JSON).")
    else:
        log.info(
            "Using GEOMETRIC-MEAN Boulder+Lead weights (w_lead=w_boulder=0.5) "
            "with RMS σ combination. Run scripts/fit_combined_weights.py to attempt "
            "learned weights."
        )

    SessionFactory = init_db()

    with SessionFactory() as session:
        # Load Boulder ratings (n_events >= MIN_EVENTS)
        boulder_ratings: dict[int, Rating] = {}
        for r in session.execute(
            select(Rating).where(
                Rating.discipline == Discipline.BOULDER,
                Rating.n_events >= MIN_EVENTS,
            )
        ).scalars():
            boulder_ratings[r.athlete_id] = r

        # Load Lead ratings (n_events >= MIN_EVENTS)
        lead_ratings: dict[int, Rating] = {}
        for r in session.execute(
            select(Rating).where(
                Rating.discipline == Discipline.LEAD,
                Rating.n_events >= MIN_EVENTS,
            )
        ).scalars():
            lead_ratings[r.athlete_id] = r

        # Athletes present in both
        combined_athlete_ids = set(boulder_ratings.keys()) & set(lead_ratings.keys())
        log.info(
            "Found %d athletes with %d+ events in both Boulder and Lead",
            len(combined_athlete_ids),
            MIN_EVENTS,
        )

        # Delete any existing BL ratings (idempotent re-run)
        existing_bl = (
            session.execute(
                select(Rating).where(Rating.discipline == Discipline.BOULDER_LEAD)
            )
            .scalars()
            .all()
        )
        if existing_bl:
            log.info(
                "Deleting %d existing BL ratings before recomputing", len(existing_bl)
            )
            for r in existing_bl:
                session.delete(r)
            session.flush()

        inserted = 0
        for aid in sorted(combined_athlete_ids):
            b = boulder_ratings[aid]
            lead = lead_ratings[aid]

            mu_combined = compute_combined_mu(b.mu, lead.mu, weights)
            sigma_combined = compute_combined_sigma(b.sigma, lead.sigma, weights)

            # Use the more recent of the two last_event_at dates
            last_event = None
            if b.last_event_at and lead.last_event_at:
                last_event = max(b.last_event_at, lead.last_event_at)
            elif b.last_event_at:
                last_event = b.last_event_at
            elif lead.last_event_at:
                last_event = lead.last_event_at

            # n_events for combined = min of the two (the athlete only qualifies
            # for combined rounds when they've competed in both)
            n_events_combined = min(b.n_events, lead.n_events)

            rating = Rating(
                athlete_id=aid,
                discipline=Discipline.BOULDER_LEAD,
                mu=mu_combined,
                sigma=sigma_combined,
                n_events=n_events_combined,
                last_event_at=last_event,
                provisional=False,  # both individual ratings already non-provisional
            )
            session.add(rating)
            inserted += 1

        try:
            session.commit()
        except IntegrityError as e:
            session.rollback()
            log.error(
                "Commit failed (likely concurrent run): %s. Run the script once at a time.",
                e,
            )
            raise
        log.info("Inserted %d combined (BL) ratings", inserted)

        # Print top 10 men and women for verification
        for gender in (Gender.F, Gender.M):
            label = "Women" if gender == Gender.F else "Men"
            print(f"\nTop 10 {label} (Boulder+Lead combined):")
            rows = session.execute(
                select(Rating, Athlete)
                .join(Athlete, Athlete.id == Rating.athlete_id)
                .where(
                    Rating.discipline == Discipline.BOULDER_LEAD,
                    Athlete.gender == gender,
                )
                .order_by(Rating.mu.desc())
                .limit(10)
            ).all()
            for rank, (rating, athlete) in enumerate(rows, 1):
                b = boulder_ratings.get(athlete.id)
                l_r = lead_ratings.get(athlete.id)
                b_mu = f"{b.mu:.1f}" if b else "N/A"
                l_mu = f"{l_r.mu:.1f}" if l_r else "N/A"
                print(
                    f"  {rank:2d}. {athlete.name:<30s} "
                    f"BL={rating.mu:.1f}  B={b_mu}  L={l_mu}  "
                    f"(n_B={b.n_events if b else 0}, n_L={l_r.n_events if l_r else 0})"
                )


if __name__ == "__main__":
    main()
