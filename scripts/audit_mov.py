#!/usr/bin/env python3
"""Audit the margin-of-victory multiplier (Issue #53).

Walks every recorded pairwise contest in ``rating_history`` and computes:

* The **legacy** MOV multiplier (unconditioned: ``min(1 + gap/max_gap, MARGIN_CAP)``),
* The **new** 538-style gap-conditioned multiplier (this is what the engine
  produces now — and what is stored in ``contributing_pairs.margin_multiplier``
  for any backfill done after Issue #53),

bucketed by the *rating gap* Δμ = μ_winner − μ_loser at the time of the contest.

The purpose is to confirm:

1. The new multiplier is uniformly ≤ the legacy multiplier for favourite wins
   (Δμ > 0) — i.e. we are damping elite-vs-junior bonuses.
2. At Δμ ≈ 0 (peer matchups) the multipliers match — i.e. we are not
   blunting genuine peer-vs-peer signal.
3. Upset wins (Δμ < 0) retain the full bonus — asymmetric.

Output: writes ``docs/MOV_AUDIT.md`` with a markdown table.

Usage::

    uv run python scripts/audit_mov.py
    uv run python scripts/audit_mov.py --db path/to/climbing_elo.db
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.database import get_engine
from climbing_elo.engine.elo import (
    DEFAULT_CONFIG,
    _gap_conditioning_factor,
)
from climbing_elo.models import Discipline, Event, RatingHistory, Round

# This audit script reads MOV / margin constants for *reporting* — it does not
# call into the engine, so it is fine to read from ``DEFAULT_CONFIG`` directly
# rather than threading an EloConfig argument through.
_CFG = DEFAULT_CONFIG
MARGIN_CAP = _CFG.margin_cap
BOULDER_MARGIN_MAX_GAP = _CFG.boulder_margin_max_gap
SPEED_MAX_GAP_SECONDS = _CFG.speed_max_gap_seconds
MOV_RATING_SCALE = _CFG.mov_rating_scale
MOV_SOFTENING = _CFG.mov_softening


def _legacy_multiplier(score_gap: float, max_gap: float) -> float:
    """Pre-#53 MOV multiplier — unconditioned on rating gap."""
    return min(1.0 + score_gap / max_gap, MARGIN_CAP)


def _max_gap_for_discipline(d: Discipline) -> float:
    if d == Discipline.BOULDER:
        return BOULDER_MARGIN_MAX_GAP
    if d == Discipline.SPEED:
        return SPEED_MAX_GAP_SECONDS
    return 20.0  # Lead default in engine/elo.py


# Rating-gap buckets (Δμ = μ_winner − μ_loser). Designed to expose the
# elite-vs-junior tail; the buckets get wider as Δμ grows.
BUCKETS: list[tuple[str, float, float]] = [
    ("upset (Δμ ≤ -100)", -1e9, -100.0),
    ("peer (-100 < Δμ < 100)", -100.0, 100.0),
    ("favourite-100-250", 100.0, 250.0),
    ("favourite-250-500", 250.0, 500.0),
    ("favourite-500-750", 500.0, 750.0),
    ("favourite-750+", 750.0, 1e9),
]


def _bucket_for(rating_gap: float) -> str | None:
    for label, lo, hi in BUCKETS:
        if lo < rating_gap <= hi:
            return label
    return None


def audit(
    session: Session,
) -> tuple[dict[str, dict[Discipline, list[tuple[float, float]]]], int]:
    """Walk rating_history pairs and tally (legacy, new) multipliers per bucket.

    Returns
    -------
    by_bucket
        ``{bucket_label: {discipline: [(legacy_mult, new_mult), ...]}}``
    total_pairs
        Number of "won"-side pair contributions audited (each pair counted
        once, not twice).
    """
    # Snapshot mu_before per (athlete_id, round_id) — we need it to read the
    # opponent's μ at the contest.
    mu_by_round: dict[tuple[int, int], float] = {}
    for rh in session.execute(select(RatingHistory)).scalars():
        mu_by_round[(rh.athlete_id, rh.round_id)] = rh.mu_before

    # Round → discipline lookup.
    discipline_by_round: dict[int, Discipline] = {}
    for round_id, discipline in session.execute(
        select(Round.id, Event.discipline).join(Event, Event.id == Round.event_id)
    ):
        discipline_by_round[round_id] = discipline

    by_bucket: dict[str, dict[Discipline, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total = 0

    for rh in session.execute(select(RatingHistory)).scalars():
        pairs = rh.contributing_pairs
        if not pairs:
            continue
        if isinstance(pairs, str):
            pairs = json.loads(pairs)
        for p in pairs:
            # Audit each contest exactly once — process the winner side only.
            if p.get("result") != "won":
                continue
            opponent_id = p["opponent_id"]
            mu_winner = rh.mu_before
            mu_loser = mu_by_round.get((opponent_id, rh.round_id))
            if mu_loser is None:
                continue  # opponent has no history row for this round (shouldn't happen)
            rating_gap = mu_winner - mu_loser

            discipline = discipline_by_round.get(rh.round_id)
            if discipline is None:
                continue

            # Reverse-engineer the score gap from the *stored* multiplier.
            # Stored mult = base * factor (where factor is the new conditioning).
            # base = min(1 + score_gap/max_gap, MARGIN_CAP).
            # We can't always reverse base out of stored — the DB at audit time
            # may have *either* old or new multipliers depending on whether a
            # post-#53 backfill ran. The robust path: derive base by *dividing*
            # the stored multiplier by the gap-conditioning factor we'd apply.
            stored = float(p.get("margin_multiplier", 1.0))
            factor_now = _gap_conditioning_factor(rating_gap)
            # If DB was generated POST-#53: stored = base * factor_now, so
            # base = stored / factor_now (recovers legacy).
            # If DB was generated PRE-#53: stored == base already; dividing
            # by factor_now would over-recover. We disambiguate by checking
            # whether base recovered this way exceeds MARGIN_CAP — which can
            # only happen if we mis-divided a legacy value.
            if factor_now > 0.0:
                candidate_base = stored / factor_now
            else:
                candidate_base = stored
            if candidate_base > MARGIN_CAP * 1.001:
                # Stored was pre-#53 (already unconditioned base). Use it as base.
                base = stored
            else:
                base = candidate_base
            base = min(base, MARGIN_CAP)
            base = max(base, 1.0)  # multiplier always ≥ 1 in the original formula

            new_mult = base * factor_now
            legacy_mult = base  # legacy = no conditioning

            bucket = _bucket_for(rating_gap)
            if bucket is None:
                continue
            by_bucket[bucket][discipline].append((legacy_mult, new_mult))
            total += 1

    return by_bucket, total


def render_markdown(
    by_bucket: dict[str, dict[Discipline, list[tuple[float, float]]]],
    total_pairs: int,
    output_path: Path,
) -> None:
    """Render an aggregate-by-bucket markdown table to ``output_path``."""
    lines: list[str] = []
    lines.append("# MOV Audit — Issue #53")
    lines.append("")
    lines.append(
        f"Generated from `rating_history.contributing_pairs` ({total_pairs:,} "
        "winner-side pairs audited)."
    )
    lines.append("")
    lines.append(f"- `MOV_RATING_SCALE = {MOV_RATING_SCALE}`")
    lines.append(f"- `MOV_SOFTENING = {MOV_SOFTENING}`")
    lines.append(f"- `MARGIN_CAP = {MARGIN_CAP}` (backstop on the base multiplier)")
    lines.append("")
    lines.append("## Aggregate — all disciplines pooled")
    lines.append("")
    lines.append(
        "| rating-gap bucket | n_pairs | mean MOV (legacy) | mean MOV (new) | Δ (new−legacy) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    # Print in the BUCKETS order for stable output.
    for label, _, _ in BUCKETS:
        all_pairs: list[tuple[float, float]] = []
        for pairs in by_bucket.get(label, {}).values():
            all_pairs.extend(pairs)
        if not all_pairs:
            lines.append(f"| {label} | 0 | — | — | — |")
            continue
        mean_legacy = statistics.fmean(p[0] for p in all_pairs)
        mean_new = statistics.fmean(p[1] for p in all_pairs)
        delta = mean_new - mean_legacy
        lines.append(
            f"| {label} | {len(all_pairs):,} | {mean_legacy:.4f} | {mean_new:.4f} | {delta:+.4f} |"
        )

    lines.append("")
    lines.append("## Per-discipline breakdown")
    lines.append("")
    for d in (Discipline.LEAD, Discipline.BOULDER, Discipline.SPEED):
        # Only emit a section if there is any data.
        any_data = any(
            d in by_bucket.get(label, {}) and by_bucket[label][d]
            for label, _, _ in BUCKETS
        )
        if not any_data:
            continue
        lines.append(f"### {d.name.title()}")
        lines.append("")
        lines.append(
            "| rating-gap bucket | n_pairs | mean MOV (legacy) | mean MOV (new) | Δ (new−legacy) |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for label, _, _ in BUCKETS:
            pairs = by_bucket.get(label, {}).get(d, [])
            if not pairs:
                lines.append(f"| {label} | 0 | — | — | — |")
                continue
            mean_legacy = statistics.fmean(p[0] for p in pairs)
            mean_new = statistics.fmean(p[1] for p in pairs)
            delta = mean_new - mean_legacy
            lines.append(
                f"| {label} | {len(pairs):,} | {mean_legacy:.4f} | {mean_new:.4f} | {delta:+.4f} |"
            )
        lines.append("")

    lines.append("## Reading the table")
    lines.append("")
    lines.append(
        "- **Upset & peer rows** — Δ should be 0 (asymmetric: no damping on upsets, "
        "no damping at Δμ=0). Small nonzero Δ in the peer row is from contests with "
        "Δμ in (0, 100) which already attract mild damping."
    )
    lines.append(
        "- **Favourite-* rows** — Δ should be negative and grow in magnitude with "
        "the rating gap. This is the elite-inflation fix: large MOV bonuses against "
        "weak fields are damped."
    )
    lines.append(
        "- **Empty rows** — disciplines that contribute no contests in that bucket "
        "(e.g. Speed rarely has Δμ > 750 because the field is small and concentrated)."
    )
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Optional SQLite path. When omitted, the script reads "
            "DATABASE_URL from the environment (required)."
        ),
    )
    parser.add_argument(
        "--output",
        default="docs/MOV_AUDIT.md",
        help="Where to write the markdown report.",
    )
    args = parser.parse_args(argv)

    engine = get_engine(Path(args.db) if args.db else None)
    with Session(engine) as session:
        by_bucket, total = audit(session)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_markdown(by_bucket, total, output_path)
    print(f"Wrote {output_path} — audited {total:,} winner-side pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
