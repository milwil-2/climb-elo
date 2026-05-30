#!/usr/bin/env python3
"""One-shot migration: add ``engine_version`` to EventForecastScore unique key.

Issue #131 — symmetric mirror of #124, applied to the score table. The original
``uq_event_forecast_score_event_gender_backfill`` constraint keyed on
``(event_id, gender, is_backfill)``; bumps to ``engine_version_tag()`` caused
the post-event upsert to overwrite the prior-version score row. The new
constraint ``uq_event_forecast_score_event_gender_backfill_version`` keys on
``(event_id, gender, is_backfill, engine_version)`` so re-scoring at a new
engine version inserts a fresh row alongside the prior one.

Idempotent: detects current shape via ``inspector.get_unique_constraints`` and
exits 0 if the new constraint is already in place. Defensive against the
existing constraint's name (uses the inspector to look up whatever Postgres
actually stored, in case it was auto-generated rather than the
``Base.metadata.create_all`` name).

Runs against ``$DATABASE_URL`` — point at the **session pooler** (port 5432).

Usage::

    uv run python scripts/migrate_forecast_score_engine_version_constraint.py
"""

from __future__ import annotations

import logging
import os
import sys

from sqlalchemy import inspect, text

from climbing_elo.database import get_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TABLE = "event_forecast_scores"
_OLD_COLS = {"event_id", "gender", "is_backfill"}
_NEW_COLS = {"event_id", "gender", "is_backfill", "engine_version"}
_NEW_NAME = "uq_event_forecast_score_event_gender_backfill_version"


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        logger.error("DATABASE_URL is required")
        return 1

    engine = get_engine()
    inspector = inspect(engine)

    if not inspector.has_table(_TABLE):
        logger.error(
            "table %s does not exist; run scripts/init_forecast_tables.py first",
            _TABLE,
        )
        return 1

    existing = inspector.get_unique_constraints(_TABLE)

    has_new = any(set(c["column_names"]) == _NEW_COLS for c in existing)
    if has_new:
        logger.info("%s already includes engine_version; nothing to do", _TABLE)
        return 0

    old_constraint = next(
        (c for c in existing if set(c["column_names"]) == _OLD_COLS),
        None,
    )
    if old_constraint is None:
        logger.error(
            "no constraint matching old shape %s found on %s; "
            "current unique constraints: %s",
            sorted(_OLD_COLS),
            _TABLE,
            [(c["name"], sorted(c["column_names"])) for c in existing],
        )
        return 1

    old_name = old_constraint["name"]
    logger.info("migrating %s: drop %s -> add %s", _TABLE, old_name, _NEW_NAME)

    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE {_TABLE} DROP CONSTRAINT "{old_name}"'))
        conn.execute(
            text(
                f"ALTER TABLE {_TABLE} "
                f"ADD CONSTRAINT {_NEW_NAME} "
                "UNIQUE (event_id, gender, is_backfill, engine_version)"
            )
        )

    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
