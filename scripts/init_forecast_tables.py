#!/usr/bin/env python3
"""Create the forecast tables on the live DB.

Idempotent: ``Base.metadata.create_all`` only emits CREATE TABLE for tables
that don't yet exist. Existing tables (athletes, events, results, rounds,
ratings, rating_history) are left untouched.

Runs against ``$DATABASE_URL`` — point at the **session pooler** (port 5432)
locally, or let the daily ``scrape-supabase.yml`` workflow invoke this with
its own secret. Safe to re-run.

Usage::

    uv run python scripts/init_forecast_tables.py
"""

from __future__ import annotations

import logging
import os
import sys

from sqlalchemy import inspect

from climbing_elo.database import get_engine
from climbing_elo.models import Base, EventForecast, EventForecastScore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        logger.error("DATABASE_URL is required")
        return 1

    engine = get_engine()
    inspector = inspect(engine)

    targets = [EventForecast.__tablename__, EventForecastScore.__tablename__]
    pre_existing = [t for t in targets if inspector.has_table(t)]
    missing = [t for t in targets if t not in pre_existing]

    if not missing:
        logger.info("forecast tables already exist: %s", ", ".join(targets))
        return 0

    logger.info("creating forecast tables: %s", ", ".join(missing))
    Base.metadata.create_all(
        engine,
        tables=[
            EventForecast.__table__,
            EventForecastScore.__table__,
        ],
    )
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
