#!/usr/bin/env python3
"""Refresh local IFSC + AscentStats ranking caches (Issue #44 + AscentStats).

Operator-side maintenance script. Hits live external sites; not invoked by
the backtest harness or tests. Writes JSON snapshots to
``data/external_rankings/`` (gitignored). To promote a fresh snapshot
into the in-repo fixture corpus, ``cp`` the file from
``data/external_rankings/<source>/...`` to
``tests/fixtures/external_rankings/<source>/``.

Usage
-----

::

    # Refresh both sources for 2024 + 2025 + 2026
    uv run python scripts/refresh_external_rankings.py --seasons 2024 2025 2026

    # Only one source
    uv run python scripts/refresh_external_rankings.py --source ifsc_official \
        --seasons 2024

Exit code is 0 if at least one fresh snapshot was written, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from climbing_elo.scraper.external_rankings import (
    DEFAULT_CACHE_DIR,
    refresh_ascentstats,
    refresh_ifsc_official,
)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="One or more four-digit years (e.g. 2024 2025 2026).",
    )
    p.add_argument(
        "--source",
        choices=("ifsc_official", "ascentstats", "all"),
        default="all",
        help="Which external source to refresh (default: all).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Where to write snapshots (default: {DEFAULT_CACHE_DIR}).",
    )
    args = p.parse_args(argv)

    written: list[Path] = []
    for season in args.seasons:
        if args.source in ("ifsc_official", "all"):
            for discipline in ("boulder", "lead", "speed"):
                for gender in ("M", "F"):
                    out = refresh_ifsc_official(
                        season, discipline, gender, cache_dir=args.cache_dir
                    )
                    if out:
                        written.append(out)
                        print(f"wrote {out}")
        if args.source in ("ascentstats", "all"):
            for gender in ("M", "F"):
                out = refresh_ascentstats(season, gender, cache_dir=args.cache_dir)
                if out:
                    written.append(out)
                    print(f"wrote {out}")

    if not written:
        print("No snapshots written — every fetch returned empty.", file=sys.stderr)
        return 1
    print(f"\nWrote {len(written)} snapshot(s) to {args.cache_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
