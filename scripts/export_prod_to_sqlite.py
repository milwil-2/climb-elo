"""Read-only export of the production Postgres DB into a local SQLite file.

For local work that needs prod-equivalent data in the SQLite form the offline
tools expect (regrids, backtests). Streams every table via SQLAlchemy Core and
bulk-inserts into a fresh SQLite DB. NEVER writes to prod.

Two things keep this from falling over on ``rating_history`` (Issue #216):

* The read is streamed with a server-side cursor and consumed in ``--batch``
  chunks. Materialising the whole table first buffered ~70 MB client-side and
  the pooler dropped the connection mid-fetch (``SSL SYSCALL error: EOF``).
* ``contributing_pairs`` is skipped by default. It averages ~1.1 KB and is the
  bulk of the table's bytes, but only the /breakdown page and the profile
  opponents preview read it, so an offline export does not need it. The ORM
  defers that column; a Core ``select(table)`` does not inherit the deferral,
  so it has to be dropped explicitly. Pass --include-contributing-pairs to
  export it anyway.

Usage:
    DATABASE_URL='<prod session-pooler URL, port 5432>' \
        uv run python scripts/export_prod_to_sqlite.py --out data/prod_export.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import Table, create_engine, insert, select

from climbing_elo.database import get_engine
from climbing_elo.models import Base

# (table, column) pairs dropped unless the caller opts back in. Keyed by name
# so the set stays readable next to the --include-contributing-pairs flag.
_HEAVY_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {("rating_history", "contributing_pairs")}
)


def export_columns(table: Table, *, include_heavy: bool) -> list:
    """Columns to pull for ``table``.

    Drops the known-heavy blobs unless ``include_heavy``. Every dropped column
    is nullable, so the omitted values land as NULL in the SQLite copy rather
    than failing the insert.
    """
    if include_heavy:
        return list(table.columns)
    return [c for c in table.columns if (table.name, c.name) not in _HEAVY_COLUMNS]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="Target SQLite path (overwritten)")
    p.add_argument("--batch", type=int, default=1000, help="Stream/insert chunk size")
    p.add_argument(
        "--include-contributing-pairs",
        action="store_true",
        help=(
            "Also export rating_history.contributing_pairs (~57 MB). Only "
            "needed if the local copy must serve the breakdown page."
        ),
    )
    args = p.parse_args()

    if args.batch < 1:
        print("error: --batch must be >= 1", file=sys.stderr)
        return 1

    prod = get_engine()  # reads DATABASE_URL from env (Postgres, sslmode=require)
    dialect = prod.dialect.name
    if dialect != "postgresql":
        print(
            f"error: DATABASE_URL is not Postgres (got {dialect!r}). "
            "Point it at the prod session pooler (5432).",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    if out.exists():
        out.unlink()
    sqlite = create_engine(f"sqlite:///{out}")
    Base.metadata.create_all(sqlite)

    with prod.connect() as pconn, sqlite.begin() as sconn:
        # Server-side cursor + yield_per bounds how much of any one table is
        # resident at a time; without it psycopg2 buffers the entire result.
        streaming = pconn.execution_options(stream_results=True, yield_per=args.batch)

        for table in Base.metadata.sorted_tables:
            cols = export_columns(table, include_heavy=args.include_contributing_pairs)
            dropped = len(table.columns) - len(cols)

            n = 0
            result = streaming.execute(select(*cols))
            for chunk in result.mappings().partitions():
                sconn.execute(insert(table), [dict(m) for m in chunk])
                n += len(chunk)

            note = f" (skipped {dropped} heavy column(s))" if dropped else ""
            print(f"  {table.name}: {n} rows{note}", flush=True)

    print(f"exported prod -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
