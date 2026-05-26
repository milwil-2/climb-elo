#!/usr/bin/env python3
"""
Create a consistent, compressed snapshot of the climbing ELO SQLite database.

Usage:
    uv run python scripts/snapshot_db.py [--db-path PATH] [--out-dir DIR]

Output:
    snapshots/climbing_elo_YYYY-MM-DD.db.gz
    snapshots/climbing_elo_YYYY-MM-DD.db.gz.sha256
"""

import argparse
import gzip
import hashlib
import shutil
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "climbing_elo.db"
DEFAULT_OUT_DIR = PROJECT_ROOT / "snapshots"


def _backup_db(src_path: Path, dst_path: Path) -> None:
    """Use SQLite's online backup API for a consistent snapshot."""
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def _gzip_file(src: Path, dst: Path) -> None:
    """Compress *src* to *dst* with gzip."""
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_counts(db_path: Path) -> dict[str, int]:
    """Return row counts for the key tables."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        counts: dict[str, int] = {}
        for table in ("events", "ratings", "results"):
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                counts[table] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = 0
        return counts
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot the climbing ELO database to a gzip-compressed file."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the source SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for snapshot files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--date",
        dest="snapshot_date",
        default=None,
        help="Override snapshot date as YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    db_path: Path = args.db_path
    out_dir: Path = args.out_dir
    snapshot_date: date = (
        date.fromisoformat(args.snapshot_date) if args.snapshot_date else date.today()
    )

    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"climbing_elo_{snapshot_date.isoformat()}"
    gz_path = out_dir / f"{stem}.db.gz"
    sha_path = out_dir / f"{stem}.db.gz.sha256"

    print(f"Source DB : {db_path}")
    print(f"Snapshot  : {gz_path}")

    # 1. Consistent backup via SQLite API into a temp file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        print("Backing up database (SQLite online backup API)…")
        _backup_db(db_path, tmp_path)

        # 2. Row counts from the clean backup copy
        counts = _row_counts(tmp_path)

        # 3. Gzip the backup
        print("Compressing…")
        _gzip_file(tmp_path, gz_path)

    finally:
        tmp_path.unlink(missing_ok=True)

    # 4. SHA-256 sidecar
    digest = _sha256(gz_path)
    sha_path.write_text(f"{digest}  {gz_path.name}\n")

    # 5. Summary
    gz_size_mb = gz_path.stat().st_size / (1024 * 1024)
    print()
    print("=== Snapshot summary ===")
    print(f"  File       : {gz_path.name}")
    print(f"  Size       : {gz_size_mb:.2f} MB")
    print(f"  SHA-256    : {digest}")
    print(f"  Events     : {counts.get('events', 0):,}")
    print(f"  Ratings    : {counts.get('ratings', 0):,}")
    print(f"  Results    : {counts.get('results', 0):,}")
    print(f"  Sidecar    : {sha_path.name}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
