#!/usr/bin/env python3
"""
Restore a climbing ELO database snapshot from the GitHub Release `db-snapshots`.

Usage:
    uv run python scripts/restore_snapshot.py [--date YYYY-MM-DD] [--db-path PATH]

    --date   Snapshot date to restore (default: latest available in the release).
    --db-path  Where to write the restored database (default: data/climbing_elo.db).

The existing database (if any) is moved to <db-path>.bak before extraction.
Requires the `gh` CLI authenticated with read access to the repo.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "climbing_elo.db"
RELEASE_TAG = "db-snapshots"
REPO = "milwil-2/climb-elo"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def _gh(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["gh", *args]
    if capture:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    return subprocess.run(cmd, check=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_release_assets() -> list[str]:
    """Return asset names from the db-snapshots release."""
    result = _gh(
        "release",
        "view",
        RELEASE_TAG,
        "--repo",
        REPO,
        "--json",
        "assets",
        capture=True,
    )
    data = json.loads(result.stdout)
    return [a["name"] for a in data.get("assets", [])]


def _latest_snapshot_date(assets: list[str]) -> str:
    """Return the ISO date string for the most recent .db.gz asset."""
    dates = []
    for name in assets:
        if name.startswith("climbing_elo_") and name.endswith(".db.gz"):
            date_str = name[len("climbing_elo_") : -len(".db.gz")]
            dates.append(date_str)
    if not dates:
        print("ERROR: no snapshot assets found in release.", file=sys.stderr)
        sys.exit(1)
    return sorted(dates)[-1]


def _verify_sha256(gz_path: Path, sha_path: Path) -> None:
    """Verify the gz file against its sidecar sha256 file."""
    expected_line = sha_path.read_text().strip()
    # Format: "<digest>  <filename>"
    expected_digest = expected_line.split()[0]
    actual_digest = _sha256(gz_path)
    if actual_digest != expected_digest:
        print(
            f"ERROR: SHA-256 mismatch!\n  expected: {expected_digest}\n  actual:   {actual_digest}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"SHA-256 verified: {actual_digest}")


def _decompress(gz_path: Path, out_path: Path) -> None:
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore a DB snapshot from the GitHub Release."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Snapshot date to restore as YYYY-MM-DD (default: latest available)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Destination path for the restored database (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    db_path: Path = args.db_path

    # 1. Resolve which snapshot to restore
    print(f"Fetching asset list from release '{RELEASE_TAG}'…")
    assets = _list_release_assets()

    if args.date:
        snapshot_date = args.date
        gz_name = f"climbing_elo_{snapshot_date}.db.gz"
        sha_name = f"{gz_name}.sha256"
        if gz_name not in assets:
            print(
                f"ERROR: snapshot '{gz_name}' not found in release. Available:\n"
                + "\n".join(f"  {a}" for a in sorted(assets) if a.endswith(".db.gz")),
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        snapshot_date = _latest_snapshot_date(assets)
        gz_name = f"climbing_elo_{snapshot_date}.db.gz"
        sha_name = f"{gz_name}.sha256"
        print(f"No date specified — using latest: {snapshot_date}")

    print(f"Restoring snapshot: {gz_name}")

    # 2. Download into a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("Downloading snapshot and sidecar…")
        _gh(
            "release",
            "download",
            RELEASE_TAG,
            "--repo",
            REPO,
            "--pattern",
            gz_name,
            "--pattern",
            sha_name,
            "--dir",
            str(tmp_path),
        )

        gz_path = tmp_path / gz_name
        sha_path = tmp_path / sha_name

        if not gz_path.exists():
            print(f"ERROR: download failed — {gz_name} not present.", file=sys.stderr)
            sys.exit(1)
        if not sha_path.exists():
            print(
                f"WARNING: sidecar {sha_name} not found; skipping integrity check.",
                file=sys.stderr,
            )
        else:
            _verify_sha256(gz_path, sha_path)

        # 3. Back up existing DB
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            bak_path = db_path.with_suffix(".db.bak")
            print(f"Backing up existing DB to {bak_path}")
            shutil.move(str(db_path), str(bak_path))

        # 4. Decompress
        print(f"Extracting to {db_path}…")
        _decompress(gz_path, db_path)

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print()
    print("=== Restore complete ===")
    print(f"  Snapshot date : {snapshot_date}")
    print(f"  Restored to   : {db_path}")
    print(f"  DB size       : {size_mb:.1f} MB")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
