#!/usr/bin/env python3
"""Seed the database from Kaggle IFSC CSV files in data/."""

import logging
import sys
from pathlib import Path

from climbing_elo.database import init_db
from climbing_elo.scraper.kaggle_loader import load_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}/")
        print("Download a Kaggle IFSC dataset and place the CSV files there.")
        print(
            "Recommended: https://www.kaggle.com/datasets/mxmlnv/ifsc-competition-climbing"
        )
        sys.exit(1)

    SessionFactory = init_db()

    total_results = 0
    for csv_path in csv_files:
        print(f"\nLoading {csv_path.name}...")
        with SessionFactory() as session:
            report = load_csv(session, csv_path)
            total_results += report.results_created
            print(f"  Events:   {report.events_created}")
            print(f"  Athletes: {report.athletes_created}")
            print(f"  Results:  {report.results_created}")
            print(f"  Skipped:  {report.rows_skipped}")
            if report.errors:
                print(f"  Errors:   {len(report.errors)}")

    print(f"\nTotal results loaded: {total_results}")


if __name__ == "__main__":
    main()
