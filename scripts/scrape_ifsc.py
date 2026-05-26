#!/usr/bin/env python3
"""Scrape IFSC results (Lead, Boulder, or Speed) from the IFSC results API into the database."""

import argparse
import logging

from climbing_elo.database import init_db
from climbing_elo.scraper.ifsc_api import scrape_all_seasons

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape IFSC competition results")
    parser.add_argument(
        "--min-year", type=int, default=2006, help="Earliest season to scrape"
    )
    parser.add_argument(
        "--max-year", type=int, default=2026, help="Latest season to scrape"
    )
    parser.add_argument(
        "--discipline",
        choices=["lead", "boulder", "speed"],
        default="lead",
        help="Discipline to scrape (default: lead)",
    )
    args = parser.parse_args()

    SessionFactory = init_db()

    print(
        f"Scraping IFSC {args.discipline.capitalize()} results for {args.min_year}–{args.max_year}..."
    )
    print("This will take a few minutes due to rate limiting.\n")

    with SessionFactory() as session:
        report = scrape_all_seasons(
            session,
            min_year=args.min_year,
            max_year=args.max_year,
            discipline=args.discipline,
        )

    print("\nScrape complete:")
    print(f"  Discipline:      {args.discipline}")
    print(f"  Seasons scraped: {report.seasons_scraped}")
    print(f"  Events scraped:  {report.events_scraped}")
    print(f"  Results created: {report.results_created}")
    print(f"  Athletes found:  {report.athletes_created}")
    if report.errors:
        print(f"  Errors:          {len(report.errors)}")
        for err in report.errors[:10]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
