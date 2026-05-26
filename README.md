# Climbing ELO

An ELO rating system for World Climbing (formerly IFSC) Lead competitions. Uses Plackett-Luce pairwise decomposition to compute field-quality-aware ratings from historical competition results.

## Why

The official World Climbing ranking system rewards attendance over strength and ignores opponent quality. This project builds predictive ELO ratings that answer "who is actually the strongest right now?" — useful for fans, coaches, and broadcast analytics.

## Features

- **Plackett-Luce ELO engine** — pairwise decomposition adapted for multi-athlete fields with margin-aware weighting
- **IFSC API scraper** — pulls Lead results directly from the official results API (2006–present)
- **Historical backfill** — processes events chronologically to build full rating histories
- **Internal dashboard** — FastAPI + Jinja2 web app with leaderboard, athlete profiles, rating charts, and pairwise breakdowns
- **Backtesting harness** — validates model accuracy against holdout seasons

## Quickstart

```bash
# Install dependencies
uv sync --all-extras

# Scrape historical Lead results (takes a few minutes)
uv run python scripts/scrape_ifsc.py --min-year 2012 --max-year 2026

# Compute ELO ratings from results
uv run python scripts/run_backfill.py

# Start the dashboard
uv run uvicorn climbing_elo.api.app:app --reload
# Open http://localhost:8000
```

## Project Structure

```
src/climbing_elo/
├── models.py              # SQLAlchemy ORM (Athlete, Event, Round, Result, Rating, RatingHistory)
├── database.py            # SQLite engine + session management
├── engine/
│   ├── elo.py             # Core ELO: Plackett-Luce, K-factors, margin weighting, time decay
│   └── backfill.py        # Chronological rating computation across all events
├── scraper/
│   ├── ifsc_api.py        # IFSC results API client
│   └── kaggle_loader.py   # Alternative: load from Kaggle CSV datasets
├── api/
│   ├── app.py             # FastAPI application factory
│   └── routes.py          # Dashboard routes (leaderboard, athlete, event, breakdown)
└── templates/             # Jinja2 HTML templates with Chart.js
```

## Rating Model

- **Algorithm**: Plackett-Luce pairwise decomposition with K/(N-1) normalization
- **K-factors**: tiered by event prestige (Olympics > Worlds > World Cup > Continental) and round (Final > Semi > Qualification)
- **Initialization**: new athletes start at μ=1500, σ=350
- **Provisional**: athletes with < 3 events use a 2x K-multiplier for faster convergence
- **Time decay**: σ widens with an 18-month exponential half-life during inactivity
- **Edge cases**: DNS excluded, DNF ranked at bottom with capped margin, ties produce zero pairwise delta

## MVP Scope

This is v0.1 — Lead discipline only, batch updates, internal dashboard. See `agent-resources/PRD.md` for the full product vision including Boulder, Speed, Boulder+Lead aggregate ratings, public API, and live projections.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/scrape_ifsc.py` | Scrape Lead results from IFSC API |
| `scripts/run_backfill.py` | Compute ratings from all results in DB |
| `scripts/run_backtest.py` | Validate model vs. holdout seasons |
| `scripts/seed_from_kaggle.py` | Alternative: load from Kaggle CSV files |

## Testing

```bash
uv run pytest           # 26 tests: ELO math, models, backfill integration
```

## Tech Stack

Python 3.11+ · FastAPI · SQLAlchemy · SQLite · Chart.js · NumPy
