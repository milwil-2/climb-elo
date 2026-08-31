# Climbing ELO

A rating system for World Cup competition climbing - Lead, Boulder, Speed, and the Olympic Boulder+Lead combined format.

**Live dashboard:** https://climb-elo.vercel.app

## Why

The official IFSC ranking rewards attendance and ignores opponent quality - winning a continental event against a thin field can score the same as placing eighth at a World Cup against the world's best. This project answers a different question: **who is actually the strongest right now?**

The engine decomposes each round's finishing order into pairwise contests (Plackett-Luce) and updates ratings with Glicko-2 uncertainty modulation, margin-of-victory conditioning, and a tier-weighted zero-sum tournament participation bonus. In backtests it predicts results substantially better than the official ranking.

## What's on the site

- **Leaderboard** - filter by discipline and gender, with active / all-time views.
- **Athlete profiles** - rating-over-time chart, recent events with pre/post rating.
- **Pairwise breakdown** - every contributing pair contest behind any rating change.
- **Monte Carlo projections** - 10,000-trial simulations of upcoming events.
- **Head-to-head** - win probability between any two athletes, with dual history chart.
- **Live events** - embedded stream alongside results during active competitions.

## How it works

Three-stage pipeline: **scrape → backfill → serve**. A daily job pulls results (2012-present) from the IFSC results API into Postgres; the backfill replays events chronologically to compute ratings; a FastAPI + Jinja2 app serves the dashboard and a read-only REST API (docs at `/docs`).

Stack: FastAPI · SQLAlchemy · Supabase Postgres · Vercel · `uv`.

## Development

```bash
uv sync --all-extras
uv run pytest                                      # tests use in-memory SQLite
export DATABASE_URL='postgresql://...'             # required for scripts + server
uv run uvicorn climbing_elo.api.app:app --reload   # http://localhost:8000
```

[CLAUDE.md](./CLAUDE.md) is the full engineering reference (architecture, operations, connection strings, data pipeline).

## License

Personal project; not currently licensed for redistribution.
