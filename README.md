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

Three-stage pipeline: **scrape → backfill → serve**. A daily job pulls results (2012-present) from the IFSC results API into Postgres; the backfill replays events chronologically to compute ratings; a FastAPI + Jinja2 app serves the public dashboard and a private read-only REST API.

Stack: FastAPI · SQLAlchemy · Supabase Postgres · Vercel · `uv`.

## Development

```bash
uv sync --all-extras
uv run pytest                                      # tests use in-memory SQLite
export DATABASE_URL='postgresql://...'             # required for scripts + server
uv run uvicorn climbing_elo.api.app:app --reload   # http://localhost:8000
```

## Private API and request limits

`/api/v1/*`, `/docs`, `/redoc`, and `/openapi.json` require an `Authorization: Bearer <token>` header matching the server's `CLIMBING_ELO_API_KEY` environment variable. With no key configured, API access is disabled. Keep this key in server environment settings; never put it in a URL, client script, or repository. All private responses use `private, no-store` and cannot enter the CDN cache.

The website remains public. Its debounced athlete pickers use the bounded `/search/athletes` endpoint, which requires two characters and returns at most 50 matches. `/health` is a public, uncached liveness check for deployment monitoring.

Application limits per IP: 120 requests/minute per ordinary route, 60/minute for athlete search and failed authentication, 10/minute for each private prediction endpoint, and a shared 12/minute bucket for the website's prediction and live projection routes. These counters are per server instance; enable Vercel Firewall rate limiting for an additional limit across instances. Only Vercel's runtime trusts its overwritten `x-vercel-forwarded-for` header. Local servers use the socket peer.

## Serving and caching

Vercel runs `scripts/build_static.py` to copy the authoritative assets from `src/climbing_elo/static/` into the generated `public/` directory. The rewrite excludes those assets, so CSS and favicons are served directly from the CDN. Local FastAPI serves the same source assets. Static browser caching lasts one hour with validators; CDN caching lasts one day.

Public data pages have a one-minute browser cache, a ten-minute CDN freshness window, and up to one hour of stale-while-revalidate grace. Live scores and private responses are uncached. Route-specific cache policies and cookie-setting responses take precedence. The app also reuses compiled templates and database engines while keeping sessions independent. Prediction page contexts expire after ten minutes and invalidate when ratings or upcoming roster counts change; custom simulations invalidate when their input ratings or uncertainty change.

## License

Personal project; not currently licensed for redistribution.
