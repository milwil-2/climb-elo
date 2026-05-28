# Climbing ELO

A rating system for World Cup–level competition climbing — Lead, Boulder, Speed, and the Olympic Boulder+Lead combined format.

**Live dashboard:** https://climb-elo.vercel.app
**Data source:** [`ifsc.results.info`](https://ifsc.results.info/api/v1) (the legacy IFSC results API — still fully populated despite the 2025 rebrand to "World Climbing")
**Stack:** FastAPI · SQLAlchemy 2.0 · Jinja2 · Supabase Postgres · Vercel `@vercel/python` · `uv`

## Why

The official IFSC ranking system rewards attendance and ignores opponent quality — winning a continental event against a thin field can score the same points as placing eighth at a World Cup against the world's best. This project answers a different question: **who is actually the strongest right now?**

The engine uses Plackett-Luce pairwise ELO with Glicko-2 rating-deviation modulation, 538-style margin-of-victory gap-conditioning, and a tier-weighted zero-sum tournament participation bonus. In backtest it beats the IFSC official ranking by +37.5 percentage points on Lead and +77.8pp on Boulder.

## What you can do with the live site

- **Leaderboard** — three view modes (active in last 12 months / smart all-time / no-filter debug). Filter by discipline + gender.
- **Athlete profiles** — Chart.js rating-over-time graph, recent events with pre/post μ, photo + body metrics where IFSC publishes them.
- **Pairwise breakdown** — for any (athlete, event) pair, every contributing pair contest is listed with expected/actual, margin multiplier, and Δμ.
- **Monte Carlo projections** — 10,000-trial simulation of upcoming events using the registered roster (or a likely-roster fallback when IFSC hasn't published one yet).
- **Head-to-head** — analytic win probability between any two athletes, plus a dual rating-history chart.
- **Live events** — sandboxed YouTube embed alongside the leaderboard during active competitions.

## Quickstart (5 minutes)

You need Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Clone + install
git clone https://github.com/milwil-2/climb-elo.git
cd climb-elo
uv sync --all-extras

# 2. Point at a Postgres database (Supabase recommended; the project assumes a
#    Supabase session pooler URL on port 5432 for local development).
export DATABASE_URL='postgresql://postgres:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres'

# 3. Run tests against in-memory SQLite (no DATABASE_URL needed for tests).
uv run pytest                          # all tests
uv run pytest tests/test_elo.py -x -q  # one file, fail fast

# 4. Pull historical results and compute ratings.
uv run python scripts/scrape_ifsc.py --min-year 2012 --max-year 2026
uv run python scripts/run_backfill.py
uv run python scripts/compute_combined_ratings.py

# 5. Start the dashboard.
uv run uvicorn climbing_elo.api.app:app --reload
# Open http://localhost:8000 — interactive API docs at /docs
```

Note: `DATABASE_URL` is **required** for all scripts (no silent SQLite fallback). Tests use an in-memory SQLite database that's created fresh per run.

## Architecture

```
┌─────────────────┐    ┌────────────────┐    ┌──────────────────┐
│  ifsc.results   │───▶│   Scraper      │───▶│  Postgres        │
│  .info API      │    │  scraper/      │    │  (Supabase)      │
└─────────────────┘    │  ifsc_api.py   │    │                  │
                       └────────────────┘    │  Athletes        │
                                             │  Events          │
                       ┌────────────────┐    │  Rounds          │
                       │   Backfill     │◀───┤  Results         │
                       │  engine/       │    │                  │
                       │  backfill.py   │───▶│  Ratings         │
                       │                │    │  RatingHistory   │
                       │  (Glicko-2 +   │    │                  │
                       │   MOV + TPB)   │    └──────────────────┘
                       └────────────────┘             │
                                                      ▼
                                             ┌──────────────────┐
                                             │   FastAPI app    │
                                             │   api/           │
                                             │                  │
                                             │   Jinja2 +       │
                                             │   Chart.js       │
                                             │   v1 REST API    │
                                             └────────┬─────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │ climb-elo.vercel  │
                                            │ .app (Vercel      │
                                            │  Fluid Compute)   │
                                            └───────────────────┘
```

Three-stage pipeline: **scrape → backfill → serve**. Each stage is independent and idempotent. The scraper writes Athlete / Event / Round / Result rows; the backfill processes events chronologically and writes Rating + RatingHistory rows; the FastAPI app reads from both.

## Where to look in the codebase

| Folder / file | What lives there |
|---|---|
| `src/climbing_elo/api/app.py` | FastAPI app factory; CORS + rate-limiter wiring. |
| `src/climbing_elo/api/routes.py` | HTML routes (leaderboard, athlete, event, breakdown, projections, head-to-head, live). |
| `src/climbing_elo/api/v1_routes.py` | Public REST API under `/api/v1/*` (10 read-only endpoints). |
| `src/climbing_elo/api/schemas.py` | Pydantic response models for the REST API. |
| `src/climbing_elo/engine/elo.py` | The core: Glicko-2, MOV gap-conditioning, K-factor table, `EloConfig`, `compute_tournament_participation_bonus`. |
| `src/climbing_elo/engine/backfill.py` | Per-event rating update orchestration; idempotent via the `(athlete_id, round_id, kind)` unique constraint. |
| `src/climbing_elo/engine/projections.py` | Monte Carlo: podium probabilities + multi-round event progression. |
| `src/climbing_elo/engine/activity.py` | `is_likely_retired_simple` classifier — query-layer fix for inactive ratings. |
| `src/climbing_elo/scraper/ifsc_api.py` | IFSC results-API client; pagination, retries, health check. |
| `src/climbing_elo/models.py` | SQLAlchemy models — six tables, enums, unique + check constraints. |
| `src/climbing_elo/templates/` | Jinja2 templates (single tree; monochrome v2 design). |
| `src/climbing_elo/live/` | Live-event poller, in-process pub/sub bus, SSE handler, YouTube URL validator. |
| `scripts/` | Operational scripts — scrape, backfill, fit combined weights, regrid K factors, scrape athlete profiles, smoke test, health check, clear cache. |
| `tests/` | pytest suite. 618 tests; uses in-memory SQLite. Live-network tests gated by `@pytest.mark.network`. |
| `api/index.py` | Vercel entry-point shim. |

## Operations

- **Continuous deployment** — Vercel auto-deploys on every push to `main`. PR previews are auto-created.
- **Daily data refresh** — `.github/workflows/scrape-supabase.yml` runs every day at 04:00 UTC: scrapes upcoming events + recent finished results, runs backfill for Lead / Boulder / Speed, refreshes combined Boulder+Lead, enriches athlete profile data.
- **Health monitoring** — `.github/workflows/health-check.yml` runs every 30 minutes: probes upstream IFSC API + own deployment; on 3+ consecutive failures opens / comments on a labeled issue and (if `DISCORD_WEBHOOK_URL` is set) posts a Discord alert.
- **CI** — `.github/workflows/ci.yml` runs pytest on Python 3.11 + 3.12 and ruff lint + format checks on every push and PR.
- **Secrets** — never commit `.env`, `.mcp.json`, or anything under `data/backtests/` / `agent-resources/` (all gitignored). GitHub push protection actively blocks accidental commits of API tokens.

## Configuration

The project reads only one environment variable at runtime: **`DATABASE_URL`**, a PostgreSQL connection URL.

Supabase exposes three URL variants for the same database. Picking the wrong one will silently break:

| Context | URL pattern | Port | Why |
|---|---|---|---|
| **Local dev / one-off scripts** | `aws-0-REGION.pooler.supabase.com` (session pooler) | 5432 | IPv4; supports long-lived transactions and bulk inserts. |
| **Vercel runtime** | `aws-0-REGION.pooler.supabase.com` (transaction pooler) | 6543 | IPv4; required for short-lived serverless connections. |
| **GitHub Actions** | `aws-0-REGION.pooler.supabase.com` (session pooler) | 5432 | Bulk-insert workflows need session state. |

The direct `db.PROJECT.supabase.co` URL is **IPv6-only**, so it works from a developer laptop but not from CI runners or Vercel functions. See [CLAUDE.md → "Connection strings (Supabase)"](../../CLAUDE.md#connection-strings-supabase) for the full explanation.

## Tech notes

- **Dependencies** — `uv.lock` is the single source of truth for production. Vercel's `@vercel/python` runtime auto-detects it and installs pinned versions. No `requirements.txt` exists.
- **Python version** — `requires-python = ">=3.11"`. CI tests both 3.11 and 3.12.
- **Linting** — `ruff check` + `ruff format --check` are gating in CI. Run `uv run ruff check src/ tests/ scripts/ && uv run ruff format src/ tests/ scripts/` locally before committing.
- **Testing** — every test starts from a fresh schema generated from `models.py`, so schema drift between fixtures and production can't happen.

## Documentation

- **Engineering reference** — [CLAUDE.md](../../CLAUDE.md) — the canonical operational + architectural document.
- **Process retrospective** — [docs/PROCESS_RETROSPECTIVE.md](../PROCESS_RETROSPECTIVE.md) — what one day of focused iteration looked like, and what was learned.
- **Technical deep-dive** — [docs/writeups/technical-deep-dive.md](./technical-deep-dive.md) — the ELO math, the deployment story, the orchestration patterns, all together.
- **Blog post** — [docs/writeups/blog-post.md](./blog-post.md) — public-facing narrative version.
- **Hiring summary** — [docs/writeups/hiring-summary.md](./hiring-summary.md) — skimmable one-pager.
- **Research synthesis** — [docs/RATING_SYSTEM_RESEARCH.md](../RATING_SYSTEM_RESEARCH.md) — the rating-system R&D backlog.

## License

Personal project; not currently licensed for redistribution.
