# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --all-extras                    # install all deps
uv run pytest                           # run all tests
uv run pytest tests/test_elo.py -k "test_zero_sum"  # run single test
uv run uvicorn climbing_elo.api.app:app --reload     # dev server on :8000
# Interactive docs at http://localhost:8000/docs

# Data pipeline (run in order)
uv run python scripts/scrape_ifsc.py --min-year 2012 --max-year 2026
uv run python scripts/run_backfill.py
uv run python scripts/run_backtest.py   # validates model beats baseline by ≥15pp
uv run python scripts/compute_combined_ratings.py  # populate BOULDER_LEAD aggregate

# Health-check monitoring
uv run python scripts/health_check_cli.py             # ping API; exit 0/1
uv run python scripts/health_check_cli.py --quiet     # no output (for cron)
uv run python scripts/health_check_cli.py --webhook "$DISCORD_WEBHOOK_URL"  # alert on failure
```

## Monitoring

The IFSC API health check runs **every 30 minutes** via `.github/workflows/health-check.yml`.

- Pings `ifsc.results.info/api/v1/` via `health_check()` in `scraper/ifsc_api.py`.
- Exits 0 (healthy) or 1 (unhealthy) — GitHub Actions emails the maintainer on failure by default.
- If `DISCORD_WEBHOOK_URL` is set as a GitHub Actions secret, a Discord embed alert is also posted (rate-limited to max 1 per hour to suppress flapping).
- On **3+ consecutive failures**, a GitHub issue is automatically opened (or commented on if one already exists) with label `health-check-alert`.
- The workflow can also be triggered manually via `workflow_dispatch`.

## Architecture

The system is a three-stage pipeline: **scrape → backfill → serve**.

### Data Source

All competition data is fetched from `ifsc.results.info` — the legacy IFSC results API (no auth required, just a Referer header). This is still the canonical, fully-populated data source despite IFSC rebranding to "World Climbing" in 2025.

**Why not worldclimbing.com?** The new `worldclimbing.com` site is a Next.js marketing/UI front-end with no public API. The underlying results data continues to be served by `ifsc.results.info`, which had full 2026 season data (both finished and upcoming events) at the time of investigation (Issue #30, May 2025).

**If the legacy API is ever deprecated:** See [Issue #30](https://github.com/milwil-2/climb-elo/issues/30) for the migration investigation notes. A scraper targeting `worldclimbing.com`'s internal Next.js data endpoints would need to be reverse-engineered at that point.

**Scrape** (`scraper/ifsc_api.py`) fetches Lead, Boulder, or Speed results from `ifsc.results.info` (no auth — just a Referer header) and writes Athlete/Event/Round/Result rows to SQLite. The API structure is: `/api/v1/` → seasons → `season_leagues/{id}` → events + d_cat IDs → `events/{id}/result/{d_cat_id}` → full rankings. Only `league_id=1` (World Cup) is scraped. Discipline categories are identified by matching the d_cat discipline field.

**Backfill** (`engine/backfill.py`) processes all events chronologically, computing ELO updates per round (qualification → semi → final). Each round calls `calculate_round_updates()` from `engine/elo.py`, which decomposes the multi-athlete finishing order into all pairwise contests using Plackett-Luce. The critical normalization is `pair_k = base_k / (n - 1)` — without this, deltas scale with field size. Rating changes across a round sum to zero. Commits are per-event (atomic). The `n_events` counter increments once per event, not per round.

**Serve** (`api/routes.py`) is a FastAPI + Jinja2 dashboard. HTML routes: `/` (leaderboard), `/athletes/{id}` (profile with Chart.js rating-over-time), `/events` and `/events/{id}` (results with pre/post μ), `/breakdown/{athlete_id}/{event_id}` (pairwise contributing-pairs table), `/projections/{event_id}` (Monte Carlo outcome projections), `/projections/new` (manual projection form). The public REST API lives under `/api/v1/` (see below).

## Data Model

Six SQLAlchemy models in `models.py`: Athlete → Event → Round → Result (competition data), Rating + RatingHistory (computed ratings). RatingHistory stores `contributing_pairs` as a JSON column for the breakdown view. Key enums: `EventTier` (olympics/world_championship/world_cup/continental), `RoundType` (qualification/semi/final), `Discipline` (L/B/S/BL).

## ELO Engine Specifics

K-factors are tiered by EventTier × RoundType, doubled from the PRD baseline after grid-search tuning (Olympics Final = 96, World Cup Final = 64, etc.). MARGIN_CAP=1.5 (tuned). Provisional athletes (< 3 events) get 2× K-multiplier. Lead score normalization: `"34+"` → 34.5, `"TOP"` → 999.0. Boulder normalization: `tops*1000 + zones*100 - top_att*10 - zone_att` (BOULDER_MARGIN_MAX_GAP=1000). Speed normalization: time in seconds (SPEED_MAX_GAP_SECONDS=2.0). σ decays toward 350 with an 18-month half-life during inactivity, and converges downward by 0.98× per event.

## Combined (Boulder+Lead) Ratings

`scripts/compute_combined_ratings.py` populates `Discipline.BOULDER_LEAD` ratings using the **geometric mean** `sqrt(mu_boulder × mu_lead)` of athletes with ≥3 events in both disciplines. The geometric mean penalizes specialists and rewards all-rounders, matching the Olympic combined format. Sigma uses RMS: `sqrt((sigma_b² + sigma_l²) / 2)`.

## Projections Engine

`engine/projections.py` provides Monte Carlo outcome prediction:

- `compute_podium_probabilities(athletes, n_simulations=10_000)` — draws N(μ, σ) performance scores per simulation, ranks athletes, and tallies win/podium/top-8 fractions. Returns `{athlete_id: {win, podium, top_8, expected_rank}}`. 10k sims for 20 athletes runs in ~15ms (numpy vectorized).
- `simulate_event_progression(athletes, rounds, n_simulations=10_000)` — multi-round Monte Carlo. Each trial draws N(μ, σ) for all athletes, advances the top-K to the next round, re-draws, and repeats until the final. Returns a list of `ProgressionResult` dataclasses with `advance_probs` (per-round), `final_podium_prob`, and `final_win_prob`. Runs pure Python per-sim (no vectorisation across rounds), so it is slower than `compute_podium_probabilities` — keep n_simulations ≤ 10k for latency-sensitive routes.
- `default_event_format(tier: str) -> list[RoundConfig]` — returns the default `RoundConfig` list for a given `EventTier` string value: Olympics/World Championship (qual→20, semi→8, final), World Cup (qual→26, semi→8, final), Continental (qual→20, final).
- `predict_winner(athletes)` — deterministic: returns athlete_id with highest μ.
- `expected_finish_ranks(athletes)` — returns athlete_ids sorted by descending μ.

Athletes with no rating for a discipline receive defaults (μ=1500, σ=350).

The `/projections/{event_id}` HTML route automatically uses `simulate_event_progression` when the event has ≥ 2 rounds recorded in the DB (detected by counting distinct `RoundType` values for the requested gender). Single-round events fall back to `compute_podium_probabilities`.

## Public REST API (v1)

All endpoints are read-only and require no authentication. CORS is open (`*`), no credentials.
Interactive docs: `http://localhost:8000/docs` — OpenAPI schema: `/openapi.json`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/disciplines` | List supported disciplines and codes |
| GET | `/api/v1/leaderboard` | Paginated ELO rankings. Query: `discipline`, `gender`, `limit` (1–100), `offset` (0–10000) |
| GET | `/api/v1/athletes/{id}` | Athlete profile with all discipline ratings and 20 most recent events |
| GET | `/api/v1/athletes/{id}/history` | Rating-over-time history for charts. Query: `discipline` |
| GET | `/api/v1/events` | Paginated event list. Query: `discipline`, `season`, `limit`, `offset` |
| GET | `/api/v1/events/{id}` | Event details with rounds and per-athlete results + pre/post ELO |

Source files: `api/v1_routes.py` (endpoints), `api/schemas.py` (Pydantic response models).

## Daily Snapshots

Daily snapshots of the DB are stored as GitHub Release artifacts on the `db-snapshots` release. The `.github/workflows/snapshot.yml` workflow runs at 03:00 UTC, scrapes recent events, runs backfill + combined ratings, and uploads a gzip-compressed snapshot + SHA-256 sidecar. Retention: 30 daily + 12 monthly (1st of month) + 5 yearly (Jan 1).

```bash
uv run python scripts/snapshot_db.py                          # create local snapshot
uv run python scripts/restore_snapshot.py                     # restore latest (backs up existing DB to .bak)
uv run python scripts/restore_snapshot.py --date 2026-06-01   # restore specific date
```

`snapshots/` is gitignored. Restore requires `gh` CLI authenticated.

## Monitoring

`.github/workflows/health-check.yml` runs `scripts/health_check_cli.py` every 30 min against `ifsc.results.info/api/v1/`. On 3+ consecutive failures it auto-creates (or comments on) an issue labeled `health-check-alert`. Set the `DISCORD_WEBHOOK_URL` repo secret for optional Discord pings (rate-limited to 1/hour). Webhook target is allowlisted to Discord hosts only.

## Caching

The `/predictions` page caches per-event Monte Carlo results via the in-memory `TTLCache` at `src/climbing_elo/cache.py` (1-hour TTL). Cache key includes a fingerprint of athletes + ratings, so stale ratings don't silently persist. Call `predictions_cache.clear()` after a scrape for immediate freshness, or run `uv run python scripts/clear_cache.py`. Multi-worker deploys get per-worker caches (acceptable for read-only data).

## Testing

Tests use an in-memory SQLite database (`conftest.py:db_session`). Fixtures `sample_event` and `eight_athletes` provide pre-built test data. `test_elo.py` validates pairwise math (zero-sum invariant across all 3 disciplines). `test_backfill.py` runs a 3-event integration test and checks reproducibility. `test_api.py` covers all v1 REST endpoints. `test_projections.py` covers Monte Carlo invariants. `test_combined.py` covers the Boulder+Lead aggregate. `test_scraper_upcoming.py` covers upcoming-event filter logic. `test_snapshot.py` covers snapshot/restore round-trips. `test_health_check.py` covers CLI exit codes + Discord rate-limiting. `test_cache.py` covers TTLCache thread-safety + expiry.
