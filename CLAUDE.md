# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Deployment

Production lives at **https://climb-elo.vercel.app**, served from Vercel with **Supabase Postgres** (project ref `micecpgpuispvdfqdtmm`) as the backing store.

- **Hosting**: Vercel, `@vercel/python` runtime. The project auto-deploys on every push to `main` (no separate CD workflow — Vercel watches the repo). Preview deployments are created automatically for PRs.
- **Entry point**: `api/index.py` — thin shim that prepends `src/` to `sys.path` and calls `climbing_elo.api.app.create_app()`. Startup failures surface as Vercel's standard `FUNCTION_INVOCATION_FAILED` page; the full traceback is in the deployment's runtime logs.
- **Vercel config**: `vercel.json` declares `api/index.py` as a `@vercel/python` build and routes all paths (`/(.*)`) to it.
- **Production deps**: `uv.lock` is the single source of truth — `@vercel/python` auto-detects it and runs `uv` to install pinned versions (build log shows `Using uv 0.10.11` → `Installing required dependencies from uv.lock...`). No `requirements.txt` exists; the previously hand-written one was removed in #72. Python version comes from `pyproject.toml` (`requires-python = ">=3.11"`).
- **Required env vars** (set in Vercel project settings):
  - `DATABASE_URL` — Supabase **transaction pooler** (port 6543, IPv4). See "Connection strings" below.
- **Local override**: when `DATABASE_URL` is unset, the app falls back to a local SQLite file (`climbing_elo.db`) — handy for offline dev.

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

## Smoke Test

`scripts/smoke_test.py` is a re-runnable end-to-end smoke test for the dashboard HTML routes.

**How to run:**
```bash
uv run python scripts/smoke_test.py                                  # starts its own server on :8080
uv run python scripts/smoke_test.py --base-url http://localhost:8080  # against an already-running server
uv run python scripts/smoke_test.py --no-screenshots                 # skip cmux browser screenshots
```

**When to run:** before deploys, after large refactors, after any template or route changes.

**What it covers (11 checks):** GET `/`, `/predictions`, `/head-to-head`, `/head-to-head/{a}/{b}?discipline=lead`, `/projections/new`, `/projections/{event_id}`, `/events`, `/events/{event_id}`, `/athletes/{id}`, `/breakdown/{a}/{e}`, and a 404 for non-existent athlete. **Does not cover** POST routes, REST API (`/api/v1/*`), live SSE streaming, or visual regressions beyond "key strings present".

**Screenshots:** when `cmux browser` is available and enabled, PNGs are saved to `/tmp/climbing_elo_smoke/YYYY-MM-DD/`. The `screenshots/` directory is gitignored.

**Exit codes:** 0 = all checks passed, 1 = one or more failures.

## Rate Limiting (Issue #34)

Application-level per-IP rate limiting is implemented via **slowapi** (in-memory backend). This is the production reality — Vercel's Python runtime does not put a reverse proxy with rate-limiting in front of the function, so slowapi is the only per-IP throttle in the request path.

| Endpoint | Limit |
|---|---|
| `POST /api/v1/projections` | 10 req/min |
| `GET /api/v1/predictions/upcoming` | 60 req/min |
| All other `GET /api/v1/*` | 120 req/min (default) |
| HTML routes (`/`, `/athletes/*`, etc.) | 120 req/min (default) |
| `GET /live/{event_id}/stream` (SSE) | 100-connection cap; no per-request limit |

Exceeded limits return HTTP 429 with `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers.

**Key files**: `src/climbing_elo/api/limiter.py` (shared `Limiter` instance), `src/climbing_elo/api/app.py` (wires `SlowAPIMiddleware` + exception handler), `src/climbing_elo/api/v1_routes.py` (`@limiter.limit()` decorators on the two stricter endpoints).

**Security note**: `get_remote_address` reads `request.client.host`. On Vercel, requests reach the function through Vercel's edge network, so `request.client.host` is an edge-node IP rather than the real client. To rate-limit by true client IP we would need to key off `X-Forwarded-For` (Vercel sets this; the leftmost untrusted hop is the client) with a custom key func — currently we accept the per-edge-IP limits. The in-memory backend is per-instance; Vercel may run multiple cold-start instances concurrently, so the per-instance limits are looser than the documented numbers under load. Acceptable for current traffic.

**Superseded by**: nothing — Vercel does not offer a built-in per-route rate limiter for serverless Python functions, so this is the production rate-limit. Future hardening would mean moving to a shared-state backend (e.g. Upstash Redis via the Vercel Marketplace) or switching to Vercel WAF rate-limit rules.

## Monitoring

`.github/workflows/health-check.yml` runs **every 30 minutes** and contains two independent jobs that fail / alert separately:

1. **`ifsc-health-check`** — pings the upstream `ifsc.results.info/api/v1/` via `health_check()` in `scraper/ifsc_api.py`. On 3+ consecutive failures it opens / comments on an issue labeled `health-check-alert`.
2. **`prod-health-check`** (added in #70) — probes our own deployment at `https://climb-elo.vercel.app/` (expects HTTP 200 + the string "Leaderboard") and `https://climb-elo.vercel.app/api/v1/disciplines` (expects HTTP 200 + JSON). On 3+ consecutive failures it opens / comments on an issue labeled `prod-health-alert`.

Shared behaviour:

- Exits 0 (healthy) or 1 (unhealthy) — GitHub Actions emails the maintainer on failure by default.
- If `DISCORD_WEBHOOK_URL` is set as a GitHub Actions secret, the IFSC check also posts a Discord embed alert (rate-limited to max 1/hour to suppress flapping). The prod-health check uses the same webhook.
- The workflow can also be triggered manually via `workflow_dispatch`.

## Architecture

The system is a three-stage pipeline: **scrape → backfill → serve**.

### Data Source

All competition data is fetched from `ifsc.results.info` — the legacy IFSC results API (no auth required, just a Referer header). This is still the canonical, fully-populated data source despite IFSC rebranding to "World Climbing" in 2025.

**Why not worldclimbing.com?** The new `worldclimbing.com` site is a Next.js marketing/UI front-end with no public API. The underlying results data continues to be served by `ifsc.results.info`, which had full 2026 season data (both finished and upcoming events) at the time of investigation (Issue #30, May 2025).

**If the legacy API is ever deprecated:** See [Issue #30](https://github.com/milwil-2/climb-elo/issues/30) for the migration investigation notes. A scraper targeting `worldclimbing.com`'s internal Next.js data endpoints would need to be reverse-engineered at that point.

**Scrape** (`scraper/ifsc_api.py`) fetches Lead, Boulder, or Speed results from `ifsc.results.info` (no auth — just a Referer header) and writes Athlete/Event/Round/Result rows through SQLAlchemy. The destination is whichever DB `DATABASE_URL` points at: local SQLite (`climbing_elo.db`) when running on a laptop, Supabase Postgres in production / GitHub Actions. The API structure is: `/api/v1/` → seasons → `season_leagues/{id}` → events + d_cat IDs → `events/{id}/result/{d_cat_id}` → full rankings. Only `league_id=1` (World Cup) is scraped. Discipline categories are identified by matching the d_cat discipline field.

**Backfill** (`engine/backfill.py`) processes all events chronologically, computing ELO updates per round (qualification → semi → final). Each round calls `calculate_round_updates()` from `engine/elo.py`, which decomposes the multi-athlete finishing order into all pairwise contests using Plackett-Luce. The critical normalization is `pair_k = base_k / (n - 1)` — without this, deltas scale with field size. Rating changes across a round sum to zero. Commits are per-event (atomic). The `n_events` counter increments once per event, not per round.

**Serve** (`api/routes.py`) is a FastAPI + Jinja2 dashboard deployed at **https://climb-elo.vercel.app**. The frontend is the monochrome "v2" design served at root — the original v1 frontend was removed in commit `9e96f8e` (templates promoted from `templates_v2/` to `templates/`, leaving a single template tree). HTML routes: `/` (leaderboard), `/athletes/{id}` (profile with Chart.js rating-over-time), `/events` and `/events/{id}` (results with pre/post μ), `/breakdown/{athlete_id}/{event_id}` (pairwise contributing-pairs table), `/projections/{event_id}` (Monte Carlo outcome projections), `/projections/new` (manual projection form), `/predictions` (upcoming events hub), `/head-to-head` (athlete selection form), `/head-to-head/{a_id}/{b_id}?discipline=lead` (head-to-head result page with analytic win probability, shared-event count, and dual rating-history chart). The public REST API lives under `/api/v1/` (see below).

### Connection strings (Supabase)

Supabase exposes three connection URLs for the same Postgres project. We use all three, each in a different context — picking the wrong one will silently break.

| Context | URL pattern | Port | Address family | Why |
|---|---|---|---|---|
| Local dev / one-off scripts | `db.<PROJECT>.supabase.co` | 5432 | **IPv6 only** | Direct connection. Fast, session-stateful (transactions, prepared statements, `SET LOCAL`). Works from a developer laptop (most home/office networks expose IPv6) but is unreachable from GitHub Actions runners. |
| Vercel runtime | `aws-0-<REGION>.pooler.supabase.com` | **6543** | IPv4 | **Transaction** pooler (pgBouncer). Required for serverless: each Vercel function invocation gets a fresh pooled connection. Caveats: no session-state features (`SET`, `LISTEN`, prepared statements outside a single statement), so app code must avoid them. |
| GitHub Actions (`scrape-supabase.yml`) | `aws-0-<REGION>.pooler.supabase.com` | **5432** | IPv4 | **Session** pooler. Needed because the scrape pipeline uses long-running transactions and bulk inserts that the transaction pooler will reject. Use the "Session pooler" tab in Supabase → Settings → Database. |

Pain points we've already paid for, so don't re-learn them:

- The direct `db.<PROJECT>.supabase.co` URL is **IPv6-only**. GitHub Actions runners are IPv4-only, so any workflow that uses it will fail with a DNS / connect timeout that looks like a Supabase outage. Always use a pooler URL in CI.
- The transaction pooler (6543) will reject any statement that depends on session state. Symptoms: random `prepared statement "..." does not exist` errors, or features that work locally but 500 in prod.
- The session pooler (5432) is fine for Actions but is **not** what we want for Vercel — long-lived sessions don't fit the serverless lifecycle and you'll exhaust the pool.

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
| GET | `/api/v1/athletes/{id}/combined` | Athlete's combined (BOULDER_LEAD) rating plus boulder/lead breakdown. 404 if no combined rating |
| GET | `/api/v1/events` | Paginated event list. Query: `discipline`, `season`, `limit`, `offset` |
| GET | `/api/v1/events/{id}` | Event details with rounds and per-athlete results + pre/post ELO |
| GET | `/api/v1/combined/leaderboard` | Paginated combined (BOULDER_LEAD) leaderboard with mu_boulder/mu_lead breakdown. Query: `gender`, `limit`, `offset` |
| POST | `/api/v1/projections` | Monte Carlo podium probabilities. Body: `{"discipline": "lead", "athlete_ids": [1,2,…]}` (2–64 athletes, no duplicates). Cached 1h. |
| GET | `/api/v1/predictions/upcoming` | Upcoming events with predicted top-3 per gender. Query: `discipline` (lead/boulder/speed), `season`. Falls back to likely-roster when no registered athletes. |

Source files: `api/v1_routes.py` (endpoints), `api/schemas.py` (Pydantic response models).

## Data freshness in production

Production data lives in **Supabase Postgres**. Two GitHub Actions workflows keep it (and a local archive) fresh:

- **`.github/workflows/scrape-supabase.yml`** — runs daily at 04:00 UTC against the Supabase session pooler. Scrapes upcoming events + recent finished results, runs the ELO backfill (idempotent via `uq_rating_history_athlete_round`), and refreshes combined Boulder+Lead ratings. Workflow-dispatchable with an optional `historical_backfill` flag for the full 2012→present rescrape. Requires the `DATABASE_URL` repo secret (session pooler URL, port 5432).
- **`.github/workflows/snapshot.yml`** — runs daily at 03:00 UTC and uploads a gzip-compressed SQLite snapshot + SHA-256 sidecar to the `db-snapshots` GitHub Release. **Archival only now** that Supabase is the production DB; kept as a recovery / forensic artifact and so local dev (`DATABASE_URL` unset) has a sensible starting point. Retention: 30 daily + 12 monthly (1st of month) + 5 yearly (Jan 1).

Local snapshot helpers:

```bash
uv run python scripts/snapshot_db.py                          # create local snapshot
uv run python scripts/restore_snapshot.py                     # restore latest (backs up existing DB to .bak)
uv run python scripts/restore_snapshot.py --date 2026-06-01   # restore specific date
```

`snapshots/` is gitignored. Restore requires `gh` CLI authenticated.

## Caching

The `/predictions` page caches per-event Monte Carlo results via the in-memory `TTLCache` at `src/climbing_elo/cache.py` (1-hour TTL). Cache key includes a fingerprint of athletes + ratings, so stale ratings don't silently persist. Call `predictions_cache.clear()` after a scrape for immediate freshness, or run `uv run python scripts/clear_cache.py`. Multi-worker deploys get per-worker caches (acceptable for read-only data).

A separate `likely_roster_cache` (also `TTLCache`, 1-hour TTL) stores results from `engine/likely_roster.py`. Cache key: `"roster:{discipline.value}:{season}:{gender.value}"`. Flushed by `scripts/clear_cache.py` alongside `predictions_cache`.

## Predictions Roster Fallback

The IFSC API publishes a registered-athletes list only ~7-14 days before an event. For upcoming events without a stored list, `/predictions` falls back to a **likely-competitor roster** computed by `engine/likely_roster.py`:

- **Definition**: an athlete is a likely competitor in discipline X, season Y, gender G if they competed in ≥ 60% of the season's finished World Cup events (for gender G) to date.
- **Finished event**: an Event row that has ≥1 non-DNS Result stored in the DB (i.e. the scraper + backfill have processed it).
- **Early-season fallback**: if fewer than 3 World Cup events have finished, the function falls back to the top-64 athletes by current μ (filtered by gender, requiring ≥3 career events).
- **Cap**: at most 64 athletes are returned, ordered by μ descending (matches `_MAX_ATHLETES_PER_PROJECTION_CARD`).
- **Tier filter**: only `EventTier.WORLD_CUP` events count toward the denominator; continental/championship events are excluded.
- **DNS exclusion**: a DNS result does not count as participation.
- When the fallback is used, the prediction card shows a "Predicted roster based on season attendance" disclaimer and the `from_likely_roster` flag is `True` in the template context.

## Live Events

Live event support allows real-time score ingestion and streaming to browser clients during active competitions.

### Architecture

- **Poller** (`live/poller.py`): `LivePoller(event_id, dcat_id, interval_seconds=15)` is an async task that polls `/api/v1/events/{event_id}/result/{dcat_id}` on the IFSC API. It diffs the API response against the DB using `(athlete_id, round_type, rank, raw_score)` tuples, inserts new `Result` rows, and publishes payloads to the `EventBus`. Stops automatically when the event status returns `"finished"`.
- **EventBus** (`live/bus.py`): in-process pub/sub. One `asyncio.Queue` per subscriber per `event_id`. Poller writes; SSE handlers read.
- **SSE endpoint** (`api/sse.py`): `GET /live/{event_id}/stream` returns `text/event-stream`. Each new result emits `data: {"type":"new_result",...}\n\n`. Heartbeat every 30 s. Auto-closes after 4 h. Cap: 100 concurrent connections per event (429 if exceeded). 404 if event not in DB.

### Poller Mutex

A file lock at `/tmp/climbing_elo_poller_<event_id>.lock` prevents duplicate pollers across processes (e.g. two uvicorn workers or a manual CLI run). The lock is released on graceful shutdown.

### ELO Updates

Mid-event ELO updates are intentionally deferred. The poller only inserts `Result` rows. Run `scripts/run_backfill.py` after the event finishes (status = `"finished"`) to compute ratings.

### Starting / Stopping Pollers

```bash
# Manual (one event, blocks until Ctrl+C or event finishes):
uv run python scripts/live_poll.py --event-id 1234 --dcat-id 567

# With custom poll interval:
uv run python scripts/live_poll.py --event-id 1234 --dcat-id 567 --interval 30

# Programmatic (inside async code, e.g. a startup hook):
from climbing_elo.live import start_polling, stop_polling, is_polling
await start_polling(event_id=1234, dcat_id=567)
stop_polling(event_id=1234)
```

SSE stream (browser / curl):
```bash
curl -N http://localhost:8000/live/1234/stream
```

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request.

- **Jobs**: `pytest (3.11)`, `pytest (3.12)`, and `ruff` (lint + format check).
- **Branch protection**: add `pytest (3.11)` and `pytest (3.12)` as required status checks via GitHub repo Settings → Branches → Branch protection rules (one-time manual step).
- **Debugging failures**: check the Actions tab; the failing test name and full traceback appear in the "Run tests" step output. Lint failures show the offending line(s) from `ruff check` / `ruff format --check`.

## Testing

Tests use an in-memory SQLite database (`conftest.py:db_session`). Fixtures `sample_event` and `eight_athletes` provide pre-built test data. `test_elo.py` validates pairwise math (zero-sum invariant across all 3 disciplines). `test_backfill.py` runs a 3-event integration test and checks reproducibility. `test_api.py` covers all v1 REST endpoints. `test_projections.py` covers Monte Carlo invariants. `test_combined.py` covers the Boulder+Lead aggregate. `test_scraper_upcoming.py` covers upcoming-event filter logic. `test_snapshot.py` covers snapshot/restore round-trips. `test_health_check.py` covers CLI exit codes + Discord rate-limiting. `test_cache.py` covers TTLCache thread-safety + expiry. `test_likely_roster.py` covers the likely-competitor fallback logic. `test_live.py` covers the live poller + SSE (new result detection, dedup, finished-status auto-stop, EventBus pub/sub, file lock mutex, SSE 404/200/429). `test_baselines.py` + `test_external_rankings.py` cover the IFSC-official and AscentStats backtest baselines (recorded JSON fixtures in `tests/fixtures/external_rankings/`; live network tests are gated by `@pytest.mark.network`, deselected by default via `pyproject.toml`).

## Issue & Project organization (GitHub)

Two GitHub Projects partition open work for the repo:

- **Climbing ELO** (project #1, https://github.com/users/milwil-2/projects/1) — every open issue *except* those labeled `research`.
- **Research** (project #3, https://github.com/users/milwil-2/projects/3) — rating-system R&D from `docs/RATING_SYSTEM_RESEARCH.md` (issues #51-#57 territory). The `research` label marks these.

**Default issue view** should filter out research items: `https://github.com/milwil-2/climb-elo/issues?q=is%3Aissue+is%3Aopen+-label%3Aresearch`.

**Adding issues to a project is currently manual.** A `.github/workflows/auto-add-to-project.yml` was attempted and removed (commit `8d9f7e3`) — neither GitHub Projects v2 built-in "Auto-add" (unavailable for personal-account user-owned projects of this age) nor a custom Actions workflow (mysteriously never triggered on `issues:` events despite the actions allowlist being updated to permit `actions/add-to-project@*`) worked. When opening a new issue, add it to the appropriate project manually via `gh project item-add <project> --owner milwil-2 --url <issue-url>`.

**Required Dependabot labels:** the repo has `dependencies` and `ci` labels (created so Dependabot can apply them per `.github/dependabot.yml`). If either label gets deleted, Dependabot will fail silently with "labels could not be found" and refuse to rebase PRs.

## Branch protection caveats

`main` requires `pytest (3.11)` and `pytest (3.12)` status checks. The owner can bypass via `--admin` on `gh pr merge`. **Merging PRs that touch `.github/workflows/*.yml` requires the `workflow` OAuth scope** — if `gh auth status` shows scopes without `workflow`, run `gh auth refresh -h github.com -s workflow` first, or merge via the web UI.

## Actions allowlist

The repo runs in `selected_actions` mode (security lockdown — see `docs/SECURITY_LOCKDOWN.md`). Current allowlist: GitHub-owned actions + verified-creator actions + the explicit patterns `astral-sh/setup-uv@*` and `peter-evans/create-issue-from-file@*`. New third-party actions must be added to this allowlist via `gh api -X PUT repos/milwil-2/climb-elo/actions/permissions/selected-actions ...` before they will run.

## Supabase MCP server

`.mcp.json` is wired to the hosted Supabase MCP server (`https://mcp.supabase.com/mcp?project_ref=micecpgpuispvdfqdtmm`, read-only) for schema introspection and ad-hoc queries against the production DB from Claude Code.
