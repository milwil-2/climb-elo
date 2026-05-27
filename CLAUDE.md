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
- **Local dev / scripts**: `DATABASE_URL` is **required everywhere** as of Issue #82 — there is no longer a silent SQLite fallback. Point it at the Supabase session pooler (port 5432) for local development. The previous "local SQLite if unset" behaviour was removed because it caused stale-schema drift (Issue #79).

## Commands

```bash
uv sync --all-extras                    # install all deps
uv run pytest                           # run all tests
uv run pytest tests/test_elo.py -k "test_zero_sum"  # run single test
uv run uvicorn climbing_elo.api.app:app --reload     # dev server on :8000
# Interactive docs at http://localhost:8000/docs

# All scripts require DATABASE_URL (point at the Supabase session pooler for local dev).

# Data pipeline (run in order)
uv run python scripts/scrape_ifsc.py --min-year 2012 --max-year 2026
uv run python scripts/run_backfill.py
uv run python scripts/run_backtest.py   # validates model beats baseline by ≥15pp
uv run python scripts/compute_combined_ratings.py  # populate BOULDER_LEAD aggregate

# Athlete profile refresh (Issue #86 / #93)
# One of --only-missing or --force is required (no implicit default).
uv run python scripts/scrape_athlete_profiles.py --only-missing  # skip rows with photo_url set (daily cron mode)
uv run python scripts/scrape_athlete_profiles.py --force         # re-scrape every athlete (use when IFSC updates photos)
uv run python scripts/scrape_athlete_profiles.py --athlete-id 5  # refresh one athlete only

# One-time prod backfill (Issue #93 — run locally; GH Actions 60-min cap is too tight).
# Takes ~10-12 min for the full ~2,700-athlete population. Steady-state ongoing
# enrichment is handled by the --only-missing step in scrape-supabase.yml.
#   export DATABASE_URL='<Supabase session pooler URL, port 5432>'
#   uv run python scripts/scrape_athlete_profiles.py --only-missing

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

**What it covers (12 checks):** GET `/`, `/leaderboard`, `/athletes`, `/predictions`, `/projections`, `/head-to-head`, `/head-to-head/{a}/{b}?discipline=lead`, `/events`, `/events/{event_id}`, `/athletes/{id}`, `/breakdown/{a}/{e}`, and a 404 for non-existent athlete. **Does not cover** POST routes, REST API (`/api/v1/*`), live SSE streaming, or visual regressions beyond "key strings present". The v1 routes `/projections/new` and `/projections/{event_id}` were removed when v2 was promoted to root — no v2 equivalent ships yet.

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

`Athlete.photo_url`, `height_cm`, `weight_kg`, `wingspan_cm` (added in #86) hold optional profile metadata for the rich `/athletes/{id}` page. All nullable — most rows are NULL until `scripts/scrape_athlete_profiles.py` runs. `photo_url` is hot-linked from `ifsc.results.info` (no Vercel Blob). `weight_kg` has no IFSC source today; column exists for future expansion.

`Event.livestream_url` (added in #23) holds an optional YouTube URL for the live broadcast. Strictly validated against a `youtube.com` / `youtu.be` allowlist in `src/climbing_elo/live/livestream.py` before being rendered into a sandboxed iframe on `/live/{event_id}`. Populated manually per event — the IFSC API does not expose stream URLs.

`Athlete.retired_at` (added 2026-05-26 in #91) is a nullable manual override for the dual-view leaderboard. See "Activity classification" below.

### Activity classification

Glicko-2's σ inflates during inactivity but μ does not decay, so 5-year-absent athletes used to sit on the leaderboard at their last μ. Issue #91 (Gap 2 from #88) introduces a query-layer fix — no re-backfill required.

- **`Athlete.retired_at` (nullable DATE)** — manual override. Non-NULL ⇒ the athlete is unconditionally hidden from the `all` and `active` view filters of the All-time leaderboard. No automated source today; populated case-by-case from news / social signals.
- **Pure-function classifier** `engine.activity.is_likely_retired_simple(last_event_at, retired_at, today=None, threshold_years=3.0) → bool`:
  1. `retired_at` set → `True` (manual wins).
  2. `last_event_at is None` → `False` (never-competed athletes aren't "retired" — different problem).
  3. `(today - last_event_at) >= threshold_years` → `True`.
  Module constants: `INACTIVE_THRESHOLD_MONTHS = 12` (for `active` view), `RETIRED_THRESHOLD_YEARS = 3.0` (for the heuristic).
- **Three view modes** on `/leaderboard` (HTML) and `GET /api/v1/leaderboard?view=…`:
  - `active` (**default since 2026-05-26**) — `WHERE last_event_at >= today − 12 months`.
  - `all` — smart all-time list, `WHERE retired_at IS NULL AND (last_event_at IS NULL OR last_event_at >= today − 3 years)`.
  - `legacy` — debug-only; no activity filter (the pre-#91 behaviour).
- **Breaking change**: `GET /api/v1/leaderboard` default flipped from `legacy` → `active` on 2026-05-26 (#91). Pass `view=all` for the smart all-time list, `view=legacy` for the pre-#91 behaviour. The HTML route's default also flipped; invalid `view` values fall back to `active` (forgiving) on the HTML route, but yield a 422 on the API.
- **Out of scope** (file follow-ups): age-aware refinement gated on `year_of_birth` coverage (waiting on #86), retirement-year tooltip, news-scraper to auto-populate `retired_at`, per-discipline thresholds (boulder peak is younger than lead).

## ELO Engine Specifics

Glicko-2 RD–weighted ELO with 538-style gap-conditioned margin-of-victory. K-factor table tiered by EventTier × RoundType (Olympics Final = 48, World Cup Final = 32 — halved post-#51 as a conservative starting point for the Glicko-2 effective-K regime). Per-round effective K is `K_base · g(φ_opponent) · margin_mult`, so opponents with high uncertainty (cold start, post-sabbatical) contribute less. The legacy 2× provisional-K cliff is gone — cold start is now driven by Glicko-2's large initial φ (σ=350) and the closed-form φ shrinkage. σ decays during inactivity via Glicko-2's Wiener-process formula (`σ_inactivity=5.0`, 30-day grace), clamped to [50, 350]. MARGIN_CAP=1.5 remains as a backstop on the base multiplier; the new gap-conditioning (#53) damps favourite-side bonuses asymmetrically — `mult = base · 2.2/(max(Δμ,0)/400 + 2.2)` — so an elite crushing a junior earns less than a peer crushing a peer, while upsets keep the full bonus. Score normalization unchanged: Lead `"34+"`→34.5, `"TOP"`→999.0; Boulder `tops*1000 + zones*100 - top_att*10 - zone_att` (max_gap=1000) for pre-2025 events, decimal pass-through for 2025+ events; Speed in seconds (max_gap=2.0).

**K_FACTOR_TABLE regrid (Issue #80).** Per-cell K values are chosen via `scripts/regrid_k_factors.py` — a coordinate-descent sweep that, for each `(tier, round_type)` cell, holds all other cells fixed and tries a multiplicative grid (default `[0.5, 0.75, 1.0, 1.5, 2.0]`) around the current value. The sweep selects the winner that maximises top-3 hit rate subject to μ-p95 landing in the elite band `[1900, 2200]` (re-tuned post-Glicko-2 to fix the elite μ inflation that the conservative-halving #51 starting point left at ~2500-3300). The canonical record of how the current K values were chosen lives at `docs/K_REGRID_REPORT.md` — re-run the script after any change to the effective-K math (e.g. adjusting σ_inactivity, the margin cap, or the MOV gap-conditioning) and apply the recommended dict to `_DEFAULT_K_FACTORS` in `engine/elo.py`. After applying new K values a **production re-backfill** is required: `gh workflow run scrape-supabase.yml --repo milwil-2/climb-elo`.

## Combined (Boulder+Lead) Ratings

`scripts/compute_combined_ratings.py` populates `Discipline.BOULDER_LEAD` ratings using the **geometric mean** `sqrt(mu_boulder × mu_lead)` of athletes with ≥3 events in both disciplines. The geometric mean penalizes specialists and rewards all-rounders, matching the Olympic combined format. Sigma uses RMS: `sqrt((sigma_b² + sigma_l²) / 2)`.

### Learned composite weights (Issues #54, #76, #77, #78)

The script supports an optional **learned-weights** mode. When `data/learned_combined_weights.json` exists and is well-formed, the formula generalises to `μ_combined = μ_lead^w_lead × μ_boulder^w_boulder` (with `w_lead + w_boulder = 1`). When the file is missing or malformed it logs a warning and falls back to the geometric mean (w_lead = w_boulder = 0.5). The script logs which mode is active at startup.

When the JSON additionally contains `w_sigma_lead` / `w_sigma_boulder` keys (Issue #78), σ also switches to a weighted form `sqrt((w_σL × σ_l² + w_σB × σ_b²) / (w_σL + w_σB))`. At equal weights this collapses to the classical RMS formula `sqrt((σ_b² + σ_l²) / 2)` — so v1 JSON payloads (no σ fields) keep the historical RMS behaviour unchanged.

Fit the weights with:

```bash
uv run python scripts/fit_combined_weights.py --source-db data/climbing_elo.db                  # scipy + 5-fold CV + σ fit (defaults)
uv run python scripts/fit_combined_weights.py --source-db data/climbing_elo.db --method grid    # 0.1 grid sweep (legacy)
uv run python scripts/fit_combined_weights.py --source-db data/climbing_elo.db --cv holdout     # original single-pass scoring
uv run python scripts/fit_combined_weights.py --source-db data/climbing_elo.db --no-fit-sigma   # μ-only (writes v1 schema)
```

The fitter:

1. Discovers "virtual combined events" — every (WCh season, gender) tuple where ≥ 6 athletes have results in BOTH Boulder and Lead. The Olympics tier is included if present; the IFSC API doesn't expose Olympic events directly, so today the folds come from WCh seasons (2012–2025).
2. Builds the ground-truth combined ranking via rank product (Tokyo 2020 / Paris 2024 format).
3. For each fold year, copies the production DB to a temp file and runs backfill on Boulder + Lead with `end_date=date(year, 1, 1)` — ratings reflect only events strictly prior, no data leakage.
4. Optimises (w_lead, 1 - w_lead) via `scipy.optimize.minimize_scalar` (`method='bounded'`, default) on the 1-D parameter `w_lead ∈ [0, 1]` so the `w_L + w_B = 1` constraint is automatic. `--method grid` reproduces the 0.1-step legacy grid for transparency. Scoring inside the optimiser is Monte Carlo podium log-loss (primary), Spearman rank correlation (tie-breaker).
5. Runs **5-fold CV** over the discovered folds (default `--cv kfold --k 5`) and reports mean ± std of log-loss / rank-corr. The ship rule uses the **CV mean** — learned weights ship only if their CV-mean log-loss beats the baseline's CV-mean log-loss AND CV-mean rank-corr stays within 5% of the baseline's. `--cv holdout` reverts to single-pass scoring.
6. With `--fit-sigma` (default on), runs a second 1-D `minimize_scalar` for `(w_sigma_lead, 1 - w_sigma_lead)` minimising Brier score on the top-3 finish probability. Brier is σ-sensitive (unlike rank correlation), so a well-calibrated σ should reduce it. `--no-fit-sigma` keeps the v1 RMS formula.
7. Writes `data/learned_combined_weights.json` only if the ship rule (above) passes; otherwise prints "Learned weights did not improve over geometric mean — keeping baseline." and leaves the JSON alone.

**When to re-run**: after each new combined-format major (Olympics, World Championships) lands in the DB. New ground-truth folds + updated ratings can shift the optimum. The JSON is checked into the repo (the `.gitignore` has an `!data/learned_combined_weights.json` exception) so the production deployment picks up the new weights on next deploy without a manual config step.

**scipy dependency note**: `scipy>=1.13` is now a runtime dep (used by both `--method scipy` and `--fit-sigma`). On Vercel this adds ~30 MB to the Python layer but stays well under the 250 MB serverless function limit. If build size becomes a concern, the fitter is a script-only entry point and could be moved to an `[optional-dependencies]` group.

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
| GET | `/api/v1/leaderboard` | Paginated ELO rankings. Query: `discipline`, `gender`, `view` (`active`/`all`/`legacy`, default `active` since 2026-05-26 — #91), `limit` (1–100), `offset` (0–10000) |
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

Production data lives in **Supabase Postgres**. One GitHub Actions workflow keeps it fresh:

- **`.github/workflows/scrape-supabase.yml`** — runs daily at 04:00 UTC against the Supabase session pooler. Scrapes upcoming events + recent finished results, runs the ELO backfill (idempotent via `uq_rating_history_athlete_round`), and refreshes combined Boulder+Lead ratings. Workflow-dispatchable with an optional `historical_backfill` flag for the full 2012→present rescrape. Requires the `DATABASE_URL` repo secret (session pooler URL, port 5432).

**Backups**: Supabase provides its own rolling backups (7-day on the free tier, PITR on paid). The previous in-repo snapshot workflow (`.github/workflows/snapshot.yml`) and `scripts/snapshot_db.py` / `scripts/restore_snapshot.py` helpers were removed in Issue #82 — they snapshotted an empty CI-local SQLite file and the `db-snapshots` GitHub Release contents were never usable for recovery.

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

### YouTube broadcast embed

When `Event.livestream_url` is populated, `/live/{event_id}` renders a sandboxed YouTube iframe alongside the leaderboard + projections (left video / right stats at ≥1024px; stacked below). URL is validated by `parse_youtube_video_id()` in `src/climbing_elo/live/livestream.py` — `javascript:`, `data:`, and non-allowlisted hosts are rejected. Europe-paywalled events fall back to YouTube's own region-block message inside the iframe; the "Open on YouTube →" link remains usable for VPN users.

### Production caveat: SSE on Vercel

The live SSE endpoint (`/live/{event_id}/stream`) does NOT work on Vercel serverless — function timeouts cap responses at 60s while the SSE handler keeps connections open for up to 4 hours. The endpoint exists in the codebase but is effectively dead in production. To re-enable live streaming, the poller + SSE handler would need to run on a long-lived host (Fly.io, Railway, or a Vercel-adjacent VPS). Tracked informally; not yet ticketed.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request.

- **Jobs**: `pytest (3.11)`, `pytest (3.12)`, and `ruff` (lint + format check).
- **Branch protection**: add `pytest (3.11)` and `pytest (3.12)` as required status checks via GitHub repo Settings → Branches → Branch protection rules (one-time manual step).
- **Debugging failures**: check the Actions tab; the failing test name and full traceback appear in the "Run tests" step output. Lint failures show the offending line(s) from `ruff check` / `ruff format --check`.

## Testing

Tests use an in-memory SQLite database (`conftest.py:db_session`). Fixtures `sample_event` and `eight_athletes` provide pre-built test data. `test_elo.py` validates pairwise math (zero-sum invariant across all 3 disciplines). `test_backfill.py` runs a 3-event integration test and checks reproducibility. `test_api.py` covers all v1 REST endpoints. `test_projections.py` covers Monte Carlo invariants. `test_combined.py` covers the Boulder+Lead aggregate. `test_scraper_upcoming.py` covers upcoming-event filter logic. `test_health_check.py` covers CLI exit codes + Discord rate-limiting. `test_cache.py` covers TTLCache thread-safety + expiry. `test_likely_roster.py` covers the likely-competitor fallback logic. `test_live.py` covers the live poller + SSE (new result detection, dedup, finished-status auto-stop, EventBus pub/sub, file lock mutex, SSE 404/200/429). `test_baselines.py` + `test_external_rankings.py` cover the IFSC-official and AscentStats backtest baselines (recorded JSON fixtures in `tests/fixtures/external_rankings/`; live network tests are gated by `@pytest.mark.network`, deselected by default via `pyproject.toml`).

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

## Iteration backlog (rating model improvements)

Sequenced plan from the R&D synthesis in `docs/RATING_SYSTEM_RESEARCH.md`. Open issues are tracked on GitHub; this section captures the **why** and the cross-cutting decisions so future work doesn't relitigate.

### Phase 1 — in flight (parallel)
- **#51 — Glicko-2 RD integration** (highest payoff). Makes σ a real Glicko-2 RD that modulates updates. Retires the 2× provisional-K cliff. Full plan: `docs/PLAN_GLICKO2_RD_INTEGRATION.md`. **Baked-in decisions** (locked in to avoid relitigation):
  1. **Inactivity inflation**: calendar-time semantics (months since last event), per Glicko-2's Wiener-process model.
  2. **Margin of victory**: stays separate from outcome `s_j` ∈ {0, 0.5, 1}. Folds into K via `K_effective = K_base × g(φ_opp) × margin_multiplier`. Keeps #53 as an independent change.
  3. **Projection σ**: reuse Glicko-2 φ in `engine/projections.py`. One source of truth.
  - **Deferred to follow-ups**: K_FACTOR_TABLE regrid sweep at variable effective-K, full Glicko-2 volatility update (using simplified closed-form in v1).
- **#54 — Learned Boulder+Lead composite weights**. Replaces fixed geometric mean with `μ_L^w_L × μ_B^w_B`. Ships only if learned weights beat geometric mean on combined-format log-loss. LA28-relevant.

### Phase 2 — after Phase 1 lands (sequential)
- **#53 — Margin-of-victory audit + G-Elo benchmark**. Conditions MOV multiplier on rating gap (538-style) to prevent elite inflation. Touches the same MOV multipliers as #51, so sequential PR after #51.

### Phase 3 — after Phase 2 stable for ~1 week
- **#56 — Bracket-native Speed model**. Replaces P-L on time-normalized Speed with actual head-to-head bracket matchups + Davidson (1970) tie-handling. Isolated to the Speed branch of `engine/elo.py`.

### Phase 4 — last
- **#52 — Whole-History Rating (WHR / ILSR) batch refit**. New batch pipeline alongside online ELO. Biggest change; held last so it compares against the post-#51 baseline.

### Cross-cutting safety scaffolding
- Every PR in this sequence must keep `scripts/run_backtest.py` ≥15pp above baseline (existing CI assertion).
- Every PR adds a cold-start trajectory test using real production athlete IDs (Anraku=5, Bertone=447, Garnbret=60).
- Rollback story per PR: single `git revert <sha>` + re-run `scrape-supabase.yml` workflow (cleaned up by `uq_*` constraints from #69).
- No feature flags — 1 maintainer, cheap revert path, complexity not worth it.

### Operational TODOs (separate from R&D)
- **#75** — Full historical backfill from 2012→present + initial backtest validation against the populated Supabase DB. Best done locally with `DATABASE_URL` pointed at the session pooler (avoids the GH Actions 60-min timeout).
- **Live SSE on persistent host** — not ticketed; would re-enable `/live/{event_id}/stream` (broken on Vercel serverless).
- **Livestream URL automation** — `livestream_url` is manually populated; could be scraped if IFSC ever exposes it.
