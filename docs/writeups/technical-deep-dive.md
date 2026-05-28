# Climbing ELO — Technical Deep-Dive

A walkthrough of how the engine, the pipeline, the deployment, and the workflow fit together. Assumes you've cloned the repo; references file paths directly. For onboarding, see [README-candidate.md](./README-candidate.md). For the full engineering reference, see [CLAUDE.md](../../CLAUDE.md).

## The pipeline

Three stages, independent, idempotent:

```
scrape → backfill → serve
```

**Scrape** (`src/climbing_elo/scraper/ifsc_api.py`). The IFSC results API has no auth, only a Referer header. Structure: `/api/v1/` → seasons → `season_leagues/{id}` → events + `d_cat` IDs → `events/{id}/result/{d_cat_id}` → full rankings. We only scrape `league_id=1` (World Cup). Discipline categories are identified by matching the `d_cat` discipline field. The scraper writes Athlete / Event / Round / Result rows through SQLAlchemy; uniqueness constraints (`uq_athlete_name_gender`, `uq_event_name_season_discipline`, `uq_round_event_type_gender`, `uq_result_round_athlete`) make re-scrapes safe.

**Backfill** (`src/climbing_elo/engine/backfill.py`). Processes events chronologically, ordered by `Event.start_date`. For each event, iterates over its rounds in the order `QUALIFICATION → SEMI → FINAL`, and for each round calls `calculate_round_updates()` from `engine/elo.py`. After all rounds have been processed, layers on the Tournament Participation Bonus. Commits are per-event (atomic). The `n_events` counter increments once per event, not per round. Idempotency is enforced by `uq_rating_history_athlete_round_kind` — re-running backfill against an already-rated event is a no-op.

**Serve** (`src/climbing_elo/api/`). FastAPI app factory (`api/app.py`) wires the rate limiter (`api/limiter.py`), CORS, exception handlers, and route modules. HTML routes are in `api/routes.py` (leaderboard, athlete profile, event, breakdown, projections, head-to-head, live event). REST API is in `api/v1_routes.py` with Pydantic schemas in `api/schemas.py`. Templates (Jinja2) live in `templates/`. Rendering uses Chart.js for athlete rating-over-time graphs.

## The rating engine

The core math lives in `src/climbing_elo/engine/elo.py`. Six small ideas, layered:

### 1. Plackett-Luce pairwise decomposition

Any multi-athlete finishing order can be decomposed into all `n(n-1)/2` pairwise contests. Each pair updates symmetrically (zero-sum on μ). The critical normalisation is `pair_k = base_k / (n-1)` — without this, an N=30 World Cup final would swing ratings ~15× more than an N=2 event, just because there are more pairs.

### 2. Glicko-2 RD modulation (issue #51)

Each athlete carries μ (rating, display scale ~1500) and σ (rating deviation, display scale ~50–350). σ is Glicko-2's φ on the display scale (`σ = φ · 173.7178`).

For each ordered pair `(i ahead of j)`:

```
g(φ_j_internal) = 1 / sqrt(1 + 3φ²/π²)
E_ij            = 1 / (1 + exp(-g(φ_j) · (μ_i - μ_j)/173.7178))
K_eff           = K_base(tier, round) · g(φ_j) · margin_mult(rating_gap)
Δ_pair          = K_eff · (1 - E_ij)   # pair-symmetric: +Δ to i, -Δ to j
```

Then accumulate variance for both sides into a per-athlete `v_inv` running total:

```
v_inv_i += g(φ_j)² · E_ij · (1 - E_ij)
```

After all pairs of the round are processed, update each athlete's φ via the simplified closed-form:

```
1/φ_new² = 1/φ_inflated² + v_inv_sum
σ_after  = clamp(φ_new · 173.7178, [50, 350])
```

(The full Glicko-2 volatility-σ iteration — Glickman 2013 Step 5 — is deferred as issue #81; we use the standard Glicko-1.5-style approximation, which buys 98–99% of the calibration of the full iteration in long-running implementations.)

Three design decisions baked in to avoid relitigation:

1. **Inactivity inflation uses calendar time** (months since last event), per Glicko-2's Wiener-process model. `φ_new² = φ_old² + σ_inactivity² · months_inactive` with a 30-day grace period and `σ_inactivity = 5.0` (display scale). This reuses the existing `Rating.last_event_at` column.
2. **Margin-of-victory stays separate from outcome score** `s_j ∈ {0, 0.5, 1}`. The MOV multiplier folds into K instead — keeps issue #53 (MOV audit) as an independent change.
3. **Projection σ reuses Glicko-2 φ** in `engine/projections.py`. One source of truth. Trade-off: φ is rating uncertainty, not performance variance — but the practical effect (wider draws for less-certain athletes) is directionally correct.

### 3. 538-style MOV gap-conditioning (issue #53)

The base MOV multiplier `min(1 + gap/max_gap, MARGIN_CAP)` is unconditioned on rating gap — an elite crushing a junior would earn the same MOV bonus as an elite crushing a peer. 538's NFL/NBA work and Kovalchik (2020) on ATP tennis both showed this drives autocorrelation drift at the top of the distribution.

Fix: damp the MOV bonus when the *favourite* wins by Δμ:

```
multiplier = base · softening / (max(Δμ, 0)/rating_scale + softening)
```

with `rating_scale = 400` and `softening = 2.2`. This is asymmetric on purpose — an upset (`Δμ < 0`, underdog wins) keeps the full bonus, since a big-margin upset is genuinely high-information.

### 4. Tournament Participation Bonus (issue #90)

A tier-weighted, zero-sum μ credit applied per event on top of the pairwise updates. After all rounds of an event are processed:

```
gross_bonus[r]  = tpb_table[tier][r-1]   # 0.0 if r > len(table)
total_bonus     = Σ gross_bonus
debit           = total_bonus / N        # uniform across the field
Δ_tpb           = gross_bonus - debit    # sums to zero
```

Tier table (default):

| Tier | Top-K | Bonus to #1 | Curve |
|---|---|---|---|
| Olympics | 8 | +30μ | linear to 0 at rank 8 |
| World Championship | 8 | +20μ | linear |
| World Cup | 6 | +12μ | linear |
| Continental | 4 | +5μ | linear |

Why a separate layer instead of folding into K? Three reasons: keeps the pairwise zero-sum invariant clean (pair tests still pass); the breakdown page renders TPB as its own line so it's easier to explain and ablate; backtests can A/B turn TPB on/off without re-running the pair updates. TPB is persisted as a synthetic `RatingHistory` row with `kind='tpb'` whose `round_id` points at the event's final round. Idempotent via `uq_rating_history_athlete_round_kind`.

### 5. Score normalisation

| Discipline | Normalisation |
|---|---|
| Lead | `"34+"` → 34.5; `"TOP"` → 999.0; otherwise as-given. |
| Boulder (pre-2025) | `tops·1000 + zones·100 - top_att·10 - zone_att` (max_gap = 1000). |
| Boulder (2025+) | Decimal pass-through (e.g. `"34.5"` → 34.5). |
| Speed | Seconds (lower is better; max_gap = 2.0). |

### 6. Combined Boulder+Lead

The Olympic combined format uses rank product. We approximate with geometric mean `μ_combined = √(μ_B · μ_L)` for athletes with ≥3 events in both disciplines. Sigma uses RMS. Athletes who specialise in one discipline get penalised by the geometric mean's sensitivity to the smaller factor — which matches the Olympic format's penalty on specialists.

Optionally, the script supports a **learned-weights mode** that fits `μ_combined = μ_lead^w_lead · μ_boulder^w_boulder` (constrained `w_L + w_B = 1`) via `scipy.optimize.minimize_scalar` with 5-fold cross-validation over WCh seasons. The ship rule is: weights ship only if their CV-mean podium log-loss beats the baseline AND CV-mean rank correlation stays within 5% of the baseline. Otherwise the script logs a warning and the file is left alone — the geometric-mean baseline ships by default. A second 1-D optimisation fits `w_sigma_lead`/`w_sigma_boulder` against Brier score on top-3 finish probability.

## Deployment

Production is Vercel `@vercel/python` Fluid Compute against Supabase Postgres.

### Vercel entry point

`api/index.py` is a thin shim that prepends `src/` to `sys.path` and calls `climbing_elo.api.app.create_app()`. The shim exists because the project layout is a `src/` package, not a flat module tree, and Vercel's runtime doesn't auto-`pip install -e .` the project. `vercel.json` declares `api/index.py` as a `@vercel/python` build and routes all paths to it.

There's a subtle gotcha: Vercel's runtime statically introspects the entry point looking for a top-level `app = ...` assignment. Wrapping `create_app()` in a try/except hides that assignment from the analyzer and the deployment fails with "could not find a top-level app." The fix (if you need defensive startup-error handling) is two assignments — one outside the try satisfies the analyzer, one inside is what actually runs.

### Supabase: three connection strings, one database

Supabase exposes three different URLs for the same Postgres project. Picking the wrong one will silently break:

| Context | URL pattern | Port | Address family |
|---|---|---|---|
| Local dev / one-off scripts | `db.PROJECT.supabase.co` (direct) | 5432 | IPv6 only |
| Local dev / scripts (IPv4) | `aws-0-REGION.pooler.supabase.com` (session) | 5432 | IPv4 |
| Vercel runtime | `aws-0-REGION.pooler.supabase.com` (transaction) | 6543 | IPv4 |
| GitHub Actions | `aws-0-REGION.pooler.supabase.com` (session) | 5432 | IPv4 |

Pain points already paid for in production outages:

- The direct URL is **IPv6-only**. GitHub Actions runners and Vercel functions are IPv4-only. A CI workflow using the direct URL fails with a DNS / connect timeout that looks like a Supabase outage.
- The transaction pooler (6543) rejects any statement that depends on session state — advisory locks, `SET LOCAL`, prepared statements outside a single statement. Symptoms: random `prepared statement "..." does not exist` errors, or features that work locally but 500 in prod.
- The session pooler (5432) is fine for Actions but is **not** what you want for Vercel — long-lived sessions don't fit the serverless lifecycle and you'll exhaust the pool.

This took four production-failure cycles to fully diagnose. It's now documented in CLAUDE.md so the next visitor doesn't relearn it.

### Rate limiting

Application-level per-IP via slowapi, in-memory backend. Vercel's Python runtime doesn't put a reverse proxy with rate-limiting in front of the function, so slowapi is the only per-IP throttle. The in-memory backend is per-instance — under high load Vercel may run multiple cold-start instances concurrently, so the per-instance limits are looser than documented. Acceptable for current traffic; the upgrade path is Upstash Redis via the Vercel Marketplace for a shared-state backend.

### Live events on Vercel: the limitation

The SSE endpoint (`/live/{event_id}/stream`) does **not** work on Vercel serverless — function timeouts cap responses at 60s while SSE keeps connections open for up to 4 hours. The endpoint exists in the codebase but is effectively dead in production. Re-enabling would mean moving the poller + SSE handler to a long-lived host (Fly.io, Railway, a Vercel-adjacent VPS). Tracked informally; not yet ticketed.

## Workflow patterns

A few patterns proved load-bearing across the project:

**Plan before you build.** For any change touching foundational math (Glicko-2 integration, field-strength + activity classifier), the first agent dispatched is a *planning* agent whose only deliverable is a written document with file:line references, exact formulas, and flagged open decisions. The *implementation* agent then runs with those decisions baked into its prompt as locked-in. Plans are cheaper than implementation re-runs. See `docs/PLAN_GLICKO2_RD_INTEGRATION.md` (488 lines) and `docs/PLAN_FIELD_STRENGTH_AND_ACTIVITY.md` (606 lines) for examples.

**Worktree-isolated parallel agents, gated by file-set disjointness.** Four agents in parallel works when their file sets don't collide. The constraint isn't intelligence; it's merge-graph thinking. Draw the dependency graph before dispatching: `R1 → R3 → EloConfig refactor → K-regrid`, with `R4` / `#82` / `#86` branching off in parallel. Wrong call = manual rebase. Right call = 4× wall-clock throughput.

**Backtest-gated changes.** Every PR touching rating math must keep `scripts/run_backtest.py` ≥ 15pp above the IFSC baseline. Currently a discipline applied per PR; an open follow-up is to promote it to a hard CI gate.

**File the follow-ups; don't bundle scope.** Each major rating-math change explicitly defers smaller improvements as their own issues. Issue #51 (Glicko-2) deferred K-regrid (#80) and full volatility iteration (#81). Issue #54 (combined weights) deferred scipy.optimize (#76), k-fold CV (#77), σ weights (#78). Each deferral became its own issue; the PRs stayed scoped.

**Doc-the-doc.** CLAUDE.md gets updated as part of the same PR as the code change that prompts it, not as a follow-up. The doc is a feedback loop, not a delivery artifact.

## Tests

`tests/conftest.py:db_session` provides an in-memory SQLite database — every test starts from a fresh schema generated from `models.py`. No fixture drift possible. Live-network tests (Wikipedia for IFSC ranking surfaces, AscentStats) are gated by `@pytest.mark.network` and deselected by default via `pyproject.toml`'s `addopts = "-m 'not network'"`. Run them with `uv run pytest -m network` to validate a scraper change.

Notable test files:

- `test_elo.py` — pairwise math, zero-sum invariants, Glicko-2 g(φ), inactivity inflation, TPB tier monotonicity.
- `test_backfill.py` — 3-event integration test; reproducibility; TPB row writing; backfill idempotency.
- `test_cold_start_glicko2.py` — real-data trajectory test using a production snapshot. Asserts Sorato Anraku (id=5) appears near the top of men's Boulder, matching the AscentStats fixture. Skips when the snapshot is missing or stale.
- `test_combined.py` — Boulder+Lead aggregate maths.
- `test_projections.py` — Monte Carlo invariants (probabilities sum to 1, expected rank decreases with rating).
- `test_likely_roster.py` — the upcoming-event roster fallback when IFSC hasn't published a registered list yet.
- `test_live.py` — poller + SSE (new result detection, dedup, finished-status auto-stop, file lock mutex).

618 tests passing as of the most recent commit.

## What's next

The open R&D backlog, in execution order:

1. **#80 K-factor regrid recommendations** — `docs/K_REGRID_REPORT.md` exists; the recommended values haven't been applied to `_DEFAULT_K_FACTORS` and re-backfilled into production yet.
2. **#84 G-Elo benchmark** — compare our rating against a published G-Elo implementation on the same fixture set.
3. **#85 MOV grid search** — the `mov_softening` and `mov_rating_scale` constants are first-pass values; a small grid sweep is filed.
4. **#56 Bracket-native Speed model** — replace Plackett-Luce on time-normalised Speed with actual head-to-head bracket matchups + Davidson (1970) tie-handling.
5. **#52 Whole-History Rating** — batch refit alongside the online ELO. Biggest scope; held last so it compares against the post-#51 baseline.

Operational TODOs: full historical backfill (#75), live SSE on a persistent host, livestream URL automation.

Pointers: [README-candidate.md](./README-candidate.md) for onboarding · [blog-post.md](./blog-post.md) for the narrative · [hiring-summary.md](./hiring-summary.md) for the one-pager · [CLAUDE.md](../../CLAUDE.md) for the canonical engineering reference · [docs/PROCESS_RETROSPECTIVE.md](../PROCESS_RETROSPECTIVE.md) for a build-day retrospective.
