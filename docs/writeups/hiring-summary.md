# Climbing ELO — Hiring Summary

**Live:** https://climb-elo.vercel.app   **Repo:** https://github.com/milwil-2/climb-elo

Solo-built, production-deployed rating system for World Cup–level competition climbing. Glicko-2-flavoured ELO over the official IFSC results API; FastAPI + Postgres on Vercel; ~2,700 athletes across Lead, Boulder, Speed, and the Olympic Boulder+Lead combined format.

---

## At a glance

| | |
|---|---|
| **Role** | Sole engineer — architecture, math, ops, frontend, CI, monitoring |
| **Stack** | Python 3.11+, FastAPI, SQLAlchemy 2.0, Jinja2, NumPy, SciPy, slowapi |
| **Data** | Supabase Postgres (prod), in-memory SQLite (tests), `uv` for deps |
| **Hosting** | Vercel `@vercel/python` runtime, auto-deploy on push to `main` |
| **CI / quality** | GitHub Actions: pytest on 3.11 + 3.12, ruff lint + format; **618 tests passing** |
| **Uptime monitoring** | GitHub Actions cron every 30 min — upstream IFSC API + own deployment, with Discord alerting |
| **Data freshness** | Nightly GitHub Actions scrape (04:00 UTC) — Lead + Boulder + Speed + combined refresh, idempotent |
| **Modelling depth** | Glicko-2 RD–weighted ELO + 538-style MOV gap-conditioning + tier-weighted zero-sum tournament participation bonus + learned composite weights |
| **Measured accuracy** | +37.5pp (Lead) / +77.8pp (Boulder) over official IFSC ranking baseline in backtest |

---

## What was built

**Three-stage pipeline.**
- **Scrape** — `scraper/ifsc_api.py` pulls Lead / Boulder / Speed results from the legacy IFSC results API (no auth, just a Referer header) and writes Athlete/Event/Round/Result rows through SQLAlchemy.
- **Backfill** — `engine/backfill.py` processes events chronologically; computes ELO via Plackett-Luce pairwise decomposition normalised by field size (`pair_k = base_k / (n-1)`); commits per-event atomically; idempotent across re-runs.
- **Serve** — FastAPI + Jinja2 dashboard at the production URL above. Routes: leaderboard, athlete profile (Chart.js rating graph, photo, body metrics), event results with pre/post μ, pairwise breakdown, Monte Carlo projections, head-to-head probabilities, upcoming-event predictions with likely-roster fallback.

**Public REST API.** 10 read-only endpoints under `/api/v1/*`, OpenAPI schema published at `/openapi.json`, per-IP rate limiting via slowapi.

**Operational systems.**
- 30-minute health-check cron (own deployment + upstream IFSC) with 3-failure threshold + Discord webhook alerting.
- Daily scrape workflow against Supabase session pooler — Lead, Boulder, Speed, combined ratings, athlete profile enrichment.
- Database migrations applied via Supabase MCP in lockstep with SQLAlchemy model changes.
- Secret-scanning lockdown: GitHub Actions allowlist restricts third-party actions; push protection blocked an accidental leak of a Vercel personal access token mid-development.

**Frontend.** Custom monochrome design (no UI framework), Jinja2 templates, Chart.js for rating-over-time, sandboxed YouTube embeds for live event broadcasts. Replaced an earlier v1 frontend in-place — single template tree, no dead code.

---

## Depth, briefly

### Modelling

The engine is Plackett-Luce pairwise ELO with Glicko-2 rating-deviation (φ) modulation. Beating a high-φ opponent (cold start, post-sabbatical) moves you less; high-φ athletes move further per round via the closed-form Glicko-2 step. Inactivity inflates φ on calendar time via Glicko-2's Wiener-process formula. Margin-of-victory is applied as a multiplier on the effective K, gap-conditioned per 538's NFL/NBA work — `mult = base × softening / (max(Δμ, 0)/scale + softening)`, asymmetric so upsets keep the full MOV bonus while elite-vs-junior blowouts are damped. Per-cell K-factor table tuned via coordinate-descent regrid (`scripts/regrid_k_factors.py`) holding μ-p95 in the elite band [1900, 2200]. A tier-weighted zero-sum tournament participation bonus layers on top (Olympics +30μ peak, Continental +5μ peak), debited uniformly across the field so per-event μ stays zero-sum.

Combined Boulder+Lead ratings start with the geometric mean `√(μ_B · μ_L)` (matches Olympic rank-product math), with an optional learned-weights mode (`μ_combined = μ_lead^w_lead × μ_boulder^w_boulder`) fit via `scipy.optimize.minimize_scalar` with 5-fold cross-validation over World Championship seasons. Weights ship only if they beat the geometric baseline on CV-mean podium log-loss — a deliberate ship-rule that prevents overfitting and gives a clear rollback path.

### Operations

Production runs on Vercel's `@vercel/python` Fluid Compute runtime against Supabase Postgres. The deployment uses Supabase's **transaction pooler** (port 6543, IPv4) because Vercel functions are short-lived and the direct connection URL is IPv6-only — Vercel doesn't have IPv6 connectivity to external hosts. The GitHub Actions scrape workflow uses Supabase's **session pooler** (port 5432, IPv4) because bulk inserts and longer transactions get rejected by the transaction pooler. Three connection strings for the same database, one per compute environment. This was paid for in production outages and is now documented as part of the project's engineering reference.

Application-level per-IP rate limiting via slowapi (in-memory backend, per Vercel instance) — `POST /api/v1/projections` at 10 req/min, upcoming-predictions at 60 req/min, everything else at 120 req/min default. Acceptable for current traffic; if it stops being acceptable, the next step is Upstash Redis via the Vercel Marketplace for a shared-state backend.

Tests use in-memory SQLite (`conftest.py:db_session`) so every test run starts from fresh schema generated from `models.py` — no fixture drift possible. Live-network tests (Wikipedia, AscentStats) are gated by `@pytest.mark.network` and deselected by default.

---

## What this demonstrates

- **End-to-end ownership** — data ingestion through production deployment, including the parts most projects skip (alerting, rate limiting, secret hygiene, schema migration discipline).
- **Engineering judgment on tradeoffs** — choosing Glicko-2 over simpler ELO (justified by the cold-start trajectory test on real production athletes), choosing geometric mean as a learned-weights baseline (with a CV-mean ship rule to prevent overfitting), choosing the in-memory rate-limit backend (acceptable for current traffic, with a documented upgrade path).
- **Operational instinct** — when local dev started drifting from production, the fix was to delete the local fallback entirely and require the real DB everywhere, not to add a sync script. 743 lines removed; project simpler.
- **Comfort with adversarial conditions** — secret nearly committed; GitHub push protection blocked it; `.gitignore` tightened the same day; reproduction of the incident lives in the engineering reference so it doesn't recur.
- **Documentation discipline** — engineering reference (`CLAUDE.md`) and process retrospective (`docs/PROCESS_RETROSPECTIVE.md`) both maintained; the reference is updated as part of the same PR as the code change that prompted it, not as a follow-up.
- **Math literacy without overreach** — the engine implements published Glicko-2 (Glickman 2013) and 538's MOV gap-conditioning rather than inventing new formulas. Where formulas were left simplified (Glicko-2 volatility update closed-form rather than full iteration), the simplification is explicit and the upgrade path is filed as an open issue.

---

## Selected pull requests / commits

- **Glicko-2 RD integration** (#51, commit `47dae01`) — added Glicko-2's φ to the rating update; cold-start trajectory test for Sorato Anraku (#1/611 men's Boulder) matched the external AscentStats ranking.
- **MOV gap-conditioning** (#53) — 538-style asymmetric MOV multiplier; preserves upset information, damps elite blowouts.
- **EloConfig dataclass** (#83 Target 3) — consolidated all tunable ELO knobs into a single frozen dataclass; eliminated module-global monkey-patching for sweep experiments.
- **K-factor regrid** (#80) — coordinate-descent sweep to tune the K-factor table at the new Glicko-2 effective-K regime; canonical run recorded in `docs/K_REGRID_REPORT.md`.
- **Learned composite weights** (#54, #76, #77, #78) — Boulder+Lead learned-weight mode with scipy.optimize, 5-fold CV, learned σ weights. Ships only if it beats the geometric mean baseline; otherwise falls back gracefully.
- **Dual-view leaderboard + retirement classifier** (#91) — pure-function `is_likely_retired_simple(last_event_at, retired_at, today, threshold_years)` + three view modes; query-layer fix, no re-backfill required.
- **Tournament Participation Bonus** (#90) — tier-weighted, zero-sum μ credit; layered separately from pair updates so the existing zero-sum invariant on pairs is unchanged.

---

## Pointers

- **Engineering reference (full)** — [CLAUDE.md](../../CLAUDE.md)
- **Build retrospective** — [docs/PROCESS_RETROSPECTIVE.md](../PROCESS_RETROSPECTIVE.md) — what one day of focused iteration looked like
- **Technical deep-dive** — [technical-deep-dive.md](./technical-deep-dive.md) in this directory
- **Public blog post** — [blog-post.md](./blog-post.md)
- **Live demo** — https://climb-elo.vercel.app
