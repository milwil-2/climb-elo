# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --all-extras                    # install all deps
uv run pytest                           # run all 50 tests
uv run pytest tests/test_elo.py -k "test_zero_sum"  # run single test
uv run uvicorn climbing_elo.api.app:app --reload     # dev server on :8000

# Data pipeline (run in order)
uv run python scripts/scrape_ifsc.py --min-year 2012 --max-year 2026
uv run python scripts/run_backfill.py
uv run python scripts/run_backtest.py   # validates model beats baseline by ≥15pp
```

## Architecture

The system is a three-stage pipeline: **scrape → backfill → serve**.

**Scrape** (`scraper/ifsc_api.py`) fetches Lead results from `ifsc.results.info` (no auth — just a Referer header) and writes Athlete/Event/Round/Result rows to SQLite. The API structure is: `/api/v1/` → seasons → `season_leagues/{id}` → events + d_cat IDs → `events/{id}/result/{d_cat_id}` → full rankings. Only `league_id=1` (World Cup) is scraped. Lead discipline categories are identified by `"lead"` in the d_cat discipline field.

**Backfill** (`engine/backfill.py`) processes all events chronologically, computing ELO updates per round (qualification → semi → final). Each round calls `calculate_round_updates()` from `engine/elo.py`, which decomposes the multi-athlete finishing order into all pairwise contests using Plackett-Luce. The critical normalization is `pair_k = base_k / (n - 1)` — without this, deltas scale with field size. Rating changes across a round sum to zero. Commits are per-event (atomic). The `n_events` counter increments once per event, not per round.

**Serve** (`api/routes.py`) is a FastAPI + Jinja2 dashboard. Routes: `/` (leaderboard), `/athletes/{id}` (profile with Chart.js rating-over-time), `/events` and `/events/{id}` (results with pre/post μ), `/breakdown/{athlete_id}/{event_id}` (pairwise contributing-pairs table), `/projections/{event_id}` (Monte Carlo outcome projections for an event), `/projections/new` (manual projection form).

## Data Model

Six SQLAlchemy models in `models.py`: Athlete → Event → Round → Result (competition data), Rating + RatingHistory (computed ratings). RatingHistory stores `contributing_pairs` as a JSON column for the breakdown view. Key enums: `EventTier` (olympics/world_championship/world_cup/continental), `RoundType` (qualification/semi/final), `Discipline` (L/B/S/BL — MVP is Lead only).

## ELO Engine Specifics

K-factors are tiered by EventTier × RoundType (e.g., Olympics Final = 48, World Cup Qual = 8). Provisional athletes (< 3 events) get 2× K-multiplier. Lead score normalization: `"34+"` → 34.5, `"TOP"` → 999.0. Margin multiplier is capped at 2.0×. σ decays toward 350 with an 18-month half-life during inactivity, and converges downward by 0.98× per event.

## Projections Engine

`engine/projections.py` provides Monte Carlo outcome prediction:

- `compute_podium_probabilities(athletes, n_simulations=10_000)` — draws N(μ, σ) performance scores per simulation, ranks athletes, and tallies win/podium/top-8 fractions. Returns `{athlete_id: {win, podium, top_8, expected_rank}}`. 10k sims for 20 athletes runs in ~15ms (numpy vectorized).
- `predict_winner(athletes)` — deterministic: returns athlete_id with highest μ.
- `expected_finish_ranks(athletes)` — returns athlete_ids sorted by descending μ.

Athletes marked DNS should be filtered by the caller before passing to the engine. Athletes with no rating for a discipline receive defaults (μ=1500, σ=350).

## Testing

Tests use an in-memory SQLite database (`conftest.py:db_session`). Fixtures `sample_event` and `eight_athletes` provide pre-built test data. `test_elo.py` validates pairwise math including zero-sum invariant. `test_backfill.py` runs a 3-event integration test and checks reproducibility (same input → identical output). `test_projections.py` covers Monte Carlo math invariants (win probs sum to 1, podium probs sum to 3, edge cases).
