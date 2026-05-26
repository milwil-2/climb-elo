# Glicko-2 RD Integration Plan (Issue #51)

**Status:** Proposal for review. No code written. References Glickman 2013 (*Example of the Glicko-2 system*, glicko.net technical paper) for formulas and `docs/RATING_SYSTEM_RESEARCH.md` §5 R1 for context.

**Companion files:** `src/climbing_elo/engine/elo.py`, `src/climbing_elo/engine/backfill.py`, `src/climbing_elo/models.py`, `src/climbing_elo/engine/evaluation.py`, `scripts/run_backtest.py`.

---

## TL;DR

Replace the cosmetic σ field with a **Glicko-2 rating deviation (φ)** that actually feeds into both the expected-score gradient and an effective per-pair K. The win: cold-start, sabbatical returns, and confidence-weighted updates all fall out of one principled mechanism, retiring the ad-hoc 2× provisional-K cliff and the 0.98×-per-event σ convergence. Key tradeoff: K-factor table will almost certainly need re-tuning at a lower base (current best `K_SCALE=2.0` was fit with σ inert; with variable effective-K the system is "hotter" per update on high-RD athletes). Estimated effort: ~7 working days including a K re-tune sweep and the R0-harness A/B.

---

## Current state

### σ today is bookkeeping, not signal

`src/climbing_elo/engine/elo.py:9-17`:
```
DEFAULT_SIGMA = 350.0
SIGMA_DECAY_HALF_LIFE_DAYS = 18 * 30  # ~18 months
SIGMA_FLOOR = 50.0
SIGMA_CEILING = 350.0
SIGMA_CONVERGENCE_FACTOR = 0.98
```

- `apply_time_decay` (`elo.py:188-197`) inflates σ during inactivity but never feeds it into the update math.
- `calculate_round_updates` (`elo.py:200-313`) ignores σ entirely; the update uses constant `pair_k = base_k / (n - 1)` (`elo.py:212-214`).
- After every round each athlete's σ is multiplied by `0.98` and floored at 50 (`elo.py:300`).

Evidence that σ carries no signal in the current production data (Supabase, May 2026):
- Lead: 1,716 athletes, σ ∈ [305.3, 350.0], mean 341.8.
- Boulder: 1,059 athletes, σ ∈ [319.4, 343.0], mean 341.3.

Even Garnbret (72 events) and Ondra (44 events) have σ values clamped against the floor/decay equilibrium — they cluster in a 50-unit band that the rating update never sees. This is the smoking gun for R1.

### Cold-start is a binary cliff

`elo.py:11`: `PROVISIONAL_THRESHOLD = 3`. `elo.py:244-245`:
```python
if rating_i.provisional or rating_j.provisional:
    k *= PROVISIONAL_K_MULTIPLIER  # 2.0×
```

A 2nd-event athlete gets a full 2× K. A 4th-event athlete gets 1×. There is nothing in between, and nothing afterward distinguishes a 4-event athlete from a 40-event athlete. Anraku (19 events from 2023-06 onward, current ID 5) and Oriane Bertone (9 events, ID 447) graduated out of the provisional regime before their ratings had stabilized — exactly the cold-start shape R1 is supposed to fix.

### Sabbaticals get σ-decay but it doesn't act

`apply_time_decay` inflates σ during inactivity, but since σ does not enter the update, the only effect of a 12-month sabbatical today is cosmetic. Garnbret's post-Olympic 2025 break and Ondra's 2024-2025 sparse calendar (44 events over 13 years) are real cases where Glicko-2 RD inflation would meaningfully reopen the rating to fresh evidence.

### Monte Carlo projector already consumes σ — but with a different meaning

`src/climbing_elo/engine/projections.py:74` draws `N(μ, σ)` per athlete per simulation. The σ here is interpreted as **per-event performance noise** (boulder-set luck, route-reading variance). Glicko-2 φ is fundamentally a **rating uncertainty** — how much we don't know about the underlying skill, not how much skill varies game-to-game.

These are different quantities. Conflating them today produces narrow, overconfident projections for veterans (σ ≈ 305) and reasonable spread for newcomers (σ = 350). After R1 the σ stored in the `Rating` table will be Glicko-2 φ — which decays faster for active athletes and inflates more aggressively during sabbaticals. **The projections engine must either (a) keep using φ as a proxy for performance noise — accepting the semantic drift — or (b) separately track a performance-σ.** See "Open questions" below.

### Models

`src/climbing_elo/models.py:151-173`:
```python
class Rating(Base):
    mu: float = 1500.0
    sigma: float = 350.0
    n_events: int = 0
    last_event_at: date | None
    provisional: bool = True
```

`RatingHistory` (`models.py:176-196`) already stores `sigma_before` and `sigma_after`. **No schema change is strictly required** if we reinterpret `sigma` as the human-scale RD (`RD = 173.7178 × φ`). Optional additive: a new `volatility` float column (Glicko-2 σ in internal units, default 0.06) — see Migration section.

### Backfill

`src/climbing_elo/engine/backfill.py:176-189` writes `ar.sigma = upd.sigma_after` to both the in-memory cache and the `Rating` row. It commits per-event (`backfill.py:265`). The `n_events` and `provisional` flag are updated at end-of-event (`backfill.py:248-263`). All integration points are already in place — the σ update value just needs to come from the Glicko-2 path.

### Backtest harness

`scripts/run_backtest.py` is a thin shim around `engine/evaluation.BacktestRunner`. The harness:
- Runs against a **copy** of the DB (`evaluation.py:10-17` — state-safe).
- Scores log-loss / Brier / calibration / Spearman / top-K hit rates per round.
- Stratifies by tenure, tier, round, discipline, season, field size.
- Already supports `--variant` plug-in via `register_variant()` (`evaluation.py:209-216`).
- Already has a `stripped_elo` baseline (`engine/baselines.py`) that turns off margin / σ-decay / provisional-K — the perfect ablation neighbour for the `glicko2` variant.

**Threshold for "no regression"**: the CLAUDE.md "+15pp" line refers to the legacy podium-hit-rate harness. The new harness has no hard pass/fail threshold — it reports the full metric × stratification cube and lets a human compare. For R1 the operational definitions are:
- **Aggregate**: log-loss-podium and brier-podium on the holdout-2s split must not regress >2% relative.
- **Cold-start stratum** (tenure 1-3, 4-10): log-loss-podium must improve (the whole point of R1).
- **Hit rates**: top-1, top-3, top-8 on holdout must stay within ±2pp of baseline.

---

## Proposed change

### Glicko-2 formulas (Glickman 2013, §1-2)

**Scale conversion** (Glicko-2 internal ↔ human-readable Elo):
```
μ  = (r - 1500) / 173.7178
φ  = RD / 173.7178
σ  : volatility, stored on the internal scale directly
```

**Weighting function**:
```
g(φ) = 1 / sqrt(1 + 3·φ²/π²)
```

**Expected score** (replaces our `expected_score(mu_a, mu_b)` on `elo.py:92-93`):
```
E(μ, μ_j, φ_j) = 1 / (1 + exp(-g(φ_j)·(μ - μ_j)))
```

**Variance estimate** over all opponents in the rating period:
```
v = [ Σ_j  g(φ_j)² · E(μ, μ_j, φ_j) · (1 - E(μ, μ_j, φ_j)) ]^(-1)
```

**Estimated improvement**:
```
Δ = v · Σ_j  g(φ_j) · (s_j - E(μ, μ_j, φ_j))
```
where `s_j ∈ {0, 0.5, 1}` is the outcome of the j-th game.

**Volatility update** (Step 5; iterative — see "Volatility iteration" below). Find σ' such that:
```
f(x) = (e^x · (Δ² - φ² - v - e^x)) / (2·(φ² + v + e^x)²)  -  (x - ln(σ²)) / τ²
```
has `f(x*) = 0`; then σ' = exp(x*/2). System constant τ ∈ [0.3, 1.2] (Glickman recommends 0.5 for "moderately" volatile sport ratings; chess uses ~0.3).

**Pre-period RD inflation**:
```
φ* = sqrt(φ² + σ'²)
```

**New RD**:
```
φ' = 1 / sqrt(1/(φ*)² + 1/v)
```

**New rating**:
```
μ' = μ + (φ')² · Σ_j  g(φ_j) · (s_j - E(μ, μ_j, φ_j))
```

**Initial values for new players** (Glickman 2013):
```
r = 1500   →  μ = 0
RD = 350   →  φ ≈ 2.014767
σ = 0.06   (volatility)
```

**Zero-game rating period**: skip Δ, v, and σ update; only inflate `φ → sqrt(φ² + σ²)`.

### Volatility iteration (Illinois bracketing algorithm)

Glickman's recommended root-finder for Step 5 is the **Illinois algorithm** (a modified regula falsi). Initial bracket:
- `A = ln(σ²)`
- `B`: if `Δ² > φ² + v`, set `B = ln(Δ² - φ² - v)`; otherwise expand by stepping `A - k·τ` for `k = 1, 2, …` until `f(A - k·τ) < 0`.
- Iterate until `|B - A| < ε`, with `ε = 1e-6` (Glickman default).
- Return `σ' = exp(A/2)`.

This converges in 5-15 iterations per player per period. At ~80 athletes × ~50 events/year × 3 rounds, total cost is ~12M float ops/year — negligible vs the current pairwise loop.

### Adaptation to our setting: "rating period" = one round

Glickman's formulation assumes a *batch* of games per player per rating period (he recommends 8-15 games per period for stability). Our reality is each round of n athletes generates n-1 pairwise opponents per athlete (via the existing P-L decomposition in `calculate_round_updates`). Each round naturally maps to a Glicko-2 rating period:

- A WC final (8 athletes) → 7 games per player per period — within Glickman's "modest" recommendation, but on the low side.
- A WC semi (~24 athletes) → 23 games per player per period — comfortably above the recommended threshold.
- A qual (60-80 athletes) → 59-79 games per player per period — very high; v will be small (high confidence in update).

The s_j outcomes are read directly from the existing pairwise loop in `calculate_round_updates` (`elo.py:225-289`): `res_i.rank < res_j.rank → s_i_j = 1.0`, ties → 0.5, lost → 0.0.

**Margin still multiplies the score input**. Today the multiplier is applied symmetrically to winner and loser deltas (`elo.py:264-265`). Under Glicko-2 we instead modulate the *score* `s_j` toward the extremes:
```
s_j_adjusted = 0.5 + margin_mult/2 · (s_j - 0.5)  # winner moves toward 1, loser toward 0
```
This keeps the symmetric-application invariant intact, but the symmetric-application bug R3 (#53) flags is still a separate decision — see "Pairing with #53".

---

## Integration points (concrete edits)

All file:line references against the current main as of 2026-05-26.

1. **`src/climbing_elo/engine/elo.py:9-17`** — constants
   - Add `GLICKO2_SCALE = 173.7178`, `GLICKO2_TAU = 0.5`, `GLICKO2_EPSILON = 1e-6`, `GLICKO2_DEFAULT_VOLATILITY = 0.06`.
   - Replace `DEFAULT_SIGMA = 350.0` with semantics "human-scale RD"; internally it converts to φ = 350/173.7178 ≈ 2.015.
   - Retire `SIGMA_CONVERGENCE_FACTOR` (the 0.98 per-event multiplier) — Glicko-2's `φ' = 1/sqrt(1/φ*² + 1/v)` replaces it.
   - Retire `SIGMA_DECAY_HALF_LIFE_DAYS` (the inactivity inflation) — Glicko-2's pre-period `φ* = sqrt(φ² + σ²·Δt_periods)` replaces it. **Need to decide how to count periods between events for an inactive athlete** — see Open Q1.
   - Retire `PROVISIONAL_K_MULTIPLIER = 2.0` and `PROVISIONAL_THRESHOLD = 3` from the update path. Keep the `provisional` flag on the Rating row for downstream UI badges (e.g. "<3 events" tag in the leaderboard template) but stop multiplying K by it.

2. **`src/climbing_elo/engine/elo.py:92-93`** — `expected_score`
   - Replace the 400-scale logistic with the Glicko-2 form on the internal scale:
     ```python
     def glicko2_expected_score(mu_a: float, mu_b: float, phi_b: float) -> float:
         return 1.0 / (1.0 + math.exp(-glicko2_g(phi_b) * (mu_a - mu_b)))
     ```
   - Keep `expected_score(mu_a, mu_b)` as a thin wrapper for legacy callers / tests (computes the 400-scale form for display purposes). The display value on `/breakdown/{a}/{e}` page should stay 400-scale (no UX disruption).

3. **`src/climbing_elo/engine/elo.py:188-197`** — `apply_time_decay`
   - Replace with `glicko2_inflate_phi(phi, sigma, periods_inactive) -> phi_star`. The "periods" count is the integer or fractional number of empty rating periods between this athlete's last event and the current one. Tentative recommendation: count one period per 30 days of inactivity (so a 6-month sabbatical = 6 empty periods of σ-inflation).

4. **`src/climbing_elo/engine/elo.py:200-313`** — `calculate_round_updates`
   - Inside the per-athlete outer loop (`elo.py:219-289`), accumulate per-athlete `v_inv_terms` and `delta_terms` lists across all opponents in the round.
   - After the inner pair loop, compute v, Δ, σ' (via Illinois iteration), φ*, φ', μ' per athlete.
   - Build `RatingUpdate` with `mu_after = μ'·SCALE + 1500`, `sigma_after = φ'·SCALE` (display-scale).
   - Margin multiplier folds into `s_j` (see "Adaptation to our setting" above) — drop the symmetric `k * margin_mult * (...)` application on `elo.py:264-265`.
   - **Zero-sum invariant**: under Glicko-2 zero-sum no longer holds exactly (variance and inflation can break it). Adjust `tests/test_elo.py:80-93` to allow tolerance — or compute updates pairwise-symmetrically (each pair contributes ±g(φ)·(s − E) for both sides) and apply per-side scaling at the end. The literature accepts non-zero-sum; this is a real change to a tested invariant — flag it for Milan.

5. **`src/climbing_elo/engine/elo.py:200-216`** — pair_k decomposition
   - **Drop `pair_k = base_k / (n - 1)`**. The field-size normalization was a hack for the constant-K world; Glicko-2's v is *already* the right normalizer (large field → small v_inv terms summed → small v → big confident step). The K_FACTOR_TABLE still controls the **base K injected via the τ system constant per tier × round** — see "K-factor retuning" below.

6. **`src/climbing_elo/engine/backfill.py:176-189`** — already-correct rating-write path
   - No structural changes. `upd.sigma_after` and `upd.mu_after` already round-trip through the cache and `Rating` row.
   - **Optional**: write the Glicko-2 internal-scale volatility (σ in internal units, ≈ 0.06) to a new `Rating.volatility` column — see Migration.

7. **`src/climbing_elo/models.py:151-173`** — Rating schema
   - **Minimum-invasive**: reinterpret existing `sigma` column as RD (display scale, 50-350 range). No migration needed; existing values are already in this range.
   - **Recommended**: add a nullable `volatility: float` column for Glicko-2 σ. Initial value 0.06. Makes the system observable and lets us tune τ per discipline if needed (Open Q4).

8. **`src/climbing_elo/engine/projections.py:74,237`** — Monte Carlo draws
   - **Status quo**: `N(μ, σ)` continues to work with σ reinterpreted as RD. The semantics shift slightly (RD reflects rating uncertainty rather than performance noise) but the practical effect is similar: high-uncertainty athletes get wider draws → reasonable behaviour for cold-start and post-sabbatical predictions.
   - **Recommended for a follow-up**: separate `performance_sigma` per athlete (could be estimated from residuals of the rating model) and use `N(μ, sqrt(RD² + perf_σ²))` for projections. **Out of scope for R1**; flag it as a follow-up under R0's stratified-calibration work.

9. **`tests/test_elo.py`** — invariants
   - `test_calculate_round_updates_zero_sum` (`tests/test_elo.py:80-93`) — relax to "|total| < tolerance(field_size)" or remove with a note that Glicko-2 explicitly does not preserve zero-sum.
   - `test_provisional_higher_k` (~`tests/test_elo.py:192+`) — replace with "high-φ athletes have larger absolute deltas than low-φ athletes" (i.e. the Glicko-2-natural cold-start test).
   - Add `test_glicko2_g_function`, `test_glicko2_expected_score_known_values`, `test_volatility_iteration_converges`, `test_scale_round_trip`.

10. **`src/climbing_elo/engine/baselines.py`** — register `glicko2` variant
    - Add a `Glicko2Engine` class implementing the `RatingEngine` protocol. Its `predict()` returns the post-backfill (μ, φ) for each athlete at training cutoff. Variant name: `"glicko2"`.
    - Run via `uv run python scripts/run_backtest.py --variant glicko2`.

---

## Migration strategy

### Two options

**Option A — Forward-only**
- Ship Glicko-2 code; the next scrape inserts new results, and the next backfill incrementally updates the existing Rating rows using Glicko-2 math.
- **Risk**: existing μ values were computed under the constant-K + symmetric-margin regime. They are not "Glicko-2 ratings" — they're the wrong reference point for the new update. A 2nd event after migration sees an athlete with `μ = 1800, φ = 305/173.7 ≈ 1.76` (low RD by Glicko-2 standards), so the update is small and conservative. **Cold-start athletes who happened to be in the provisional bucket at cutover get treated as established overnight** — the exact failure mode R1 is trying to fix.
- **Verdict**: not recommended for production. Possibly viable for a quick correctness sanity check (does the math run without exceptions on real data) before committing to Option B.

**Option B — Full re-backfill (RECOMMENDED)**
- Wipe `Rating` and `RatingHistory` rows for the affected disciplines and re-run `scripts/run_backfill.py` from event 1 with the new engine. Total backfill walltime today is ~30s for 14 years of Lead + Boulder; this is cheap.
- **Risk**: any client caching the leaderboard for >30s sees an inconsistent snapshot during the swap. Mitigation: run backfill into a staging DB, copy the rows in one transaction, then clear `predictions_cache` and `likely_roster_cache` (`scripts/clear_cache.py`).
- **Risk**: the old `RatingHistory` (used by `/athletes/{id}` charts and `/breakdown/...`) is wiped. **This is a feature, not a bug** — the new chart actually reflects the production model. Users who screenshot the old charts have no protection guarantee.
- **Recommendation**: do a snapshot (`scripts/snapshot_db.py`) immediately before swap so we can roll back. Update CLAUDE.md to note the discontinuity at the cut date.

### Migration commands (sketch)

```bash
# 1. Snapshot the current production DB
uv run python scripts/snapshot_db.py
# 2. Run backfill into a copy
uv run python scripts/run_backfill.py --db /tmp/glicko2-staging.db --from-scratch
# 3. Validate against backtest harness
uv run python scripts/run_backtest.py --db /tmp/glicko2-staging.db --variant glicko2
# 4. Compare top-30 leaderboard between staging and prod (visual diff)
# 5. Atomic file replace + cache clear
mv /tmp/glicko2-staging.db data/climbing.db
uv run python scripts/clear_cache.py
```

The `--from-scratch` flag does not exist today; `run_backfill.py` would need a small CLI addition to clear `Rating`/`RatingHistory` rows before running. Trivial.

### What about live events?

If an event is being polled live at the moment of migration, the file lock at `/tmp/climbing_elo_poller_<event_id>.lock` (per CLAUDE.md "Live Events" section) prevents the migration from racing with new Result inserts — but the live poller only writes Result rows, not Rating updates. Safe to migrate during a live event; the next post-event backfill will pick up the new results under the Glicko-2 engine.

---

## K-factor retuning

### Why the existing table needs to change

Current `K_FACTOR_TABLE` (`elo.py:25-46`) was grid-searched with the constant-K + 1.5-margin-cap + 2× provisional regime. The values (Olympics Final = 96, WC Final = 64, WC Qual = 24) implicitly absorb the average update aggressiveness across all athletes — including provisional ones moving 2× faster.

Under Glicko-2:
- New athletes (φ ≈ 2.0) get **much** larger effective steps via `μ' = μ + φ'² · Σ g(φ)(s - E)` — the φ'² factor dominates. Empirically Glicko-2 implementations report initial-period swings of 200-400 rating points for a single 8-game period. Our current 2× provisional gives ~120-point swings.
- Established athletes (φ ≈ 0.5-1.0) get **smaller** effective steps than today — the small φ² shrinks the update, which is the whole point.

If we kept the current K_FACTOR_TABLE while adding Glicko-2, the new-athlete updates would be **far too aggressive** and the veteran updates **too sluggish**.

### Concrete proposal

Move the K_FACTOR_TABLE off of the per-pair scalar and onto a **per-tier × per-round volatility budget τ_tier_round**, with the existing relative ratios preserved. Starting point:

```
τ_tier_round = K_FACTOR_TABLE[tier][round] × K_TAU_COUPLING
K_TAU_COUPLING ≈ 0.005  (initial guess; grid-search target)
```

This puts Olympics Final at τ ≈ 0.48 and WC Qual at τ ≈ 0.12 — within Glickman's recommended τ ∈ [0.3, 1.2] for Olympics/WCh and below it for Qualification (which makes sense: a qual round has 60+ pairwise comparisons per athlete, so per-comparison signal is low and we want a gentler τ).

### Re-tune sweep

After the engine works end-to-end, run a grid search analogous to `scripts/tune_kfactors.py:60-61`:
```python
K_TAU_COUPLING_VALUES = [0.003, 0.005, 0.008, 0.012]
TAU_BASE_VALUES = [0.3, 0.5, 0.8]  # Glickman's range
MARGIN_CAP_VALUES = [1.0, 1.2, 1.5]  # MOV may matter less under Glicko-2
```

Target metric: aggregate **log-loss-podium** on the holdout-2s split (lower is better), with the constraint that no stratum regresses by more than 5%. Reuse the existing tune script's structure; swap the inner `run_evaluation` call for one that builds a `BacktestRunner` with `variant="glicko2"`.

**Estimated sweep cost**: 4 × 3 × 3 = 36 backfill+score runs. Backfill is ~30s; backtest scoring is ~60s. Total ~55 minutes single-threaded. Trivially parallelizable.

---

## Test plan

### New unit tests (`tests/test_elo.py`)

1. `test_glicko2_g_function_values` — assert `g(0) = 1.0`, `g(350/173.7) ≈ 0.351` (matches Glickman 2013 worked example to 3dp).
2. `test_glicko2_expected_score_known_values` — assert the expected score matches the standard Glicko-2 reference numbers (Glickman 2013 §3 worked example: player at r=1500, RD=200 vs r=1400, RD=30 → E ≈ 0.376).
3. `test_volatility_iteration_converges` — random (Δ, φ, v) triplets; assert Illinois converges in <20 iterations with `|f(σ')| < 1e-6`.
4. `test_scale_round_trip` — `from_internal(to_internal(r, RD)) == (r, RD)` to floating-point.
5. `test_zero_games_period` — call the update with no opponents; assert μ unchanged, σ unchanged, φ inflates to `sqrt(φ² + σ²)`.
6. `test_high_phi_athlete_moves_more_than_low_phi` — replaces `test_provisional_higher_k`. Two athletes with same (μ, n_events) but different φ; after the same round, higher-φ athlete has larger |Δμ|.

### New integration tests (`tests/test_backfill.py`)

7. `test_glicko2_backfill_3_event_integration` — analogue of the existing 3-event integration test, using the Glicko-2 path. Asserts ratings stabilize within expected Glickman-range, σ shrinks as expected.
8. `test_glicko2_reproducibility` — same seed, same data, same final ratings to 1e-6.

### Cold-start trajectory tests (NEW)

The acceptance criterion in #51 is "emerging athletes (Anraku, Bertone) climb to top-10 in fewer events than current". Use real production athletes:

| Athlete | ID | n_events | First | Last |
|---|---|---|---|---|
| Sorato ANRAKU | 5 | 19 | 2023-06-14 | 2026-05-08 |
| Oriane BERTONE | 447 | 9 | 2021-07-01 | 2024-06-26 |
| Max BERTONE | 15 | 13 | 2024-06-26 | 2026-05-08 |

**Test design** (`tests/test_elo.py:test_cold_start_climbs_faster`):
1. Run the current-engine backfill against the production DB copy → record event-by-event μ trajectory for athlete IDs 5 and 447.
2. Run the Glicko-2 backfill → record the same trajectory.
3. Assert that the event-index at which Anraku first crosses top-10 (Boulder discipline) under Glicko-2 is ≤ event-index under current engine. Same for Oriane Bertone.

**Implementation note**: the test can run against a snapshot of the production DB (`scripts/restore_snapshot.py --date 2026-05-01`) checked into `tests/fixtures/` as a small SQLite file, or it can be marked `@pytest.mark.slow` and run against the live DB during CI nightlies. The latter is preferred — it always reflects the current data.

### Sabbatical-return test (NEW)

`test_sabbatical_return_widens_uncertainty` using Garnbret (ID 60, 72 events 2015-2026, took a post-Olympic 2025 break):
1. Find the longest gap in Garnbret's Boulder event history (likely ~9 months in 2025).
2. Under current engine, σ before and σ after the gap differ by <5 RD units (cosmetic decay).
3. Under Glicko-2, φ before and φ after should differ noticeably (the Wiener-process inflation actually acts). Acceptance: post-gap RD ≥ pre-gap RD × 1.1.

### "No regression" check (CI gate)

Run the full backtest harness pre- and post-migration:
```bash
uv run python scripts/run_backtest.py --variant current --output-dir data/backtests/pre/
uv run python scripts/run_backtest.py --variant glicko2 --output-dir data/backtests/post/
```

Pass conditions (`tests/test_baselines.py` analogue):
- `report.aggregate["log_loss_podium"]` for `glicko2` ≤ `current` × 1.02 (max 2% regression).
- `report.aggregate["hit_rate_top3"]` for `glicko2` ≥ `current` − 0.02 (max 2pp regression).
- Cold-start stratum (tenure 1-3, 4-10): `log_loss_podium` *improves* by ≥ 5% relative.

These thresholds are tighter than the legacy "+15pp vs baseline" because the new harness scores against the production engine, not a random baseline.

### Supabase MCP for the cold-start test

The lookup query for Anraku/Bertone IDs is:
```sql
SELECT id, name, gender FROM athletes
WHERE name ILIKE '%anraku%' OR name ILIKE '%bertone%'
ORDER BY name;
```
(Already confirmed: Anraku = 5, Oriane Bertone = 447, Max Bertone = 15. IDs are stable in the production DB and used directly in the test.)

---

## Open questions

### Open Q1 — How long is a "Glicko-2 rating period" for inactivity inflation?

**The choice**: when an athlete has been inactive for 6 months between events, how many empty periods do we inflate φ across?

**Options**:
- (a) One period per event-the-athlete-skipped (need to enumerate events globally per discipline per gender and count those between this athlete's last event and now).
- (b) One period per 30 days of inactivity (simple, doesn't require global event enumeration).
- (c) One period per discipline-season-bin (e.g. 6 WC events per Boulder season → ~60 days/period).

**Tentative recommendation**: (b) — calendar-driven, simple, robust to schedule changes. Set rate so that 18 months of inactivity inflates a φ ≈ 0.5 athlete to φ ≈ 1.5 — matching the current SIGMA_DECAY_HALF_LIFE_DAYS = 540 behaviour at the boundaries.

**What would let Milan decide for sure**: Glickman's chess implementation uses (a)-style (per FIDE-rating period of one month). Lichess uses a fixed 30-day calendar period. Our event cadence is irregular enough that (b) is the safer default; (a) would require a non-trivial query per athlete per backfill step.

### Open Q2 — Sole symmetric application of margin vs. asymmetric (R3 #53 territory)

**The choice**: keep applying the margin multiplier symmetrically to winner and loser (current behaviour), or move to winner-only (the #53 conjecture)?

**Tentative recommendation**: keep symmetric for #51 — folded into the `s_j` adjustment as described in "Adaptation to our setting". Defer asymmetric to #53 so it can be A/B'd cleanly via the backtest harness.

**What would let Milan decide for sure**: results of the diagnostic plot in #53 step 1 (do top-10 ratings drift up over multi-year windows?). If yes, asymmetric is justified; otherwise we leave the symmetric application as a tested invariant.

### Open Q3 — Margin folding into s_j vs. into K

**The choice**: incorporate the margin multiplier by (a) skewing `s_j` toward extremes (the "score-shifting" interpretation), or (b) scaling the per-pair contribution to v and Δ (the "weighted observation" interpretation).

**Tentative recommendation**: (a). It preserves the variance-update interpretation and keeps the v formula clean (an upset of 1 hold ≈ same evidential weight as an upset of 15 holds, with the latter shifting the apparent outcome).

**What would let Milan decide for sure**: G-Elo (Szczecinski 2022, cited in R3) explicitly chooses (b). If we end up implementing G-Elo as the R3 deliverable, we should align — flag this for revisit when #53 lands.

### Open Q4 — Per-discipline τ?

**The choice**: one global τ or per-discipline τ values?

**Tentative recommendation**: one global τ ∈ [0.3, 0.5] for the initial implementation. The grid sweep will tell us if per-discipline τ is worth the additional configuration surface.

**What would let Milan decide for sure**: backtest result. If `log_loss_podium` for Boulder is materially worse than for Lead under a shared τ, that's evidence per-discipline τ is needed.

### Open Q5 — Projections σ semantics

**The choice**: re-use Glicko-2 φ as the Monte Carlo draw σ in `projections.py:74`, or introduce a separate `performance_sigma` field?

**Tentative recommendation**: re-use for the R1 PR. The semantic drift is real but the practical effect (wider draws for less-certain athletes) is directionally correct. Flag for follow-up work as part of R0's calibration stratification.

**What would let Milan decide for sure**: the R0 calibration plot. If the new Glicko-2 + projection combo is materially over- or under-confident in any stratum, separating the two σ values becomes necessary.

---

## Pairing with #53

**Recommendation: ship #51 first, then #53 as a follow-up — NOT one combined PR.**

### Reasons

1. **Both touch `engine/elo.py` expected-score and update math, but they touch different parts.** #51 replaces the update mechanism entirely (variance-based, iterative volatility). #53 reshapes the MOV multiplier (rating-gap conditioning, optional asymmetric application). These are orthogonal in the code.

2. **#53 wants a diagnostic plot first** (per its acceptance criterion: "Diagnostic plot or test demonstrating presence/absence of drift"). That diagnostic should be run **after** #51 lands, because the Glicko-2 update changes the drift profile by construction (φ²-scaled updates shrink veteran movement). Running the diagnostic on the current engine and then immediately deprecating that engine wastes the analysis.

3. **Backtest harness A/B is cleaner with one variable at a time.** Combining R1 and R3 into one PR means a regression in `log_loss_podium` cannot be attributed to the wrong half. Sequential PRs let us measure each effect independently — R0's whole purpose is exactly this.

4. **Effort estimate matches.** #51 = ~1 week per Milan's issue; #53 = ~2-3 days per Milan's issue. The combined PR would be the worst of both worlds (large diff, multiple variables, slower review).

### Coordination

When #51 lands, immediately open #53 against the new engine. The MOV conditioning formula in #53 (`mov_mult = ln(|score_gap| + 1) × 2.2 / (0.001 × elo_diff + 2.2)`) plugs cleanly into the score-shifting `s_j_adjusted` adapter defined in §"Adaptation to our setting" above — so #53 becomes a one-line change to that adapter plus a re-tune.

---

## Effort breakdown

Total: **7 working days** (~1 week, matching the issue estimate).

| # | Task | Days |
|---|---|---|
| 1 | Implement Glicko-2 primitives (`g`, `E`, scale conversion, Illinois iteration) + unit tests | 1.0 |
| 2 | Wire into `calculate_round_updates` (per-athlete v/Δ accumulation, post-loop σ'/φ'/μ' computation) | 1.5 |
| 3 | Inactivity inflation (`apply_time_decay` → `glicko2_inflate_phi`) + tests | 0.5 |
| 4 | Adapt margin multiplier (fold into s_j) + retire `pair_k` normalization | 0.5 |
| 5 | Register `glicko2` variant in `engine/baselines.py` | 0.25 |
| 6 | Migration script + atomic-swap workflow + snapshot/rollback rehearsal | 0.5 |
| 7 | K-factor → τ retune sweep (36 runs, single-threaded ~1h) + result writeup | 1.0 |
| 8 | Cold-start trajectory test (Anraku, Bertone) using snapshot/live DB | 0.5 |
| 9 | Sabbatical-return test (Garnbret) | 0.25 |
| 10 | Backtest harness A/B against `current` + `stripped_elo` baselines, regression-gate check | 0.5 |
| 11 | Documentation update (CLAUDE.md "ELO Engine Specifics" + RATING_SYSTEM_RESEARCH.md §5 R1 status) | 0.5 |
| 12 | PR review + iteration buffer | 0.5 |

**Risk of overrun**: K-factor retune (#7) is the biggest single time-sink. If the initial grid points all show >5% regression in some stratum, we may need a finer second-pass sweep — add another 0.5-1.0 day.

---

## Out of scope

Explicitly **not** in this PR:

1. **Whole-History Rating** (R2 / #52). WHR is a batch refit pass, not incremental. Should land as a separate workstream.
2. **MOV redesign** (R3 / #53). See "Pairing with #53" — follow-up PR.
3. **Per-pair tie parameter for Speed** (R6 / Davidson 1970). Speed pipeline already has its own margin function; updating it to Glicko-2 is a separate concern.
4. **Separating projection σ from rating φ** (Open Q5). Out of scope; follow-up on R0 work.
5. **Per-discipline τ tuning** (Open Q4). Initial implementation uses one global τ; revisit if backtest shows discipline-level miscalibration.
6. **TrueSkill / OpenSkill migration** (R&D §3.2). Not on the recommendation list; this PR commits us to Glicko-2 specifically.
7. **Schema changes** beyond an optional `Rating.volatility` column. No new tables, no new enums.
8. **Live-event mid-event ELO updates** (CLAUDE.md "Live Events"). Still deferred to post-event backfill; Glicko-2 doesn't change that.
9. **Frontend changes** to the leaderboard/profile views. The σ→RD semantic shift is invisible to users (same numbers, similar range).

---

*End of plan. Updates should preserve the section numbering and integration-point numbering for stable cross-references with PR review comments.*
