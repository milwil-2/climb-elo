# σ-Ceiling Investigation (Issue #89)

## TL;DR

**Root cause is not the ceiling — it's that the prod DB has never been re-backfilled since R1 (#51) landed.** 80.3% of `ratings` rows (3,342 of 4,160) sit at exactly **σ=343.000**, which is **350 × 0.98** — the value produced by the *pre-R1* per-event σ-convergence step (`σ ← σ × 0.98`). The R1 commit (47dae01, today 2026-05-26) retired `apply_time_decay()` and the 0.98 multiplier, but `backfill.py` has an idempotency guard that skips any round already present in `rating_history`, so every existing row is frozen with legacy math. The 7-point bias (350→343) shows up in every event back through May 2026.

**Recommended fix:** trigger a full re-backfill against prod (purge `rating_history` + reset `ratings` to defaults, then run `run_backfill.py` for L/B/S + `compute_combined_ratings.py`). Effort ≈ 30 min wall-clock, low risk. Optionally pair with a config tweak (`glicko2_sigma_inactivity` raise from 5.0 → ~25.0, see Phase 4) because the current value barely moves σ even under the *new* code.

## Phase 1 — Data audit

All numbers from `mcp__supabase__execute_sql` against `micecpgpuispvdfqdtmm` on 2026-05-26.

### Overall distribution

| Metric | Value |
|---|---|
| Total rated-athlete-discipline rows | 4,160 |
| Min / Max σ | 305.27 / 350.00 |
| Mean σ | 342.14 |
| Stddev σ | 2.27 |
| Rows with σ exactly = 343.00 | **3,342 (80.3%)** |
| Rows with σ exactly = 350.00 | 13 |
| Rows with σ < 340 | 580 (13.9%) |
| Rows with σ < 300 | **0** |
| Rows with σ < 200 | 0 |
| Rows with σ ≤ floor (50) | 0 |

The entire population spans only **45 RD points** (305 → 350). The floor (50) is **untouched**. The σ band carries no useful information across the leaderboard.

### Per-discipline breakdown

| Discipline | Total | At σ=343 | At σ=350 | Mean σ | Min σ |
|---|---:|---:|---:|---:|---:|
| Lead (L) | 1,716 | 1,315 (76.6%) | 5 | 341.78 | 305.27 |
| Boulder (B) | 1,744 | 1,523 (87.3%) | 8 | 342.49 | 326.60 |
| Speed (S) | 227 | 190 (83.7%) | 0 | 342.29 | 336.14 |
| Boulder+Lead (BL) | 473 | 314 (66.4%) | 0 | 342.10 | 333.22 |

Effect is **uniform across disciplines**. Issue #89's "80-91%" framing matches Boulder; Lead is slightly lower (76.6%) because more long-inactive lead-only specialists drift into the 305-340 band.

### Top-10 σ values (per discipline, n_events≥3)

Every name on every top-10 list except 4 outliers sits at σ=343.00. Concretely:

- **Lead M/F top-10:** Garnbret, Schubert, Seo, Mori, Verhoeven, Ondra (343), Ja In Kim (336.25), Anraku, Ginés López, Amma — all but Kim at 343.
- **Boulder top-10:** Garnbret, Anraku, Grossman (339.72), Schalck, Noguchi, Bertone, Lee, Raboutou, Sanders, Fujii — all at 343 except Grossman.
- **Speed top-10:** all 10 at exactly 343.00.

### Sanity check — the mechanism does work for *some* rounds

`rating_history` (31,328 rows) min/max σ_after: **301.18 / 343.00** — `sigma_after` is *never* above 343 anywhere in the history. The minimum 301.18 confirms the post-update shrinkage formula does fire, just rarely enough that the ceiling dominates.

## Phase 2 — Source audit

Both clamps are in place and correct relative to spec — the bug is upstream.

### Where σ_ceiling clamps

1. **`src/climbing_elo/engine/elo.py:377`** — `glicko2_inflate_phi` returns `min(sigma_new, config.sigma_ceiling)`.
2. **`src/climbing_elo/engine/elo.py:717-719`** — `calculate_round_updates` post-update clamp `max(sigma_floor, min(sigma_ceiling, sigma_after_display))`.
3. **`src/climbing_elo/engine/backfill.py:194-195`** — writes `db_rating.sigma = upd.sigma_after` (no extra clamp).

### Inactivity-inflation math — sanity check

`σ_new² = σ_display² + σ_inactivity² · months_inactive` with `σ_inactivity = 5.0` (display scale, per docstring line 372).

Local arithmetic (`python3 -c …`):

| Days gap from σ=343 | σ_new |
|---:|---:|
| 7 | 343.0085 |
| 42 | 343.051 |
| 300 | 343.37 |
| 2,433 (Verhoeven's gap) | 348.85 |

**With the current σ_inactivity=5.0, inflation can never reach the 350 ceiling from σ=343 — not even for 10-year sabbaticals.** This is a soft secondary finding (Phase 3 cause #2 partially confirmed): even with a fresh re-backfill, inflation is far too weak to make σ visually meaningful.

### Trace of one athlete

`SELECT * FROM rating_history WHERE athlete_id = 60 AND e.discipline='L'`, ordered chronologically, shows the diagnostic transition:

| Date | Round | days_gap | prev_sa | sb | sa |
|---|---|---:|---:|---:|---:|
| 2022-09-09 | quali | 7 | 343.000 | **346.10** | 339.17 |
| 2022-09-09 | final | 0 | 339.17 | 342.24 | 335.39 |
| 2022-09-24 | quali | 15 | 335.39 | 341.91 | 335.07 |
| 2022-09-24 | final | 0 | 335.07 | 341.59 | 334.76 |
| **2023-06-14** | **quali** | 263 | 334.76 | **350.00** | **343.00** |
| 2023-06-14 → 2026-05-08 (every subsequent row) | all | … | 343 | **350.00** | **343.00** |

The cutover is at the 2023 season opener. The pre-cutover σ values (335-346) are produced by either the current Glicko-2 code *or* are old enough that they happen to be in the same range. The post-cutover values are **all identical to `350 × 0.98 = 343`**, which is the *exact* output of the pre-R1 per-event multiplier (`σ ← σ × 0.98` from the original PRD §σ-decay).

`350.0 * 0.98 = 343.000` — verified in SQL. This is not a coincidence.

## Phase 3 — Root cause

**Cause #4 wins (legacy pre-R1 data, never re-rated).** Evidence:

1. **Math doesn't add up under current code.** With σ_inactivity=5.0 (display), no realistic inactivity gap produces σ_before=350.0 from any starting σ. Yet **19,302 of 31,328** `rating_history` rows have sb=350.000 exactly. The current `glicko2_inflate_phi` simply cannot produce that value.
2. **The 0.98 multiplier matches exactly.** `350 × 0.98 = 343.000` — the *exact* value seen in 80% of rows. The R1 commit message (47dae01) explicitly states: "apply_time_decay() retired and replaced by glicko2_inflate_phi()". The git log shows that function was the pre-R1 owner of σ updates.
3. **Idempotency guard prevents re-rating.** `backfill.py:146-150` skips any round with an existing `RatingHistory` row. So once a round was rated with the old code, the new code never touches it.
4. **The R1 commit landed *today* (2026-05-26).** Every `rating_history` row for events dated ≤ 2026-05-22 (yesterday's Bern Boulder event has 201/201 rows at sa=343.000) was written by the pre-R1 code path. No scheduled scrape has run with the new code yet.
5. **`Rating.sigma` carries forward the last rated round.** Athletes whose most recent round produced sa=343 will keep σ=343 indefinitely until either (a) a new round happens, or (b) the data is re-rated. Most active top-30 athletes' last round used the old formula → they sit at 343.

**Rejected alternatives:**

- **Cause #1 (ceiling too low):** the ceiling clamp is rarely the binding constraint — sigma_after is *never* recorded above 343 anywhere in history. Raising sigma_ceiling alone changes nothing for existing rows.
- **Cause #2 (inactivity inflation too aggressive):** the opposite is true — inflation with σ_inactivity=5.0 is *too weak* to matter even over multi-year gaps. Still a real secondary issue (see Phase 4).
- **Cause #3 (storage-side clamp wrong layer):** the clamp is correctly at the engine layer; moving it to display would hide the symptom but not fix the legacy data.
- **Cause #5 (initial assignment instant-inflates):** verified false — `_get_or_create_rating` writes σ=350 and the first round drops it; no double-inflation path exists.
- **Cause #6 (R1 design deviations):** symmetric K and σ_inactivity=5.0 are R1 design choices flagged in `docs/PLAN_GLICKO2_RD_INTEGRATION.md`. The σ_inactivity=5.0 number is too small to be useful (see Phase 4 fix #2), but it's not the root cause of the ceiling-binding observation.

## Phase 4 — Recommended fixes

### Fix 1 — Re-backfill prod from scratch (REQUIRED)

**What:** Purge `rating_history`, reset `ratings` to defaults (μ=1500, σ=350, n_events=0, last_event_at=NULL), then run `scripts/run_backfill.py` for L/B/S followed by `scripts/compute_combined_ratings.py`.

| Dimension | Score |
|---|---|
| Risk | **Low** — fully recoverable from `db-snapshots` GitHub Release if anything goes wrong; the operation is deterministic given the same code+config. |
| Reversibility | **High** — single SQL transaction can be rolled back; snapshot can be restored. |
| User-visible impact | **Very high** — every athlete's σ becomes whatever the new Glicko-2 math produces. Top athletes (Garnbret/Schubert/Ondra) should drop to σ ≈ 150-250 instead of 343. |
| Effort | ~30 min wall-clock (scrape was confirmed up-to-date; backfill is the main runtime). |

**Sketch:**
```sql
TRUNCATE rating_history;
UPDATE ratings SET mu=1500, sigma=350, n_events=0, last_event_at=NULL, provisional=true;
```
then `uv run python scripts/run_backfill.py --discipline lead`, `--discipline boulder`, `--discipline speed`, then `uv run python scripts/compute_combined_ratings.py`.

**Prerequisite:** Confirm #80 (K-regrid) status. The CLAUDE.md and Issue #89 itself note: "Trigger re-backfill on production (verify σ values then become differentiated) … gated on #80 K-regrid landing first." If #80 is still open and likely to land soon with a new K-factor table, defer this fix until then to avoid a double re-backfill.

### Fix 2 — Raise `glicko2_sigma_inactivity` from 5.0 to ~25.0 (RECOMMENDED follow-up)

**What:** Change `EloConfig.glicko2_sigma_inactivity: float = 5.0` → `25.0` (or run a small sweep [10, 15, 25, 50] against the backtest harness).

**Why:** Even with Fix 1, σ_inactivity=5.0 is too weak. Computed inflation over a 12-month sabbatical from σ=200:
- σ_new² = 200² + 5²·12 = 40,300 → σ_new = 200.75 — i.e. **negligible**.
- With σ_inactivity=25.0: σ_new² = 40,000 + 625·12 = 47,500 → σ_new = 217.9. Modest, plausible.
- With σ_inactivity=50.0: σ_new² = 40,000 + 2,500·12 = 70,000 → σ_new = 264.6. Clear "this athlete has been away" signal.

The CLAUDE.md and module docstring claim "with σ_inactivity=5.0, a 12-month gap inflates φ=0.5 (RD≈87) to φ≈0.85 (RD≈148)" — that arithmetic is wrong by ~30× and is the legacy of a misread-then-copy-pasted Glickman constant. The actual produced value is σ ≈ 87.5 after 12 months, not 148.

| Dimension | Score |
|---|---|
| Risk | Medium — affects all future backfills; could shift backtest log-loss. Run regression first. |
| Reversibility | High — one-line config change, re-run backfill. |
| User-visible impact | Medium — makes σ band on graph (#86) actually wide enough to see between active and inactive athletes. |

### Fix 3 — Update `last_event_at` per-round, not per-event (MINOR)

**What:** In `backfill.py:247-260`, move the `last_event_at` update inside the per-round loop instead of the per-event loop.

**Why:** Currently, multi-round events (quali → semi → final) all see the same stale `last_event_at` (= prior event's date), so the inflation function applies the same months-of-inactivity to every round of an event. With σ_inactivity=5.0 this is invisible, but if Fix 2 is applied it could over-inflate σ for the final relative to the quali.

| Dimension | Score |
|---|---|
| Risk | Very low — a 2-line move. |
| Reversibility | Trivial. |
| User-visible impact | Low — only matters if Fix 2 is taken. |

### Fix 4 — Raise `sigma_ceiling` from 350 to 500-600 (NOT RECOMMENDED on its own)

**Why not:** The ceiling isn't the binding constraint in current data. The post-update formula in `calculate_round_updates` (line 717-719) clamps `sigma_after_display` and the only ceiling-binding rows in history come from the *pre-R1* per-event multiplier that no longer exists. Raising the ceiling without Fix 1 changes nothing. With Fix 1, the ceiling may need a small raise (450-500) only if Fix 2 is also taken and starts producing σ values >350 for long-inactive athletes.

## Phase 5 — Test plan

All tests are after Fix 1 is applied. Run against the re-backfilled prod DB or a staged Supabase branch (`mcp__supabase__create_branch`).

### Test 1 — Active athletes should have *low* σ (well below 343)

Athletes who competed in ≥5 rounds in the last 6 months should have σ ≪ 343 because Glicko-2 v_inv shrinkage compounds.

| Athlete ID | Name | Activity | Expected post-fix σ |
|---|---|---|---|
| 60 | Janja GARNBRET | 56 Lead events, last 19 days ago | < 200 |
| 335 | Jakob SCHUBERT | 67 Lead events, last 248 days ago | < 250 |
| 5 | Sorato ANRAKU | 19 Lead events, last 19 days ago | < 220 |

```sql
SELECT athlete_id, discipline, sigma FROM ratings
WHERE athlete_id IN (60, 335, 5) AND discipline = 'L';
-- ASSERT: all rows have sigma < 280 (much lower than current 343)
```

### Test 2 — Long-inactive athletes should have *high* σ

After Fix 1 alone, sigma will likely also be ≈ 343 (because their last rated round was when they were active, so v_inv shrinkage applied). After Fix 1 + Fix 2 (σ_inactivity=25.0), the inflation from then-to-now should push them notably higher.

| Athlete ID | Name | Days inactive | Expected post-fix-1 σ | Expected post-fix-1+2 σ |
|---|---|---:|---:|---:|
| 1441 | Sachi AMMA (L) | 3,847 (10.5 yr) | ~250 | > 320 |
| 1006 | Anak VERHOEVEN (L) | 2,433 (6.7 yr) | ~230 | > 290 |
| 932 | Shauna COXSEY (B) | 1,832 (5.0 yr) | ~230 | > 280 |
| 769 | Akiyo NOGUCHI (B) | 1,799 | ~230 | > 280 |

Note: the "active" σ is the σ *immediately after* their last rated round (before display-time inflation). The displayed value at query time should include the inactivity inflation. The `/api/v1/athletes/{id}` endpoint and `Rating.sigma` column give the *frozen* value as of the last rated round — they do NOT recompute inflation at read time. So Fix 2 alone won't help until either (a) re-backfill happens, or (b) we change the read path to apply inflation on the fly.

### Test 3 — Distribution should be well-spread

```sql
SELECT 
  COUNT(*) FILTER (WHERE sigma < 200) AS lt_200,
  COUNT(*) FILTER (WHERE sigma BETWEEN 200 AND 250) AS bucket_200_250,
  COUNT(*) FILTER (WHERE sigma BETWEEN 250 AND 300) AS bucket_250_300,
  COUNT(*) FILTER (WHERE sigma BETWEEN 300 AND 343) AS bucket_300_343,
  COUNT(*) FILTER (WHERE sigma > 343) AS gt_343
FROM ratings;
-- ASSERT: no bucket holds > 50% of the population
-- ASSERT: at least 10% of rows are < 250 (i.e. some athletes really do have confident ratings)
```

### Test 4 — Acceptance criterion from Issue #89

> σ-band on the ELO graph (from #86) is visually distinct between active and inactive athletes.

Pull `/athletes/60/history` and `/athletes/1441/history` after the fix. Garnbret's σ trajectory should be visually flat-low (~150-200) across her 56 events; Amma's should be flat-then-rising (his last fit σ + Wiener growth) and visibly higher than Garnbret's.

## Decision needed from Milan

**Two decisions, one blocking and one optional.**

1. **(Blocking) Should the prod re-backfill wait for #80 (K-regrid) to land, or proceed now?** Issue #89 explicitly says "gated on #80 K-regrid landing first" but also frames #89 as urgent. If #80 is weeks away, doing the re-backfill now is the right call — the σ ceiling-binding is the user-visible symptom and the K table is independently re-tunable later. If #80 is days away, batching avoids a double-backfill.

2. **(Optional) Bump `glicko2_sigma_inactivity` from 5.0 to 25.0 as part of the same change?** Doing it together avoids a second re-backfill. The downside is that the value hasn't been swept against the backtest yet — the safe path is: Fix 1 now, run backtest, then Fix 2 if the baseline numbers hold.

## Out of scope

- The CLAUDE.md docstring claim "σ decays toward 350 with an 18-month half-life during inactivity, and converges downward by 0.98× per event" describes the *retired* pre-R1 behaviour. Updating the docstring is a follow-up to whichever fix lands.
- Display-time σ inflation (recomputing σ at read time using `last_event_at` and the inflation formula) would let inactive athletes' uncertainty grow continuously between events without re-backfilling. This is a cleaner long-term architecture but is a larger change and is deferred.
- The `last_event_at` per-round vs per-event question (Fix 3) is a real but minor bug; deferred unless Fix 2 is taken.
- The arithmetic claim in the module docstring at `elo.py:362-365` ("12-month gap inflates φ=0.5 (RD≈87) to φ≈0.85 (RD≈148)") is off by ~30× and should be corrected in the docstring fix that accompanies whichever code change lands.
