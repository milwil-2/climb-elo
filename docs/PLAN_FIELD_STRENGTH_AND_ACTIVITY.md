# Field-Strength Credit + Activity-Weighted Leaderboard (Issue #88)

**Status:** Proposal for review. No code written. References Issue #88 framing (Gaps 1+2, candidate options A-H) and follows the structure of `docs/PLAN_GLICKO2_RD_INTEGRATION.md`.

**Companion files:** `src/climbing_elo/engine/elo.py`, `src/climbing_elo/api/routes.py`, `src/climbing_elo/api/v1_routes.py`, `src/climbing_elo/templates/leaderboard.html`, `src/climbing_elo/engine/likely_roster.py`, `src/climbing_elo/models.py`.

---

## TL;DR

Ship two small, independently-reversible changes that address both gaps without touching the ELO update math:

1. **Gap 1 (field strength)** — Implement **Option A** (Tournament Participation Bonus, TPB). Add a tier-weighted, zero-sum participation credit applied once per athlete per event during backfill. Calibrated so an Olympic top-8 earns ≈ +6 to +20 μ over a Continental top-8.
2. **Gap 2 (activity weighting)** — Implement **Option G** (Active / All-time dual view) with a **smart "All-time" classifier**. Default view = "Active" (last event within 12 months). All-time view filters out a `likely_retired` bucket using a 3-year-no-event heuristic plus a manual `Athlete.retired_at` escape valve (year-of-birth is currently 0% populated in production, so age-based thresholds are infeasible without a scraper enrichment).

Both ship as **display-layer + backfill-side** changes — no schema breaking changes, no migration window, fully reversible by feature-flag.

**Cross-gap calibration:** TPB increment is intentionally small (top-line ≤ +20 μ/event for Olympics) so an athlete who skips a year and then gets filtered out of the default view does **not** also see their μ artificially boosted by a TPB they didn't earn. The two changes operate on different axes (μ vs filter), so double-punishment is structural, not numeric.

**Total effort:** ≈ **4 working days** end-to-end (1.5d Gap 1 + 1.5d Gap 2 + 0.5d cross-gap test + 0.5d UX/copy + buffer). Both gaps can ship as separate PRs in series.

---

## Current state

### Gap 1 — Field strength is invisible to μ

`src/climbing_elo/engine/elo.py:532-734` (`calculate_round_updates`):

- μ deltas are computed pair-by-pair as `delta_pair = k_pair * (1.0 - e_i)` (line 668), then accumulated symmetrically:
  ```python
  deltas[res_i.athlete_id] += delta_pair
  deltas[res_j.athlete_id] -= delta_pair
  ```
- After R1 (Glicko-2 integration, Issue #51), the *only* tier-dependent input to μ is `base_k` looked up from `config.k_factor_table[tier][round_type]` (lines 94-115). Olympic Final K = 48 vs WC Final K = 32; the ratio applies symmetrically to winners and losers.
- **There is no participation credit anywhere.** A 9th-place at Olympics gets exactly the same μ trajectory as a 9th-place at a Continental event, modulo who their pairwise opponents were and the per-pair K factor.
- The `_gap_conditioning_factor` (line 252) damps the *MOV multiplier* for favourites — it does not add credit for showing up to a strong field.

This is provably zero-sum across each pair, hence across each round, hence across each event — by design of pairwise Elo and reinforced explicitly in module docstring (line 50: "μ updates remain pairwise-symmetric and therefore zero-sum across a round").

### Gap 2 — μ never decays, σ ceiling is binding

**Inactivity handling today** (`elo.py:345-377`, `glicko2_inflate_phi`):

- During inactivity, φ inflates per Wiener-process: `φ_new² = φ_old² + σ_inactivity² · months_inactive`.
- **The result is then clamped to `config.sigma_ceiling = 350.0`** (line 377). For an athlete who was already near the ceiling (the common case post-#51 — see audit below), inactivity inflation is a no-op.
- **μ never decays.** `deltas` (line 592) is initialised to zero per athlete and only mutated by the pairwise loop — there is no time-decay term.

**Audit of production state (Supabase, 2026-05-27):**

Even after R1 shipped, σ values are *still* clamped at the ceiling for the overwhelming majority of athletes — meaning the σ-based inactivity signal has effectively no headroom:

| Discipline | n | min σ | avg σ | max σ | σ > 340 (near ceiling) | σ in 200–300 (mid-band) |
|---|---|---|---|---|---|---|
| Lead     | 1,716 | 305.3 | 341.8 | 350.0 | 1,369 (80%) | 0 |
| Boulder  | 1,744 | 326.6 | 342.5 | 350.0 | 1,591 (91%) | 0 |
| Speed    |   227 | 336.1 | 342.3 | 343.0 |   203 (89%) | 0 |
| BL       |   473 | 333.2 | 342.1 | 343.0 |   417 (88%) | 0 |

Even the most-rated athletes (Garnbret 56 events, Schubert 67 events, Ondra 37 events) all show `σ = 343` per the World Championships Seoul 2025 result table. This confirms the σ-ceiling is binding for the simplified closed-form Glicko-2 update and that *any* user-visible activity weighting today is invisible. (Whether this is a φ-update bug is filed as a separate concern under #51 follow-ups; for this plan, we assume σ continues to behave as it does today.)

**Ghost athletes in the top-30 (Supabase, 2026-05-27):**

Using `last_event_at < 2024-05-27` (24m+ gap) as a "ghost" filter:

| Discipline | Gender | Ghosts in top-30 (>24m gap) | Semi-inactive (12-24m gap) |
|---|---|---|---|
| Lead     | M | **8**  | 1 |
| Lead     | F | **14** | 2 |
| Boulder  | M | **8**  | 2 |
| Boulder  | F | **7**  | 4 |

Concrete examples from the men's lead top-30: **Sachi AMMA at rank 5** (last event 2015-11-14, μ=2000), **Gautier SUPPER at rank 16** (last event 2018-07-06), **Manuel ROMAIN at rank 21** (last event 2014-09-08, μ=1802), **Magnus MIDTBOE at rank 23** (left climbing for YouTube, last 2015-08), **Keiichiro KORENAGA at rank 27** (last 2019-10), **Yuki HADA at rank 30** (last 2019-10).

Women's lead is more severe — nearly half the top-30 are ghosts including pre-2020 retirees who never had their μ deflated by subsequent results.

### Existing infrastructure that we can lean on

- `Rating.last_event_at` (`models.py:164`) is already populated on every event commit (`backfill.py:259`). No new schema for the activity classifier.
- `engine/likely_roster.py:30-91` already implements an "active-this-season" filter for the predictions page (athletes with ≥1 non-DNS WC result in the current season). The same pattern is the seed of the activity classifier here.
- `_get_rankings_v2` (`api/routes.py:169-193`) and `/api/v1/leaderboard` (`api/v1_routes.py:173-232`) are the only two places in the codebase that rank by `Rating.mu` for display. Both already return `last_event_at` in their payloads — a filter parameter is purely additive.

---

## Gap 1 — Field strength

### Options

| Option | Mechanism | μ-impact | Backtest impact | Reversibility |
|---|---|---|---|---|
| **A — Tournament Participation Bonus (TPB)** | Per event, every non-DNS finisher receives a small μ credit; total credit summed to zero across the field. Credit scales with `tier × round_reached`. | Direct, visible. Top finishers at Olympics get measurably more credit. | Predicts unchanged outcomes (zero-sum), so log-loss/Brier unchanged. | Trivial — feature flag, run backfill from scratch. |
| **B — Stronger field-strength MOV** | Extend R3's gap-conditioning so wins against deep (high-μ) fields get a multiplier on top of MOV. | Indirect, mediated through pairwise wins. Pretty cosmetic for back-of-field. | Likely small log-loss improvement (closer to G-Elo / field-quality models). | Medium — interacts with #53 MOV audit; tuning required. |
| **C — Parallel ranking system** | Keep μ as-is, ship a separate ATP-style season-points leaderboard. | Zero impact on μ. Two leaderboards now. | Doesn't apply (separate metric). | Easy — drop the table if unused. |
| **D — Hybrid display** | `μ_display = μ_elo × (1 - α) + season_points × α` | Most disruptive — changes the displayed number for every athlete. Highest user confusion risk. | Doesn't apply (display arithmetic). | Hard — once users see the hybrid number they internalise it. |

### Recommended: Option A (Tournament Participation Bonus)

**Why A over B/C/D:**

- **A is single-axis** — it changes one number (μ) in a way fans already understand intuitively ("Olympic top-8 should count for more than Continental top-8"). No new dashboard concept, no new column on the leaderboard.
- **A preserves the zero-sum invariant**: total credit is balanced across the field, so the leaderboard sum across all athletes stays constant. Existing R1 tests (`tests/test_elo.py:test_calculate_round_updates_zero_sum`) only need extension, not rewrite.
- **B requires another pass through the K-factor / MOV grid** (already done twice — for R1 #51 and R3 #53). Calibrating a third factor risks overfitting and would invalidate the existing tuned K-table.
- **C/D introduce a second visible metric** which Milan has historically resisted (issue framing notes "fans browsing the leaderboard get confused by ghost rankings" — adding a second leaderboard *adds* confusion, doesn't reduce it).

### Concrete TPB formula

Per event, after the pairwise updates have been computed (i.e. as a final step in `backfill.run_backfill`, not inside `calculate_round_updates`):

```
tier_weight = {olympics: 1.0, world_championship: 0.75, world_cup: 0.5, continental: 0.25}
round_weight = {final_reached: 1.0, semi_reached_only: 0.5, qual_only: 0.2}

# Total budget per event (Olympics: 100 μ-points pooled across field)
event_budget = TPB_BASE_BUDGET * tier_weight[event.tier]  # TPB_BASE_BUDGET = 100.0

# Per-athlete share: weighted by their best round reached
athlete_share[a] = round_weight[best_round_reached(a)] / Σ round_weight[best_round_reached(*)]

# Bonus per athlete (zero-sum after the field-average subtraction)
field_avg = event_budget / n_athletes_at_event
tpb[a] = event_budget * athlete_share[a] - field_avg
```

Concrete magnitudes for an Olympic event with 20 finishers (final = 8, semi-only = 12, qual-only = 0):

- Event budget = 100 μ-points.
- Finalist share: `1.0 / (8·1.0 + 12·0.5) = 1.0 / 14 ≈ 0.0714` → bonus ≈ `100·0.0714 - 5 = +2.14`.
- Wait — that's tiny. Need to revisit so finalists clearly get more credit. Revised formula uses **expected per-rank credit** instead:

```
# Better: per-rank credit decays from rank 1 down
per_rank_credit[k] = event_budget * exp(-LAMBDA * (k-1)) / Σ exp(-LAMBDA * (j-1))
# LAMBDA = 0.15 → rank 1 gets ~3× rank 8's credit; tail vanishes.
# Subtract field_avg = event_budget / n so total = 0.
```

For Olympics (budget 100, 20 athletes, λ=0.15):
- Rank 1 → +14 μ (≈ +19 raw, − 5 field-avg)
- Rank 3 → +9 μ
- Rank 8 → +0 μ (break-even)
- Rank 20 → −3.5 μ (small penalty for showing up but finishing back)

For Continental (budget 25, 20 athletes, λ=0.15):
- Rank 1 → +3.5 μ
- Rank 8 → +0 μ
- Rank 20 → −0.9 μ

**Final calibration is a tunable** (TPB_BASE_BUDGET, LAMBDA). The above is the recommended starting point; expose both as `EloConfig` fields so a grid search can tune them.

**Why this matches fan intuition:** Janja's Olympic gold gets visibly more credit than her IFSC WC win. A 9th-place at Olympics earns ≈ −0.5 μ vs a 9th-place at Continental earns ≈ −0.1 μ — but a top-3 at Olympics opens a real μ gap vs top-3 at Continental.

**Trade-off acknowledgement:** TPB conflates "Olympic results matter more" (true — that's the credit asymmetry) with "Olympic *participation* should be rewarded" (which is the actual mechanism). A 20th-place at Olympics is treated identically to a 20th-place at Continental in expected pairwise outcome (since the field is generally similar quality at the bottom) — but TPB doesn't *care* about the field composition, only the event tier and finish position. This is intentional simplification; Option B (field-quality multiplier) would be the more "correct" but expensive answer.

---

## Gap 2 — Activity weighting

### Options

| Option | Mechanism | μ-impact | UX cost | Reversibility |
|---|---|---|---|---|
| **E — μ decay during inactivity** | After N months of no events, μ pulls toward 1500 by some rate (FIDE-style). | Direct, changes the stored μ. | Low — invisible to fans, leaderboard "fixes itself". | Hard — once you've decayed an athlete's μ, restoring it requires re-backfill. |
| **F — Activity-weighted filter view** | Default leaderboard hides athletes inactive >X months; "Show all" toggle. | Zero μ impact. | Medium — one toggle, one URL param. | Trivial. |
| **G — Active vs All-time dual view** | Two named leaderboards. Default = Active. All-time = filtered, smart. | Zero μ impact. | Medium-high — two URLs, two mental models. | Trivial. |
| **H — Recency-weighted scoring** | Each event's contribution to μ decays with age (sliding window). | Indirect — rewards recent results in incremental backfill. | Low. | Hard — requires retaining full event history, alternative engine entirely. |

### Recommended: Option G with a smart "All-time" classifier

The coordinator's clarification (mid-task) refines G as follows: **two leaderboard views**, but "All-time" should **not** dump raw historical data — it should distinguish:

1. **"On a break"** — could plausibly return competitive (Garnbret took 2024 off, returned 2025; Ondra-style sabbaticals). → **Include** in All-time.
2. **"Likely retired"** — won't realistically return at top level (Coxsey post-Tokyo 2020, Noguchi 2021 retirement, Amma 2015-stopped). → **Exclude** from All-time.

Hence two filters: **Active** (`last_event_at ≥ 12 months`), **All-time** (`NOT is_likely_retired`). The classifier sits on the *backfill side* (computed once and stored) or the *display side* (computed on the fly per request — cheaper, more flexible).

#### Smart "likely retired" classifier — proposal

**Hard data constraint:** the production `athletes.year_of_birth` column is **0/2680 populated** (verified via Supabase 2026-05-27). The scraper doesn't pull date-of-birth from `ifsc.results.info` today. **All age-based heuristics are infeasible without first enriching the scraper** (separate prerequisite, ~0.5 day's work to add a DOB scrape against the `/api/v1/athletes/{id}` endpoint).

For now, the classifier must work on signals we *do* have: `last_event_at`, `n_events`, and the result trajectory before disappearance.

**Recommended classifier (year-of-birth-free, v1):**

```python
def is_likely_retired(rating: Rating, today: date, results_recent: list[Result]) -> bool:
    """
    Heuristic — flag athletes who won't realistically return to top-level comp.

    Signals (all must be available without year_of_birth):
      - long gap (≥3 years since last event)
      - OR  manual override (Athlete.retired_at is not NULL)
      - OR  declining trajectory + ≥2y gap: μ trended down in their last
        3 events AND ≥2 years since last event.

    Garnbret's 2024 break (was at peak before, returned strong) does NOT
    trigger. Coxsey post-Tokyo (was on podium, then 5y silence) does trigger
    via the 3y-gap rule.
    """
    if rating.last_event_at is None:
        return True
    if rating.athlete.retired_at is not None:
        return True

    days_since = (today - rating.last_event_at).days
    if days_since >= 3 * 365:
        return True

    if days_since >= 2 * 365:
        # Trajectory check on last 3 events
        recent_mu_deltas = [r.mu_after - r.mu_before for r in results_recent[-3:]]
        if len(recent_mu_deltas) >= 2 and all(d < 0 for d in recent_mu_deltas):
            return True

    return False
```

**Sanity check against known names (production data):**

| Name | Disc | Last event | Years since | Classification | Expected |
|---|---|---|---|---|---|
| Janja GARNBRET   | L | 2026-05-08 | 0 | active | ✓ |
| Adam ONDRA       | L | 2025-09-21 | 0 | active | ✓ |
| Jakob SCHUBERT   | L | 2025-09-21 | 0 | active | ✓ |
| Sachi AMMA       | L | 2015-11-14 | 10 | **likely_retired** | ✓ |
| Magnus MIDTBOE   | L | 2015-08-21 | 10 | **likely_retired** | ✓ |
| Tomoa NARASAKI   | L | 2024-06-26 | 1 | on_break (in All-time) | ✓ (still competing in B) |
| Sean BAILEY      | L | 2023-08-01 | 2 | on_break (in All-time) | ✓ (still competing) |
| Akiyo NOGUCHI    | L | 2021-06-23 | 4 | **likely_retired** | ✓ (officially retired 2021) |
| Shauna COXSEY    | B | 2021-05-21 | 5 | **likely_retired** | ✓ (officially retired 2021) |
| Sasha DIGIULIAN  | L | 2012-07-20 | 13 | **likely_retired** | ✓ |
| Petra KLINGLER   | L | 2024-04-12 | 2 | on_break | ⚠️ borderline (effectively retired) |

The Klingler case is the failure mode — she's effectively retired but the 3-year rule won't catch her until 2027-04. The escape valve is the manual `Athlete.retired_at` column.

**Impact on the leaderboard (Supabase numbers, top-100 ranked athletes, year-of-birth-free v1 heuristic, classifier = `last_event_at IS NULL OR years_since >= 3`):**

| Discipline | Gender | Retired in top-100 | Retired in top-30 | On-break in top-100 | On-break in top-30 | Truly active in top-100 |
|---|---|---|---|---|---|---|
| Lead    | M | 33 | 7  | 20 | 2 | 47 |
| Lead    | F | 36 | 11 | 20 | 5 | 44 |
| Boulder | M | 30 | 5  | 21 | 5 | 49 |
| Boulder | F | 30 | 6  | 22 | 5 | 48 |

In other words: roughly **a third of every top-100 leaderboard is a "likely retired" ghost** under this classifier, but the on-break population (Garnbret, Narasaki, etc.) is mostly preserved. The **Active** (12-month) view would show ≈ 47-49 athletes per top-100 page; the **All-time** (excludes retired) view shows ≈ 65-70 per top-100 page.

**Why G over E:**

- E (μ decay) is destructive and asymmetric in time — once Coxsey's μ is decayed to 1500 over 10 years, she shows up nowhere even when fans want to look at "where did the 2019 athletes rank back then?" The historical view is gone.
- G is *purely additive*: μ stays a true skill estimate; the filter is a display decision. Reversible per request via URL param.
- E doubles down on the gap with #51 — Glicko-2 already inflates φ during inactivity. Adding μ decay on top means an inactive athlete loses both ways. The "skill estimate" interpretation degrades to a mush.

**Why G over F:**

- F is "Active toggle" only — but Milan explicitly wants the All-time view to *also* be smart (filter out likely retired). G is F with a second filter level on top.

**Why G over H:**

- H requires re-architecting the backfill engine to retain per-event contributions and weight them on read. That's an order of magnitude more complexity for a display problem. Defer until #52 (WHR/ILSR) lands, since WHR naturally implements time-weighted updates.

---

## Cross-gap interactions

### Recommended combination is safe

The recommended pair (TPB + Dual-view) is structurally non-interfering:

- TPB affects **μ** (a stored, persistent value updated during backfill).
- Dual-view affects **display filtering** (a request-time predicate over Rating rows).

The two never share a code path. The only overlap is that a "likely retired" athlete will continue to accrue TPB if they ever come back — which is correct behaviour, since participation in a tournament is what TPB rewards.

### What about Option E + Option A combined?

The issue framing flags this as the double-punishment concern: "an athlete who skipped one season gets double-punished." This is real if both ship:

- E decays an athlete's μ by ≈ 30 points per year of inactivity (FIDE-typical rate).
- A doesn't decay anything but their last-event TPB is now several seasons stale; the athlete missed three event-bonuses they would otherwise have earned.

A theoretical Garnbret-style 1-year break would see her μ drop ~30 pts purely from E, plus miss out on 4-6 events of TPB (∼+50 μ in total) — a net ~80-pt swing for *taking time off*. That's overcorrection.

**We recommend against shipping E at all** in favour of G. If a future Milan wants E anyway, the calibration constraint is:

> The combined annual μ loss from E + foregone TPB should not exceed the typical 12-month μ drift of an actively-competing top-10 athlete (∼20-30 points).

### Combinatorial UI explosion check

The issue worries about "if we ship Option C (parallel ranking) AND Option F (filtered view), are we showing 4 leaderboards now?" Our recommendation collapses to:

- 1 leaderboard, 2 view-modes via pill toggle: **Active** (default) / **All-time**.
- Per discipline: 4 disciplines × 2 genders × 2 views = 16 paths, but UX is just one new pill alongside the existing Boulder/Lead/Speed/BL pills.
- No new leaderboard table types. Pagination/sort/columns unchanged.

### Interaction with #52 (WHR batch refit)

If #52 ever lands, it would supersede TPB by virtue of being a fundamentally different engine:

- WHR is a *batch* maximum-likelihood refit that already weights observations by time and field strength (via opponent ratings at the time of contest). The motivation for TPB ("Olympic top-8 should count for more") is partly subsumed because WHR weights opponents by their then-current Whole-History rating.
- WHR does **not** natively address the "ghost retired athletes" problem. The smart All-time classifier remains relevant.

Recommended sequencing: **TPB and Dual-view first** (this plan); **#52 WHR** only if there's appetite for a full engine re-architecture *after* this plan ships and we know whether the cosmetic problems are resolved.

### Interaction with #51 σ-ceiling bug (or possible bug)

The audit revealed that 80-91% of rated athletes have σ > 340 in production. This means R1's Glicko-2 φ-inflation is not visible in current state. **This plan does not depend on σ being fixed** — both TPB and Dual-view operate independently of σ. But once R1's σ behaves as designed, the "ghost athletes" problem will mostly self-correct (high-σ athletes will fall in any uncertainty-weighted display). The dual-view classifier remains useful as a hard cutoff for genuinely-retired athletes who would otherwise sit at high μ + high σ.

---

## Data audit

Captured from Supabase (`micecpgpuispvdfqdtmm`, 2026-05-27, today = 2026-05-26):

### Athlete activity distribution per discipline

| Discipline | Total rated | Active 12m | Active 24m | Active 36m | Inactive 36m+ |
|---|---|---|---|---|---|
| Lead     | 1,716 | 354 (21%) | 481 (28%) | 637 (37%) | 1,079 (63%) |
| Boulder  | 1,744 | 373 (21%) | 515 (30%) | 682 (39%) | 1,062 (61%) |
| Speed    |   227 | 211 (93%) | 227 (100%) | 227 (100%) | 0 (0%) — Speed scraping is 2023+ only |
| BL       |   473 | 218 (46%) | 256 (54%) | 306 (65%) | 167 (35%) |

### Top-50 men's lead — drop-off under activity filters

| Filter | Kept | Dropped |
|---|---|---|
| 12-month active | 28 | 22 |
| 24-month active | 32 | 18 |

That is, **44% of the men's lead top-50 is inactive >12 months**. The drop is heavily concentrated in the 12-24 month window (people who competed in 2024 but not 2025); the residual 18 inactive >24m are genuine ghosts.

### Top-30 ghost counts per discipline/gender (>24m inactive)

| Discipline | Gender | Ghosts in top-30 | Semi-inactive in top-30 |
|---|---|---|---|
| Lead    | M |  8 | 1 |
| Lead    | F | 14 | 2 |
| Boulder | M |  8 | 2 |
| Boulder | F |  7 | 4 |

### Year-of-birth coverage

- **0 / 2,680** athletes in production have `year_of_birth` populated.
- This **blocks** any age-based retirement heuristic (e.g. "athlete past 30 + 2y absence = retired"). The plan recommends the year-of-birth-free heuristic above. Filing a follow-up to enrich the scraper with DOB is a separate ~0.5 day task; until then the classifier degrades to a flat 3-year rule + manual `retired_at` override.

### Sigma distribution (informational — explains why σ-only inactivity is invisible today)

| Discipline | min | avg | max | σ > 340 |
|---|---|---|---|---|
| Lead     | 305 | 342 | 350 | 80% |
| Boulder  | 327 | 343 | 350 | 91% |
| Speed    | 336 | 342 | 343 | 89% |
| BL       | 333 | 342 | 343 | 88% |

---

## Reference systems

### ATP / WTA tennis — 52-week rolling points

Each Monday the rankings refresh: points earned 52 weeks ago drop off, replaced by new results. Players must "defend" their previous-year points by matching that performance. Total ranking = sum of points from the four Grand Slams + eight mandatory Masters 1000 events + ATP Finals + best 7 of remaining Tour/Challenger results. **Pure points system — no opponent-strength term.** This is the cleanest implementation of Option C; the cost is two parallel rankings (the ATP Rankings vs the Race to Year-End), which Milan has explicitly rejected for the climbing dashboard.

Source: [ATP Rankings (Wikipedia)](https://en.wikipedia.org/wiki/ATP_rankings), [PIF ATP Rankings rulebook 2025](https://www.atptour.com/-/media/files/rulebook/2025/2025-rulebook-chapter-9_pif-atp-rankings_23dec.pdf).

### Chess FIDE — no formal decay (yet)

Despite popular perception, FIDE does **not** currently decay an inactive player's rating. Players who stop playing are marked "inactive" but retain their rating indefinitely. FIDE CEO Emil Sutovsky has announced (October 2025) that a rating-decay system is "under development" but **no formal formula has been published**. The motivating case was Hikaru Nakamura retaining a 2780+ rating during a multi-year online-only competitive period.

Implication for this plan: FIDE has historically gone in the same direction we're proposing — μ stays as a *skill estimate*, the leaderboard is what gets filtered — and only now considering decay. This is supporting evidence for Option G over Option E.

Source: [Quora: What happens to FIDE rating if a player doesn't play](https://www.quora.com/What-happens-to-the-FIDE-rating-of-a-player-if-he-doesnt-play-for-a-long-time), [Khel Now on FIDE rating decay](https://khelnow.com/chess/fide-rating-decay-explained-202510).

### 538 NFL Elo — offseason regression

At the start of each NFL season, every team's Elo reverts **one-third of the way toward the mean of 1505**. This is an explicit "inactivity adjustment" implementing partial μ decay — but for the very specific reason that NFL roster turnover (draft, free agency, coaching changes) is genuinely high. **Not directly transferable to individual sports** where the "roster" is one athlete and their off-season skill change is much smaller than a team's.

Per-game K-factor = 20; games at end-of-season weight more.

Source: [How Our NFL Predictions Work (FiveThirtyEight)](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/).

### TrueSkill — Bayesian σ that grows with inactivity in theory

TrueSkill ([Microsoft Research](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/), [TrueSkill 2 paper](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)) models skill as a Gaussian (μ, σ); σ shrinks with consistent play and is supposed to grow during inactivity. In practice, the production implementations don't explicitly decay σ during inactivity — that's left as an integration concern.

TrueSkill Through Time is the Microsoft Research extension that explicitly tracks skill over time. It's closer in spirit to WHR/ILSR than to a real-time Elo system.

### Olympic combined formats — no participation bonus precedent

The Olympic combined format (Tokyo 2020 boulder+lead+speed combined → Paris 2024 boulder+lead) uses **multiplicative scoring** (product of per-discipline ranks) without any participation credit. The "field strength" credit is implicit in the format being Olympic — there's no per-discipline tier-weighting at the Olympics themselves. So there's no direct precedent for tier-weighted participation bonuses in Olympic climbing scoring; TPB would be a model-side decision specific to climbing-elo.

---

## Integration points (concrete edits — IF implemented)

### Gap 1 — TPB

1. **`src/climbing_elo/engine/elo.py:128-204`** — Add to `EloConfig`:
   ```
   tpb_base_budget: float = 100.0          # μ-points pooled per Olympic event
   tpb_tier_weights: dict[EventTier,float] # olympics=1.0, wch=0.75, wc=0.5, cont=0.25
   tpb_rank_lambda: float = 0.15           # exponential decay per-rank inside event
   tpb_enabled: bool = True                # feature flag
   ```
   These would live alongside `glicko2_tau` etc. so a future #80 grid search can tune them without monkey-patching.

2. **`src/climbing_elo/engine/backfill.py:247-279`** — New post-event step. After the `event_had_updates` block but before `session.commit()`, compute `tpb[athlete_id]` for the event, apply it to `db_rating.mu` (and to in-memory `ar.mu`), and write a new `RatingHistory` row with `event_id=event.id` and a synthetic `round_id=NULL` plus a `kind='tpb'` tag. **Schema change required**: add `RatingHistory.kind: str = 'pairwise'` column (nullable, default `'pairwise'`).

3. **`src/climbing_elo/models.py:176-196`** — Add `RatingHistory.kind: Mapped[str | None]` column. Migration: ALTER TABLE rating_history ADD COLUMN kind TEXT DEFAULT 'pairwise'. Cheap on PG.

4. **`src/climbing_elo/api/routes.py:529-720`** — `/breakdown/{athlete_id}/{event_id}` template: show a "Participation credit" line if a TPB rating-history row exists for that (athlete, event). Display the tier weight, finish rank, and computed bonus. Helps the explainability story.

5. **`tests/test_elo.py`** — New test `test_tpb_zero_sum_across_event` asserting `sum(tpb[a] for a in event.athletes) ≈ 0`. New test `test_tpb_olympics_outscales_continental` asserting rank-1 at Olympics gets ≥ 4× the TPB of rank-1 at Continental for the same field size.

6. **`tests/test_backfill.py`** — Extend `test_backfill_3_event_integration` to assert TPB rows are written per event when `tpb_enabled=True`.

7. **Migration**: re-run `scripts/run_backfill.py --from-scratch` (the same flag that #51's plan introduced) on a staging DB, copy to prod atomically. Total walltime ≈ 30s.

### Gap 2 — Dual-view + smart classifier

8. **`src/climbing_elo/models.py:55-72`** — Add `Athlete.retired_at: Mapped[date | None]` column. Migration: ALTER TABLE athletes ADD COLUMN retired_at DATE.

9. **`src/climbing_elo/engine/activity.py`** (NEW file, ~80 lines) — Module exporting:
   - `is_likely_retired(rating, today, recent_history=None) -> bool`
   - `is_active(rating, today, window_days=365) -> bool`
   - `LeaderboardView` enum: `ACTIVE`, `ALL_TIME`, `LEGACY`.
   - Optional: an in-memory `TTLCache` (1-hour, mirroring `predictions_cache`) keyed on `(athlete_id, today)` to avoid recomputing the trajectory check per request.

10. **`src/climbing_elo/api/routes.py:169-193`** — Update `_get_rankings_v2` to accept a `view: LeaderboardView = ACTIVE` parameter and filter the SQL accordingly:
    - `ACTIVE`: `WHERE last_event_at >= today - INTERVAL '12 months' AND retired_at IS NULL`
    - `ALL_TIME`: `WHERE NOT is_likely_retired(...)` — push to Python post-filter on the top-200 SQL slab to keep SQL portable. ~hundreds of rows post-SQL, cheap.
    - `LEGACY`: no filter (escape hatch for fans who want the raw historical view; not advertised in nav).

11. **`src/climbing_elo/api/routes.py:377-418`** — `/leaderboard` route: accept a `view` query param defaulting to `active`. Wire through to `_get_rankings_v2`.

12. **`src/climbing_elo/templates/leaderboard.html:51-73`** — Add a third pill row (alongside discipline + gender) with three pills: Active / All-time / Legacy. Update the JS scroll-to-top logic if needed. UX copy: "Active = competed in the last 12 months. All-time = excludes athletes who appear to have retired."

13. **`src/climbing_elo/api/v1_routes.py:173-232`** — `/api/v1/leaderboard`: add `view: str = "active"` query param with the same three values. Document in OpenAPI. Backward-compatible if we make the default `legacy` (matches today's behaviour) and have the HTML default to `active`. *Recommended:* keep API default = `legacy` for backward compatibility, force HTML default = `active`. This is a UX decision — flag it as Open Q3.

14. **`src/climbing_elo/templates/leaderboard.html:86-103`** — Optionally show a small "RETIRED" badge next to the athlete name when `is_likely_retired` is True and we're in `LEGACY` view. Helps fans recognize which athletes have been computed as retired.

15. **`tests/test_activity.py`** (NEW) — Cover `is_likely_retired` against fixtures: Garnbret-on-break (false), Coxsey-3y-gap (true), Klingler-2y-with-trajectory (true), Anraku-active (false). Cover the LeaderboardView filtering in `_get_rankings_v2`.

16. **`tests/test_api.py`** — Add coverage for `?view=active|all_time|legacy` against `/api/v1/leaderboard` (count + ordering invariants).

17. **`scripts/clear_cache.py`** — If we add the `is_likely_retired` cache, flush it alongside `predictions_cache` and `likely_roster_cache`.

---

## Test plan

### Unit tests (Gap 1 — TPB)

- `test_tpb_zero_sum_across_event` — for each event in `tests/fixtures/`, assert `abs(sum(tpb[a])) < 1e-6`.
- `test_tpb_tier_ordering` — same field, same rank: Olympics TPB[rank 1] > WCh TPB[rank 1] > WC TPB[rank 1] > Continental TPB[rank 1].
- `test_tpb_rank_decay` — within one event, TPB is monotonically decreasing in rank.
- `test_tpb_disabled_flag` — when `EloConfig(tpb_enabled=False)`, backfill produces the same μ values as today (regression guard).

### Unit tests (Gap 2 — Classifier)

- `test_is_likely_retired_gap_only` — synthetic rating with `last_event_at = today - 4y` → True.
- `test_is_likely_retired_recent_active` — synthetic rating with `last_event_at = today - 6m` → False.
- `test_is_likely_retired_on_break` — `last_event_at = today - 14m`, no declining trajectory → False (lets Garnbret pass).
- `test_is_likely_retired_declining` — `last_event_at = today - 27m`, last 3 μ-deltas all negative → True (catches early-retirement).
- `test_is_likely_retired_manual_override` — `Athlete.retired_at = today - 1d` → True regardless of `last_event_at`.

### Integration tests

- `test_leaderboard_view_active_filters_ghosts` — using a real-data snapshot, assert that Sachi AMMA is NOT in the men's lead top-10 under `view=active` but IS under `view=legacy`.
- `test_leaderboard_view_all_time_keeps_on_break` — assert that Tomoa NARASAKI (lead, last 2024) IS in the men's lead top-30 under `view=all_time`.
- `test_leaderboard_view_all_time_excludes_retired` — assert that Akiyo NOGUCHI (officially retired 2021) is NOT in any view ≠ legacy.

### Backtest impact

- TPB is zero-sum and tier-deterministic, so it can change individual μ-trajectories but should not change the *order* of pairwise predictions on average. Run `scripts/run_backtest.py --variant tpb` and confirm aggregate `log_loss_podium` does NOT regress by >1% vs baseline.
- Dual-view does not touch μ; backtest is unaffected. Skip.

### The "feels right" challenge

This is the hardest acceptance criterion to formalize. Log-loss won't move much for either gap (TPB doesn't change pairwise expectations; dual-view doesn't change μ at all). Propose three "feels right" checkpoints:

1. **Top-10 reasonableness audit.** A human checks the men's/women's lead/boulder top-10 under `view=active` and rates each entry: "obviously belongs there" / "questionable" / "clearly wrong". Acceptance: ≤1 questionable per top-10, 0 clearly wrong.
2. **Comparative ranking screenshot.** Generate `before/` (current leaderboard) and `after/` (proposed leaderboard) HTML for the 4 disciplines × 2 genders. Have Milan eyeball and approve.
3. **Anti-regression check on big events.** When a major event lands, the post-event leaderboard should make sense (e.g. the Olympic gold medallist must end the season top-3 in the relevant discipline). Add a synthetic test: simulate the Tokyo 2020 boulder event with current data + TPB enabled, assert podium athletes finish the season top-5.

### Migration / rollback

- TPB: feature-flag in `EloConfig.tpb_enabled`. If we ship it on and then want to roll back, re-run backfill with `tpb_enabled=False`. Walltime ~30s.
- Dual-view: no backfill needed — it's a display filter. Toggle by changing the default URL param.
- Classifier: cache flush via `scripts/clear_cache.py`.

---

## Effort breakdown

Total: **≈ 4 working days** (4d, plus 0.5d buffer = 4.5d). Both gaps can ship as **two independent PRs** in series.

### Gap 1 — TPB (≈1.5 days)

| # | Task | Hours |
|---|---|---|
| 1 | Add `tpb_*` fields to `EloConfig` + module-level defaults | 0.5 |
| 2 | Implement `compute_tpb_credits(event, results) -> dict[athlete_id, float]` in `engine/elo.py` | 2 |
| 3 | Wire TPB into `backfill.run_backfill` post-event-update block + new `RatingHistory.kind` migration | 2 |
| 4 | New unit tests in `test_elo.py` (zero-sum, tier ordering, rank decay) | 1.5 |
| 5 | Extend `test_backfill.py` integration test with TPB enabled assertion | 1 |
| 6 | `/breakdown/...` template update to show participation-credit line | 1 |
| 7 | Backtest regression check (run + diff) | 1 |
| 8 | Documentation update (CLAUDE.md "ELO Engine Specifics" + new TPB section) | 0.5 |
| 9 | Migration rehearsal on staging snapshot | 0.5 |
| **Subtotal** | | **10h (1.25d)** |

### Gap 2 — Dual-view + classifier (≈1.5 days)

| # | Task | Hours |
|---|---|---|
| 10 | New `Athlete.retired_at` column migration | 0.5 |
| 11 | New `engine/activity.py` module with classifier + `LeaderboardView` enum | 2 |
| 12 | Update `_get_rankings_v2` to accept `view` param + SQL/Python filter | 1.5 |
| 13 | Update `/leaderboard` HTML route + template (pill row) | 1.5 |
| 14 | Update `/api/v1/leaderboard` + REST tests | 1.5 |
| 15 | New `tests/test_activity.py` with classifier fixtures | 2 |
| 16 | Smoke-test pass + UX copy review | 1 |
| 17 | Documentation (CLAUDE.md + plan-status update) | 0.5 |
| **Subtotal** | | **10.5h (1.3d)** |

### Cross-gap + buffer (≈1 day)

| # | Task | Hours |
|---|---|---|
| 18 | Combined integration test (TPB + view filter coexist correctly) | 1 |
| 19 | End-to-end smoke (`scripts/smoke_test.py`) verifying new view works on all HTML routes | 1 |
| 20 | Cmux screenshot pass for top-30 before/after comparison | 1 |
| 21 | Buffer for review + iteration | 4 |
| **Subtotal** | | **7h (0.9d)** |

---

## Open questions

### Open Q1 — Should TPB write to a separate `RatingHistory.kind='tpb'` row or mutate the per-round rows?

**The choice:** (a) one synthetic RatingHistory per (athlete, event) with `kind='tpb'`, `round_id=NULL`; or (b) fold the TPB delta into the final round's `mu_after` and don't write a separate row.

**Tentative recommendation:** (a). Easier to audit, easier to disable retroactively, doesn't conflate two semantically different updates (pairwise outcome vs participation credit) in one row. Requires the `RatingHistory.kind` column + a nullable `round_id` (currently NOT NULL).

**What would let Milan decide for sure:** look at how the `/breakdown/...` page would render under each option. (a) means an extra section labeled "Tournament participation credit"; (b) means the existing pair list is missing some μ to account for the round's mu_after. (a) is the more transparent option.

### Open Q2 — Year-of-birth enrichment: prerequisite to "age-aware" classifier, or independent?

**The choice:** ship the dual-view *now* with the year-of-birth-free heuristic, or wait until the scraper is enriched.

**Tentative recommendation:** ship now. The simple 3-year-gap + manual-override heuristic catches all the clear retirees in production data. Age-awareness would let us catch e.g. Petra KLINGLER (2-year gap at age 36) earlier than the 3-year rule does, but the manual `retired_at` column is the right escape valve for those edge cases.

**What would let Milan decide for sure:** count the false-negatives under the simple heuristic. If Klingler-style cases exceed 5% of top-100, the age enrichment is worth the extra 0.5 day.

### Open Q3 — Default `view` for the REST `/api/v1/leaderboard` endpoint

**The choice:** does `view=active` become the default for `/api/v1/leaderboard`, or does the REST API stay backward-compatible with `view=legacy` while the HTML page defaults to `active`?

**Tentative recommendation:** REST stays `view=legacy` (backward compat for any external clients), HTML defaults to `view=active` (sane default for the people using the dashboard).

**What would let Milan decide for sure:** count known consumers of `/api/v1/leaderboard`. If there are none, default both to `active`.

### Open Q4 — Should "likely retired" badges show in the UI?

**The choice:** in `view=legacy` (the escape hatch), should an athlete classified as retired get a small "Retired" pill next to their name?

**Tentative recommendation:** yes, with a tooltip explaining the heuristic. Helps users understand why an athlete is in `legacy` but not `all_time`. Important: never show this on the official leaderboard view (`active`).

**What would let Milan decide for sure:** mock the UI in the design handoff directory and eyeball.

### Open Q5 — TPB tunables (BUDGET, LAMBDA) — start with proposed values or grid search?

**The choice:** the recommended starting point is `TPB_BASE_BUDGET = 100.0, LAMBDA = 0.15`. These are intuition-based — should we instead grid search to maximize some calibration metric?

**Tentative recommendation:** ship with the proposed values first. TPB is zero-sum so it doesn't change pairwise prediction quality — backtest can't really pick a winner. Tune for "feels right" against the top-10 reasonableness audit instead.

**What would let Milan decide for sure:** if the "feels right" audit fails (e.g. WC podiums end up indistinguishable from Continental podiums under the chosen LAMBDA), we expose the params in `EloConfig` so a tune script can sweep them.

### Open Q6 — Does `retired_at` need a UI for manual entry?

**The choice:** the `Athlete.retired_at` column is the manual override. How does it get populated?

**Tentative recommendation:** initially via a SQL one-shot. For ongoing maintenance, add a simple admin endpoint (auth gated) — or simply commit a `data/retirements.yml` file and have `scripts/sync_retirements.py` reconcile to DB. The YAML approach is auditable in git.

**What would let Milan decide for sure:** count expected retirements per year. If <10/yr (likely), a YAML + script suffices. If higher, an admin UI is worth building.

---

## Out of scope

Explicitly **not** in this plan:

1. **μ decay (Option E).** Recommended against; if a future ticket wants it, this plan flags the calibration constraint.
2. **Recency-weighted scoring (Option H).** Belongs to a WHR/ILSR rewrite (#52), not a display-layer enhancement.
3. **Parallel ATP-style points system (Option C/D).** Adding a second ranking metric duplicates the user's mental model; conflicts with the "fans get confused" framing.
4. **Field-quality MOV multiplier (Option B).** Would require yet another K-factor / MOV re-tune sweep; TPB achieves most of the user-visible improvement with much less risk.
5. **Year-of-birth scraper enrichment.** Filed as a separate prerequisite issue. The dual-view classifier degrades gracefully without it.
6. **`retired_at` admin UI.** YAML-and-script is the proposed v1; build the UI when retirement volume warrants.
7. **Smart "likely retired" trajectory analysis using deep learning / clustering.** Stick with the rule-based heuristic; revisit if false-positive/false-negative rates are unacceptable.
8. **Per-discipline TPB tuning.** Initial implementation uses one global `tpb_tier_weights` table; per-discipline tuning is a follow-up if the "feels right" audit shows discipline-specific mismatches.
9. **Backend cache invalidation hooks for the new `is_likely_retired` cache.** Reuse `predictions_cache` flush pattern; full hook system out of scope.
10. **Backfilling `RatingHistory.kind='tpb'` rows for historical events.** This happens automatically when we re-run backfill from scratch under the new engine; no separate backfill needed.

---

*End of plan. Updates should preserve the numbered integration points (1-17) for stable cross-references with PR review comments.*
