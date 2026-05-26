# Climbing ELO — Rating System Research & Improvement Plan

**Status:** Research synthesis, May 2026 (revised after CLAUDE.md update revealing that the Boulder+Lead geometric-mean composite, Monte Carlo projections engine, Speed P-L wiring, and v1 REST API are already implemented).
**Audience:** Project owner or a future agent picking up rating-engine work.
**Companion files:** `engine/elo.py`, `engine/backfill.py`, `models.py`, `scripts/run_backtest.py`.

This document combines a deep read of the current implementation, an academic-literature pass on rating-system methodology, and IFSC-domain specifics into a single ranked improvement plan. Section 1 is the TL;DR; sections 2–4 are the evidence; sections 5–6 are the recommendations and open questions.

---

## 1. Executive Summary

The current engine is well-tuned for what it is — Plackett-Luce pairwise decomposition with grid-searched K-factors hitting 87.5% podium prediction on a 2025–26 holdout, with a Boulder+Lead geometric-mean composite, Monte Carlo projection probabilities, and a v1 REST API already shipped. The four highest-leverage improvements, in order:

0. **Expand the backtest harness** to use proper scoring rules (log-loss, Brier, calibration) on the existing Monte Carlo projection outputs, stratified by athlete tenure / event tier / round / discipline / season. Prerequisite to everything else — without it, we can't tell which of the changes below actually help.
1. **Make σ a real Glicko-2 RD inside the update**, not just cosmetic decay. This single change retires the ad-hoc 2× provisional-K multiplier, fixes the cold-start / "rating burn-in" problem for emerging athletes, and handles sabbatical returns (Garnbret, Raboutou, Ondra) without further tuning.
2. **Add a Whole-History Rating (WHR) batch refit as the canonical historical pass.** The literature is consistent: for our data size (~14 years × ~50 events × ~80 athletes), batch Bayesian methods dominate incremental Elo on every published prediction benchmark. Keep the current engine for live updates; WHR becomes the authoritative leaderboard after each season.
3. **Audit the margin-of-victory multiplier against G-Elo (Szczecinski 2022).** Our current MOV formula is ad-hoc and unconditioned on rating gap, which the 538 NFL/NBA experience shows induces autocorrelation drift at the top. G-Elo gives a principled cumulative-link MOV model with proper-scoring-rule guarantees.

Tier-2: learned composite weights replacing the fixed geometric-mean Boulder+Lead aggregate (relevant for Olympic combined-format prediction); validation against AscentStats' independently-computed Bayesian Bradley-Terry ratings as a cheap external sanity check; bracket-native Speed model replacing the current P-L-on-time approximation.

Tier-3 (skip): TrueSkill / TrueSkill 2 migration; neural rating systems; score-based Gaussian likelihoods for Boulder.

---

## 2. Current Implementation Snapshot

### 2.1 ELO math (`engine/elo.py`)

K-factor table tiered by `EventTier × RoundType`, recently scaled 2× via grid search:

```
OLYMPICS:          Final 96   Semi 72   Qual 36
WORLD_CHAMPIONSHIP: Final 80   Semi 60   Qual 30
WORLD_CUP:         Final 64   Semi 48   Qual 24
CONTINENTAL:       Final 48   Semi 36   Qual 18
PROVISIONAL_K_MULTIPLIER: 2.0× for athletes with n_events < 3
```

Standard logistic expected-score formula:
`E(A,B) = 1 / (1 + 10^((μ_B - μ_A) / 400))`

**σ is computed but not used in the rating update.** It's pure bookkeeping: 350 initial, exponential decay toward 350 during inactivity with 540-day half-life, multiplicative 0.98× convergence per event, bounded `[50, 350]`. Confidence display only — does not modulate K, does not enter the expected score.

**Pairwise decomposition:** Each round of n athletes generates n(n-1)/2 pairwise contests with field-size normalization `pair_k = base_k / (n - 1)`. This keeps total rating movement per round roughly constant across field sizes. Zero-sum is maintained to <0.0001 in tests.

**Margin multiplier:** Capped at 1.5×, applied symmetrically to winner and loser deltas. Lead uses score gap with `max_gap = 20` (hold count); Boulder uses `max_gap = 1000` against the normalized score `tops × 1000 + zones × 100 - top_attempts × 10 - zone_attempts`. DNF athletes get `margin_mult = 1.0`.

### 2.2 Backfill (`engine/backfill.py`)

Events processed chronologically; rounds within event ordered qual → semi → final. Per-event atomic commit. `n_events` increments once per event (not per round). Provisional flag flips after 3 events.

### 2.3 Discipline handling

`Discipline` enum: `L`, `B`, `S`, `BL`. Lead, Boulder, and Speed are all wired into the pairwise P-L engine with discipline-specific score normalization. Speed uses time in seconds with `SPEED_MAX_GAP_SECONDS=2.0` as the margin scale, treating each round as a P-L finishing order — pragmatic, but ignores the genuine single-elimination bracket structure of the sport. `Discipline.BOULDER_LEAD` ratings are populated by `scripts/compute_combined_ratings.py` as a **geometric mean** `sqrt(μ_boulder × μ_lead)` for athletes with ≥3 events in both disciplines; σ combines via RMS. The geometric mean penalizes specialists and rewards all-rounders, matching the Olympic combined format.

### 2.4 Projections and public API

**Projections engine (`engine/projections.py`).** Monte Carlo outcome prediction: `compute_podium_probabilities(athletes, n_simulations=10_000)` draws N(μ, σ) per simulation, ranks athletes, and tallies win/podium/top-8 fractions. 10k sims for 20 athletes runs in ~15ms (numpy-vectorized). **Probabilistic forecasts are already a first-class output** — proper scoring rules (log-loss, Brier, calibration) can be applied to backtesting without additional engineering. Athletes with no rating for a discipline receive defaults (μ=1500, σ=350).

**Public REST API (v1).** Read-only, no-auth endpoints under `/api/v1/` (`api/v1_routes.py`, `api/schemas.py`): leaderboard, athlete profile + history, event list + details. External validation work (e.g., comparing to AscentStats) can be done against a stable interface, and future evaluation tooling can consume the same surface human users see.

### 2.5 Known gaps from code inspection

- **σ doesn't feed back into updates** — biggest single architectural gap.
- **Margin multiplier is unconditioned on rating gap** — pure score gap, so an upset by 15 holds is rewarded identically to a favorite winning by 15 holds. This is the autocorrelation-drift hazard.
- **No pre-2025 Boulder scoring translation** despite the IFSC's mid-2025 system change (old tops/zones/attempts → new 25/10/–0.1 cardinal). Old data is currently re-normalized through the new-format margin function, which is a silent mismatch.
- **Boulder + Lead aggregate uses a fixed geometric mean**, not weights learned against the actual prediction target (Olympic combined-format log-loss).
- **Speed uses pairwise P-L on time normalization** rather than the bracket-native pairwise structure the sport actually has.
- **Backtest validates a single metric (podium hit-rate)** without log-loss, Brier, calibration, or stratification by tenure / tier / round / discipline / season. The Monte Carlo projection engine produces probability distributions that the harness does not yet score.
- **Tests cover zero-sum, reproducibility, projections invariants, and the BL composite** but not σ-decay correctness, numerical-accumulation stability, large-field (>32) scaling, or cold-start trajectory shape.

---

## 3. Methods Landscape (Academic Read)

What the rating-system literature actually says, ordered by relevance to our problem.

### 3.1 Foundational facts (settled)

**The choice of link function does not matter empirically.** Stern (1992, *Mathematical Social Sciences* 23) showed that gamma-distributed performance models — with Bradley-Terry (logistic) and Thurstone (probit) as limiting cases — fit sports data nearly identically. Do not waste effort on logistic-vs-probit; do focus on the rating-update *structure* (online vs batch, MOV vs rank, dynamic vs static).

**Pairwise decomposition of a ranked finish is *approximately* Plackett-Luce MLE, not identical.** Plackett (1975, *Applied Statistics* 24) gives the canonical ranking likelihood:
`P(π) = Π_i v_{π_i} / (v_{π_i} + v_{π_{i+1}} + ... + v_{π_r})`
The score function of pairwise B-T fits on a full ranking converges to the P-L score for moderate field sizes, but is biased for large fields. Practical implication: our 80-athlete qualification rounds are where the approximation is weakest. A proper P-L MLE (Hunter 2004 MM algorithm, or Maystre & Grossglauser 2015 ILSR) would be more accurate.

**Comparisons within an event are not independent.** Cattelan (2012, *Statistical Science* 27) — athletes facing the same boulder set share information; treating their pairwise contests as i.i.d. overstates evidence and biases standard errors downward. Pairwise composite likelihood with sandwich-variance estimation is the recommended fix. **Not in our code; arguably explains why grid-searched K-factors are needed at all** — we're correcting variance miscalibration empirically.

### 3.2 Sequential / online methods

**Glicko / Glicko-2 (Glickman 1999, 2013).** RD (rating deviation, our σ) modulates K-factor and the opponent's RD enters the expected-score gradient via `g(RD_j) = 1/√(1 + 3·RD_j²/π²)`. Three concrete consequences:

- A high-RD athlete's K is effectively large (they move fast).
- Beating a high-RD opponent moves *you* less (you don't trust the signal yet).
- RD inflates during inactivity with a known Wiener kernel — no hand-tuned "rust" multiplier needed.

Glicko-2 adds a per-player volatility τ for genuinely erratic form. For elite climbers competing 6–12 events/year, this is unlikely to add much over Glicko-1. The valuable upgrade is *integrating RD into the update at all* — we currently get zero benefit from σ in updates.

**TrueSkill / TrueSkill 2 (Herbrich 2007, Minka 2018).** Same Thurstonian family as P-L, fit by expectation propagation over a factor graph. Handles multi-player free-for-all natively. TrueSkill 2's gains are FPS-specific (squad bonuses, kill counts) — irrelevant to climbing. **Not worth migrating to**, but it confirms the multi-player ranked-finish setting is a solved problem class.

**OpenSkill / Weng-Lin (2011, *JMLR* 12).** Closed-form Gaussian approximations for multi-player ranking. Lighter-weight TrueSkill that natively handles ranks. **Probably the closest off-the-shelf match for our problem** if we want an upgrade without writing custom math — multi-athlete rounds with explicit ranks, online updates, Bayesian uncertainty intervals included.

### 3.3 Batch / full-history methods

**Whole-History Rating, Coulom 2008.** Computes the MAP of *all* players' rating trajectories jointly. Per-player time series is Newton's method on the Bradley-Terry likelihood with a Wiener-process prior on rating drift. Sub-second convergence for 10⁴ players × 10⁵ games — our problem fits easily. **Order-invariance is the key property:** a back-dated event correctly updates *all* prior estimates; incremental Elo cannot do this. Beats Elo, Glicko, and TrueSkill on the Go-game prediction benchmark in Coulom's paper.

**TrueSkill Through Time, Dangauthier et al. 2007 (*NeurIPS* 20).** Factor-graph + EP smoothing of the entire history. Conceptually similar to WHR — both fit the full history as a single Bayesian model. Community consensus mildly favors WHR for simplicity. TTT is the better choice only if you want full posterior uncertainty (we might, for the σ display).

**Luce Spectral Ranking, Maystre & Grossglauser 2015 (*NeurIPS*).** Recasts P-L MLE as the stationary distribution of a Markov chain. ILSR (iterative variant) converges in 5–10 steps, O(comparisons) per step. Strictly faster than Hunter MM. If we move to batch, this is the fastest known P-L estimator.

### 3.4 Margin-of-victory

**G-Elo, Szczecinski 2022 (*JQAS* 18, arXiv:2010.11187).** Discretizes MOV into k bins, models bin probabilities with a cumulative-link (ordered-logit) function of skill gap. Update is structurally identical to standard Elo but with bin-specific score functions derived from the proper-scoring-rule of the categorical model. Tested on EPL and NFL; "modest" improvements over rank-only Elo.

**Kovalchik 2020 (*Int. J. Forecasting* 36).** Tested four MOV-Elo variants on ATP tennis: linear, additive, multiplicative, logistic. All improved on rank-only Elo (contradicting a naive reading of Stern 1992), but **only the joint-additive form had stable variance.** Standard Elo plateaus at ~70% prediction accuracy on tennis; MOV variants push 1–3pp higher.

**Settled position:** MOV is genuinely informative beyond rank, *if* you model the link function carefully. Our current `1.0 + gap/max_gap` capped at 1.5× is the unstable kind of MOV multiplier — it adds noise without the conditioning on rating gap that 538's NFL formula uses.

### 3.5 Tied outcomes

**Davidson 1970 (*JASA* 65).** Bradley-Terry with tie parameter ν: `P(i beats j) = v_i / (v_i + v_j + ν√(v_i v_j))`. Directly applicable to Speed climbing, where 0.001s gaps are functionally ties. Not needed for Lead/Boulder — their countback tiebreaks resolve to a strict order.

### 3.6 Climbing-specific academic work

**Villatoro-Paz 2025 (ResearchGate 395113022), *An Adapted Elo Framework for Skill Rating and Probabilistic Forecasting in Elite Bouldering*.** The only peer-published-adjacent academic work on competition climbing rating. **Not retrievable through public fetch** (ResearchGate returns 403 to non-authenticated requests). From the abstract and the author's Medium/Substack: multi-player Elo adapted for ranked-finish bouldering; outputs feed a hierarchical Bayesian "Elo Power Curve" giving P(advance | rating); two-stage forecast `Final = P(Engine) + P(Modifier)` where Engine is a power-curve fit and Modifier is a pairwise dominance score. **The substantive update equations, K-factors, normalization, and validation metrics are not publicly disclosed (patent-pending).** We cannot reimplement against this paper; it does directionally validate the multi-player-Elo-on-ranked-finishes design class.

**AscentStats (ascentstats.com), non-academic but methodologically explicit.** Bayesian dynamic Bradley-Terry MCMC on IFSC bouldering 2008–2026. Reported all-time peaks: Anraku 3092, Garnbret 3039, Grossman 2995, Narasaki 2964, Ondra 2885, Meignan 2700, Bertone 2672. **Use as a free external sanity check on our top-N.**

**Aubin et al. 2021, arXiv:2111.08140.** Bayesian inference of *outdoor route grades*, not competition rating. The dynamic-B-T MCMC machinery is reusable for our problem.

### 3.7 Modern / neural

No published evidence that neural rating systems beat WHR or TTT on prediction benchmarks at our data scale. RNN-based ratings are useful as feature extractors *over* a rating history, not as replacements. **Skip.**

---

## 4. IFSC Domain Factors

Why some methods fit climbing better than they fit chess or tennis.

**Sparse competition.** Top athletes compete 6–12 events/year; sabbatical years are common (post-Olympic blues 2025: Garnbret cut to ~3 events, Raboutou skipped Boulder season, Schubert finger injury). Online incremental methods need K-factor tricks to handle this; Glicko-style RD inflation handles it natively.

**Boulder scoring rule change in 2025.** Old format: ordinal tops/zones/attempts. New format: cardinal 25/zone-10/-0.1 per attempt, 100 max per round. **Score margins are only meaningful within the new format and within a round.** Our current Boulder margin function applies the new-format normalization to pre-2025 data — silent bug.

**Three rounds per event, different fields.** Qualification 60–80 athletes (post-2025: two-route format), semi 24 (down from 26 in 2025), final 8. Field-size normalization `pair_k/(n-1)` handles this correctly.

**Combined Olympic format.** Paris 2024 used additive Boulder + Lead scoring; LA 2028 is expected to repeat. This is a real prediction target, not academic. Per-discipline + learned-weight composite is the right model.

**Disciplines are genuinely different sports.** Lead is endurance + route-reading on a single attempt. Boulder is power + creativity across 4 problems. Speed is a 5-second time trial. A single composite Elo would be wrong; per-discipline with optional composite weighting is correct.

**IFSC's official ranking is points-based with worst-result drops** — known issues: penalizes absentees disproportionately, undifferentiates dominant athletes who max out base points, opaque (the "strength factor" is rarely published). An ELO is a strict improvement on every axis except institutional acceptance.

---

## 5. Recommendations, Ranked

Each recommendation lists payoff, effort, and concrete next steps.

### R0 — Expand the backtest harness

**Payoff:** Foundational. Without finer-grained metrics, we can't tell whether R1–R7 actually help. The current harness reports one number (podium hit-rate ≥ baseline + 15pp). The existing projections engine already produces probability distributions that proper scoring rules can score directly — we're leaving the most informative validation signal on the floor.

**Effort:** ~1–2 weeks.

**Concrete steps:**

1. **Add metrics:**
   - Log-loss and Brier score on `compute_podium_probabilities` outputs (proper scoring rules).
   - Calibration / reliability plots — bucket predicted probability, compare to empirical frequency. "When the model says 70% podium, do they podium ~70% of the time?"
   - Spearman rank correlation between predicted μ-order and actual finish-order.
   - Top-1, top-3, top-8 hit-rates separately.

2. **Add stratifications:**
   - **By athlete tenure** (`n_events` buckets: 1–3, 4–10, 11–30, 30+) — direct cold-start diagnostic.
   - By event tier, round, discipline, season, field size.

3. **Add baselines:**
   - Random; persistence (predict same finish as last event); previous-season IFSC official ranking; stripped-down Elo (no margin / provisional / σ) to isolate the value of each feature.

4. **Add out-of-sample modes:**
   - Walk-forward chronological folds (train through season N, predict N+1).
   - Leave-one-event-out within a season.
   - Leave-one-athlete-out specifically for cold-start measurement — hide athlete X's first N events, measure recovery time.

5. **Add variant harness:** `--variant glicko2`, `--variant whr`, `--variant g_elo`, `--variant bracket_speed`, so each of R1–R7 can be A/B'd against the current engine on the same holdout with the same metric matrix.

6. **Output:** a single JSON or markdown report with the metric × stratification × variant cube, suitable for human review and future automated regression checks.

**Risk:** Scope creep. Build the minimum that scores the existing engine across the metric/stratification matrix, then iterate. Probabilistic metrics (log-loss, Brier, calibration) are the priority since they unlock evaluation of every subsequent recommendation.

### R1 — Glicko-2 RD integration into the update

**Payoff:** Highest. Solves three problems at once: cold-start (replaces the 2× provisional-K cliff), sabbatical returns (Garnbret/Ondra-pattern), and the underused σ field.

**Effort:** ~1 week. Isolated change in `engine/elo.py`. Backfill re-runs to populate new σ trajectories.

**Concrete steps:**
1. Initialize new athletes with `σ = 350`, retire the `n_events < 3 → 2× K` rule.
2. Replace constant K in `calculate_round_updates` with K scaled by current σ: `effective_K = K_base × g(σ_self)` where `g` follows Glickman's formulation.
3. Multiply expected-score gradient by `g(σ_opponent)` — beating a high-σ opponent moves you less.
4. Update σ per Glicko-2's closed-form rule (or the iterative volatility update if we want Glicko-2 specifically).
5. Re-run backfill, re-validate against the same podium-hit-rate holdout. Should not regress.
6. Eyeball check: emerging athletes (Anraku, Roberts, Bertone, Sanders) should climb to top-10 faster.

**Risk:** Variable K may interact with our grid-tuned K-factor table — likely needs re-tuning at a lower base.

### R2 — Whole-History Rating as a batch refit pass

**Payoff:** High. Order-invariant historical ratings; eliminates burn-in entirely for retroactive views; standard academic answer for our problem class.

**Effort:** ~2 weeks. New module (`engine/whr.py`), nightly/weekly job, separate authoritative leaderboard. Online engine continues to serve live updates between passes.

**Concrete steps:**
1. Implement WHR per Coulom 2008: Newton's method on the per-player log-likelihood with Wiener-process σ prior between events. Discipline-specific drift σ is a free parameter to tune.
2. Validate WHR's top-N against AscentStats' published Bayesian-BT top-N (free sanity check).
3. Decide presentation: WHR as the leaderboard, online Elo as "live, may shift at next refit"? Or two columns?
4. Tests: order-invariance (shuffle event order, ratings should match), reproducibility, calibration on holdout.

**Alternative:** ILSR (Maystre & Grossglauser 2015) is faster, fits multi-player P-L directly, lacks the smooth-drift prior. **Probably try ILSR first** — simpler, gets us most of the WHR win, and only fails if temporal drift turns out to dominate the signal.

### R3 — Margin-of-victory audit and G-Elo benchmark

**Payoff:** Moderate. Fixes a latent autocorrelation drift at the top of the leaderboard; aligns with the literature consensus on principled MOV.

**Effort:** ~2–3 days. Self-contained in `engine/elo.py`.

**Concrete steps:**
1. Diagnose: in the current backfill output, do top-10 ratings drift up over multi-year windows holding skill roughly constant? If yes, MOV-induced.
2. Condition the multiplier on rating gap, 538-style: `mov_mult = ln(|score_gap| + 1) × 2.2 / (0.001 × elo_diff + 2.2)`.
3. Optionally implement G-Elo (Szczecinski 2022) as a separate code path and A/B against the simpler conditioned multiplier on the same holdout.
4. Drop the symmetric application of margin to winner and loser — it's a bug (a 50-hold gap doesn't tell us the loser is *that* much worse; it tells us the winner is that much stronger *for this route*).

### R4 — Learned composite weights (replacing the fixed geometric mean)

**Payoff:** Moderate. The current Boulder+Lead aggregate uses a fixed geometric mean (effectively equal weights in log-space) — a sensible default that has not been validated against the actual prediction target. Learned weights tuned to Olympic combined-format log-loss should outperform on the prediction task even if they look similar in rank order.

**Effort:** ~1 week. Modify `scripts/compute_combined_ratings.py` (or a parallel script); weight-tuning loop against R0's backtest harness.

**Concrete steps:**
1. Keep the geometric mean as a baseline variant.
2. Parameterize: either linear `μ_combined = w_L × μ_L + w_B × μ_B` or multiplicative `μ_combined = μ_L^w_L × μ_B^w_B` (the latter generalizes the geometric mean) with `w_L + w_B = 1`.
3. Optimize `w_L`, `w_B` by minimizing log-loss on held-out Olympic + WCh combined-format events.
4. Compare against the current geometric mean using R0's metric matrix. Ship the learned version only if it wins on log-loss without regressing on rank correlation.

### R5 — AscentStats external validation

**Payoff:** Low individual payoff, but free. **Do this before any of R1–R4** — it tells us whether the current model is broken in ways we don't know about.

**Effort:** ~half day.

**Concrete steps:**
1. Scrape or eyeball AscentStats' top-N for Boulder men and women circa 2025.
2. Compare to our current top-N. Acceptable: same set of names, within a few rank positions. Concerning: missing names, large rank inversions.
3. Document any discrepancies as test cases.

### R6 — Bracket-native Speed model

**Payoff:** Speed is currently rated via pairwise P-L on time normalization (`SPEED_MAX_GAP_SECONDS=2.0`). This treats each round as a free-for-all when Speed is actually a single-elimination bracket — athletes face only a subset of opponents per event, not all of them. The structural mismatch likely inflates variance and dilutes the signal from genuine head-to-heads.

**Effort:** ~1–2 weeks.

**Concrete steps:**
1. Replace the per-round P-L decomposition with bracket-aware pairwise updates: only actual head-to-head matchups generate Elo deltas, not all pairwise positions.
2. Two-player Elo as the base, Davidson (1970) tie parameter for sub-0.01s gaps.
3. Time-margin weighting: a 4.58s vs 5.10s race carries information beyond who won — reuse the G-Elo-style conditioned MOV from R3.
4. Validate against R0's harness — bracket-native should improve Speed-stratified log-loss without regressing on aggregate metrics.
5. Re-check structural assumptions for Krakow 2026's 4-lane format before committing.

### R7 — Boulder pre-2025 scoring translation

**Payoff:** Removes a known silent bug.

**Effort:** ~3 days. Translation layer in scraper or normalization pass.

**Concrete steps:**
1. Decide: discard pre-2025 Boulder, or implement a deterministic mapping from old (tops, zones, attempts) → new (25/10/-0.1) scores?
2. If translating, document the mapping in code comments and add tests on a hand-validated event.

### Sequencing recommendation

`R5 (AscentStats validate, ~½ day) → R0 (backtest harness, ~1–2 wk) → R1 (Glicko-2) → R3 (MOV audit) → R2 (WHR/ILSR batch) → R4 (learned composite) → R7 (Boulder translation) → R6 (Speed bracket)`

R5 first: cheapest diagnostic; may surface issues that change priorities. R0 next: without it we cannot measure whether R1–R7 actually help, and the rest of the work is uninterpretable. R1 and R3 are both inside `engine/elo.py` and should be done together since they interact (MOV conditioning depends on the rating-gap term which Glicko-2 affects via expected score). R2 is independent of R1/R3 and parallelizable with them. R6 last because Speed is structurally different and isolated from the Lead/Boulder pipeline.

---

## 6. Open Methodological Questions

The literature does not resolve these. We'd have to decide on engineering judgment.

1. **Composite-rating combination across disciplines.** No academic work on cross-discipline Elo combination; tennis surface-specific Elos use learned weights, but no theoretical justification exists for what the right combination function *should* be.
2. **σ evolution under heterogeneous K.** Glicko-2's RD update assumes a fixed rating-period structure; our K varies by event tier × round. The right closed-form RD update for our setting isn't published.
3. **Tie tolerance for Speed.** Davidson's ν parameter is ML-fit; nobody has done this at 0.001s resolution.
4. **Discipline-specific drift σ for WHR.** Plausibly Speed ratings drift less than Boulder ratings (technique vs creativity), but no published evidence either way.
5. **Whether the "dependent comparisons" correction (Cattelan 2012) materially changes our K-tuning.** Worth a one-shot experiment: re-tune K under composite-likelihood SEs and see if the result differs.

---

## 7. Bibliography

Papers cited with retrievability notes. PDFs marked "not retrieved" are summarized from abstracts and downstream citations; treat those conclusions with appropriate uncertainty.

**Foundational**
- Plackett 1975, *Applied Statistics* 24(2):193-202, DOI 10.2307/2346567 — abstract retrieved.
- Hunter 2004, *Annals of Statistics* 32(1):384-406, DOI 10.1214/aos/1079120141 — content via secondary.
- Cattelan 2012, *Statistical Science* 27(3):412-433, arXiv:1210.1016 — abstract retrieved.
- Stern 1992, *Mathematical Social Sciences* 23(1):103-117 — abstract retrieved via citing papers.

**Sequential / online**
- Glickman 1999, *Applied Statistics* 48(3):377-394, DOI 10.1111/1467-9876.00159 — content via Wikipedia + glicko.net.
- Glickman 2013, *Example of the Glicko-2 system* (technical paper, glicko.net) — retrieved.
- Herbrich, Minka, Graepel 2007, *TrueSkill: A Bayesian Skill Rating System*, MS Research — retrieved.
- Minka et al. 2018, *TrueSkill 2: An improved Bayesian skill rating system*, MSR-TR-2018-8 — retrieved.
- Weng & Lin 2011, *JMLR* 12:267-300 — abstract retrieved.

**Batch / full-history**
- Coulom 2008, *Whole-History Rating*, *Computers and Games* (Springer LNCS) — content via author site + abstract.
- Dangauthier, Herbrich, Minka, Graepel 2007, *TrueSkill Through Time*, *NeurIPS* 20 — partial PDF retrieved.
- Maystre & Grossglauser 2015, *Fast and accurate inference of Plackett-Luce models*, *NeurIPS* — partial PDF retrieved.
- Caron & Doucet 2012, *JCGS* 21(1):174-196, arXiv:1011.1761 — abstract retrieved.

**Margin / ties**
- Szczecinski 2022, *JQAS* 18(1), DOI 10.1515/jqas-2020-0115, arXiv:2010.11187 — content via abstract + Kovalchik.
- Kovalchik 2020, *Int. J. Forecasting* 36(4):1329-1341, DOI 10.1016/j.ijforecast.2020.01.006 — abstract + secondary.
- Davidson 1970, *JASA* 65(329):317-328, DOI 10.1080/01621459.1970.10481082 — abstract retrieved.

**Climbing-specific**
- Villatoro-Paz 2025, *An Adapted Elo Framework for Skill Rating and Probabilistic Forecasting in Elite Bouldering*, ResearchGate publication 395113022 — **paper not retrieved (RG 403)**; content via author Medium + Substack only.
- Villatoro-Paz 2025, *The Competition Paradox*, ResearchGate publication 395524464 — not retrieved.
- Aubin et al. 2021, *Bayesian inference of the climbing grade scale*, arXiv:2111.08140 — retrieved (outdoor grades, not competition).
- AscentStats blog (ascentstats.com) — non-peer-reviewed Bayesian dynamic Bradley-Terry on IFSC bouldering 2008–2026.

**Review articles worth ordering through a library**
- Glickman & Sonas 2024, *Models and Rating Systems for Head-to-Head Competition*, *Annu. Rev. Stat. Appl.*, DOI 10.1146/annurev-statistics-040722-061813 — paywalled, would be the single best survey.

**Industry / blog references (cite cautiously)**
- 538 NBA Elo methodology (fivethirtyeight.com) — origin of the rating-gap-conditioned MOV formula.
- Lichess Glicko-2 rating-period discussion — useful precedent for period-length choice.
- Steven Morse, *Elo as a statistical learning model* — pairwise Elo as SGD on the B-T likelihood.

---

*End of document. Updates should preserve section numbering for stable cross-references.*
