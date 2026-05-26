# PRD: World Climbing ELO Rating & Projection System

**Status:** Draft v0.4 · **Owner:** Product · **Last updated:** 2026-05-24
**Engineering lead:** TBD · **Data science lead:** TBD · **Sport-domain advisor:** TBD

---

## 1. Problem Statement & Goals

### The problem
World Climbing publishes cumulative ranking points across its World Cup season: an athlete's points are the sum of their top N event placings, sometimes with a "drop the worst" rule. This system has three structural weaknesses for fans, coaches, and broadcasters:

1. **Attendance bias.** A mid-tier specialist who attends every event can outrank a stronger athlete who skipped two World Cups for injury or training block.
2. **Field-quality blindness.** Finishing 4th at a stacked World Championship is rewarded identically (or worse) than 4th at a depleted continental cup.
3. **Not predictive.** Ranking points are designed to award the season, not to forecast the next event. They make poor priors for "who wins on Saturday?"

### Why ELO
ELO and its successors (Glicko, TrueSkill, Plackett-Luce) maintain a latent strength estimate that updates based on result *and* opponent quality. They produce naturally calibrated win probabilities, which is the unit of currency for projections, broadcast graphics, and fantasy products.

### User personas & jobs-to-be-done
- **Maya, superfan.** "Before the semis broadcast starts, I want to know which qualifier-round upsets were flukes vs. signals." → Pre-event matchup probabilities and trajectory charts.
- **Jiro, national-team coach.** "I want to know which of my athlete's upcoming-event opponents have been quietly peaking, and which look beatable on current form." → Per-discipline ratings with confidence bands and recent-form deltas.
- **Sam, broadcast producer.** "I need a live podium-probability ticker for the final round, and three pre-show talking points grounded in numbers." → Real-time projection API + pre-canned narrative hooks.

### Goals (12-month)
- Ship publicly queryable ratings for Boulder, Lead, Speed, and Boulder+Lead aggregate
- Beat the official-ranking baseline on backtested podium prediction by ≥15 percentage points
- Get one broadcast partner using the projection API in production

### Non-goals
- Replacing the federation's official ranking (this is an analytical layer, not a governance layer)
- Outdoor climbing performance modeling

---

## 2. Competitive Landscape

| Effort | What it does | Gap we fill |
|---|---|---|
| **climbing-stats.com** | Aggregates and visualizes historical results | No predictive model; descriptive only |
| **Vertical-Life** | Training, community, route logging | Not competition analytics |
| **8a.nu** | Outdoor ascent tracking | Different sport entirely |
| **Federation rankings** | Official cumulative points | Attendance-biased, not predictive |
| **Independent stats Twitter/Substack** | One-off blog analysis | Not productized, no API |

**Partnership candidates:** climbing-stats.com (data complementarity), one national federation as design partner. **Competitor risk:** low today; federation could in-house this within 2–3 years if we don't establish the category.

---

## 3. Data Model

### Core entities
```
Athlete         (id, name, dob, nationality, gender, disciplines[])
Event           (id, name, tier, host_country, season, start_date, end_date,
                 disciplines[], format_notes)
Round           (id, event_id, discipline, round_type {qual|semi|final},
                 athlete_count)
Result          (id, round_id, athlete_id, rank, raw_score, score_max,
                 dnf, dns, withdrawn, qualification_only)
Rating          (athlete_id, discipline {B|L|S|BL}, mu, sigma, n_events,
                 last_event_at, provisional_flag)
RatingHistory   (rating_id, event_id, mu_before, mu_after, sigma_before,
                 sigma_after, contributing_pairs[])
```

The `BL` discipline value designates the Boulder+Lead aggregate as a first-class rating, not a view.

### Source of truth
- **Primary:** the federation's official results pages and PDFs, scraped post-event
- **Secondary:** manual-entry admin UI for results delayed or missing from the primary source
- **Schema-change resilience:** ingestion runs through a validation layer; any parse failure raises an alert, never silently drops data
- **DECISION NEEDED:** Do we negotiate a direct data feed with the federation, or rely on scrape-first in perpetuity? Affects v2 SLAs.

### Normalization
Every result, regardless of discipline, normalizes into a shared shape that the rating engine consumes:
```
{
  round_id, athlete_id, ordinal_rank,
  margin_to_next, score_normalized, dnf, dns
}
```
- Boulder margins use point gaps under the 2025 scoring system
- Lead margins use hold-count differences with "+" treated as +0.5
- Speed produces both a pairwise-bracket representation and a time-based normalized score

**Acceptance criteria:**
- *Given* a federation results PDF, *when* ingested, *then* every athlete in the result list appears in the normalized output with a non-null `ordinal_rank` or an explicit `dnf`/`dns` flag.

---

## 4. Rating Model Design

### Recommendation (not a survey)

**Lead & Boulder: Plackett-Luce update over the full round field.** Each round's finishing order is modeled as a sequence of "the winner beats all remaining athletes, then the runner-up beats all remaining athletes," etc. Pairwise ELO deltas are computed and summed per athlete. Margin (Boulder point gap, Lead hold gap) modulates the K-factor on each pairwise contest — a 50-point Boulder gap counts more than a 1-point gap.

**Speed: standard ELO on head-to-head bracket matches**, with a Bayesian prior derived from qualification-round times (calibrated against the discipline's time→rank relationship).

**K-factor structure:**

| Event tier | Final | Semi | Qualification |
|---|---|---|---|
| Olympics | 48 | 36 | 18 |
| World Championship | 40 | 30 | 15 |
| World Cup | 32 | 24 | 12 |
| Continental Champs | 24 | 18 | 9 |

Provisional athletes (<3 events) use a K-multiplier of 2.0×.

**Time decay:** exponential, half-life 18 months on σ (uncertainty), gently widening confidence bands during inactivity rather than mean-reverting μ.

### Four first-class ratings per athlete
Boulder, Lead, Speed, Boulder+Lead aggregate. The aggregate is a maintained rating with its own update path, history, and confidence band — not a UI sum.

**DECIDED (1): Boulder+Lead aggregate computation — Hybrid.** The aggregate uses a derived blend of the Boulder and Lead sub-ratings as a Bayesian prior, then updates from actual combined-format events (Tokyo '20, Paris '24, Olympic qualifiers, future LA28 qualifier circuits) as they occur. This gives the rating dense default coverage *and* lets athletes who genuinely over- or under-perform in combined formats diverge from the pure derivation over time. Default blend weights start at 0.5/0.5 and are re-fit annually against observed combined-event rank-prediction accuracy.

**DECIDED (2): K-factor calibration — empirical fit with domain-advisor lock.** The K-factor table values are fit by maximizing log-likelihood on backtested results, then reviewed by the sport-domain advisor before being locked for the season. Re-fit is performed annually after the season closes. Sanity checks must flag any K value that swings >50% from the previous season as requiring explicit advisor sign-off.

**DECIDED (3): 2025 Boulder scoring rule change — translation layer + forward re-fit.** Build a translation function that maps pre-2025 (tops, zones, attempts) results to the new 25/10/5 point system for historical continuity. *Separately*, fit margin-weighting parameters on post-2025 data only for forward-going K-calibration, so the new scoring distribution drives the active model without being diluted by translated historical results. The translation function itself is fit on the 2024–2025 subset of results scored under both systems.

### Initialization
- Athletes appear for the first time at μ = 1500 with σ = 350
- Regional/age-group prior nudges μ ±100 based on the athlete's youth-circuit or continental-cup history if available
- First 3 senior events are flagged `provisional` with K-multiplier 2.0×

### Edge cases
- **DNS:** no update; appearance count not incremented
- **DNF mid-round:** ranked at bottom of round; margin weighting capped to prevent outlier-driven swings
- **Withdrawal between rounds:** treated as DNS for subsequent rounds
- **Ties:** athletes split the average rank position; their pairwise contest produces zero net delta
- **Qualification-only appearances:** count toward ratings at qualification K-factor
- **Protest reversals:** trigger a recomputation from the affected event forward; RatingHistory preserves both pre- and post-reversal states

---

## 5. Projection Features (v1)

1. **Pairwise win probability** — P(athlete A beats athlete B) for any pair in any of the 4 ratings
2. **Podium probability** — Monte Carlo (n=10,000) over a given start list, returning P(gold), P(silver), P(bronze), P(podium)
3. **Season-end ranking projection** — using remaining scheduled events
4. **Athlete trajectory chart** — μ over time with σ-derived confidence band

All four queryable against any of the four ratings. The Boulder+Lead aggregate drives combined-format event projections.

**Acceptance criteria:**
- *Given* an upcoming event with a published start list, *when* the projection API is called, *then* podium probabilities sum to 3.0 (±0.001) across the field.
- *Given* two athletes with identical ratings, *when* pairwise probability is requested, *then* the response is 0.5 ± 0.005.

---

## 6. Functional Requirements

### Pipeline stages

| Stage | Input | Output | Latency target | Failure mode |
|---|---|---|---|---|
| Ingestion | Federation results page/PDF | Raw result rows | <30 min post-event publication | Alert on parse failure; queue for manual review |
| Normalization | Raw results | Normalized `(rank, margin, score)` tuples | <5 min | Per-row rejection with logged reason; partial success allowed |
| Rating engine | Normalized results + current ratings | Updated ratings + RatingHistory entries | <2 min for one event | Atomic per-event; rollback on failure |
| Projection service | Start list + ratings | Probabilities, trajectories | <500 ms p95 | Stale-cache fallback up to 24h |
| Query API | HTTP request | JSON response | <200 ms p95 | Standard 5xx with retry-after |

### API surface (v1)
- `GET /athletes/{id}/ratings` — current ratings across all 4 disciplines
- `GET /athletes/{id}/history?discipline=B` — rating timeline
- `GET /matchup?a={id}&b={id}&discipline=L` — pairwise win prob
- `GET /events/{id}/projection` — podium probabilities for upcoming event
- `GET /ratings/{id}/breakdown?event={event_id}` — rating change explainability

---

## 7. Non-Functional Requirements

- **Accuracy.** On 2022–2024 backtests, podium hit-rate (correctly predicting any podium finisher in top-3 ranked predictions) must exceed official-ranking baseline by ≥15 pp per discipline. Reported separately for Boulder, Lead, Speed, Boulder+Lead.
- **Explainability.** Every rating change must be attributable to specific pairwise match-ups within an event. `GET /ratings/{id}/breakdown` returns the contributing-pairs list with per-pair Δμ.
- **Reproducibility.** Re-running the full pipeline on identical input data must produce byte-identical ratings. Implies deterministic RNG seeding for Monte Carlo projections (seeds stored per projection request).
- **Scale.** ~500 active athletes × 4 ratings × ~10 yrs history → manageable on a single Postgres instance. Projection service horizontally scalable behind a load balancer.

---

## 8. Phasing

| Phase | Scope | Why |
|---|---|---|
| **MVP (v0.1)** | Lead ELO only, batch updates, internal dashboard | Fastest path to validate model quality on the discipline with the cleanest scoring; no API exposure risk |
| **v1** | All four ratings (B, L, S, B+L aggregate), public read API, podium projections | The full product story |
| **v2+** | Live in-event updates, fantasy-league hooks, athlete-trajectory charts with confidence bands, broadcast graphics SDK | Once accuracy is proven |

---

## 9. Success Metrics

**Leading (model quality):**
- Podium prediction hit-rate vs. official-ranking baseline (target: +15 pp)
- Log-loss on pairwise predictions (target: <0.55 across all disciplines)
- Rating stability — σ for athletes with ≥10 events should converge below 80 within 2 seasons

**Lagging (product usage):**
- Monthly active queries to public API (target: 100k by month 6 of v1)
- Number of broadcast/media partners using the API (target: 1 by month 12)
- Press citations of model predictions per quarter (target: TBD pending baseline)

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Federation results feed instability | M | H | Manual-entry fallback; alert on parse failure; multiple-source cross-check |
| Overfitting K-factors to small history | M | M | Hold out one season for true OOS validation; refit annually |
| 2025 Boulder scoring rule change disrupts backfill | H | M | Translation layer for pre-2025 results; separate K-fitting eras |
| Sparse combined-format data weakens aggregate rating | H | M | Hybrid model (§4 Decision 1) anchors aggregate via derived prior even when combined events absent |
| Athlete privacy / likeness concerns | L | M | Public data only; opt-out mechanism for non-public profile elements |
| Federation builds competing in-house product | L | H | Build broadcast/coach distribution lock-in; offer partnership |

---

## 11. Out of Scope (v1)

- Youth and Junior circuits
- Paraclimbing
- National-level domestic competitions
- Betting and odds integrations
- Outdoor (non-comp) climbing performance modeling
- Route-setter difficulty modeling (potentially relevant for v3+)

---

## 12. Open Questions Requiring Domain Expert Input

- **Paris 2024 vs. LA 2028 format weighting.** Paris ran Boulder+Lead combined; LA will split them. How should historical combined results be weighted for the separated-format era projections? *Default assumption:* combined results inform the B+L aggregate at full weight, and the separate B and L ratings at 0.5× K (since athletes are pacing across two disciplines).
- **Minimum events before "published."** At what point does a provisional rating get exposed publicly? *Default:* 3 senior international events.
- **Cross-gender rating space.** Maintain separate male/female rating pools (recommended) or shared with a fitted offset? *Default:* separate pools; never compute cross-gender matchups in v1.
- **Format weight asymmetry within Speed.** Should bracket round wins count more than qualification time-trials, given the head-to-head is the "real" race? *Default:* bracket K-factor 2× qualification K-factor.
- **2025 Boulder rule translation function.** What's the empirical mapping from (tops, zones, attempts) to the new 25/10/5 point system? *Default:* fit on the subset of 2024–2025 results that were scored under both systems for calibration.

---

# Appendix

## A. Example Event JSON Schema (ingested form)

```json
{
  "event_id": "wc_innsbruck_2026",
  "name": "IFSC World Cup Innsbruck 2026",
  "tier": "world_cup",
  "host_country": "AUT",
  "season": 2026,
  "start_date": "2026-06-12",
  "end_date": "2026-06-14",
  "disciplines": ["boulder", "lead"],
  "format_notes": null,
  "rounds": [
    {
      "round_id": "wc_innsbruck_2026_b_final_m",
      "discipline": "boulder",
      "round_type": "final",
      "gender": "M",
      "athlete_count": 8,
      "results": [
        {
          "athlete_id": "ondra_a",
          "rank": 1,
          "raw_score": {"tops": 3, "zones": 4, "points": 89.4},
          "score_max": 100.0,
          "dnf": false, "dns": false,
          "withdrawn": false, "qualification_only": false
        }
      ]
    }
  ]
}
```

## B. Example Rating Change Breakdown Response

```json
GET /ratings/ondra_a/breakdown?event=wc_innsbruck_2026&discipline=B

{
  "athlete_id": "ondra_a",
  "event_id": "wc_innsbruck_2026",
  "discipline": "B",
  "mu_before": 1742.3,
  "mu_after": 1761.8,
  "delta_mu": 19.5,
  "sigma_before": 92.0,
  "sigma_after": 88.4,
  "k_effective": 32,
  "contributing_pairs": [
    {"opponent": "schubert_j", "result": "won", "expected": 0.54,
     "actual": 1.0, "delta": 14.7, "margin_multiplier": 1.0},
    {"opponent": "narasaki_t", "result": "won", "expected": 0.61,
     "actual": 1.0, "delta": 12.5, "margin_multiplier": 1.0},
    {"opponent": "anraku_s", "result": "lost", "expected": 0.49,
     "actual": 0.0, "delta": -15.7, "margin_multiplier": 1.0},
    "..."
  ]
}
```

## C. Worked Numerical Example — Boulder Final

8-athlete final. Pre-event μ values:

| Rank pre | Athlete | μ_before |
|---|---|---|
| 1 | A | 1750 |
| 2 | B | 1700 |
| 3 | C | 1680 |
| 4 | D | 1650 |
| 5 | E | 1620 |
| 6 | F | 1600 |
| 7 | G | 1570 |
| 8 | H | 1540 |

Final result: **B, A, D, C, E, F, H, G** (an upset; B wins over A, H jumps G).

K-factor for World Cup final = 32. Margin multipliers = 1.0 for simplicity.

**Sample pairwise contest (B vs. A, B wins):**
- Expected_B = 1 / (1 + 10^((1750-1700)/400)) = 1 / (1 + 10^0.125) = **0.429**
- Δμ_B from this contest = 32 × (1 − 0.429) = **+18.3**
- Δμ_A from this contest = 32 × (0 − 0.571) = **−18.3**

Applying the full Plackett-Luce decomposition (summing each athlete's pairwise contests against all athletes they finished ahead of, minus all athletes they finished behind):

| Athlete | μ_before | Net Δμ | μ_after |
|---|---|---|---|
| B (1st) | 1700 | +52.1 | 1752.1 |
| A (2nd) | 1750 | +1.4  | 1751.4 |
| D (3rd) | 1650 | +21.0 | 1671.0 |
| C (4th) | 1680 | −8.7  | 1671.3 |
| E (5th) | 1620 | −2.1  | 1617.9 |
| F (6th) | 1600 | −5.4  | 1594.6 |
| H (7th) | 1540 | +9.8  | 1549.8 |
| G (8th) | 1570 | −68.1 | 1501.9 |

Key takeaways the model captures: B's upset win delivers the largest single gain; A barely loses rating despite finishing 2nd because they were expected to win and lost only to one stronger-than-expected opponent; G's collapse from rank-5 expected to last is the largest negative swing.

## D. Worked Example — Boulder+Lead Aggregate Update (Option C, Hybrid)

Athlete X, post-Paris-2024 combined event (a real combined-format result):

- μ_B = 1680 (σ_B = 90), μ_L = 1720 (σ_L = 85)
- **Derived prior for B+L** (default weights 0.5/0.5): μ_BL_prior = 1700, σ_BL_prior = √((90² + 85²)/2) ≈ 87.5
- Combined event result: X finishes 3rd of 8, beating a field whose mean μ_BL = 1690
- Apply Plackett-Luce update at Olympic K=48 to derived prior → μ_BL_posterior = 1714.6

The aggregate is now slightly higher than the pure derivation — X has demonstrated above-prior combined-format performance, and the system records that signal independently of their separate B and L ratings.

---

*End of PRD. §4 Decisions 1–3 are now locked. Pending only the items in §12, this document is ready for engineering scoping.*
