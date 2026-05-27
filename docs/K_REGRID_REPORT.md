# K_FACTOR_TABLE regrid report (Issue #80)

- Started: `2026-05-27T01:45:32Z`
- Finished: `2026-05-27T02:08:15Z`
- Disciplines: L, B
- Holdout seasons: 2
- MC simulations per round: 2000
- RNG seed: 2026
- Grid: [0.5, 0.75, 1.0, 1.5, 2.0]
- Passes: 2
- μ-p95 target band: [1900, 2200]
- Tolerance: 0.00pp

## Summary

| Metric | Current (baseline) | Recommended |
|---|---|---|
| Top-3 hit rate | 0.6765 | 0.8676 |
| Top-1 hit rate | 0.2206 | 0.2941 |
| Log-loss (win) | 0.4220 | 0.2254 |
| Log-loss (podium) | 1.0730 | 0.6096 |
| Brier (podium) | 0.0734 | 0.0589 |
| μ min  | 721 | 1082 |
| μ p50  | 1568 | 1523 |
| μ p95  | 2478 | 2045 |
| μ p99  | 2872 | 2289 |
| μ max  | 3223 | 2668 |

## Per-cell sweep results

### world_cup / final — **UPDATED** (in band)

- Baseline K: **32.00** (top-3=0.6765, μ-p95=2168)
- Recommended K: **12.00** (top-3=0.8676, μ-p95=2136)

| Mult | K | Top-3 | LL podium | μ p50 | μ p95 | μ p99 |
|---|---|---|---|---|---|---|
| 0.50 | 12.00 | 0.8676 | 0.6340 | 1538 | 2136 | 2404 | ←
| 0.75 | 18.00 | 0.7206 | 0.6449 | 1535 | 2148 | 2424 |
| 1.00 | 24.00 | 0.6765 | 0.8190 | 1532 | 2168 | 2464 |
| 1.50 | 36.00 | 0.6765 | 1.0303 | 1527 | 2227 | 2578 |
| 2.00 | 48.00 | 0.6765 | 1.2696 | 1520 | 2294 | 2701 |

### world_cup / qualification — **UPDATED** (in band)

- Baseline K: **12.00** (top-3=0.8676, μ-p95=2136)
- Recommended K: **4.50** (top-3=0.8676, μ-p95=2076)

| Mult | K | Top-3 | LL podium | μ p50 | μ p95 | μ p99 |
|---|---|---|---|---|---|---|
| 0.50 | 3.00 | 0.8529 | 0.6372 | 1528 | 2017 | 2269 |
| 0.75 | 4.50 | 0.8676 | 0.6321 | 1538 | 2076 | 2339 | ←
| 1.00 | 6.00 | 0.8676 | 0.6340 | 1538 | 2136 | 2404 |
| 1.50 | 9.00 | 0.8382 | 0.6159 | 1553 | 2262 | 2514 |
| 2.00 | 12.00 | 0.8088 | 0.6294 | 1557 | 2391 | 2654 |

### world_cup / semi — **UNCHANGED — no data** (out of band)

- No rounds with this (tier, round_type) appear in the source DB; K held at the current default of **24.00**.

### world_championship / final — **UPDATED** (in band)

- Baseline K: **40.00** (top-3=0.8676, μ-p95=2076)
- Recommended K: **15.00** (top-3=0.8676, μ-p95=2084)

| Mult | K | Top-3 | LL podium | μ p50 | μ p95 | μ p99 |
|---|---|---|---|---|---|---|
| 0.50 | 10.00 | 0.8529 | 0.5980 | 1538 | 2073 | 2317 |
| 0.75 | 15.00 | 0.8676 | 0.6199 | 1538 | 2084 | 2331 | ←
| 1.00 | 20.00 | 0.8676 | 0.6321 | 1538 | 2076 | 2339 |
| 1.50 | 30.00 | 0.8676 | 0.6816 | 1537 | 2081 | 2373 |
| 2.00 | 40.00 | 0.8529 | 0.7406 | 1537 | 2084 | 2399 |

### world_championship / qualification — **UPDATED** (in band)

- Baseline K: **15.00** (top-3=0.8676, μ-p95=2084)
- Recommended K: **3.75** (top-3=0.8676, μ-p95=2045)

| Mult | K | Top-3 | LL podium | μ p50 | μ p95 | μ p99 |
|---|---|---|---|---|---|---|
| 0.50 | 3.75 | 0.8676 | 0.6097 | 1523 | 2045 | 2289 | ←
| 0.75 | 5.62 | 0.8676 | 0.6170 | 1528 | 2058 | 2310 |
| 1.00 | 7.50 | 0.8676 | 0.6199 | 1538 | 2084 | 2331 |
| 1.50 | 11.25 | 0.8529 | 0.6485 | 1543 | 2129 | 2384 |
| 2.00 | 15.00 | 0.8529 | 0.6737 | 1556 | 2215 | 2454 |

### world_championship / semi — **UNCHANGED — no data** (out of band)

- No rounds with this (tier, round_type) appear in the source DB; K held at the current default of **30.00**.

### continental / final — **UPDATED** (in band)

- Baseline K: **24.00** (top-3=0.8676, μ-p95=2045)
- Recommended K: **6.00** (top-3=0.8676, μ-p95=2045)

| Mult | K | Top-3 | LL podium | μ p50 | μ p95 | μ p99 |
|---|---|---|---|---|---|---|
| 0.50 | 6.00 | 0.8676 | 0.6096 | 1523 | 2045 | 2289 | ←
| 0.75 | 9.00 | 0.8676 | 0.6097 | 1523 | 2045 | 2289 |
| 1.00 | 12.00 | 0.8676 | 0.6097 | 1523 | 2045 | 2289 |
| 1.50 | 18.00 | 0.8676 | 0.6097 | 1522 | 2045 | 2289 |
| 2.00 | 24.00 | 0.8676 | 0.6100 | 1522 | 2046 | 2290 |

### continental / qualification — **UPDATED** (in band)

- Original baseline K: **9.00**
- Recommended K: **4.50** (top-3=0.8676, μ-p95=2045)
- Pass 1 selected 0.5x (→ K=4.50) on a μ-p95=2168 baseline; pass 2 confirmed
  K=4.50 as still optimal under the further-converged neighbour cells.

Pass-2 sensitivity at K=4.50 (every multiplier produces identical metrics →
the cell has near-zero observable signal):

| Mult | K | Top-3 | LL podium | μ p50 | μ p95 | μ p99 |
|---|---|---|---|---|---|---|
| 0.50 | 2.25 | 0.8676 | 0.6096 | 1522 | 2045 | 2288 |
| 0.75 | 3.38 | 0.8676 | 0.6096 | 1523 | 2045 | 2288 |
| 1.00 | 4.50 | 0.8676 | 0.6096 | 1523 | 2045 | 2289 | ←
| 1.50 | 6.75 | 0.8676 | 0.6097 | 1522 | 2045 | 2289 |
| 2.00 | 9.00 | 0.8676 | 0.6098 | 1522 | 2045 | 2289 |

### continental / semi — **UNCHANGED — no data** (out of band)

- No rounds with this (tier, round_type) appear in the source DB; K held at the current default of **18.00**.

### olympics / final — **UNCHANGED — no data** (out of band)

- No rounds with this (tier, round_type) appear in the source DB; K held at the current default of **48.00**.

### olympics / qualification — **UNCHANGED — no data** (out of band)

- No rounds with this (tier, round_type) appear in the source DB; K held at the current default of **18.00**.

### olympics / semi — **UNCHANGED — no data** (out of band)

- No rounds with this (tier, round_type) appear in the source DB; K held at the current default of **36.00**.

## Recommended K_FACTOR_TABLE (paste into `_DEFAULT_K_FACTORS`)

```python
# Recommended K_FACTOR_TABLE (regrid 2026-05-27 against backtest):
_DEFAULT_K_FACTORS: dict[EventTier, dict[RoundType, float]] = {
    EventTier.WORLD_CUP: {
        RoundType.FINAL: 12.00,  # was 32.0
        RoundType.SEMI: 24.00,  # unchanged
        RoundType.QUALIFICATION: 4.50,  # was 12.0
    },
    EventTier.WORLD_CHAMPIONSHIP: {
        RoundType.FINAL: 15.00,  # was 40.0
        RoundType.SEMI: 30.00,  # unchanged
        RoundType.QUALIFICATION: 3.75,  # was 15.0
    },
    EventTier.CONTINENTAL: {
        RoundType.FINAL: 6.00,  # was 24.0
        RoundType.SEMI: 18.00,  # unchanged
        RoundType.QUALIFICATION: 4.50,  # was 9.0
    },
    EventTier.OLYMPICS: {
        RoundType.FINAL: 48.00,  # unchanged
        RoundType.SEMI: 36.00,  # unchanged
        RoundType.QUALIFICATION: 18.00,  # unchanged
    },
}
```

## Next steps

1. Review the per-cell tables above for sanity (any cell with
   top-3 dropping more than 2pp deserves a closer look).
2. Apply the recommended `_DEFAULT_K_FACTORS` dict to
   `src/climbing_elo/engine/elo.py`.
3. Re-run the full backtest: `uv run python scripts/run_backtest.py --db data/climbing_elo.db`.
4. Trigger a prod re-backfill: `gh workflow run scrape-supabase.yml --repo milwil-2/climb-elo`.
