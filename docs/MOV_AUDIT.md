# MOV Audit — Issue #53

Generated from `rating_history.contributing_pairs` (1,197,829 winner-side pairs audited).

- `MOV_RATING_SCALE = 400.0`
- `MOV_SOFTENING = 2.2`
- `MARGIN_CAP = 1.5` (backstop on the base multiplier)

## Aggregate — all disciplines pooled

| rating-gap bucket | n_pairs | mean MOV (legacy) | mean MOV (new) | Δ (new−legacy) |
|---|---:|---:|---:|---:|
| upset (Δμ ≤ -100) | 59,553 | 1.2394 | 1.2394 | +0.0000 |
| peer (-100 < Δμ < 100) | 662,348 | 1.2757 | 1.2391 | -0.0366 |
| favourite-100-250 | 276,146 | 1.3907 | 1.1703 | -0.2204 |
| favourite-250-500 | 176,557 | 1.4325 | 1.0334 | -0.3991 |
| favourite-500-750 | 23,055 | 1.2930 | 0.7868 | -0.5062 |
| favourite-750+ | 170 | 1.4265 | 0.7595 | -0.6669 |

## Per-discipline breakdown

### Lead

| rating-gap bucket | n_pairs | mean MOV (legacy) | mean MOV (new) | Δ (new−legacy) |
|---|---:|---:|---:|---:|
| upset (Δμ ≤ -100) | 20,627 | 1.1708 | 1.1708 | +0.0000 |
| peer (-100 < Δμ < 100) | 229,018 | 1.1961 | 1.1622 | -0.0339 |
| favourite-100-250 | 98,730 | 1.3274 | 1.1153 | -0.2122 |
| favourite-250-500 | 68,118 | 1.3940 | 1.0032 | -0.3909 |
| favourite-500-750 | 13,392 | 1.1772 | 0.7112 | -0.4660 |
| favourite-750+ | 137 | 1.4088 | 0.7494 | -0.6594 |

### Boulder

| rating-gap bucket | n_pairs | mean MOV (legacy) | mean MOV (new) | Δ (new−legacy) |
|---|---:|---:|---:|---:|
| upset (Δμ ≤ -100) | 26,720 | 1.3289 | 1.3289 | +0.0000 |
| peer (-100 < Δμ < 100) | 318,181 | 1.3157 | 1.2779 | -0.0378 |
| favourite-100-250 | 128,844 | 1.4350 | 1.2086 | -0.2264 |
| favourite-250-500 | 78,252 | 1.4662 | 1.0593 | -0.4069 |
| favourite-500-750 | 7,861 | 1.4555 | 0.8909 | -0.5645 |
| favourite-750+ | 33 | 1.5000 | 0.8017 | -0.6983 |

### Speed

| rating-gap bucket | n_pairs | mean MOV (legacy) | mean MOV (new) | Δ (new−legacy) |
|---|---:|---:|---:|---:|
| upset (Δμ ≤ -100) | 12,206 | 1.1592 | 1.1592 | +0.0000 |
| peer (-100 < Δμ < 100) | 115,149 | 1.3235 | 1.2850 | -0.0385 |
| favourite-100-250 | 48,572 | 1.4016 | 1.1803 | -0.2214 |
| favourite-250-500 | 30,187 | 1.4319 | 1.0342 | -0.3978 |
| favourite-500-750 | 1,802 | 1.4446 | 0.8943 | -0.5503 |
| favourite-750+ | 0 | — | — | — |

## Reading the table

- **Upset & peer rows** — Δ should be 0 (asymmetric: no damping on upsets, no damping at Δμ=0). Small nonzero Δ in the peer row is from contests with Δμ in (0, 100) which already attract mild damping.
- **Favourite-* rows** — Δ should be negative and grow in magnitude with the rating gap. This is the elite-inflation fix: large MOV bonuses against weak fields are damped.
- **Empty rows** — disciplines that contribute no contests in that bucket (e.g. Speed rarely has Δμ > 750 because the field is small and concentrated).

