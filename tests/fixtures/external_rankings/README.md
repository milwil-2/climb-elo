# External Rankings Fixtures

These JSON snapshots are the **offline test corpus** for the
`ifsc_official` and `ascentstats` backtest baselines (Issue #44 + the
AscentStats addition tracked under #55).

## Why fixtures?

The two external sources are scrapeable but unsuitable for hitting on every
CI run:

* **IFSC** — The official rankings widget lives at
  `components.ifsc-climbing.org/rankings/` and requires an interactive
  browser session (the legacy `ifsc.results.info` API has no rankings
  endpoint). We instead consume the end-of-season Wikipedia summary tables
  for the IFSC World Cup, which mirror the IFSC's own published ranking.
* **AscentStats** — Static HTML pages at
  `https://ascentstats.com/ranking-pages/ranking-{year}-{men,women}.html`.
  Boulder only, all the way back to 2008.

Both surfaces are slow / flaky from CI runners and have no contract — a
recorded fixture is safer than a live fetch.

## Updating fixtures

A maintainer-side script `scripts/refresh_external_rankings.py` calls
`scraper.external_rankings.refresh_all(seasons=[…])` and writes new JSON
files to `data/external_rankings/` (gitignored). To promote a fresh
snapshot into the repo, copy the file from `data/external_rankings/...` to
this directory:

```bash
uv run python scripts/refresh_external_rankings.py --seasons 2024 2025 2026
cp data/external_rankings/ifsc_official/2025-boulder-M.json \
   tests/fixtures/external_rankings/ifsc_official/
```

## File naming

`{source}/{season}-{discipline}-{gender}.json` where:

- `source` ∈ {`ifsc_official`, `ascentstats`}
- `season` ∈ four-digit year
- `discipline` ∈ {`boulder`, `lead`, `speed`} (AscentStats is boulder only)
- `gender` ∈ {`M`, `F`}

## Schema

```json
{
  "source": "ifsc_official",
  "season": 2024,
  "discipline": "boulder",
  "gender": "M",
  "fetched_at": "YYYY-MM-DD",
  "source_url": "<canonical upstream URL>",
  "ranking": [
    {"rank": 1, "name": "Sorato Anraku", "country": "JPN", "rating": 3365.0},
    ...
  ]
}
```

`rating` is the score the upstream source publishes — IFSC World Cup
points for `ifsc_official`, AscentStats' bit-grade rating for
`ascentstats`. Engines use rank-based μ by default for cross-source
comparability; `rating` is kept for downstream experiments.
