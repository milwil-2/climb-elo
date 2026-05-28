"""Output formatting — markdown rendering + default output-directory helper.

Consumes :class:`~climbing_elo.engine.evaluation.runner.BacktestReport`
instances. The JSON serialiser lives on the report dataclass itself
(``BacktestReport.to_json``); this module owns the human-readable markdown
rendering and the convenience entry point for the default output directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Report dataclass + JSON serialisation
# ---------------------------------------------------------------------------


@dataclass
class BacktestReport:
    """Top-level report — serialised to JSON + markdown."""

    generated_at: str
    variant: str
    oos_mode: str
    rng_seed: int
    n_simulations: int
    disciplines: list[str]
    splits: list[dict[str, Any]] = field(default_factory=list)
    # Aggregate (across all splits + disciplines).
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise with sorted keys + fixed float repr for byte-stability."""
        return json.dumps(asdict(self), sort_keys=True, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Custom JSON encoder hook.

    NaNs are rendered as the literal string ``"NaN"`` so the JSON loads in
    any standards-compliant parser (Python's allow-nan is non-portable).
    Stable across runs.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Unserialisable {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def render_markdown(report: BacktestReport) -> str:
    """Human-readable markdown rendering of a :class:`BacktestReport`."""
    lines: list[str] = []
    lines.append("# Backtest report")
    lines.append("")
    lines.append(f"- Generated: {report.generated_at}")
    lines.append(f"- Variant: `{report.variant}`")
    lines.append(f"- OOS mode: `{report.oos_mode}`")
    lines.append(f"- Disciplines: {', '.join(report.disciplines)}")
    lines.append(f"- RNG seed: {report.rng_seed}")
    lines.append(f"- MC simulations per round: {report.n_simulations}")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    agg = report.aggregate
    lines.append(f"- Rounds scored: {agg.get('n_rounds', 0)}")
    lines.append(f"- Athlete-rounds: {agg.get('n_athlete_rounds', 0)}")
    lines.append("")
    lines.append("| Metric | Win | Podium | Top-8 |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| Log-loss | "
        f"{_fmt(agg.get('log_loss_win'))} | "
        f"{_fmt(agg.get('log_loss_podium'))} | "
        f"{_fmt(agg.get('log_loss_top8'))} |"
    )
    lines.append(
        "| Brier | "
        f"{_fmt(agg.get('brier_win'))} | "
        f"{_fmt(agg.get('brier_podium'))} | "
        f"{_fmt(agg.get('brier_top8'))} |"
    )
    lines.append("")
    lines.append(f"- Mean Spearman ρ: {_fmt(agg.get('mean_spearman'))}")
    lines.append(f"- Top-1 hit rate: {_fmt(agg.get('hit_rate_top1'))}")
    lines.append(f"- Top-3 hit rate: {_fmt(agg.get('hit_rate_top3'))}")
    lines.append(f"- Top-8 hit rate: {_fmt(agg.get('hit_rate_top8'))}")
    lines.append("")

    lines.append("## Splits")
    lines.append("")
    for s in report.splits:
        m = s["metrics"]
        lines.append(
            f"- **{s['discipline']} / {s['label']}** "
            f"(train < {s['train_end_date']}, "
            f"n_events={s['n_eval_events']}): "
            f"log-loss podium={_fmt(m.get('log_loss_podium'))}, "
            f"top-3 hit={_fmt(m.get('hit_rate_top3'))}"
        )
    lines.append("")

    strata = agg.get("stratifications", {})

    def _table(title: str, key: str, group_label: str) -> None:
        s = strata.get(key, {})
        if not s:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"| {group_label} | n | LL win | LL pod | LL top-8 | Brier pod |")
        lines.append("|---|---|---|---|---|---|")
        for k in sorted(s.keys()):
            row = s[k]
            lines.append(
                f"| {k} | {row.get('n_athlete_rounds', row.get('n_rounds', 0))} | "
                f"{_fmt(row.get('log_loss_win'))} | "
                f"{_fmt(row.get('log_loss_podium'))} | "
                f"{_fmt(row.get('log_loss_top8'))} | "
                f"{_fmt(row.get('brier_podium'))} |"
            )
        lines.append("")

    lines.append("## Stratifications")
    lines.append("")
    _table("By tier", "by_tier", "Tier")
    _table("By round", "by_round", "Round")
    _table("By discipline", "by_discipline", "Discipline")
    _table("By season", "by_season", "Season")
    _table("By field size", "by_field_size", "Field size")
    _table("By tenure", "by_tenure", "Tenure (n_events)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


# Project root: src/climbing_elo/engine/evaluation/reports.py → repo root is 4 parents up.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def make_default_output_dir(root: Path | None = None) -> Path:
    """Return ``data/backtests/<utc-timestamp>/`` (creating parents)."""
    root = root or (_PROJECT_ROOT / "data" / "backtests")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out
