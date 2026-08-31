#!/usr/bin/env python3
"""Periodic backtest monitor — wrapper around ``scripts/run_backtest.py``.

This is the helper invoked by ``.github/workflows/backtest-monitor.yml`` to
run the existing backtest harness against a fresh snapshot of the production
data and emit machine-readable metrics for trend tracking + alerting.

The script is intentionally tight:
  - It does *not* re-implement backtesting (that lives in
    ``climbing_elo.engine.evaluation`` and is driven by ``run_backtest.py``).
  - It does *not* persist metrics to a DB. Output is a single JSON blob the
    workflow uploads as an artifact, plus a Discord-formatted payload on
    regression.
  - It does *not* invent its own baseline. Trend comparison happens in the
    workflow against artifacts of prior runs; this script only emits today's
    numbers and (optionally) compares them against a list of recent log-loss
    values passed in via ``--trailing-log-loss``.

The harness in ``run_backtest.py`` only supports SQLite sources (the
``BacktestDataset.source_db_path`` field). When ``DATABASE_URL`` points at
Postgres (the GH Actions case), this script first snapshots the relevant
tables into a temporary SQLite file using SQLAlchemy + the existing ORM.
That keeps the snapshot logic in one place and avoids depending on
``pg_dump`` / ``sqlite3`` binaries inside CI.

Usage
-----

::

    # Local SQLite source (fast path for tests):
    uv run python scripts/backtest_monitor.py \\
        --db data/climbing_elo.db --out /tmp/metrics.json

    # Postgres source (GH Actions path — reads DATABASE_URL):
    uv run python scripts/backtest_monitor.py --out /tmp/metrics.json

    # Walk-forward (workflow_dispatch path):
    uv run python scripts/backtest_monitor.py --mode walk_forward --out /tmp/metrics.json

    # Compare against trailing-N log-loss for regression alerting:
    uv run python scripts/backtest_monitor.py --out /tmp/metrics.json \\
        --trailing-log-loss 0.41 0.42 0.405 --regression-threshold-absolute 0.02

Exit codes
----------

  0 — backtest completed; metrics written. No regression detected (or
       regression detection disabled).
  1 — backtest failed (subprocess error, missing report, missing DB).
  2 — regression detected (today's log-loss exceeds trailing mean by more
       than the threshold). Metrics JSON is still written before exit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Postgres → SQLite snapshot (only used when DATABASE_URL points at Postgres).
# ---------------------------------------------------------------------------


def _snapshot_postgres_to_sqlite(sqlite_path: Path) -> None:
    """Copy the backtest's input tables from Postgres into a fresh SQLite file.

    Only athletes/events/rounds/results are copied — the backtest replays
    ratings from raw results, so Rating / RatingHistory / EventForecast* rows
    are recomputed inside the harness's private working copy anyway. We used
    to copy *all* ORM tables "for faithfulness", which pulled the ~70 MB
    ``rating_history`` table from Supabase weekly just to discard it; the
    input tables total ~8 MB. The skipped tables still exist (empty) in the
    SQLite file because ``init_db`` creates the full schema, which is exactly
    the reset state the harness starts each fold from (#193).

    Streams rows in chunks via SQLAlchemy Core so a single large table
    doesn't blow the runner's memory budget.
    """
    # Imports are local so the script can be imported (for unit tests) on
    # machines without DATABASE_URL set / without Postgres drivers configured.
    from sqlalchemy import select

    from climbing_elo.database import get_engine, init_db
    from climbing_elo.models import Base

    src_engine = get_engine()  # Reads DATABASE_URL.
    # ``init_db`` creates the schema + returns a sessionmaker we don't use.
    init_db(sqlite_path)
    dst_engine = get_engine(sqlite_path)

    # The backtest's inputs. Everything else (ratings, rating_history,
    # event_forecast*) is computed state the harness rebuilds from these.
    input_tables = {"athletes", "events", "rounds", "results"}

    chunk_size = 5_000
    with src_engine.connect() as src_conn, dst_engine.begin() as dst_conn:
        # Iterate tables in FK-safe order (parents first, children last).
        for table in Base.metadata.sorted_tables:
            if table.name not in input_tables:
                print(
                    f"[backtest-monitor] snapshot: {table.name} skipped "
                    "(computed state — rebuilt by the harness)",
                    flush=True,
                )
                continue
            row_count = 0
            stmt = select(table)
            result = src_conn.execution_options(stream_results=True).execute(stmt)
            while True:
                rows = result.fetchmany(chunk_size)
                if not rows:
                    break
                # ``Row`` → plain dict for engine-agnostic insert.
                dst_conn.execute(
                    table.insert(),
                    [dict(r._mapping) for r in rows],
                )
                row_count += len(rows)
            print(
                f"[backtest-monitor] snapshot: {table.name} → {row_count} rows",
                flush=True,
            )


def _resolve_source_db(explicit_db: str | None, workdir: Path) -> Path:
    """Return a path to a SQLite DB the backtest harness can read.

    Precedence:
      1. ``--db`` explicitly passed → use as-is (must exist).
      2. ``DATABASE_URL`` set and starts with ``sqlite`` → strip prefix.
      3. ``DATABASE_URL`` set (Postgres) → snapshot to ``workdir/snapshot.db``.
      4. None of the above → error.
    """
    if explicit_db:
        path = Path(explicit_db)
        if not path.exists():
            raise FileNotFoundError(f"--db path does not exist: {path}")
        return path

    db_url = os.environ.get("DATABASE_URL") or ""
    if db_url.startswith("sqlite:///"):
        return Path(db_url.removeprefix("sqlite:///"))
    if db_url.startswith("sqlite://"):
        # ``sqlite://`` (no path) = in-memory; can't snapshot that.
        raise RuntimeError("In-memory sqlite:// URLs are not supported as a source.")
    if db_url:
        # Assume Postgres / anything that needs snapshotting.
        snapshot = workdir / "snapshot.db"
        print(
            "[backtest-monitor] DATABASE_URL is non-sqlite — snapshotting to "
            f"{snapshot}",
            flush=True,
        )
        _snapshot_postgres_to_sqlite(snapshot)
        return snapshot

    raise RuntimeError("No source DB available. Pass --db or set DATABASE_URL.")


# ---------------------------------------------------------------------------
# run_backtest.py invocation + metric extraction
# ---------------------------------------------------------------------------


def _invoke_run_backtest(
    db_path: Path,
    output_dir: Path,
    mode: str,
    n_sims: int,
    extra_args: list[str] | None = None,
) -> None:
    """Run ``scripts/run_backtest.py`` as a subprocess. Raises on non-zero."""
    script = REPO_ROOT / "scripts" / "run_backtest.py"
    cmd = [
        sys.executable,
        str(script),
        "--db",
        str(db_path),
        "--output-dir",
        str(output_dir),
        "--n-sims",
        str(n_sims),
    ]
    if mode == "holdout":
        cmd += ["--oos", "holdout"]
    elif mode == "walk_forward":
        cmd += ["--oos", "walk-forward"]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    if extra_args:
        cmd += list(extra_args)

    print(f"[backtest-monitor] running: {' '.join(cmd)}", flush=True)
    # No capture — let stdout/stderr stream to the workflow log.
    subprocess.run(cmd, check=True)


def extract_metrics(report_path: Path) -> dict[str, Any]:
    """Pull the headline metrics out of a ``report.json`` produced by the harness.

    Returns a flat dict suitable for JSON serialisation + Discord embed fields.
    Missing / non-numeric values are passed through as ``None`` so downstream
    consumers can decide how to render them.
    """
    raw = json.loads(report_path.read_text())
    agg = raw.get("aggregate") or {}

    def _num(key: str) -> float | None:
        v = agg.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        # The harness emits the string "NaN" for un-computable metrics.
        return None

    return {
        "generated_at": raw.get("generated_at"),
        "variant": raw.get("variant"),
        "oos_mode": raw.get("oos_mode"),
        "disciplines": raw.get("disciplines"),
        "n_simulations": raw.get("n_simulations"),
        "n_rounds": _num("n_rounds"),
        "log_loss_win": _num("log_loss_win"),
        "log_loss_podium": _num("log_loss_podium"),
        "log_loss_top8": _num("log_loss_top8"),
        "hit_rate_top1": _num("hit_rate_top1"),
        "hit_rate_top3": _num("hit_rate_top3"),
        "hit_rate_top8": _num("hit_rate_top8"),
        "n_splits": len(raw.get("splits") or []),
    }


# ---------------------------------------------------------------------------
# Trend / regression detection
# ---------------------------------------------------------------------------


def detect_regression(
    today_log_loss: float | None,
    trailing: list[float],
    absolute_threshold: float,
    relative_threshold: float,
) -> tuple[bool, str]:
    """Compare today's log-loss against trailing-mean.

    Regression if either:
      - ``today - mean(trailing) > absolute_threshold`` (e.g. +0.02 absolute), or
      - ``today / mean(trailing) - 1 > relative_threshold`` (e.g. +15%).

    Returns ``(is_regression, reason)``. ``reason`` is a one-line human string
    suitable for a Discord embed field.
    """
    if today_log_loss is None:
        return False, "today's log-loss is missing — skipping comparison"
    if not trailing:
        return False, "no trailing data — first run, baseline established"
    mean = sum(trailing) / len(trailing)
    if mean <= 0:
        return False, f"trailing mean is non-positive ({mean:.4f}) — skipping"
    abs_delta = today_log_loss - mean
    rel_delta = abs_delta / mean
    if abs_delta > absolute_threshold:
        return (
            True,
            f"log-loss {today_log_loss:.4f} exceeds trailing-mean {mean:.4f} "
            f"by {abs_delta:+.4f} (> +{absolute_threshold:.4f} absolute)",
        )
    if rel_delta > relative_threshold:
        return (
            True,
            f"log-loss {today_log_loss:.4f} is {rel_delta * 100:+.1f}% over "
            f"trailing-mean {mean:.4f} (> +{relative_threshold * 100:.1f}%)",
        )
    return (
        False,
        f"log-loss {today_log_loss:.4f} within tolerance of "
        f"trailing-mean {mean:.4f} (Δ={abs_delta:+.4f})",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Periodic backtest monitor (Issue #120).",
    )
    p.add_argument(
        "--mode",
        choices=("holdout", "walk_forward"),
        default="holdout",
        help="Backtest OOS mode. Walk-forward is slow — reserve for manual runs.",
    )
    p.add_argument(
        "--db",
        default=None,
        help=(
            "Path to a SQLite source DB. If omitted, falls back to DATABASE_URL "
            "(snapshotting Postgres → temp SQLite if needed)."
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help="Path to write the metrics JSON artifact.",
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Monte Carlo simulations per round (passed through to run_backtest.py).",
    )
    p.add_argument(
        "--trailing-log-loss",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Trailing log-loss values from prior runs (space-separated). "
            "If non-empty, today's log-loss is compared against the mean. "
            "Used by the workflow to drive regression alerting."
        ),
    )
    p.add_argument(
        "--regression-threshold-absolute",
        type=float,
        default=0.02,
        help=(
            "Absolute log-loss delta over trailing mean that triggers a "
            "regression alert (default: 0.02)."
        ),
    )
    p.add_argument(
        "--regression-threshold-relative",
        type=float,
        default=0.15,
        help=(
            "Relative log-loss delta over trailing mean that triggers a "
            "regression alert (default: 0.15 = +15%%)."
        ),
    )
    p.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="Don't delete the temp snapshot DB on exit (useful for debugging).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ``src`` on sys.path so the local imports inside _snapshot_postgres_to_sqlite work.
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    workdir = Path(tempfile.mkdtemp(prefix="backtest_monitor_"))
    report_dir = workdir / "report"
    metrics_path = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        source_db = _resolve_source_db(args.db, workdir)
        _invoke_run_backtest(
            db_path=source_db,
            output_dir=report_dir,
            mode=args.mode,
            n_sims=args.n_sims,
        )

        report_json = report_dir / "report.json"
        if not report_json.exists():
            print(
                f"[backtest-monitor] ERROR: no report.json at {report_json}",
                file=sys.stderr,
            )
            return 1

        metrics = extract_metrics(report_json)
        metrics["monitor_started_at"] = started_at
        metrics["monitor_finished_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        metrics["mode"] = args.mode

        is_regression, reason = detect_regression(
            today_log_loss=metrics.get("log_loss_podium"),
            trailing=list(args.trailing_log_loss),
            absolute_threshold=args.regression_threshold_absolute,
            relative_threshold=args.regression_threshold_relative,
        )
        metrics["regression_detected"] = is_regression
        metrics["regression_reason"] = reason
        metrics["trailing_log_loss_n"] = len(args.trailing_log_loss)
        metrics["trailing_log_loss_mean"] = (
            sum(args.trailing_log_loss) / len(args.trailing_log_loss)
            if args.trailing_log_loss
            else None
        )

        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"[backtest-monitor] wrote metrics → {metrics_path}", flush=True)
        print(f"[backtest-monitor] regression: {is_regression} ({reason})", flush=True)

        return 2 if is_regression else 0
    except subprocess.CalledProcessError as e:
        print(
            f"[backtest-monitor] run_backtest.py exited {e.returncode}",
            file=sys.stderr,
        )
        return 1
    finally:
        if not args.keep_snapshot:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"[backtest-monitor] kept workdir at {workdir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
