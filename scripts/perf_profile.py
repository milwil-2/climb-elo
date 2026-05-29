#!/usr/bin/env python3
"""Performance profiler for the climbing-elo dashboard hot routes (Issue #97).

Times the slow HTML routes called out in #97, reports min/p50/mean/max
wall-clock latency per route, and asserts the #97 acceptance targets
(``/`` and ``/leaderboard`` p50 < 800ms, ``/predictions`` p50 < 2000ms).

Like ``smoke_test.py`` it can either profile a pre-running server
(``--base-url``) or spin up its own uvicorn instance on a port.

Usage:
    uv run python scripts/perf_profile.py
    uv run python scripts/perf_profile.py --runs 10
    uv run python scripts/perf_profile.py --base-url http://localhost:8080
    uv run python scripts/perf_profile.py --base-url https://climb-elo.vercel.app --json

Exit code 0 = all targeted routes meet their p50 budget, 1 otherwise
(non-200 responses also count as a failure for that route).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT = 8080
TIMEOUT = 30.0  # seconds for a single HTTP request
SERVER_STARTUP_TIMEOUT = 20  # seconds to wait for the server to start

#: Routes called out by #97, plus an API control to isolate template/render cost.
ROUTES: list[str] = [
    "/",
    "/leaderboard",
    "/leaderboard?disc=L&gender=F&view=all",
    "/athletes",
    "/events",
    "/predictions",
    "/api/v1/disciplines",  # control: lightweight JSON endpoint
]

#: Default p50 budgets in milliseconds (#97 acceptance criteria). Routes absent
#: from this table are profiled but never fail the run.
DEFAULT_TARGETS_MS: dict[str, float] = {
    "/": 800.0,
    "/leaderboard": 800.0,
    "/predictions": 2000.0,
}


# ---------------------------------------------------------------------------
# Pure statistics helpers (unit-tested in tests/test_perf_profile.py)
# ---------------------------------------------------------------------------


def percentile(samples: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``samples`` via linear interpolation.

    ``pct`` is in [0, 100]. Matches the common "linear interpolation between
    closest ranks" method (numpy's default). Raises ``ValueError`` on empty
    input or an out-of-range percentile.
    """
    if not samples:
        raise ValueError("percentile() requires at least one sample")
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct must be in [0, 100]")

    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]

    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def summarize(samples: list[float]) -> dict[str, float]:
    """Return min / p50 / mean / max for a list of latency samples (ms)."""
    if not samples:
        raise ValueError("summarize() requires at least one sample")
    return {
        "min": min(samples),
        "p50": percentile(samples, 50.0),
        "mean": sum(samples) / len(samples),
        "max": max(samples),
    }


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class RouteResult(NamedTuple):
    path: str
    stats: dict[str, float] | None  # None when the route errored out
    status: int  # last observed HTTP status (0 = transport error)
    target_ms: float | None  # p50 budget, or None if untargeted
    passed: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# HTTP timing
# ---------------------------------------------------------------------------


def time_route(
    client: httpx.Client,
    base_url: str,
    path: str,
    runs: int,
) -> tuple[list[float], int, str]:
    """Warm up once, then time ``runs`` requests.

    Returns ``(samples_ms, last_status, detail)``. ``samples_ms`` is empty if
    any request failed (transport error or non-200); ``detail`` explains why.
    """
    url = base_url + path

    # Warm-up request (discarded). A failure here is reported immediately.
    try:
        warm = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return [], 0, f"request error: {exc}"
    if warm.status_code != 200:
        return [], warm.status_code, f"HTTP {warm.status_code} on warm-up"

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        try:
            resp = client.get(url)
        except Exception as exc:  # noqa: BLE001
            return [], 0, f"request error: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if resp.status_code != 200:
            return [], resp.status_code, f"HTTP {resp.status_code}"
        samples.append(elapsed_ms)

    return samples, 200, ""


def profile_routes(
    base_url: str,
    routes: list[str],
    runs: int,
    targets_ms: dict[str, float],
) -> list[RouteResult]:
    """Profile every route and evaluate it against its p50 budget."""
    results: list[RouteResult] = []
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        for path in routes:
            # Match a target by the path portion (ignore query string).
            target = targets_ms.get(path.split("?", 1)[0])
            samples, status, detail = time_route(client, base_url, path, runs)

            if not samples:
                results.append(RouteResult(path, None, status, target, False, detail))
                continue

            stats = summarize(samples)
            if target is None:
                passed = True
            else:
                passed = stats["p50"] < target
                detail = f"p50 {stats['p50']:.0f}ms vs <{target:.0f}ms budget"
            results.append(RouteResult(path, stats, status, target, passed, detail))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(results: list[RouteResult], runs: int) -> None:
    print(f"\n── Route latency ({runs} warm samples each) ─────────────────────")
    header = (
        f"  {'route':<42} {'min':>8} {'p50':>8} {'mean':>8} {'max':>8}  "
        f"{'budget':>8}  result"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in results:
        if r.stats is None:
            status_txt = f"HTTP {r.status}" if r.status else "ERR"
            print(
                f"  {r.path:<42} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  "
                f"{'—':>8}  FAIL ({status_txt})"
            )
            continue
        budget = f"{r.target_ms:.0f}ms" if r.target_ms is not None else "—"
        verdict = "PASS" if r.passed else "FAIL"
        if r.target_ms is None:
            verdict = "—"
        print(
            f"  {r.path:<42} "
            f"{r.stats['min']:>7.0f}m "
            f"{r.stats['p50']:>7.0f}m "
            f"{r.stats['mean']:>7.0f}m "
            f"{r.stats['max']:>7.0f}m  "
            f"{budget:>8}  {verdict}"
        )


def to_json(results: list[RouteResult], base_url: str, runs: int) -> str:
    payload = {
        "base_url": base_url,
        "runs": runs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "routes": [
            {
                "path": r.path,
                "status": r.status,
                "stats_ms": r.stats,
                "target_ms": r.target_ms,
                "passed": r.passed,
                "detail": r.detail,
            }
            for r in results
        ],
        "all_passed": all(r.passed for r in results),
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Server lifecycle (mirrors smoke_test.py)
# ---------------------------------------------------------------------------


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


def start_server(port: int) -> subprocess.Popen | None:
    """Start uvicorn in a subprocess. Returns the Popen object, or None on failure."""
    env = os.environ.copy()
    src_dir = str(Path(__file__).parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "climbing_elo.api.app:app",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        if _port_open(port):
            return proc
        if proc.poll() is not None:
            out, _ = proc.communicate()
            print(f"  Server exited early: {out[:500]}")
            return None
        time.sleep(0.25)

    proc.terminate()
    print(f"  Server did not start within {SERVER_STARTUP_TIMEOUT}s")
    return None


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Performance profiler for the climbing-elo dashboard (#97)"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Port to bind (default {PORT})"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Profile this URL instead of starting a server "
        "(e.g. http://localhost:8080 or https://climb-elo.vercel.app)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed (warm) samples per route (default 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table",
    )
    parser.add_argument(
        "--target-root",
        type=float,
        default=DEFAULT_TARGETS_MS["/"],
        help="p50 budget in ms for / (default 800)",
    )
    parser.add_argument(
        "--target-leaderboard",
        type=float,
        default=DEFAULT_TARGETS_MS["/leaderboard"],
        help="p50 budget in ms for /leaderboard (default 800)",
    )
    parser.add_argument(
        "--target-predictions",
        type=float,
        default=DEFAULT_TARGETS_MS["/predictions"],
        help="p50 budget in ms for /predictions (default 2000)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    targets_ms = {
        "/": args.target_root,
        "/leaderboard": args.target_leaderboard,
        "/predictions": args.target_predictions,
    }

    base_url = (args.base_url or f"http://localhost:{args.port}").rstrip("/")
    server_managed = args.base_url is None

    if not args.json:
        print("Climbing-ELO Dashboard Performance Profile")
        print(f"Base URL : {base_url}")
        print(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runs     : {args.runs} warm samples per route")

    proc: subprocess.Popen | None = None
    try:
        if server_managed:
            if _port_open(args.port):
                print(
                    f"\n⚠  Port {args.port} already in use — cannot start profile "
                    "server.\n   Use --base-url http://localhost:<port> instead."
                )
                return 1
            if not args.json:
                print(f"\nStarting server on port {args.port}…")
            proc = start_server(args.port)
            if proc is None:
                print("FATAL: Could not start server.")
                return 1
            if not args.json:
                print(f"  Server PID {proc.pid} ready.")
        else:
            # Verify reachability before profiling.
            try:
                httpx.Client(timeout=TIMEOUT, follow_redirects=True).get(base_url + "/")
            except Exception as exc:  # noqa: BLE001
                print(f"FATAL: Cannot reach {base_url} — {exc}")
                return 1

        results = profile_routes(base_url, ROUTES, args.runs, targets_ms)
    finally:
        if server_managed and proc is not None:
            if not args.json:
                print("\nStopping server…")
            stop_server(proc)

    if args.json:
        print(to_json(results, base_url, args.runs))
    else:
        print_table(results, args.runs)
        targeted = [r for r in results if r.target_ms is not None]
        failed = [r for r in results if not r.passed]
        within = len([r for r in targeted if r.passed])
        print("\n" + "─" * 60)
        print(f"  {within}/{len(targeted)} targeted routes within budget")
        print("─" * 60)
        if failed:
            print("\nFailures:")
            for r in failed:
                print(f"  FAIL  {r.path}" + (f"  — {r.detail}" if r.detail else ""))
        print(f"\n{'ALL TARGETS MET' if not failed else 'SOME TARGETS MISSED'}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
