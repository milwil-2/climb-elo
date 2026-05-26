#!/usr/bin/env python3
"""Smoke test for the climbing-elo dashboard.

Starts the FastAPI server on port 8765, walks through every HTML route,
asserts HTTP 200 and key content strings, optionally takes screenshots via
cmux browser automation, then tears down the server.

Usage:
    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --port 8080
    uv run python scripts/smoke_test.py --no-screenshots
    uv run python scripts/smoke_test.py --base-url http://localhost:8080  # pre-running server

Exit code 0 = all tests passed, 1 = one or more failures.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT = 8080
BASE_URL = f"http://localhost:{PORT}"
SCREENSHOT_DIR = Path("/tmp/climbing_elo_smoke") / datetime.now().strftime("%Y-%m-%d")
TIMEOUT = 30  # seconds for HTTP requests (event-detail pages can be slow on first load)
SERVER_STARTUP_TIMEOUT = 20  # seconds to wait for the server to start

# Test fixtures (athlete/event IDs from the real DB)
LEAD_ATHLETE_A = 120  # Jakob SCHUBERT — Lead rating ✓
LEAD_ATHLETE_B = 232  # Adam ONDRA     — Lead rating ✓  (both Male, Lead rated)
POPULAR_ATHLETE = 61  # Janja GARNBRET — most events + Lead rating
FIRST_EVENT_ID = 93  # IFSC Worldcup Chamonix 2012 — has rounds + RatingHistory
BREAKDOWN_ATHLETE = 79  # athlete with contributing_pairs in event 93
BREAKDOWN_EVENT = 93

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class TestResult(NamedTuple):
    name: str
    passed: bool
    detail: str = ""


results: list[TestResult] = []


def record(name: str, passed: bool, detail: str = "") -> bool:
    results.append(TestResult(name, passed, detail))
    status = "PASS" if passed else "FAIL"
    marker = "✓" if passed else "✗"
    print(f"  [{status}] {marker} {name}" + (f"  — {detail}" if detail else ""))
    return passed


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def http_get(url: str, timeout: int = TIMEOUT) -> tuple[int, str]:
    """Return (status_code, body_text). On error returns (0, error_message)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, str(exc)
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def assert_route(
    test_name: str,
    path: str,
    must_contain: list[str],
    *,
    base_url: str = BASE_URL,
    must_not_contain: list[str] | None = None,
) -> bool:
    """GET base_url+path, assert 200 and that every string in must_contain is in the body."""
    url = base_url + path
    status, body = http_get(url)
    if status != 200:
        return record(test_name, False, f"HTTP {status} for {url}")

    missing = [s for s in must_contain if s not in body]
    if missing:
        return record(test_name, False, f"Missing strings: {missing!r}")

    if must_not_contain:
        found_bad = [s for s in must_not_contain if s in body]
        if found_bad:
            return record(test_name, False, f"Unexpected strings: {found_bad!r}")

    return record(test_name, True)


# ---------------------------------------------------------------------------
# Server lifecycle
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
    # Make sure the module search path includes the project src dir
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
    # Wait for port to open
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
# cmux browser automation helpers
# ---------------------------------------------------------------------------

# Module-level cache for the browser surface ref (resolved once on first use)
_browser_surface: str | None = None


def cmux_available() -> bool:
    try:
        result = subprocess.run(
            ["cmux", "browser", "status"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "enabled" in result.stdout
    except Exception:  # noqa: BLE001
        return False


def _get_or_open_browser_surface(base_url: str) -> str | None:
    """Open (or reuse) a browser surface and return its surface ref (e.g. 'surface:36').

    cmux `browser open` prints `OK surface=surface:N ...` on stdout.
    """
    global _browser_surface  # noqa: PLW0603
    if _browser_surface is not None:
        return _browser_surface
    try:
        result = subprocess.run(
            ["cmux", "browser", "open", base_url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        # Parse: "OK surface=surface:36 pane=pane:21 placement=reuse"
        for token in result.stdout.split():
            if token.startswith("surface="):
                _browser_surface = token.split("=", 1)[1]
                return _browser_surface
    except Exception:  # noqa: BLE001
        pass
    return None


def cmux_navigate(url: str, surface: str | None) -> bool:
    """Navigate the cmux browser surface to a URL. Returns True on success."""
    if surface is None:
        return False
    try:
        result = subprocess.run(
            ["cmux", "browser", surface, "goto", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def cmux_screenshot(path: Path, surface: str | None) -> bool:
    """Take a screenshot with cmux browser. Returns True on success."""
    if surface is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["cmux", "browser", surface, "screenshot", "--out", str(path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def navigate_and_screenshot(
    url: str,
    label: str,
    *,
    take_screenshots: bool,
    surface: str | None,
) -> None:
    """Navigate to URL and optionally take a screenshot."""
    if not take_screenshots or surface is None:
        return
    cmux_navigate(url, surface)
    # Brief pause for page render
    time.sleep(0.8)
    safe_label = label.replace("/", "_").replace("?", "_").strip("_")
    out_path = SCREENSHOT_DIR / f"{safe_label}.png"
    ok = cmux_screenshot(out_path, surface)
    if ok:
        print(f"    screenshot -> {out_path}")
    else:
        print(f"    screenshot failed for {label}")


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def run_tests(base_url: str, take_screenshots: bool, surface: str | None) -> None:
    print("\n── Route smoke tests ──────────────────────────────────────────")

    def ss(url: str, label: str) -> None:
        navigate_and_screenshot(
            url, label, take_screenshots=take_screenshots, surface=surface
        )

    # 1. Landing /
    assert_route(
        "GET / — monochrome landing",
        "/",
        must_contain=["Climbing ELO", "Ratings, not", "Boulder"],
        base_url=base_url,
    )
    ss(base_url + "/", "landing")

    # 2. Leaderboard
    assert_route(
        "GET /leaderboard — full board",
        "/leaderboard",
        must_contain=["Climbing ELO", "Leaderboard", "Boulder"],
        base_url=base_url,
    )
    ss(base_url + "/leaderboard", "leaderboard")

    # 3. Predictions hub
    assert_route(
        "GET /predictions — landing hub",
        "/predictions",
        must_contain=["Climbing ELO", "Predictions", "Head-to-head"],
        base_url=base_url,
    )
    ss(base_url + "/predictions", "predictions")

    # 4. Projections
    assert_route(
        "GET /projections — projection cards",
        "/projections",
        must_contain=["Climbing ELO", "Projections", "simulated"],
        base_url=base_url,
    )
    ss(base_url + "/projections", "projections")

    # 5. Head-to-head form
    assert_route(
        "GET /head-to-head — selection form",
        "/head-to-head",
        must_contain=["Climbing ELO", "Head-to-head", "probability"],
        base_url=base_url,
    )
    ss(base_url + "/head-to-head", "head_to_head_form")

    # 6. Head-to-head result — Schubert vs Ondra, Lead
    h2h_path = f"/head-to-head/{LEAD_ATHLETE_A}/{LEAD_ATHLETE_B}?discipline=lead"
    assert_route(
        "GET /head-to-head/{a}/{b} — result page",
        h2h_path,
        must_contain=["Climbing ELO", "%", "wins"],
        base_url=base_url,
    )
    ss(base_url + h2h_path, "head_to_head_result")

    # 7. Events list
    assert_route(
        "GET /events — paginated event list",
        "/events",
        must_contain=["Lead", "2024"],
        base_url=base_url,
    )
    ss(base_url + "/events", "events_list")

    # 8. Event detail
    assert_route(
        f"GET /events/{FIRST_EVENT_ID} — event detail with rounds",
        f"/events/{FIRST_EVENT_ID}",
        must_contain=["Qualification", "Final"],
        base_url=base_url,
    )
    ss(base_url + f"/events/{FIRST_EVENT_ID}", f"event_detail_{FIRST_EVENT_ID}")

    # 9. Athlete profile
    status_ath, body_ath = http_get(base_url + f"/athletes/{POPULAR_ATHLETE}")
    record(
        f"GET /athletes/{POPULAR_ATHLETE} — athlete profile",
        status_ath == 200 and "Climbing ELO" in body_ath,
        f"HTTP {status_ath}",
    )
    ss(base_url + f"/athletes/{POPULAR_ATHLETE}", f"athlete_{POPULAR_ATHLETE}")

    # 10. Breakdown page
    assert_route(
        f"GET /breakdown/{BREAKDOWN_ATHLETE}/{BREAKDOWN_EVENT} — pairwise pairs",
        f"/breakdown/{BREAKDOWN_ATHLETE}/{BREAKDOWN_EVENT}",
        must_contain=["Opponent", "Expected"],
        base_url=base_url,
    )
    ss(
        base_url + f"/breakdown/{BREAKDOWN_ATHLETE}/{BREAKDOWN_EVENT}",
        f"breakdown_{BREAKDOWN_ATHLETE}_{BREAKDOWN_EVENT}",
    )

    # 11. API reference page
    assert_route(
        "GET /api — API reference page",
        "/api",
        must_contain=["Climbing ELO", "leaderboard", "no auth"],
        base_url=base_url,
    )
    ss(base_url + "/api", "api")

    # 12. 404 behaviour — athlete that doesn't exist
    status, _ = http_get(base_url + "/athletes/999999")
    record(
        "GET /athletes/999999 → 404",
        status == 404,
        f"HTTP {status}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for the climbing-elo dashboard"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Port to bind (default {PORT})"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Skip server startup and use this URL instead",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Disable cmux browser screenshots",
    )
    args = parser.parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}"
    take_screenshots = not args.no_screenshots

    proc: subprocess.Popen | None = None
    server_managed = args.base_url is None  # we own the server lifecycle

    print("Climbing-ELO Dashboard Smoke Test")
    print(f"Base URL : {base_url}")
    print(f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    browser_surface: str | None = None
    if take_screenshots:
        using_cmux = cmux_available()
        if using_cmux:
            print("cmux browser : enabled (opening surface…)")
        else:
            print("cmux browser : unavailable — screenshots disabled")
            take_screenshots = False
    else:
        print("cmux browser : disabled (--no-screenshots)")

    try:
        # ── Start server ───────────────────────────────────────────────────
        if server_managed:
            if _port_open(args.port):
                print(
                    f"\n⚠  Port {args.port} already in use — cannot start test server."
                )
                print(
                    "   Use --base-url http://localhost:<port> to test an existing server."
                )
                return 1

            print(f"\nStarting server on port {args.port}…")
            proc = start_server(args.port)
            if proc is None:
                print("FATAL: Could not start server.")
                return 1
            print(f"  Server PID {proc.pid} ready.\n")
        else:
            # Verify the provided server is reachable
            status, _ = http_get(base_url + "/")
            if status == 0:
                print(f"FATAL: Cannot reach {base_url} — is the server running?")
                return 1
            print(f"\nUsing pre-running server at {base_url}\n")

        # ── Open browser surface (once, before tests) ─────────────────────
        if take_screenshots:
            browser_surface = _get_or_open_browser_surface(base_url)
            if browser_surface:
                print(f"  Browser surface: {browser_surface}")
                time.sleep(
                    1.0
                )  # let the first page fully load before we start navigating
            else:
                print("  Could not open browser surface — screenshots disabled")
                take_screenshots = False

        # ── Run tests ──────────────────────────────────────────────────────
        run_tests(base_url, take_screenshots, browser_surface)

    finally:
        # ── Cleanup ────────────────────────────────────────────────────────
        if server_managed and proc is not None:
            print("\nStopping server…")
            stop_server(proc)

    # ── Summary ────────────────────────────────────────────────────────────
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print("\n" + "─" * 60)
    print(f"  RESULTS: {len(passed)}/{len(results)} passed")
    print("─" * 60)

    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  FAIL  {r.name}" + (f"\n        {r.detail}" if r.detail else ""))
        print()

    if take_screenshots:
        print(f"Screenshots saved to: {SCREENSHOT_DIR}")

    overall = len(failed) == 0
    print(f"\n{'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
