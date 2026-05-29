"""Simple in-memory TTL cache for expensive computations.

Design notes
------------
- Pure Python, no external dependencies (no Redis, memcached, etc.).
- Thread-safe via ``threading.RLock`` — safe for single-process deployments.
- Multi-worker (e.g. gunicorn) caveat: each worker holds its own in-memory
  cache, so the first request in each worker will still pay the Monte Carlo
  cost.  For a read-only predictions page this is acceptable (staleness is
  bounded by TTL anyway).  If per-worker duplication becomes a problem, swap
  ``predictions_cache`` for a Redis-backed implementation without changing
  call sites (Issue #29).

TTL approach vs. key-versioning
--------------------------------
A smarter invalidation strategy would embed ``max(Result.id)`` for an event
into the cache key so that newly ingested results automatically bust the cache.
That requires an extra DB query on every request, which largely defeats the
purpose for a page that already queries the DB heavily.  The simpler TTL-only
approach is chosen for the MVP:

- 1-hour staleness is acceptable for the /predictions page.
- Scraper runs are infrequent (manual or scheduled), so a 1-hour window is
  rarely problematic in practice.
- If tighter freshness is needed, callers can call ``predictions_cache.clear()``
  after a scrape run, or reduce ``ttl_seconds``.
"""

from __future__ import annotations

import time
from threading import RLock


class TTLCache:
    """A thread-safe dictionary cache with per-entry time-to-live expiry.

    Parameters
    ----------
    ttl_seconds:
        How long (in seconds) a cached value is considered fresh.  Expired
        entries are lazily evicted on ``get``.

    Usage::

        cache = TTLCache(ttl_seconds=3600)
        cache.set("my_key", expensive_result)
        value = cache.get("my_key")   # returns result while fresh
        cache.invalidate("my_key")    # force removal
        cache.clear()                 # remove all entries
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 2048) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        # Maps key → (expires_at_unix_timestamp, value)
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> object | None:
        """Return the cached value for *key*, or ``None`` if missing/expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() >= expires_at:
                # Lazy eviction on read
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: object) -> None:
        """Store *value* under *key* with the configured TTL.

        If the cache is at ``max_entries`` capacity, the oldest-by-insertion
        entry is evicted (FIFO) to prevent unbounded memory growth from
        pathological/adversarial key churn.
        """
        with self._lock:
            if key not in self._data and len(self._data) >= self._max_entries:
                # FIFO eviction — pop the first inserted key.
                # dict preserves insertion order in Python 3.7+.
                oldest_key = next(iter(self._data))
                self._data.pop(oldest_key, None)
            self._data[key] = (time.time() + self._ttl, value)

    def invalidate(self, key: str) -> None:
        """Remove a single entry (no-op if the key does not exist)."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._data.clear()

    # ------------------------------------------------------------------
    # Introspection helpers (useful for tests and health-check endpoints)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries currently in the cache (may include expired)."""
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        """Return True if *key* is present and not expired."""
        return self.get(key) is not None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

#: Shared cache for /predictions Monte Carlo results.
#:
#: TTL=1 hour — acceptable staleness for a predictions page that is refreshed
#: by the scraper at most a few times per day.
predictions_cache: TTLCache = TTLCache(ttl_seconds=3600)

#: Cache for likely-competitor roster lookups (Issue #33).
#:
#: Key format: ``"roster:{discipline.value}:{season}:{gender.value}"``
#: TTL=1 hour — roster membership changes only when new event results are
#: ingested (which happens via the scraper, at most a few times per day).
#: Call ``likely_roster_cache.clear()`` after a scrape for immediate freshness,
#: or run ``uv run python scripts/clear_cache.py``.
likely_roster_cache: TTLCache = TTLCache(ttl_seconds=3600)

#: Server-side response cache for the read-heavy HTML landing/leaderboard/
#: athletes pages (Issue #97).  TTL=10 min — short enough that a manual nudge is
#: rarely needed, long enough to absorb traffic bursts.  Keys embed a *ratings
#: fingerprint* (see :func:`ratings_fingerprint`) so a re-backfill that mutates
#: μ/σ invalidates every entry automatically (mirrors how ``predictions_cache``
#: folds μ/σ into its key).  Flushed alongside the others by
#: ``scripts/clear_cache.py`` after the daily scrape.
html_page_cache: TTLCache = TTLCache(ttl_seconds=600)


def ratings_fingerprint(session) -> str:
    """Cheap whole-table fingerprint of the ``ratings`` table.

    Returns a stable string derived from aggregate stats (row count plus the
    sum/max of μ, rounded to keep float noise out).  Any backfill that changes a
    rating shifts the count or the μ-sum, so embedding this in a cache key makes
    stale ratings impossible to serve — without paying for a full per-row scan
    on every request.  This is the HTML-page analogue of the per-athlete μ/σ
    fingerprint that ``predictions_cache`` folds into its key.

    Computed with one aggregate query (no row materialisation).
    """
    # Imported lazily to keep the cache module dependency-free at import time
    # and avoid any import-order coupling with the ORM models.
    from sqlalchemy import func, select

    from climbing_elo.models import Rating

    count, mu_sum, mu_max = session.execute(
        select(
            func.count(Rating.id),
            func.coalesce(func.sum(Rating.mu), 0.0),
            func.coalesce(func.max(Rating.mu), 0.0),
        )
    ).one()
    return f"{int(count)}:{round(float(mu_sum), 2)}:{round(float(mu_max), 4)}"
