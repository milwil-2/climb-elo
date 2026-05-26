"""Tests for the TTLCache in-memory cache (Issue #28).

Covers:
- get/set roundtrip
- Expiration after TTL elapses (small TTL in test)
- Invalidation removes the entry
- clear() removes all entries
- Thread safety (concurrent get/set from multiple threads)
- __contains__ and __len__ helpers
- Global predictions_cache singleton exists and is a TTLCache
"""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from climbing_elo.cache import TTLCache, predictions_cache


# ---------------------------------------------------------------------------
# Basic get / set
# ---------------------------------------------------------------------------

class TestTTLCacheBasic:
    def test_get_set_roundtrip(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("foo", {"data": 42})
        result = cache.get("foo")
        assert result == {"data": 42}

    def test_get_missing_key_returns_none(self):
        cache = TTLCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_set_overwrites_existing(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_multiple_keys_independent(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------

class TestTTLExpiration:
    def test_value_available_before_expiry(self):
        cache = TTLCache(ttl_seconds=10)
        cache.set("fresh", "still good")
        assert cache.get("fresh") == "still good"

    def test_value_expired_after_ttl(self):
        """Use a 0.1-second TTL so the test completes quickly."""
        cache = TTLCache(ttl_seconds=0)  # expires immediately
        cache.set("stale", "gone")
        # Even ttl=0 means expires_at == now; sleep a tiny bit to guarantee
        time.sleep(0.05)
        assert cache.get("stale") is None

    def test_expired_entry_evicted_from_data(self):
        cache = TTLCache(ttl_seconds=0)
        cache.set("bye", "value")
        time.sleep(0.05)
        cache.get("bye")  # triggers lazy eviction
        assert len(cache) == 0

    def test_not_expired_entry_still_present(self):
        cache = TTLCache(ttl_seconds=3600)
        cache.set("alive", True)
        assert len(cache) == 1

    def test_short_ttl_expires(self):
        """TTL of 0.2s — confirm expiry without flakiness."""
        cache = TTLCache(ttl_seconds=1)
        # Manually set a past expiry to simulate elapsed TTL
        key = "short_lived"
        cache.set(key, "data")
        # Poke the internal dict to back-date the expiry
        with cache._lock:
            expires_at, val = cache._data[key]
            cache._data[key] = (time.time() - 1, val)  # already expired
        assert cache.get(key) is None


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

class TestTTLCacheInvalidation:
    def test_invalidate_removes_entry(self):
        cache = TTLCache(ttl_seconds=3600)
        cache.set("target", "delete me")
        cache.invalidate("target")
        assert cache.get("target") is None

    def test_invalidate_nonexistent_key_is_noop(self):
        cache = TTLCache(ttl_seconds=3600)
        cache.invalidate("ghost")  # should not raise

    def test_invalidate_only_removes_target_key(self):
        cache = TTLCache(ttl_seconds=3600)
        cache.set("keep", "me")
        cache.set("remove", "bye")
        cache.invalidate("remove")
        assert cache.get("keep") == "me"
        assert cache.get("remove") is None


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

class TestTTLCacheClear:
    def test_clear_removes_all_entries(self):
        cache = TTLCache(ttl_seconds=3600)
        for i in range(10):
            cache.set(f"key_{i}", i)
        cache.clear()
        assert len(cache) == 0
        for i in range(10):
            assert cache.get(f"key_{i}") is None

    def test_clear_on_empty_cache_is_noop(self):
        cache = TTLCache(ttl_seconds=3600)
        cache.clear()  # should not raise
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# __contains__ and __len__
# ---------------------------------------------------------------------------

class TestTTLCacheHelpers:
    def test_contains_true_for_fresh_key(self):
        cache = TTLCache(ttl_seconds=3600)
        cache.set("present", 99)
        assert "present" in cache

    def test_contains_false_for_missing_key(self):
        cache = TTLCache(ttl_seconds=3600)
        assert "absent" not in cache

    def test_len_counts_entries(self):
        cache = TTLCache(ttl_seconds=3600)
        assert len(cache) == 0
        cache.set("x", 1)
        assert len(cache) == 1
        cache.set("y", 2)
        assert len(cache) == 2
        cache.invalidate("x")
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestTTLCacheThreadSafety:
    def test_concurrent_set_get_no_exception(self):
        """Many threads hammering the same cache must not raise."""
        cache = TTLCache(ttl_seconds=3600)
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                key = f"key_{n % 10}"
                cache.set(key, n)
                val = cache.get(key)
                # val may be None (overwritten by another thread) or an int
                assert val is None or isinstance(val, int)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(worker, i) for i in range(200)]
            for f in futures:
                f.result()

        assert errors == [], f"Thread safety violations: {errors}"

    def test_concurrent_clear_and_set(self):
        """clear() mid-write must not corrupt state."""
        cache = TTLCache(ttl_seconds=3600)
        errors: list[Exception] = []
        done = threading.Event()

        def setter() -> None:
            for i in range(50):
                try:
                    cache.set(f"s_{i}", i)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
            done.set()

        def clearer() -> None:
            while not done.is_set():
                try:
                    cache.clear()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        t_set = threading.Thread(target=setter)
        t_clr = threading.Thread(target=clearer)
        t_set.start()
        t_clr.start()
        t_set.join(timeout=5)
        t_clr.join(timeout=5)

        assert errors == [], f"Thread safety violations: {errors}"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

class TestGlobalSingleton:
    def test_predictions_cache_is_ttl_cache(self):
        assert isinstance(predictions_cache, TTLCache)

    def test_predictions_cache_ttl_is_one_hour(self):
        assert predictions_cache._ttl == 3600

    def test_predictions_cache_is_writable(self):
        """Confirm the singleton can be used as a cache (set → get → clear)."""
        predictions_cache.set("__test__", "ok")
        assert predictions_cache.get("__test__") == "ok"
        predictions_cache.invalidate("__test__")
