"""Parity regression: scraper's ``_parse_boulder_score`` (raw-string branch,
no ``ascents``) must produce the same ``score_normalized`` as the engine's
``normalize_boulder_score`` for every string form the DB stores.

Motivation (#168): the two regexes silently diverged for the pre-2018
lowercase feed with omitted attempts. #115 relaxed the engine to accept
``\\d*``; the scraper stayed on ``\\d+``. Once #155 let the scraper reach
placeholder pre-2018 events, ~2,610 rows landed with NULL score_normalized
and boulder μ-p95 dropped 56 points out of band.

If either side changes format handling in the future, this test fires
before the divergence reaches prod.
"""

import pytest

from climbing_elo.engine.elo import normalize_boulder_score
from climbing_elo.scraper.ifsc_api import _parse_boulder_score

# (raw_score, expected_normalized) — cover every format the DB stores.
# Modern decimal cases are NOT included here because they require ``ascents``
# to normalize; the scraper stores NULL for decimal-without-ascents (see #117),
# and the engine returns None for decimal-only strings (#117) — they DO agree,
# both return None, and the ascents-derived path is tested elsewhere.
PARITY_CASES = [
    # Post-2018 upper-case ``NTMz A B``
    ("1T2z 3 4", 1166.0),
    ("2T2z 2 2", 2178.0),
    ("0T1z 0 5", 95.0),
    # Post-2018 alt ``NT A MBB``
    ("2T3 4B5", 2365.0),
    # Pre-2018 lowercase with attempt counts present
    ("5t6 5b6", 5434.0),
    # Pre-2018 lowercase with attempt counts OMITTED (#115/#168 relaxation).
    # Separator can be regular space OR non-breaking space (U+00A0).
    ("0t 4b10", 390.0),
    ("0t\xa04b10", 390.0),
    ("0t 0b", 0.0),
    ("0t 3b7", 293.0),
    # DNF / DNS / empty — both sides must yield None
    ("DNF", None),
    ("DNS", None),
    ("-", None),
    ("", None),
    # Unparseable garbage — None on both sides
    ("not a score", None),
]


@pytest.mark.parametrize("raw,expected", PARITY_CASES)
def test_scraper_and_engine_parse_identically(raw, expected):
    _, scraper_out = _parse_boulder_score(raw, ascents=None)
    engine_out = normalize_boulder_score(raw)
    assert scraper_out == expected, (
        f"scraper returned {scraper_out!r} for {raw!r}, expected {expected!r}"
    )
    assert engine_out == expected, (
        f"engine returned {engine_out!r} for {raw!r}, expected {expected!r}"
    )
    assert scraper_out == engine_out, (
        f"scraper/engine divergence on {raw!r}: "
        f"scraper={scraper_out!r} engine={engine_out!r}"
    )
