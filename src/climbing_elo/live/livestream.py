"""YouTube livestream URL parsing & validation (Issue #23).

Used by the /live/{event_id} HTML route to safely embed an IFSC YouTube live
stream when an Event has a non-null ``livestream_url``.

Only ``youtube.com`` / ``www.youtube.com`` / ``m.youtube.com`` and the
``youtu.be`` short-link host are accepted. Anything else (including
``javascript:`` URIs, arbitrary HTTP hosts, or YouTube look-alikes) is
rejected by returning ``None``.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import parse_qs, urlparse

# Strict allowlist: only these hostnames may produce a non-None video id.
_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }
)
_YOUTU_BE_HOSTS = frozenset({"youtu.be", "www.youtu.be"})

# YouTube video IDs are exactly 11 chars from [A-Za-z0-9_-]. Validating the
# length + character set defends against quirky inputs (smuggled query
# params, path traversal, etc.) before we interpolate the id into an iframe
# src attribute.
_VIDEO_ID_LEN = 11
_VIDEO_ID_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)


def _is_valid_video_id(vid: str) -> bool:
    return len(vid) == _VIDEO_ID_LEN and all(c in _VIDEO_ID_CHARS for c in vid)


def parse_youtube_video_id(url: Optional[str]) -> Optional[str]:
    """Return the 11-character YouTube video ID, or ``None`` if invalid.

    Accepts the common URL shapes:

    - ``https://www.youtube.com/watch?v=VIDEO_ID``
    - ``https://youtube.com/live/VIDEO_ID``
    - ``https://www.youtube.com/embed/VIDEO_ID``
    - ``https://youtu.be/VIDEO_ID``

    Returns ``None`` for any other host, any non-HTTPS/HTTP scheme,
    or any malformed video id.
    """
    if not url or not isinstance(url, str):
        return None

    try:
        parsed = urlparse(url.strip())
    except (ValueError, AttributeError):
        return None

    # Only allow http(s); blocks javascript:, data:, file:, etc.
    if parsed.scheme not in {"http", "https"}:
        return None

    host = (parsed.hostname or "").lower()
    if not host:
        return None

    # youtu.be short links: /<video_id>
    if host in _YOUTU_BE_HOSTS:
        vid = parsed.path.lstrip("/").split("/", 1)[0]
        return vid if _is_valid_video_id(vid) else None

    if host not in _YOUTUBE_HOSTS:
        return None

    path = parsed.path or ""

    # /watch?v=VIDEO_ID
    if path == "/watch":
        qs = parse_qs(parsed.query)
        vids = qs.get("v") or []
        if vids:
            vid = vids[0]
            return vid if _is_valid_video_id(vid) else None
        return None

    # /live/VIDEO_ID, /embed/VIDEO_ID, /shorts/VIDEO_ID
    for prefix in ("/live/", "/embed/", "/shorts/", "/v/"):
        if path.startswith(prefix):
            vid = path[len(prefix) :].split("/", 1)[0].split("?", 1)[0]
            return vid if _is_valid_video_id(vid) else None

    return None


def youtube_embed_url(url: Optional[str]) -> Optional[str]:
    """Return a safe ``https://www.youtube.com/embed/<id>`` URL, or ``None``.

    Always uses HTTPS and the canonical embed host. ``?rel=0`` keeps the
    end-screen suggestions inside the same channel (per YouTube docs),
    and we deliberately omit ``autoplay`` to comply with the embeddable-
    player rules and the issue's "no autoplay with sound" requirement.
    """
    video_id = parse_youtube_video_id(url)
    if video_id is None:
        return None
    return f"https://www.youtube.com/embed/{video_id}?rel=0"
