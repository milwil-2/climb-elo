"""Application-level rate limiter (Issue #34).

A single Limiter instance shared across app.py and route modules.

Security note
-------------
``get_remote_address`` reads ``request.client.host``.  Behind a reverse proxy
this returns the *proxy* IP, not the real client.  Production deployments should
either:

  a) Use ``X-Forwarded-For`` + a custom key_func restricted to trusted proxy IPs
     (never blindly trust ``X-Forwarded-For`` for security limits without
     validating the source).
  b) Rely on the reverse-proxy's own rate limiting (preferred — Issue #29).

Backend
-------
In-memory (default ``MemoryStorage``).  In multi-worker deploys each worker
maintains its own counter, so the effective per-IP limit is
``default_limit × num_workers``.  This is acceptable: the Issue #29 deployment
runs a single uvicorn worker behind a reverse proxy, and the reverse-proxy
limiter will supersede this in production.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

#: Default limit applied to all routes via SlowAPIMiddleware.
#: Stricter per-endpoint limits are applied with @limiter.limit() decorators.
#: headers_enabled=True adds X-RateLimit-* and Retry-After headers to responses.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],
    headers_enabled=True,
)
