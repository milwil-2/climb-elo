"""Application-level per-IP limits shared by HTML and private API routes.

Vercel overwrites its client-IP header; only that runtime trusts it. Local
servers use the socket peer. In-memory counters are separate in each server
instance, so edge firewall throttling is the additional boundary across
instances. See README.md for the endpoint limits and deployment behavior.
"""

import os
from ipaddress import ip_address

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_ip(request: Request) -> str:
    """Trust Vercel's overwritten client-IP header only in its runtime.

    Local servers continue to use their socket peer; user-supplied forwarding
    headers cannot change the rate-limit bucket outside Vercel.
    """
    if os.environ.get("VERCEL") == "1":
        forwarded = request.headers.get("x-vercel-forwarded-for", "").strip()
        try:
            return str(ip_address(forwarded))
        except ValueError:
            pass
    return get_remote_address(request)


#: Default limit applied to all routes via SlowAPIMiddleware.
#: Stricter per-endpoint limits are applied with @limiter.limit() decorators.
#: headers_enabled=True adds X-RateLimit-* and Retry-After headers to responses.
limiter = Limiter(
    key_func=client_ip,
    default_limits=["120/minute"],
    headers_enabled=True,
)
