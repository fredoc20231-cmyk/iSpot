"""
Optional API-key authentication.

A coarse gate for the compute-triggering and mutating endpoints (upload,
benchmark, plugin registration). When ``ISPOT_API_KEY`` is unset, auth is
disabled and the server behaves as the open MVP (default for local/self-hosted
single-tenant use). When set, requests must present the key in the
``X-API-Key`` header.

This is not full multi-tenancy (roadmap item 11) — there is one shared key and
jobs are not partitioned per user — but it keeps a public beta from exposing
unauthenticated compute. Pure stdlib so the check is unit-testable without a
web framework.
"""
from __future__ import annotations

import hmac
import os
from typing import Optional


def get_configured_key() -> Optional[str]:
    """Return the configured API key, or None if auth is disabled."""
    key = os.environ.get("ISPOT_API_KEY", "").strip()
    return key or None


def is_authorized(expected: Optional[str], provided: Optional[str]) -> bool:
    """Constant-time key check.

    Returns True when auth is disabled (no expected key). Otherwise requires a
    provided key that matches, compared with ``hmac.compare_digest`` to avoid
    timing leaks.
    """
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(str(expected), str(provided))
