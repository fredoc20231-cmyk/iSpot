"""Trivial, dependency-free targets used to test ispot.sandbox in isolation.

Kept in the package (not the tests dir) so the sandbox child process can import
them by module path without the tests directory being importable.
"""
from __future__ import annotations

import time


def echo(payload):
    """Return the payload unchanged."""
    return payload


def boom(payload):
    """Raise, to exercise error propagation from the child."""
    raise ValueError("intentional failure in sandbox child")


def sleep_forever(payload):
    """Sleep well past any reasonable timeout, to exercise the wall clock."""
    time.sleep(3600)
    return "should not reach here"


def hog(payload):
    """Allocate more memory than the sandbox's cap allows."""
    mb = int(payload.get("mb", 4096)) if isinstance(payload, dict) else 4096
    blob = bytearray(mb * 1024 * 1024)  # noqa: F841 - forces allocation
    return len(blob)
