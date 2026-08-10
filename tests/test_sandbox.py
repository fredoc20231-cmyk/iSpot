"""Unit tests for the subprocess sandbox (process/memory/timeout isolation)."""
import sys

import pytest

from ispot.sandbox import run_in_subprocess, SandboxError, SandboxTimeout

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")


def test_echo_roundtrip():
    payload = {"a": 1, "b": [1, 2, 3], "c": "hello"}
    assert run_in_subprocess("ispot._sandbox_selftest:echo", payload) == payload


def test_error_in_child_raises_sandbox_error():
    with pytest.raises(SandboxError) as exc:
        run_in_subprocess("ispot._sandbox_selftest:boom", {})
    assert "intentional failure" in str(exc.value)


def test_timeout_kills_child():
    with pytest.raises(SandboxTimeout):
        run_in_subprocess("ispot._sandbox_selftest:sleep_forever", {}, timeout=2)


def test_unknown_target_raises():
    with pytest.raises(SandboxError):
        run_in_subprocess("ispot._sandbox_selftest:does_not_exist", {})


@posix_only
def test_memory_limit_enforced():
    # Child tries to allocate ~1 GB but is capped at 256 MB -> must fail
    # (MemoryError reported back, or the interpreter aborts nonzero). Either
    # way the parent sees a SandboxError rather than a returned value.
    with pytest.raises(SandboxError):
        run_in_subprocess(
            "ispot._sandbox_selftest:hog", {"mb": 1024},
            timeout=30, memory_mb=256,
        )
