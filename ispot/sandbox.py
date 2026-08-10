"""
Subprocess sandbox for running untrusted plugin code (roadmap item 6).

Community plugins are arbitrary Python. Running them in the API process means a
plugin crash, memory blow-up, or infinite loop takes the server down (and, in a
multi-tenant deployment, exposes other users' in-process data). This module
runs a target callable in a separate short-lived subprocess with:

  * a separate address space (a crash or segfault cannot corrupt the parent),
  * an address-space (memory) rlimit and a CPU-time rlimit,
  * a wall-clock timeout that kills the child.

Communication is via pickled temp files. The target is given as
``"module:function"`` and receives a single picklable ``payload`` argument.

Limitation — network isolation: fully blocking network access requires OS-level
namespaces/containers and elevated privileges, which are out of scope for a
pure-Python sandbox. For untrusted multi-tenant use, run the API/worker itself
inside a locked-down container (no egress, read-only FS) so this subprocess
inherits that isolation. The process/memory/CPU/timeout isolation here is real
and enforced.
"""
from __future__ import annotations

import importlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover - non-POSIX
    resource = None

_REPO_ROOT = Path(__file__).resolve().parent.parent


class SandboxError(RuntimeError):
    """Raised when sandboxed execution fails (crash, nonzero exit, or error)."""


class SandboxTimeout(SandboxError):
    """Raised when the child exceeds its wall-clock timeout."""


def _make_preexec(memory_bytes: int | None, cpu_seconds: int | None):
    if resource is None or os.name != "posix":
        return None

    def _apply():  # pragma: no cover - runs in the child process
        if memory_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        if cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    return _apply


def _child_run(input_path: str, output_path: str, target: str) -> None:
    """Entry point executed inside the child process."""
    try:
        with open(input_path, "rb") as f:
            payload = pickle.load(f)
        mod_name, func_name = target.split(":", 1)
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        result = func(payload)
        out = {"ok": True, "result": result}
    except BaseException as e:  # noqa: BLE001 - report everything back to parent
        out = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    with open(output_path, "wb") as f:
        pickle.dump(out, f)


def run_in_subprocess(
    target: str,
    payload: Any,
    timeout: int = 300,
    memory_mb: int | None = 2048,
    cpu_seconds: int | None = None,
) -> Any:
    """Run ``target`` ("module:function") on ``payload`` in a sandboxed child.

    Returns the target's (picklable) return value. Raises ``SandboxTimeout`` on
    timeout and ``SandboxError`` on crash, nonzero exit, or an exception raised
    inside the target.
    """
    if cpu_seconds is None and timeout:
        cpu_seconds = int(timeout) + 5

    tmpdir = tempfile.mkdtemp(prefix="ispot-sbx-")
    in_path = os.path.join(tmpdir, "in.pkl")
    out_path = os.path.join(tmpdir, "out.pkl")
    try:
        with open(in_path, "wb") as f:
            pickle.dump(payload, f)

        preexec = _make_preexec(
            memory_mb * 1024 * 1024 if memory_mb else None, cpu_seconds
        )
        cmd = [
            sys.executable,
            "-c",
            "import sys; import ispot.sandbox as s; "
            "s._child_run(sys.argv[1], sys.argv[2], sys.argv[3])",
            in_path,
            out_path,
            target,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO_ROOT),
                capture_output=True,
                timeout=timeout,
                preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired:
            raise SandboxTimeout(f"Sandboxed target exceeded {timeout}s timeout")

        if not os.path.exists(out_path):
            stderr = proc.stderr.decode("utf-8", "replace")[-2000:]
            raise SandboxError(
                f"Sandbox produced no output (exit code {proc.returncode}). "
                f"stderr tail: {stderr}"
            )
        with open(out_path, "rb") as f:
            out = pickle.load(f)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not out.get("ok"):
        raise SandboxError(out.get("error", "unknown error") + "\n" + out.get("traceback", ""))
    return out["result"]


def _plugin_entry(payload: dict) -> dict:
    """Child-side entry that loads and runs a plugin on an on-disk AnnData.

    payload keys: name, filepath (optional), adata_path, n_clusters, seed.
    Returns {"labels": [...], "runtime": float, "embedding": None}.
    """
    import time

    import anndata as ad
    import numpy as np

    from ispot import plugins as _pl

    filepath = payload.get("filepath")
    if filepath:
        _pl._load_plugin_file(filepath, source="local")

    info = _pl._PLUGIN_REGISTRY.get(payload["name"])
    if info is None or info.run_func is None:
        raise RuntimeError(f"Plugin {payload['name']!r} not available in sandbox")

    adata = ad.read_h5ad(payload["adata_path"])
    t0 = time.time()
    result = info.run_func(adata, payload["n_clusters"], seed=payload.get("seed", 42))
    labels = np.array(result["labels"]).astype(str).tolist()
    return {
        "labels": labels,
        "runtime": float(result.get("runtime", time.time() - t0)),
        "embedding": None,
    }
