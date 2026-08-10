"""
Upload validation and resource limits.

Guards the compute-heavy benchmark pipeline against bad or oversized inputs
before any method is dispatched: an accepted file must have a supported
extension and stay under a configurable byte cap, and a profiled dataset must
stay under a configurable spot-count cap. Limits are read from the environment
so deployments can tune them without code changes:

    ISPOT_MAX_UPLOAD_MB   maximum upload size in megabytes (default 500)
    ISPOT_MAX_SPOTS       maximum number of spots per dataset (default 500000)

Pure Python (stdlib only) so the logic is unit-testable in isolation.
"""
from __future__ import annotations

import os
from typing import BinaryIO, Iterable

ALLOWED_EXTENSIONS: tuple[str, ...] = (".h5ad", ".h5", ".csv")
DEFAULT_MAX_UPLOAD_MB = 500
DEFAULT_MAX_SPOTS = 500_000


class ValidationError(Exception):
    """Raised when an upload or dataset violates a configured limit.

    ``status_code`` mirrors the HTTP status the API should return.
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def get_max_upload_bytes() -> int:
    return _env_int("ISPOT_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024


def get_max_spots() -> int:
    return _env_int("ISPOT_MAX_SPOTS", DEFAULT_MAX_SPOTS)


def has_allowed_extension(
    filename: str | None, allowed: Iterable[str] = ALLOWED_EXTENSIONS
) -> bool:
    name = (filename or "").lower()
    return name.endswith(tuple(a.lower() for a in allowed))


def validate_extension(
    filename: str | None, allowed: Iterable[str] = ALLOWED_EXTENSIONS
) -> None:
    if not has_allowed_extension(filename, allowed):
        allowed_str = ", ".join(allowed)
        raise ValidationError(
            f"Unsupported file type '{filename}'. Allowed extensions: {allowed_str}.",
            status_code=400,
        )


def validate_spot_count(n_spots: int | None, max_spots: int | None = None) -> None:
    if max_spots is None:
        max_spots = get_max_spots()
    if n_spots is not None and n_spots > max_spots:
        raise ValidationError(
            f"Dataset has {n_spots} spots, exceeding the limit of {max_spots}. "
            f"Raise ISPOT_MAX_SPOTS to allow larger inputs.",
            status_code=413,
        )


def safe_child_path(base_dir: str, name: str) -> str:
    """Resolve ``name`` under ``base_dir``, rejecting path traversal.

    Prevents ``../`` sequences and absolute paths from escaping ``base_dir``
    (e.g. a download request for ``../../meta_learning.db``). Returns the
    resolved absolute path; raises ``ValidationError`` (400) if it would fall
    outside ``base_dir``. Does not require the path to exist.
    """
    base = os.path.realpath(str(base_dir))
    target = os.path.realpath(os.path.join(base, name))
    if target != base and not target.startswith(base + os.sep):
        raise ValidationError(f"Invalid path: {name!r}", status_code=400)
    return target


def stream_to_file(
    fileobj: BinaryIO,
    dest_path: str,
    max_bytes: int | None = None,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Copy ``fileobj`` to ``dest_path`` in chunks, enforcing a byte cap.

    Streaming avoids loading the whole upload into memory. If the cap is
    exceeded the partial file is removed and a ``ValidationError`` (413) is
    raised. Returns the number of bytes written.
    """
    if max_bytes is None:
        max_bytes = get_max_upload_bytes()

    total = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = fileobj.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValidationError(
                        "Upload exceeds the maximum allowed size of "
                        f"{max_bytes // (1024 * 1024)} MB.",
                        status_code=413,
                    )
                out.write(chunk)
    except ValidationError:
        try:
            os.remove(dest_path)
        except OSError:
            pass
        raise
    return total
