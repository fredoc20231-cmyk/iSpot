"""Unit tests for upload validation and resource limits."""
import io

import pytest

from ispot import validation
from ispot.validation import ValidationError


# --- extension checks ------------------------------------------------------

@pytest.mark.parametrize("name", ["sample.h5ad", "DATA.H5", "counts.csv", "a.b.h5ad"])
def test_allowed_extensions(name):
    assert validation.has_allowed_extension(name)
    validation.validate_extension(name)  # must not raise


@pytest.mark.parametrize("name", ["sample.txt", "archive.zip", "noext", "", None])
def test_rejected_extensions(name):
    assert not validation.has_allowed_extension(name)
    with pytest.raises(ValidationError) as exc:
        validation.validate_extension(name)
    assert exc.value.status_code == 400


# --- spot-count checks -----------------------------------------------------

def test_spot_count_within_limit_ok():
    validation.validate_spot_count(1000, max_spots=5000)  # no raise


def test_spot_count_over_limit_raises_413():
    with pytest.raises(ValidationError) as exc:
        validation.validate_spot_count(6000, max_spots=5000)
    assert exc.value.status_code == 413


def test_spot_count_none_is_ignored():
    validation.validate_spot_count(None, max_spots=1)  # no raise


# --- streamed size cap -----------------------------------------------------

def test_stream_to_file_under_cap_writes_all(tmp_path):
    data = b"x" * 2048
    dest = tmp_path / "out.bin"
    written = validation.stream_to_file(io.BytesIO(data), str(dest), max_bytes=4096)
    assert written == 2048
    assert dest.read_bytes() == data


def test_stream_to_file_over_cap_raises_and_cleans_up(tmp_path):
    data = b"x" * 5000
    dest = tmp_path / "out.bin"
    with pytest.raises(ValidationError) as exc:
        validation.stream_to_file(io.BytesIO(data), str(dest), max_bytes=1024)
    assert exc.value.status_code == 413
    assert not dest.exists()  # partial file removed


# --- env-configured limits -------------------------------------------------

def test_env_overrides_limits(monkeypatch):
    monkeypatch.setenv("ISPOT_MAX_UPLOAD_MB", "10")
    monkeypatch.setenv("ISPOT_MAX_SPOTS", "1234")
    assert validation.get_max_upload_bytes() == 10 * 1024 * 1024
    assert validation.get_max_spots() == 1234


def test_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ISPOT_MAX_SPOTS", "not-a-number")
    assert validation.get_max_spots() == validation.DEFAULT_MAX_SPOTS
