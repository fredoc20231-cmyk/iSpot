"""Unit tests for optional API-key auth."""
from ispot import auth


def test_auth_disabled_allows_everything(monkeypatch):
    monkeypatch.delenv("ISPOT_API_KEY", raising=False)
    assert auth.get_configured_key() is None
    assert auth.is_authorized(None, None) is True
    assert auth.is_authorized(None, "anything") is True


def test_missing_key_rejected_when_configured():
    assert auth.is_authorized("secret", None) is False
    assert auth.is_authorized("secret", "") is False


def test_wrong_key_rejected():
    assert auth.is_authorized("secret", "nope") is False


def test_correct_key_accepted():
    assert auth.is_authorized("secret", "secret") is True


def test_configured_key_is_trimmed(monkeypatch):
    monkeypatch.setenv("ISPOT_API_KEY", "  k123  ")
    assert auth.get_configured_key() == "k123"


def test_blank_key_means_disabled(monkeypatch):
    monkeypatch.setenv("ISPOT_API_KEY", "   ")
    assert auth.get_configured_key() is None
