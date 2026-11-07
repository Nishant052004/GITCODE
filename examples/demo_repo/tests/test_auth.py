"""Tests for acme.auth password hashing and login."""
import pytest

from acme.auth import (
    AuthenticationError,
    UserNotFoundError,
    hash_password,
    login,
    verify_password,
)


def test_hash_and_verify_roundtrip():
    stored = hash_password("s3cret!")
    assert stored.startswith("pbkdf2$")
    assert verify_password("s3cret!", stored)


def test_verify_rejects_wrong_password():
    stored = hash_password("s3cret!")
    assert not verify_password("wrong", stored)


def test_login_returns_session_token():
    users = {"ada@example.com": hash_password("pw")}
    token = login("ada@example.com", "pw", users)
    assert token.startswith("ada@example.com|")


def test_login_unknown_user():
    with pytest.raises(UserNotFoundError):
        login("ghost@example.com", "pw", {})


def test_login_bad_password():
    users = {"ada@example.com": hash_password("pw")}
    with pytest.raises(AuthenticationError):
        login("ada@example.com", "not-the-password", users)
