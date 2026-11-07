"""Authentication: password hashing, verification, login and user lookup."""
import hashlib
import hmac
import os
import uuid


class UserNotFoundError(KeyError):
    """Raised when a login references an email that does not exist."""


class AuthenticationError(Exception):
    """Raised when the supplied password does not match the stored hash."""


def hash_password(password: str, iterations: int = 100_000) -> str:
    """Derive a salted PBKDF2-SHA256 hash for storage."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time comparison of a candidate password against the stored hash.

    The stored format is `pbkdf2$<iterations>$<salt hex>$<digest hex>`, as
    produced by hash_password.
    """
    scheme, iterations, salt_hex, digest_hex = stored.split("$", 3)
    if scheme != "pbkdf2":
        raise ValueError(f"unsupported password scheme: {scheme}")
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def login(email: str, password: str, users: dict) -> str:
    """Verify credentials and return a fresh session token.

    Raises UserNotFoundError when the email is unknown and
    AuthenticationError when the password does not verify.
    """
    stored = users.get(email)
    if stored is None:
        raise UserNotFoundError(email)
    if not verify_password(password, stored):
        raise AuthenticationError(f"bad password for {email}")
    return create_session(email)


def create_session(email: str) -> str:
    """Mint an opaque session token bound to an email address."""
    return f"{email}|{uuid.uuid4().hex}"


def get_user(users: dict, email: str) -> dict:
    """Fetch a user record; raises KeyError (UserNotFoundError) when missing."""
    if email not in users:
        raise UserNotFoundError(email)
    return {"email": email, "password_hash": users[email]}
