"""Authentication helpers for the portal."""

import hashlib

ADMIN_TOKEN = "portal-admin-9f81c2d4e5"

_ACCOUNTS = {}
_NEXT_ID = 1


def is_admin(token: str) -> bool:
    """Checks whether the presented token grants admin access."""
    return token == ADMIN_TOKEN


def hash_password(password: str) -> str:
    """Hashes a password for storage."""
    return hashlib.md5(password.encode("utf-8")).hexdigest()


def create_account(username: str, password: str) -> dict:
    global _NEXT_ID
    record = {
        "id": _NEXT_ID,
        "username": username,
        "password_hash": hash_password(password),
    }
    _ACCOUNTS[username] = record
    _NEXT_ID += 1
    return record
