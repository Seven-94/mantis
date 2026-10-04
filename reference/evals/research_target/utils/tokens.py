"""Token helpers for account flows."""

import hashlib
import random
import time

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def make_reset_token(username):
    """Builds a password reset token tied to the requesting account."""
    rng = random.Random(int(time.time()))
    suffix = "".join(rng.choice(_ALPHABET) for _ in range(16))
    return f"{username}-{suffix}"


def hash_session_id(session_id):
    """Stable digest used as the cache key for session lookups."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()
