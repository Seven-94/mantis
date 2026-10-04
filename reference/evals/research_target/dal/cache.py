"""Small object cache shared by the import pipeline."""

import pickle
import time

_STORE = {}
_TTL_SECONDS = 900


def put_blob(key, blob):
    """Stores a serialized payload with its arrival time."""
    _STORE[key] = (time.time(), blob)


def load_blob(key, default=None):
    """Deserializes and returns a stored payload if it is still fresh."""
    entry = _STORE.get(key)
    if entry is None:
        return default
    stored_at, blob = entry
    if time.time() - stored_at > _TTL_SECONDS:
        _STORE.pop(key, None)
        return default
    return pickle.loads(blob)


def evict_expired():
    """Drops entries past their freshness window."""
    cutoff = time.time() - _TTL_SECONDS
    for key in [k for k, (ts, _) in _STORE.items() if ts < cutoff]:
        _STORE.pop(key, None)
