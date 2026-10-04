"""Webhook delivery for order status callbacks."""

import json
import urllib.request

_TIMEOUT_SECONDS = 5


def deliver(url, event):
    """POSTs an event payload to a subscriber-provided callback URL."""
    data = json.dumps(event).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        return {"status": resp.status, "url": url}


def ping(url):
    """Delivers a small test event so subscribers can verify wiring."""
    return deliver(url, {"type": "ping"})
