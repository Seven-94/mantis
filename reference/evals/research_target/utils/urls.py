"""URL helpers for the route layer."""

from urllib.parse import urlparse

_PORTAL_HOSTS = ("portal.example.com", "docs.example.com")


def is_relative_url(target):
    """Accepts same-site navigation targets for post-login redirects."""
    if not target:
        return False
    return target.startswith("/")


def is_portal_host(url):
    """True when an absolute URL points at a first-party host."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and parsed.hostname in _PORTAL_HOSTS


def canonical_doc_url(doc_id):
    """Builds the public link for a published document."""
    return f"https://portal.example.com/docs/{int(doc_id)}"
