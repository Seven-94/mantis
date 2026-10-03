"""Link preview fetcher."""

import urllib.request

MAX_PREVIEW_BYTES = 4096


def preview(url):
    """Fetches the first bytes of a URL to render a link preview."""
    if not url:
        return ""
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read(MAX_PREVIEW_BYTES).decode("utf-8", errors="replace")
