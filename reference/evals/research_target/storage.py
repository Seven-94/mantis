"""Document storage for the portal."""

import os

DOC_ROOT = "/srv/portal/documents"


def fetch_document(doc_name):
    """Resolves a document name to a path under DOC_ROOT."""
    path = os.path.join(DOC_ROOT, doc_name)
    if not os.path.exists(path):
        raise FileNotFoundError(doc_name)
    return path


def store_upload(basename: str, data: bytes) -> str:
    """Stores an uploaded file under the uploads directory."""
    safe_name = os.path.basename(basename)
    path = os.path.join(DOC_ROOT, "uploads", safe_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return safe_name
