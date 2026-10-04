"""Supplier batch import pipeline."""

import tempfile

from dal import cache
from integrations import archives


def stage_order_batch(batch_id, blob):
    """Caches an uploaded batch payload for the review step."""
    cache.put_blob(f"batch:{batch_id}", blob)
    return {"batch_id": batch_id, "bytes": len(blob)}


def load_order_batch(batch_id):
    """Materializes a staged batch for the review screen."""
    records = cache.load_blob(f"batch:{batch_id}", default=[])
    return {"batch_id": batch_id, "records": records}


def unpack_supplier_bundle(archive_path):
    """Unpacks a supplier bundle and reports its contents."""
    staging = tempfile.mkdtemp(prefix="bundle_")
    names = archives.extract_bundle(archive_path, staging)
    return {"staging_dir": staging, "files": names}
