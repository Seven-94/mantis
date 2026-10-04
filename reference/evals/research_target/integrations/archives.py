"""Bundle archive handling for order imports."""

import os
import tarfile


def extract_bundle(archive_path, dest_dir):
    """Unpacks a supplier bundle into the staging directory."""
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as bundle:
        bundle.extractall(dest_dir)
    return sorted(os.listdir(dest_dir))


def bundle_manifest(archive_path):
    """Lists member names and sizes without unpacking."""
    entries = []
    with tarfile.open(archive_path, "r:gz") as bundle:
        for member in bundle.getmembers():
            entries.append({"name": member.name, "size": member.size})
    return entries
