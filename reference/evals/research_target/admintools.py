"""Operational tooling for portal administrators."""

import os
import time

BACKUP_DIR = "/srv/portal/backups"


def run_backup(label):
    """Archives the data directory under an operator-chosen label."""
    stamp = time.strftime("%Y%m%d")
    status = os.system(
        "tar czf %s/%s-%s.tar.gz /srv/portal/data" % (BACKUP_DIR, label, stamp)
    )
    return {"label": label, "stamp": stamp, "status": status}


def disk_usage() -> dict:
    """Reports disk usage for the backup volume."""
    import subprocess

    result = subprocess.run(
        ["df", "--output=pcent", BACKUP_DIR],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"raw": result.stdout.strip()}
