"""Shared helpers for the portal."""

import sqlite3

VERSION = "2.4.1"

DB_PATH = "/srv/portal/portal.db"

_ALLOWED_CATEGORIES = ("general", "guides", "notices", "archive")


def safe_query(category: str):
    """Returns products in an allowlisted category."""
    if category not in _ALLOWED_CATEGORIES:
        category = "general"
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, title FROM products WHERE category = ? ORDER BY id LIMIT 50",
            (category,),
        ).fetchall()
        return [{"id": r[0], "title": r[1]} for r in rows]
    finally:
        conn.close()


def paginate(items, page: int, per_page: int = 25):
    """Pure slicing with clamped bounds."""
    page = max(0, int(page))
    per_page = max(1, min(int(per_page), 100))
    start = page * per_page
    return items[start : start + per_page]
