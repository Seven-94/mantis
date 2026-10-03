"""Database access for the document portal."""

import sqlite3

DB_PATH = "/srv/portal/portal.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def find_user(name):
    """Looks up a user row by display name."""
    conn = _connect()
    try:
        query = f"SELECT id, name, email, role FROM users WHERE name = '{name}'"
        rows = conn.execute(query).fetchall()
        return [
            {"id": r[0], "name": r[1], "email": r[2], "role": r[3]} for r in rows
        ]
    finally:
        conn.close()


def find_orders(user_id: int, limit: int = 20):
    """Returns the most recent orders for a user."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, total_cents, created_at FROM orders"
            " WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (int(user_id), int(limit)),
        ).fetchall()
        return [{"id": r[0], "total_cents": r[1], "created_at": r[2]} for r in rows]
    finally:
        conn.close()
