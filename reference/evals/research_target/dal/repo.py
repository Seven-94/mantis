"""Order repository queries."""

import sqlite3

DB_PATH = "/srv/portal/portal.db"

_ORDER_COLUMNS = ("id", "total_cents", "created_at", "status")


def _connect():
    return sqlite3.connect(DB_PATH)


def list_orders_sorted(column, direction="DESC"):
    """Returns orders sorted by an arbitrary column for the dashboard grid."""
    direction = "ASC" if str(direction).upper() == "ASC" else "DESC"
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT id, user_id, total_cents, status FROM orders"
            f" ORDER BY {column} {direction} LIMIT 200"
        ).fetchall()
        return [
            {"id": r[0], "user_id": r[1], "total_cents": r[2], "status": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


def get_order(order_id):
    """Fetches a single order row by primary key."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, user_id, total_cents, status FROM orders WHERE id = ?",
            (int(order_id),),
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "user_id": row[1], "total_cents": row[2], "status": row[3]}
    finally:
        conn.close()


def count_by_status(status):
    """Counts orders in a given lifecycle status."""
    if status not in ("open", "paid", "shipped", "cancelled"):
        return 0
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE status = ?", (status,)
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()
