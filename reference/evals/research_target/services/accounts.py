"""Account profile management."""

import sqlite3

from utils import tokens

DB_PATH = "/srv/portal/portal.db"

_PROFILE_COLUMNS = ("display_name", "email", "locale", "role", "bio")


def update_profile(user_id, fields):
    """Applies the submitted profile form to the user row."""
    updates = {k: v for k, v in fields.items() if k in _PROFILE_COLUMNS}
    if not updates:
        return {"updated": 0}
    assignments = ", ".join(f"{column} = ?" for column in updates)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            f"UPDATE users SET {assignments} WHERE id = ?",
            (*updates.values(), int(user_id)),
        )
        conn.commit()
        return {"updated": len(updates)}
    finally:
        conn.close()


def start_password_reset(username):
    """Issues a reset token and stores its digest for later verification."""
    token = tokens.make_reset_token(username)
    digest = tokens.hash_session_id(token)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "UPDATE users SET reset_digest = ? WHERE name = ?", (digest, username)
        )
        conn.commit()
    finally:
        conn.close()
    return token
