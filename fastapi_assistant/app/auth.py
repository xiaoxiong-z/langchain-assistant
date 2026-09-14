from __future__ import annotations

import hashlib
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "users.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)")
    conn.commit()
    return conn


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


def register(username: str, password: str) -> None:
    salt = secrets.token_hex(16)
    value = salt + ":" + _hash(password, salt)
    with _connect() as conn:
        try:
            conn.execute("INSERT INTO users VALUES (?, ?)", (username, value))
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("用户名已存在") from exc


def authenticate(username: str, password: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        return False
    salt, expected = row[0].split(":", 1)
    return secrets.compare_digest(_hash(password, salt), expected)


TOKENS: dict[str, str] = {}


def issue_token(username: str) -> str:
    token = secrets.token_urlsafe(32)
    TOKENS[token] = username
    return token


def username_from_token(token: str | None) -> str | None:
    return TOKENS.get(token or "")
