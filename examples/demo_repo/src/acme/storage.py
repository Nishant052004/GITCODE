"""SQLite storage helpers and the user repository."""
import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  email TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions(
  token TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  expires_at TEXT
);
"""


def connect_db(path: str) -> sqlite3.Connection:
    """Open a sqlite connection with WAL mode and the app schema applied."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


class UserRepository:
    """CRUD for user rows; passwords are stored as pbkdf2 hashes only."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def save_user(self, email: str, password_hash: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO users(email, password_hash) VALUES (?, ?)",
            (email, password_hash),
        )
        self.conn.commit()

    def get_password_hash(self, email: str) -> str | None:
        row = self.conn.execute(
            "SELECT password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
        return row[0] if row else None

    def save_session(self, token: str, email: str, expires_at: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions(token, email, expires_at) VALUES (?, ?, ?)",
            (token, email, expires_at),
        )
        self.conn.commit()
