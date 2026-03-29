import sqlite3

DB_NAME = "users.db"

def init_db():
    """Initialize the database and create table if not exists"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            chat_id TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_user(email: str, password: str) -> bool:
    """Add a new user. Returns True if successful, False if email already exists."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        c.execute(
            "INSERT INTO users (email, password, chat_id) VALUES (?, ?, ?)",
            (email, password, None)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:  # Email already exists
        return False
    except Exception as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        conn.close()


def login_user(email: str, password: str):
    """Login user. Returns user tuple if successful, else None."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        c.execute(
            "SELECT * FROM users WHERE email = ? AND password = ?",
            (email, password)
        )
        user = c.fetchone()
        return user
    finally:
        conn.close()


def update_chat_id(email: str, chat_id: str) -> bool:
    """Update chat_id for a user. Returns True if updated."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        c.execute(
            "UPDATE users SET chat_id = ? WHERE email = ?",
            (chat_id, email)
        )
        conn.commit()
        return c.rowcount > 0  # Return True if any row was updated
    finally:
        conn.close()


def get_chat_id(email: str):
    """Get chat_id for a user. Returns chat_id or None."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        c.execute("SELECT chat_id FROM users WHERE email = ?", (email,))
        data = c.fetchone()
        return data[0] if data else None
    finally:
        conn.close()


# Initialize database when module is imported
init_db()
