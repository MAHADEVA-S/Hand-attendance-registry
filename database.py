import sqlite3
import time
import logging
from datetime import datetime
from typing import Optional

class Database:
    def __init__(self):
        self.conn = sqlite3.connect(
            "attendance.db",
            check_same_thread=False,
            timeout=30,
            isolation_level=None
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-10000")
        self.cursor = self.conn.cursor()
        self.setup_tables()

    def _execute_with_retry(self, query: str, params: tuple = (), retries: int = 3) -> bool:
        """Execute SQL query with retry logic"""
        logger = logging.getLogger(__name__)
        for attempt in range(retries):
            try:
                self.cursor.execute(query, params)
                self.conn.commit()
                return True
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    if attempt < retries - 1:
                        logger.warning(f"Database locked, retrying... (attempt {attempt + 1})")
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    logger.error("Max retries reached for database operation")
                logger.error(f"Database operation failed: {e}")
                raise
            except sqlite3.Error as e:
                logger.error(f"Database error: {e}")
                raise
        return False

    def setup_tables(self):
        """Create database tables"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sequence (
                last_number INTEGER DEFAULT 7000
            )
        """)
        
        self.cursor.execute("INSERT OR IGNORE INTO user_sequence (last_number) VALUES (7000)")
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                hand_landmarks BLOB,
                created_at TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                "S.No" INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                confidence REAL,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        """)
        self.conn.commit()

    def mark_attendance(self, user_id: str) -> bool:
        """Mark attendance for a user with timestamp"""
        now = datetime.now()
        timestamp = now.isoformat()
        
        query = """
            INSERT INTO attendance (user_id, entry_date, entry_time)
            VALUES (?, ?, ?)
        """
        params = (
            user_id,
            now.date().isoformat(),
            now.time().isoformat(),
            timestamp
        )
        
        logger = logging.getLogger(__name__)
        try:
            return self._execute_with_retry(query, params)
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            return False

    def get_next_user_id(self) -> str:
        """Generate next user ID in ESF7001 format"""
        # Get and increment sequence number
        self.cursor.execute("""
            UPDATE user_sequence
            SET last_number = last_number + 1
            RETURNING last_number
        """)
        seq_num = self.cursor.fetchone()[0]
        self.conn.commit()
        return f"ESF{seq_num}"
        
    def get_user_count(self) -> int:
        """Get number of users who marked attendance today"""
        today = datetime.now().date().isoformat()
        self.cursor.execute("""
            SELECT COUNT(*) FROM attendance
            WHERE entry_date = ?
        """, (today,))
        return self.cursor.fetchone()[0]

    def register_user(self, user_id: str, landmarks: list) -> bool:
        """Register a new user with their hand landmarks"""
        try:
            now = datetime.now().isoformat()
            # Serialize landmarks
            serialized = self._serialize_landmarks(landmarks)
            
            self.cursor.execute("""
                INSERT INTO users (user_id, hand_landmarks, created_at, last_seen)
                VALUES (?, ?, ?, ?)
            """, (user_id, serialized, now, now))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error registering user: {e}")
            return False

    def identify_user(self, landmarks: list) -> Optional[str]:
        """Identify user based on hand landmarks"""
        try:
            serialized = self._serialize_landmarks(landmarks)
            
            # Compare with existing users
            self.cursor.execute("""
                SELECT user_id FROM users
                WHERE hand_landmarks = ?
                LIMIT 1
            """, (serialized,))
            
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error identifying user: {e}")
            return None

    def _serialize_landmarks(self, landmarks: list) -> bytes:
        """Serialize hand landmarks for storage"""
        import pickle
        return pickle.dumps(landmarks)

    def _deserialize_landmarks(self, data: bytes) -> list:
        """Deserialize stored hand landmarks"""
        import pickle
        return pickle.loads(data)

    def update_user_last_seen(self, user_id: str) -> bool:
        """Update user's last seen timestamp"""
        try:
            now = datetime.now().isoformat()
            self.cursor.execute("""
                UPDATE users
                SET last_seen = ?
                WHERE user_id = ?
            """, (now, user_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating user: {e}")
            return False

    def close(self):
        """Close database connection"""
        self.conn.close()

if __name__ == "__main__":
    db = Database()
    print("Database setup complete!")
    db.close()
