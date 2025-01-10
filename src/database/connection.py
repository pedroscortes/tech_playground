import sqlite3
from pathlib import Path

def get_db_path():
    """Get the database file path"""
    project_root = Path(__file__).parent.parent.parent
    return project_root / 'data' / 'processed' / 'employee_feedback.db'

def get_db_connection():
    """Create a database connection"""
    try:
        db_path = get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise

if __name__ == "__main__":
    conn = get_db_connection()
    print("Successfully connected to the database!")
    conn.close()