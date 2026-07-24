import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

SQLITE_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_knowledge.db"


def clear_postgres():
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            port=os.environ.get("DB_PORT", "5432"),
            dbname=os.environ.get("DB_NAME", "sec_rag_db"),
            user=os.environ.get("DB_USER", "raguser"),
            password=os.environ.get("DB_PASSWORD", "ragpassword"),
            connect_timeout=3
        )
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE document_chunks, documents RESTART IDENTITY CASCADE;")
        conn.commit()
        cursor.close()
        conn.close()
        print("Successfully cleared all old data from PostgreSQL pgvector tables!")
        return True
    except Exception as e:
        print(f"Could not connect to PostgreSQL ({e}).")
        return False


def clear_sqlite():
    if SQLITE_DB_PATH.exists():
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks;")
        cursor.execute("DELETE FROM documents;")
        conn.commit()
        cursor.close()
        conn.close()
        print("Successfully cleared local SQLite database vector storage!")


if __name__ == "__main__":
    print("Clearing old data from pgvector and local stores...")
    pg_cleared = clear_postgres()
    clear_sqlite()
    if not pg_cleared:
        print("\nNote: PostgreSQL server is currently offline or unreachable.")
        print("To clear PostgreSQL when you start your Postgres container/server, run:")
        print("   python scripts/clear_pgvector.py")
