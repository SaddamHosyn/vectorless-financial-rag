import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.config import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("DELETE FROM documents WHERE id IN (96, 98);")
conn.commit()

print(f"Deleted {cursor.rowcount} document(s).")

cursor.close()
conn.close()
