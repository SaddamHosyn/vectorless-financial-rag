import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.config import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute(
    "SELECT chunk_text FROM document_chunks WHERE chunk_text LIKE '%early repayment%' LIMIT 5;"
)
rows = cursor.fetchall()

print(f"Found {len(rows)} rows with 'early repayment' content")
for r in rows:
    print("---")
    print(r[0][:200])

cursor.close()
conn.close()
