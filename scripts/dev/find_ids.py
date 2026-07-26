import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.config import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute(
    "SELECT id, filename FROM documents WHERE filename LIKE '%Complaints%' OR filename LIKE '%repayment%' OR filename LIKE '%hardship%';"
)
rows = cursor.fetchall()

for r in rows:
    print(r)

cursor.close()
conn.close()
