import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.config import get_connection

conn = get_connection()
cursor = conn.cursor()

import sys

# Usage: python scripts/dev/delete_docs.py <doc_id1> <doc_id2> ...
# Example: python scripts/dev/delete_docs.py 5 12
if len(sys.argv) < 2:
    print("Usage: python delete_docs.py <doc_id1> [doc_id2 ...]")
    sys.exit(1)

doc_ids = [int(x) for x in sys.argv[1:]]
placeholders = ", ".join(["%s"] * len(doc_ids))

cursor.execute(f"DELETE FROM documents WHERE id IN ({placeholders});", doc_ids)
conn.commit()

print(f"Deleted {cursor.rowcount} document(s) with IDs: {doc_ids}.")

cursor.close()
conn.close()
