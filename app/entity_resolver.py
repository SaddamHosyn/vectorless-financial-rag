import psycopg2
from app.config import DB_CONFIG

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def resolve_entity(user_input: str):
    """
    Takes a raw string like 'Apple' or 'AAPL' and returns the matching CIK + official name.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT cik, entity_name, ticker FROM companies WHERE ticker = %s LIMIT 1;",
            (user_input.upper(),)
        )
        result = cursor.fetchone()
        if result:
            return {"cik": result[0], "entity_name": result[1], "ticker": result[2], "match_type": "exact_ticker"}
        
        cursor.execute(
            """SELECT cik, entity_name, ticker, similarity(entity_name, %s) AS sim
               FROM companies
               WHERE entity_name %% %s
               ORDER BY sim DESC
               LIMIT 1;""",
            (user_input, user_input)
        )
        result = cursor.fetchone()
        if result:
            return {"cik": result[0], "entity_name": result[1], "ticker": result[2], "match_type": "fuzzy_name", "similarity": result[3]}
        
        return None
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    test_queries = ["Apple", "AAPL", "Abbot Labs", "Microsft"]
    for q in test_queries:
        result = resolve_entity(q)
        print(f"Query: '{q}' -> {result}")
