import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "sec_rag_db",
    "user": "raguser",
    "password": "ragpassword",
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)
