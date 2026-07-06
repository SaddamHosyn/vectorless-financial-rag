import os
import time
from edgar import Company, set_identity
from sec2md import convert_to_markdown, extract_sections, chunk_section, Item10K
from openai import OpenAI
from db_connector import get_connection

set_identity("Your Name your-email@example.com")
client = OpenAI()

DEMO_TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

TARGET_SECTIONS = {
    Item10K.RISK_FACTORS: "Risk Factors",
    Item10K.MANAGEMENT_DISCUSSION: "MD&A",
    Item10K.LEGAL_PROCEEDINGS: "Legal Proceedings"
}

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def process_company(ticker, cursor):
    print(f"\nProcessing {ticker}...")
    company = Company(ticker)
    filing = company.get_filings(form="10-K").latest()
    
    cik = str(company.cik).zfill(10)
    filing_date = filing.filing_date
    filing_url = filing.filing_url

    cursor.execute(
        """INSERT INTO filing_documents (cik, filing_type, filing_date, filing_url)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (cik, "10-K", filing_date, filing_url)
    )
    filing_id = cursor.fetchone()[0]

    html = filing.html()
    pages = convert_to_markdown(html, return_pages=True)
    sections = extract_sections(pages, filing_type="10-K")

    for item_enum, label in TARGET_SECTIONS.items():
        section = sections.get(item_enum)
        if not section:
            continue

        header = f"# {ticker} 10-K | {label}"
        chunks = chunk_section(section, header=header, chunk_size=512)

        for chunk in chunks:
            embedding = get_embedding(chunk.embedding_text)
            cursor.execute(
                """INSERT INTO filing_chunks (filing_id, section_name, chunk_text, embedding)
                   VALUES (%s, %s, %s, %s)""",
                (filing_id, label, chunk.content, embedding)
            )
            time.sleep(0.1)

    print(f"Finished {ticker}: inserted sections for {len(TARGET_SECTIONS)} categories")

def run():
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for ticker in DEMO_TICKERS:
            try:
                process_company(ticker, cursor)
                conn.commit()
            except Exception as e:
                print(f"Failed on {ticker}: {e}")
                conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run()