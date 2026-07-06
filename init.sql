CREATE TABLE companies (
    cik VARCHAR(10) PRIMARY KEY,
    entity_name TEXT NOT NULL,
    ticker VARCHAR(10),
    sic_code VARCHAR(4),
    sic_description TEXT
);

CREATE TABLE financials (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) REFERENCES companies(cik),
    metric_name VARCHAR(50) NOT NULL,
    fiscal_year INT,
    fiscal_period VARCHAR(2),
    unit VARCHAR(10) DEFAULT 'USD',
    value NUMERIC,
    filed_date DATE
);

CREATE TABLE evaluation_logs (
    id SERIAL PRIMARY KEY,
    user_query TEXT,
    generated_sql TEXT,
    raw_result TEXT,
    final_answer TEXT,
    sql_accuracy_score INT,
    faithfulness_score INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_companies_name_trgm ON companies USING gin (entity_name gin_trgm_ops);
