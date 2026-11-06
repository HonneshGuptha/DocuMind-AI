"""
Knowledge Base Database Module
Handles SQLite schema creation, connection pooling, and indexing.
Enforces the Canonical First-Mention per Document policy.
"""

import sqlite3
import os
from typing import Optional
from config import Config


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    extraction_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    subject TEXT NOT NULL COLLATE NOCASE,
    predicate TEXT NOT NULL COLLATE NOCASE,
    object TEXT NOT NULL COLLATE NOCASE,
    source_sentence TEXT NOT NULL,
    sentence_index INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    UNIQUE (document_id, subject COLLATE NOCASE, predicate COLLATE NOCASE, object COLLATE NOCASE)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_facts_doc_spo ON facts(
    document_id,
    subject COLLATE NOCASE,
    predicate COLLATE NOCASE,
    object COLLATE NOCASE
);

CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_facts_document_id ON facts(document_id);

CREATE TABLE IF NOT EXISTS query_history (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question_text TEXT NOT NULL,
    category TEXT,
    expected_answer_field TEXT,
    parsing_status TEXT,
    answer_status TEXT,
    answer_type TEXT,
    answer_text TEXT,
    confidence TEXT,
    supporting_fact_ids TEXT,
    applied_rule_ids TEXT,
    explanation TEXT,
    execution_time_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_query_history_timestamp ON query_history(timestamp DESC);
"""


def get_db_path(db_path: Optional[str] = None) -> str:
    """Resolve the SQLite database path."""
    if db_path:
        return db_path
    return getattr(Config, 'DATABASE_PATH', os.path.join(Config.BASE_DIR, 'database', 'documind.db'))


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Establish a connection to the SQLite database with row factory and foreign keys enabled.
    """
    resolved_path = get_db_path(db_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    
    conn = sqlite3.connect(resolved_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite database schema, creating tables and indexes if they do not exist.
    Safely migrates existing schemas to ensure the Canonical First-Mention per Document
    uniqueness constraint: UNIQUE (document_id, subject, predicate, object).
    """
    conn = get_connection(db_path)
    try:
        with conn:
            # 1. Execute initial DDL
            conn.executescript(SCHEMA_SQL)

            # 2. Safety check: ensure unique index uq_facts_doc_spo is present even on pre-existing tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='uq_facts_doc_spo';")
            if not cursor.fetchone():
                # Deduplicate pre-existing rows if any before applying unique index
                conn.execute("""
                    DELETE FROM facts
                    WHERE fact_id NOT IN (
                        SELECT MIN(fact_id)
                        FROM facts
                        GROUP BY document_id, LOWER(subject), LOWER(predicate), LOWER(object)
                    );
                """)
                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_facts_doc_spo ON facts(
                        document_id,
                        subject COLLATE NOCASE,
                        predicate COLLATE NOCASE,
                        object COLLATE NOCASE
                    );
                """)
    finally:
        conn.close()
