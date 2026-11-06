"""
Knowledge Base Subsystem
Provides structured knowledge representation using Subject-Predicate-Object (SPO) triples,
deterministic pattern-based fact extraction, and persistent SQLite relational storage.

Exports:
- Fact: Model representing an SPO triple with source sentence attribution.
- DocumentRecord: Model representing stored document metadata.
- FactExtractor: Rule-based deterministic fact extraction engine.
- FactPattern: Declarative factual pattern specification.
- FACTUAL_PATTERNS: List of canonical supported patterns.
- KnowledgeBaseRepository: Relational repository for CRUD and search operations.
- init_db: Database initialization and schema creation utility.
- get_connection: Database connection manager.
"""

from .models import Fact, DocumentRecord
from .database import init_db, get_connection
from .fact_extractor import (
    FactExtractor,
    FactPattern,
    FACTUAL_PATTERNS,
    split_sentences
)
from .repository import KnowledgeBaseRepository

__all__ = [
    'Fact',
    'DocumentRecord',
    'FactExtractor',
    'FactPattern',
    'FACTUAL_PATTERNS',
    'split_sentences',
    'KnowledgeBaseRepository',
    'init_db',
    'get_connection',
]
