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
from .entity_registry import Entity, EntityRegistry, GLOBAL_ENTITY_REGISTRY
from .relationship_registry import RelationshipDefinition, RelationshipRegistry, GLOBAL_RELATIONSHIP_REGISTRY
from .context_resolver import DocumentContext, ContextResolver, GLOBAL_CONTEXT_RESOLVER

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
    'Entity',
    'EntityRegistry',
    'GLOBAL_ENTITY_REGISTRY',
    'RelationshipDefinition',
    'RelationshipRegistry',
    'GLOBAL_RELATIONSHIP_REGISTRY',
    'DocumentContext',
    'ContextResolver',
    'GLOBAL_CONTEXT_RESOLVER',
]
