"""
Question Processing and Information Retrieval Subsystem
Provides deterministic question classification, parsing into StructuredQuery representations,
and fact retrieval from the SQLite Knowledge Base.

Exports:
- StructuredQuery: Data model representing a parsed question.
- RetrievalResult: Data model holding retrieved facts and provenance metadata.
- QuestionClassifier: Rule-based deterministic question classifier.
- QuestionParser: Rule-based question parser.
- parse_question: Single-call helper to parse a question.
- QuestionRetriever: Relational search retriever connecting to KnowledgeBaseRepository.
- retrieve_facts: Single-call helper to retrieve facts for a StructuredQuery.
- process_and_retrieve: End-to-end question processing and fact retrieval pipeline.
"""

from typing import Optional
from modules.knowledge_base.repository import KnowledgeBaseRepository
from .query_models import StructuredQuery, RetrievalResult
from .classifier import QuestionClassifier
from .parser import QuestionParser, parse_question
from .retrieval import QuestionRetriever, retrieve_facts


def process_and_retrieve(question: str,
                         repository: Optional[KnowledgeBaseRepository] = None) -> RetrievalResult:
    """
    End-to-end pipeline:
    1. Parse natural language question into a StructuredQuery.
    2. Query KnowledgeBaseRepository for matching facts.
    3. Return RetrievalResult with full source sentence provenance.
    """
    query = parse_question(question)
    return retrieve_facts(query, repository)


__all__ = [
    'StructuredQuery',
    'RetrievalResult',
    'QuestionClassifier',
    'QuestionParser',
    'parse_question',
    'QuestionRetriever',
    'retrieve_facts',
    'process_and_retrieve',
]
