"""
Information Retrieval Module
Bridges structured question queries with the existing KnowledgeBaseRepository.
Retrieves matching facts using parameterized SQLite queries without semantic/vector search.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly relational and deterministic.
Uses indexed SQL queries against the existing SQLite database.
No vector similarity, no embeddings, no RAG frameworks, no LLMs.
"""

from typing import List, Optional, Union
from modules.knowledge_base.repository import KnowledgeBaseRepository
from modules.knowledge_base.models import Fact
from .query_models import StructuredQuery, RetrievalResult


class QuestionRetriever:
    """
    Executes relational information retrieval over the SQLite Knowledge Base
    using a parsed StructuredQuery.
    """

    def __init__(self, repository: Optional[KnowledgeBaseRepository] = None):
        self.repository = repository or KnowledgeBaseRepository()

    def retrieve(self, query: StructuredQuery) -> RetrievalResult:
        """
        Retrieve facts matching the StructuredQuery from the knowledge base.
        
        Args:
            query: Parsed StructuredQuery instance.
            
        Returns:
            RetrievalResult containing matched Fact objects, status, and metadata.
        """
        # 1. Check if the query itself is invalid, empty, or unsupported
        if query.parsing_status != 'success':
            return RetrievalResult(
                query=query,
                facts=[],
                status=query.parsing_status,
                error_message=query.error_message
            )

        # 2. Execute targeted indexed search based on query parameters
        facts: List[Fact] = []

        # Case A: Boolean verification query (Both Subject and Object known)
        if query.expected_answer_field == 'boolean' and query.subject and query.object:
            facts = self.repository.search_facts(
                subject=query.subject,
                predicate=query.predicate,
                object=query.object
            )

        # Case B: Subject inquiry (Object and Predicate known, e.g. "What is the capital of Karnataka?")
        elif query.expected_answer_field == 'subject' and query.object:
            facts = self.repository.search_facts(
                predicate=query.predicate,
                object=query.object
            )

        # Case C: Object inquiry (Subject known, e.g. "Where is Tokyo located?")
        elif query.expected_answer_field == 'object' and query.subject:
            if query.predicate:
                facts = self.repository.search_facts(
                    subject=query.subject,
                    predicate=query.predicate
                )
                # Fallback for related predicates (e.g. founded in / established in)
                if not facts and query.question_type == 'establishment':
                    for alt_pred in ['established in', 'founded in', 'created in', 'built in']:
                        if alt_pred != query.predicate:
                            alt_facts = self.repository.search_facts(subject=query.subject, predicate=alt_pred)
                            if alt_facts:
                                facts.extend(alt_facts)
                                break
                elif not facts and query.question_type == 'creator':
                    for alt_pred in ['developed by', 'invented by', 'authored by', 'written by', 'discovered by']:
                        if alt_pred != query.predicate:
                            alt_facts = self.repository.search_facts(subject=query.subject, predicate=alt_pred)
                            if alt_facts:
                                facts.extend(alt_facts)
                                break
            else:
                # Open search (Identity queries: find facts where entity is subject OR object)
                subj_facts = self.repository.search_by_subject(query.subject)
                obj_facts = self.repository.search_by_object(query.subject)
                # Deduplicate by fact_id
                seen_ids = set()
                for f in subj_facts + obj_facts:
                    if f.fact_id not in seen_ids:
                        seen_ids.add(f.fact_id)
                        facts.append(f)

        # Case D: Open Predicate search
        elif query.predicate and not query.subject and not query.object:
            facts = self.repository.search_by_predicate(query.predicate)

        # 3. Determine status
        if facts:
            status = 'success'
            error_msg = None
        else:
            status = 'no_match'
            error_msg = "No matching facts found in the knowledge base."

        return RetrievalResult(
            query=query,
            facts=facts,
            status=status,
            error_message=error_msg
        )


def retrieve_facts(query: StructuredQuery, repository: Optional[KnowledgeBaseRepository] = None) -> RetrievalResult:
    """Helper function to execute retrieval for a query in a single call."""
    retriever = QuestionRetriever(repository)
    return retriever.retrieve(query)
