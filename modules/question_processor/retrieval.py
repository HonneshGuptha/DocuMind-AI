"""
Information Retrieval Module
Bridges structured question queries with the existing KnowledgeBaseRepository.
Retrieves matching facts using parameterized SQLite queries without semantic/vector search.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly relational, graph-based in Python, and deterministic.
Uses indexed SQL queries against the existing SQLite database.
No vector similarity, no embeddings, no RAG frameworks, no LLMs.
"""

from typing import List, Optional, Union, Set
from modules.knowledge_base.repository import KnowledgeBaseRepository
from modules.knowledge_base.models import Fact
from modules.knowledge_base.entity_registry import GLOBAL_ENTITY_REGISTRY
from modules.knowledge_base.relationship_registry import GLOBAL_RELATIONSHIP_REGISTRY
from .query_models import StructuredQuery, RetrievalResult


class QuestionRetriever:
    """
    Executes relational information retrieval over the SQLite Knowledge Base
    using a parsed StructuredQuery, supporting speaker resolution, alias handling,
    and reverse lookups.
    """

    def __init__(self, repository: Optional[KnowledgeBaseRepository] = None):
        self.repository = repository or KnowledgeBaseRepository()
        self.entity_reg = GLOBAL_ENTITY_REGISTRY
        self.rel_reg = GLOBAL_RELATIONSHIP_REGISTRY

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

        # 2. Resolve Speaker references if query is first-person ("Who am I?", "Where do I live?")
        eff_subject = query.subject
        if eff_subject in ('Speaker', 'I', 'me'):
            speaker = self.repository.get_primary_speaker()
            if speaker:
                eff_subject = speaker

        facts: List[Fact] = []
        seen_fact_ids: Set[int] = set()

        def add_fact_if_new(f: Fact):
            if f.fact_id is not None:
                if f.fact_id not in seen_fact_ids:
                    seen_fact_ids.add(f.fact_id)
                    facts.append(f)
            else:
                facts.append(f)

        # Helper to get candidate entity names (subject and aliases)
        def get_entity_candidates(name: Optional[str]) -> List[str]:
            if not name:
                return []
            return list(self.entity_reg.get_all_aliases(name))

        # Helper to get predicate variations
        def get_predicate_candidates(pred: Optional[str]) -> List[str]:
            if not pred:
                return []
            return self.rel_reg.get_all_variations(pred)

        # ----------------------------------------------------------------------
        # Case A: Boolean verification query (Both Subject and Object known)
        # ----------------------------------------------------------------------
        if query.expected_answer_field == 'boolean' and eff_subject and query.object:
            subj_candidates = get_entity_candidates(eff_subject)
            obj_candidates = get_entity_candidates(query.object)
            pred_candidates = get_predicate_candidates(query.predicate)

            # Step A1: Exact match search
            for s in subj_candidates:
                for p in pred_candidates:
                    for o in obj_candidates:
                        matched = self.repository.search_facts(subject=s, predicate=p, object=o)
                        for f in matched:
                            add_fact_if_new(f)

            # Step A2: Substring / Role matching for classifications (e.g. "student" in "AI and ML engineering student")
            if not facts:
                for s in subj_candidates:
                    for p in pred_candidates:
                        cand_facts = self.repository.search_facts(subject=s, predicate=p)
                        for f in cand_facts:
                            for o in obj_candidates:
                                if o.lower() in f.object.lower() or f.object.lower() in o.lower():
                                    add_fact_if_new(f)

        # ----------------------------------------------------------------------
        # Case B: Subject inquiry / Reverse Lookup (Object known, Subject queried)
        # e.g., "What is the capital of Karnataka?" or "Who works at KPMG?"
        # ----------------------------------------------------------------------
        elif query.expected_answer_field == 'subject' and query.object:
            obj_candidates = get_entity_candidates(query.object)
            pred_candidates = get_predicate_candidates(query.predicate)

            for o in obj_candidates:
                for p in pred_candidates:
                    matched = self.repository.search_facts(predicate=p, object=o)
                    for f in matched:
                        add_fact_if_new(f)

        # ----------------------------------------------------------------------
        # Case C: Object inquiry (Subject known, Object queried)
        # e.g., "Where is Tokyo located?", "Where do I live?", "What is my name?"
        # ----------------------------------------------------------------------
        elif query.expected_answer_field == 'object' and eff_subject:
            subj_candidates = get_entity_candidates(eff_subject)
            pred_candidates = get_predicate_candidates(query.predicate)

            if pred_candidates:
                for s in subj_candidates:
                    for p in pred_candidates:
                        matched = self.repository.search_facts(subject=s, predicate=p)
                        for f in matched:
                            add_fact_if_new(f)

                # Fallback for establishment related predicates
                if not facts and query.question_type == 'establishment':
                    for s in subj_candidates:
                        for alt_pred in ['established in', 'founded in', 'created in', 'built in']:
                            if alt_pred not in pred_candidates:
                                matched = self.repository.search_facts(subject=s, predicate=alt_pred)
                                for f in matched:
                                    add_fact_if_new(f)
                                if facts:
                                    break

                # Fallback for creator related predicates
                elif not facts and query.question_type == 'creator':
                    for s in subj_candidates:
                        for alt_pred in ['developed by', 'invented by', 'authored by', 'written by', 'discovered by']:
                            if alt_pred not in pred_candidates:
                                matched = self.repository.search_facts(subject=s, predicate=alt_pred)
                                for f in matched:
                                    add_fact_if_new(f)
                                if facts:
                                    break
            else:
                # Open search (Identity queries: find facts where entity is subject OR object)
                for s in subj_candidates:
                    for f in self.repository.search_by_subject(s) + self.repository.search_by_object(s):
                        add_fact_if_new(f)

        # ----------------------------------------------------------------------
        # Case D: Open Predicate search
        # ----------------------------------------------------------------------
        elif query.predicate and not eff_subject and not query.object:
            for p in get_predicate_candidates(query.predicate):
                for f in self.repository.search_by_predicate(p):
                    add_fact_if_new(f)

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
    """Convenience helper function to retrieve facts in a single call."""
    retriever = QuestionRetriever(repository)
    return retriever.retrieve(query)

