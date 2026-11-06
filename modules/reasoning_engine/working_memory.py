"""
Working Memory Module
Implements the in-memory Working Knowledge Base for classical AI inference.

ACADEMIC EXPLANATION:
In rule-based expert systems (such as OPS5, CLIPS, or Prolog engines),
Working Memory represents the current state of knowledge during problem-solving.
It holds:
1. Ground facts loaded from sources (Original Facts).
2. Deduced facts derived during the inference cycle (Inferred Facts).

CRITICAL DESIGN PROPERTY:
Working Memory operates strictly in transient RAM. It never writes to or alters
the persistent SQLite database, ensuring non-destructive reasoning and total isolation.
"""

from dataclasses import replace
from typing import List, Optional, Set, Tuple, Dict, Any, Union
from .reasoning_models import Fact


class WorkingMemory:
    """
    Transient, in-memory fact base that manages original and inferred knowledge
    propositions without database side effects.
    """

    def __init__(self):
        # Separate collections for original and inferred facts
        self._original_facts: List[Fact] = []
        self._inferred_facts: List[Fact] = []
        
        # Set of normalized (subject, predicate, object) keys for O(1) duplicate detection
        self._fact_index: Set[Tuple[str, str, str]] = set()

        # Counter for generating unique IDs for newly inferred facts
        self._inferred_counter: int = 1

    # ==========================================================================
    # 1. Fact Ingestion & Duplicate Handling
    # ==========================================================================

    def add_fact(self, fact: Fact) -> bool:
        """
        Add a Fact to Working Memory.
        
        Enforces case-insensitive SPO uniqueness:
        If an identical proposition already exists in memory (as either original
        or inferred), the addition is safely skipped to avoid redundant derivations.
        
        Returns:
            True if the fact was newly added, False if skipped as a duplicate.
        """
        norm_key = fact.normalized_key
        if norm_key in self._fact_index:
            return False

        # If it's an inferred fact without an ID, assign an automatic identifier (using replace for frozen dataclass)
        fact_to_store = fact
        if fact.is_inferred and fact.fact_id is None:
            fact_to_store = replace(fact, fact_id=f"inf_{self._inferred_counter}")
            self._inferred_counter += 1

        self._fact_index.add(norm_key)

        if fact_to_store.is_inferred:
            self._inferred_facts.append(fact_to_store)
        else:
            self._original_facts.append(fact_to_store)

        return True

    def add_facts(self, facts: List[Fact]) -> int:
        """
        Batch add multiple facts to Working Memory.
        Returns count of newly added facts.
        """
        added_count = 0
        for f in facts:
            if self.add_fact(f):
                added_count += 1
        return added_count

    def add_original_fact(self,
                          subject: str,
                          predicate: str,
                          object: str,
                          fact_id: Optional[Union[int, str]] = None,
                          document_id: Optional[int] = None,
                          source_sentence: Optional[str] = None,
                          sentence_index: Optional[int] = None) -> bool:
        """
        Convenience helper to create and insert an original ground fact.
        """
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object,
            fact_type='original',
            fact_id=fact_id,
            document_id=document_id,
            source_sentence=source_sentence,
            sentence_index=sentence_index
        )
        return self.add_fact(fact)

    def add_inferred_fact(self,
                          subject: str,
                          predicate: str,
                          object: str,
                          fact_id: Optional[Union[int, str]] = None) -> bool:
        """
        Convenience helper to insert a derived proposition produced by an inference rule.
        """
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object,
            fact_type='inferred',
            fact_id=fact_id
        )
        return self.add_fact(fact)

    def load_from_kb_facts(self, kb_facts: List[Any]) -> int:
        """
        Load facts retrieved from Module 2 (SQLite Knowledge Base) into working memory.
        Automatically tags them as 'original' while preserving their fact_id,
        document_id, source_sentence, and sentence_index.
        """
        count = 0
        for kbf in kb_facts:
            # Accepts both Module 2 Fact instances or any object with SPO attributes
            fact = Fact(
                subject=kbf.subject,
                predicate=kbf.predicate,
                object=kbf.object,
                fact_type='original',
                fact_id=getattr(kbf, 'fact_id', None),
                document_id=getattr(kbf, 'document_id', None),
                source_sentence=getattr(kbf, 'source_sentence', None),
                sentence_index=getattr(kbf, 'sentence_index', None)
            )
            if self.add_fact(fact):
                count += 1
        return count

    # ==========================================================================
    # 2. Fact Retrieval & Separation
    # ==========================================================================

    def get_all_facts(self) -> List[Fact]:
        """Return all facts currently in working memory (original + inferred)."""
        return list(self._original_facts + self._inferred_facts)

    def get_original_facts(self) -> List[Fact]:
        """Return strictly ground facts from source documents."""
        return list(self._original_facts)

    def get_inferred_facts(self) -> List[Fact]:
        """Return strictly deduced facts derived by reasoning rules."""
        return list(self._inferred_facts)

    def count_original(self) -> int:
        """Count of ground facts."""
        return len(self._original_facts)

    def count_inferred(self) -> int:
        """Count of deduced facts."""
        return len(self._inferred_facts)

    def count_total(self) -> int:
        """Total number of facts in working memory."""
        return len(self._original_facts) + len(self._inferred_facts)

    def contains_triple(self, subject: str, predicate: str, object: str) -> bool:
        """Check if a normalized triple already exists in working memory."""
        norm_key = (subject.strip().lower(),
                    predicate.strip().lower(),
                    object.strip().lower())
        return norm_key in self._fact_index

    # ==========================================================================
    # 3. Filtering and Searching
    # ==========================================================================

    def search_facts(self,
                     subject: Optional[str] = None,
                     predicate: Optional[str] = None,
                     object: Optional[str] = None,
                     fact_type: Optional[str] = None,
                     exact: bool = False) -> List[Fact]:
        """
        Filter facts in Working Memory by subject, predicate, object, and fact_type.
        
        Args:
            subject: Filter value for subject (case-insensitive).
            predicate: Filter value for predicate (case-insensitive).
            object: Filter value for object (case-insensitive).
            fact_type: Filter by 'original', 'inferred', or None for all.
            exact: True for exact match, False for substring containment.
            
        Returns:
            List of matching Fact instances.
        """
        # Determine candidate pool based on fact_type
        if fact_type == 'original':
            candidates = self._original_facts
        elif fact_type == 'inferred':
            candidates = self._inferred_facts
        else:
            candidates = self._original_facts + self._inferred_facts

        results = []
        s_norm = subject.strip().lower() if subject else None
        p_norm = predicate.strip().lower() if predicate else None
        o_norm = object.strip().lower() if object else None

        for f in candidates:
            # Subject match
            if s_norm:
                f_s = f.subject.lower()
                if exact and f_s != s_norm:
                    continue
                if not exact and s_norm not in f_s:
                    continue

            # Predicate match
            if p_norm:
                f_p = f.predicate.lower()
                if exact and f_p != p_norm:
                    continue
                if not exact and p_norm not in f_p:
                    continue

            # Object match
            if o_norm:
                f_o = f.object.lower()
                if exact and f_o != o_norm:
                    continue
                if not exact and o_norm not in f_o:
                    continue

            results.append(f)

        return results

    def search_by_subject(self, subject: str, exact: bool = False) -> List[Fact]:
        """Filter facts by subject."""
        return self.search_facts(subject=subject, exact=exact)

    def search_by_predicate(self, predicate: str, exact: bool = False) -> List[Fact]:
        """Filter facts by predicate."""
        return self.search_facts(predicate=predicate, exact=exact)

    def search_by_object(self, object: str, exact: bool = False) -> List[Fact]:
        """Filter facts by object."""
        return self.search_facts(object=object, exact=exact)

    # ==========================================================================
    # 4. Memory Reset & Lifecycle
    # ==========================================================================

    def clear(self) -> None:
        """
        Reset Working Memory, clearing all original facts, inferred facts,
        and indexing data structures.
        """
        self._original_facts.clear()
        self._inferred_facts.clear()
        self._fact_index.clear()
        self._inferred_counter = 1

    def __repr__(self) -> str:
        return (f"<WorkingMemory: {len(self._original_facts)} original, "
                f"{len(self._inferred_facts)} inferred>")
