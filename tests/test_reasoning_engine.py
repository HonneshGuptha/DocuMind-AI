"""
Unit tests for Module 4 Stage 1: Reasoning Models and Working Memory.
Tests classical AI data structures:
- Fact (original vs inferred, provenance preservation)
- InferenceRule, InferenceTrace, ReasoningResult
- WorkingMemory (in-memory fact base, case-insensitive duplicate prevention,
  separation of original/inferred, filtering, clear, and SQLite independence)
"""

import unittest
from dataclasses import dataclass, FrozenInstanceError
from typing import Optional

from modules.reasoning_engine.reasoning_models import (
    Fact,
    InferenceRule,
    InferenceTrace,
    ReasoningResult,
)
from modules.reasoning_engine.working_memory import WorkingMemory


@dataclass
class MockKBFact:
    """Mock object simulating a Module 2 Knowledge Base Fact record."""
    fact_id: int
    document_id: int
    subject: str
    predicate: str
    object: str
    source_sentence: str
    sentence_index: int


class TestReasoningModels(unittest.TestCase):
    """Tests for reasoning data models."""

    def test_fact_initialization_original(self):
        """Verify ground fact initialization and default properties."""
        fact = Fact(
            subject="Tokyo",
            predicate="capital of",
            object="Japan",
            fact_type="original",
            fact_id=101,
            document_id=1,
            source_sentence="Tokyo is the capital of Japan.",
            sentence_index=0
        )
        self.assertEqual(fact.subject, "Tokyo")
        self.assertEqual(fact.predicate, "capital of")
        self.assertEqual(fact.object, "Japan")
        self.assertEqual(fact.fact_type, "original")
        self.assertTrue(fact.is_original)
        self.assertFalse(fact.is_inferred)
        self.assertEqual(fact.triple, ("Tokyo", "capital of", "Japan"))
        self.assertEqual(fact.normalized_key, ("tokyo", "capital of", "japan"))
        self.assertEqual(fact.fact_id, 101)
        self.assertEqual(fact.document_id, 1)
        self.assertEqual(fact.source_sentence, "Tokyo is the capital of Japan.")
        self.assertEqual(fact.sentence_index, 0)

    def test_fact_initialization_inferred(self):
        """Verify inferred fact initialization."""
        fact = Fact(
            subject="Tokyo",
            predicate="located in",
            object="Japan",
            fact_type="inferred",
            fact_id="inf_1"
        )
        self.assertEqual(fact.fact_type, "inferred")
        self.assertTrue(fact.is_inferred)
        self.assertFalse(fact.is_original)
        self.assertEqual(fact.fact_id, "inf_1")
        self.assertIsNone(fact.document_id)
        self.assertIsNone(fact.source_sentence)

    def test_fact_invalid_type_raises(self):
        """Verify invalid fact_type raises ValueError."""
        with self.assertRaises(ValueError):
            Fact(subject="A", predicate="B", object="C", fact_type="hypothetical")

    def test_fact_immutability_raises_frozen_instance_error(self):
        """REGRESSION TEST: Verify Fact instances cannot be mutated after creation."""
        fact = Fact(
            subject="Tokyo",
            predicate="capital of",
            object="Japan",
            fact_type="original",
            fact_id=101
        )
        with self.assertRaises(FrozenInstanceError):
            fact.subject = "Kyoto"

        with self.assertRaises(FrozenInstanceError):
            fact.predicate = "located in"

        with self.assertRaises(FrozenInstanceError):
            fact.object = "Asia"

        with self.assertRaises(FrozenInstanceError):
            fact.fact_id = 999

    def test_fact_to_dict(self):
        """Verify serialization of Fact."""
        fact = Fact(
            subject="Paris",
            predicate="capital of",
            object="France",
            fact_id=5,
            document_id=2
        )
        d = fact.to_dict()
        self.assertEqual(d["subject"], "Paris")
        self.assertEqual(d["predicate"], "capital of")
        self.assertEqual(d["object"], "France")
        self.assertEqual(d["fact_type"], "original")
        self.assertEqual(d["fact_id"], 5)
        self.assertEqual(d["document_id"], 2)

    def test_inference_rule_model(self):
        """Verify InferenceRule model properties and serialization."""
        rule = InferenceRule(
            rule_id="R_CAPITAL_LOCATED_IN",
            rule_name="Capital Location Rule",
            premises=[("?X", "capital of", "?Y")],
            conclusion=("?X", "located in", "?Y"),
            description="If X is the capital of Y, then X is located in Y."
        )
        self.assertEqual(rule.rule_id, "R_CAPITAL_LOCATED_IN")
        self.assertEqual(len(rule.premises), 1)
        self.assertEqual(rule.conclusion, ("?X", "located in", "?Y"))
        d = rule.to_dict()
        self.assertEqual(d["rule_id"], "R_CAPITAL_LOCATED_IN")
        self.assertEqual(d["rule_name"], "Capital Location Rule")

    def test_inference_trace_model(self):
        """Verify InferenceTrace model attributes and audit serialization."""
        derived = Fact(
            subject="Tokyo",
            predicate="located in",
            object="Japan",
            fact_type="inferred",
            fact_id="inf_1"
        )
        trace = InferenceTrace(
            derived_fact=derived,
            applied_rule_id="R_CAPITAL_LOCATED_IN",
            supporting_fact_ids=[101],
            reasoning_method="forward_chaining",
            explanation="Applied rule: Tokyo is the capital of Japan -> Tokyo is located in Japan."
        )
        self.assertEqual(trace.applied_rule_id, "R_CAPITAL_LOCATED_IN")
        self.assertEqual(trace.supporting_fact_ids, [101])
        self.assertEqual(trace.reasoning_method, "forward_chaining")
        
        d = trace.to_dict()
        self.assertEqual(d["applied_rule_id"], "R_CAPITAL_LOCATED_IN")
        self.assertEqual(d["supporting_fact_ids"], [101])
        self.assertEqual(d["derived_fact"]["subject"], "Tokyo")

    def test_reasoning_result_model(self):
        """Verify ReasoningResult structure and properties."""
        fact_orig = Fact(subject="Tokyo", predicate="capital of", object="Japan", fact_id=1)
        fact_inf = Fact(subject="Tokyo", predicate="located in", object="Japan", fact_type="inferred", fact_id="inf_1")
        trace = InferenceTrace(
            derived_fact=fact_inf,
            applied_rule_id="R_CAPITAL_LOCATED_IN",
            supporting_fact_ids=[1]
        )

        result = ReasoningResult(
            status="success",
            original_facts=[fact_orig],
            inferred_facts=[fact_inf],
            inference_traces=[trace],
            explanation="Deduced 1 new fact via forward chaining."
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.total_facts_count, 2)
        self.assertEqual(result.inferred_facts_count, 1)
        
        d = result.to_dict()
        self.assertEqual(d["status"], "success")
        self.assertEqual(d["original_facts_count"], 1)
        self.assertEqual(d["inferred_facts_count"], 1)
        self.assertEqual(d["total_facts_count"], 2)


class TestWorkingMemory(unittest.TestCase):
    """Tests for WorkingMemory in-memory knowledge store."""

    def setUp(self):
        self.wm = WorkingMemory()

    def test_add_and_separate_original_and_inferred_facts(self):
        """Verify that ground facts and inferred facts are kept strictly separated."""
        orig = Fact(subject="Paris", predicate="capital of", object="France", fact_type="original", fact_id=1)
        inf = Fact(subject="Paris", predicate="located in", object="France", fact_type="inferred", fact_id="inf_1")

        self.assertTrue(self.wm.add_fact(orig))
        self.assertTrue(self.wm.add_fact(inf))

        self.assertEqual(self.wm.count_original(), 1)
        self.assertEqual(self.wm.count_inferred(), 1)
        self.assertEqual(self.wm.count_total(), 2)

        original_facts = self.wm.get_original_facts()
        inferred_facts = self.wm.get_inferred_facts()
        all_facts = self.wm.get_all_facts()

        self.assertEqual(len(original_facts), 1)
        self.assertEqual(original_facts[0].subject, "Paris")
        self.assertEqual(original_facts[0].predicate, "capital of")

        self.assertEqual(len(inferred_facts), 1)
        self.assertEqual(inferred_facts[0].subject, "Paris")
        self.assertEqual(inferred_facts[0].predicate, "located in")

        self.assertEqual(len(all_facts), 2)

    def test_convenience_add_helpers(self):
        """Verify add_original_fact and add_inferred_fact helpers."""
        res1 = self.wm.add_original_fact(
            subject="Python",
            predicate="is a",
            object="programming language",
            fact_id=42,
            document_id=5,
            source_sentence="Python is a programming language.",
            sentence_index=2
        )
        self.assertTrue(res1)

        # Inferred fact without explicit ID gets automatic ID
        res2 = self.wm.add_inferred_fact(
            subject="Python",
            predicate="category",
            object="software"
        )
        self.assertTrue(res2)

        inferred = self.wm.get_inferred_facts()
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0].fact_id, "inf_1")

        # Second inferred fact gets inf_2
        self.wm.add_inferred_fact(
            subject="Python",
            predicate="used for",
            object="AI"
        )
        inferred = self.wm.get_inferred_facts()
        self.assertEqual(len(inferred), 2)
        self.assertEqual(inferred[1].fact_id, "inf_2")

    def test_batch_add_facts(self):
        """Verify add_facts batch method returns count of non-duplicate additions."""
        facts = [
            Fact(subject="A", predicate="relates to", object="B"),
            Fact(subject="B", predicate="relates to", object="C"),
            Fact(subject="A", predicate="relates to", object="B"), # duplicate
        ]
        added = self.wm.add_facts(facts)
        self.assertEqual(added, 2)
        self.assertEqual(self.wm.count_total(), 2)

    def test_case_insensitive_duplicate_prevention(self):
        """Verify duplicate facts with different casing/whitespace are rejected."""
        f1 = Fact(subject="Bangalore", predicate="capital of", object="Karnataka")
        f2 = Fact(subject="  bangalore  ", predicate="CAPITAL OF", object="karnataka  ")

        self.assertTrue(self.wm.add_fact(f1))
        self.assertFalse(self.wm.add_fact(f2))  # Should be rejected
        self.assertEqual(self.wm.count_total(), 1)

    def test_cross_type_duplicate_prevention(self):
        """Verify that an inferred fact identical to an existing original fact is rejected."""
        orig = Fact(subject="Rome", predicate="capital of", object="Italy", fact_type="original")
        inf = Fact(subject="rome", predicate="capital of", object="italy", fact_type="inferred")

        self.assertTrue(self.wm.add_fact(orig))
        self.assertFalse(self.wm.add_fact(inf))
        self.assertEqual(self.wm.count_original(), 1)
        self.assertEqual(self.wm.count_inferred(), 0)

    def test_provenance_preservation(self):
        """Verify ground facts retain exact source sentence, sentence_index, and document_id."""
        self.wm.add_original_fact(
            subject="Linux",
            predicate="created by",
            object="Linus Torvalds",
            fact_id=7,
            document_id=3,
            source_sentence="Linux was created by Linus Torvalds in 1991.",
            sentence_index=4
        )
        fact = self.wm.get_original_facts()[0]
        self.assertEqual(fact.fact_id, 7)
        self.assertEqual(fact.document_id, 3)
        self.assertEqual(fact.source_sentence, "Linux was created by Linus Torvalds in 1991.")
        self.assertEqual(fact.sentence_index, 4)

    def test_load_from_kb_facts(self):
        """Verify loading facts from Module 2 KB fact objects."""
        kb_facts = [
            MockKBFact(
                fact_id=1,
                document_id=10,
                subject="Java",
                predicate="developed by",
                object="Sun Microsystems",
                source_sentence="Java was developed by Sun Microsystems.",
                sentence_index=0
            ),
            MockKBFact(
                fact_id=2,
                document_id=10,
                subject="Java",
                predicate="released in",
                object="1995",
                source_sentence="Java was released in 1995.",
                sentence_index=1
            ),
            # Duplicate
            MockKBFact(
                fact_id=3,
                document_id=11,
                subject="java",
                predicate="developed by",
                object="sun microsystems",
                source_sentence="Java was developed by Sun Microsystems.",
                sentence_index=0
            ),
        ]
        loaded_count = self.wm.load_from_kb_facts(kb_facts)
        self.assertEqual(loaded_count, 2)
        self.assertEqual(self.wm.count_original(), 2)
        self.assertEqual(self.wm.count_inferred(), 0)

        facts = self.wm.get_original_facts()
        self.assertEqual(facts[0].subject, "Java")
        self.assertEqual(facts[0].document_id, 10)
        self.assertEqual(facts[0].fact_id, 1)

    def test_search_and_filtering(self):
        """Verify search by subject, predicate, object, and fact_type with exact and substring modes."""
        self.wm.add_original_fact(subject="Canada", predicate="borders", object="United States")
        self.wm.add_original_fact(subject="Mexico", predicate="borders", object="United States")
        self.wm.add_inferred_fact(subject="United States", predicate="borders", object="Canada")

        # Subject search
        results_canada = self.wm.search_by_subject("Canada")
        self.assertEqual(len(results_canada), 1)
        self.assertEqual(results_canada[0].subject, "Canada")

        # Predicate search
        results_borders = self.wm.search_by_predicate("borders")
        self.assertEqual(len(results_borders), 3)

        # Object search
        results_us = self.wm.search_by_object("United States")
        self.assertEqual(len(results_us), 2)

        # fact_type search
        orig_only = self.wm.search_facts(fact_type="original")
        self.assertEqual(len(orig_only), 2)

        inferred_only = self.wm.search_facts(fact_type="inferred")
        self.assertEqual(len(inferred_only), 1)
        self.assertEqual(inferred_only[0].subject, "United States")

        # Exact vs substring search
        sub_match = self.wm.search_facts(subject="Can", exact=False)
        self.assertEqual(len(sub_match), 1)
        exact_match = self.wm.search_facts(subject="Can", exact=True)
        self.assertEqual(len(exact_match), 0)

    def test_contains_triple(self):
        """Verify contains_triple checking."""
        self.wm.add_original_fact("Berlin", "capital of", "Germany")
        self.assertTrue(self.wm.contains_triple("Berlin", "capital of", "Germany"))
        self.assertTrue(self.wm.contains_triple("berlin", "CAPITAL OF", "germany"))
        self.assertFalse(self.wm.contains_triple("Munich", "capital of", "Germany"))

    def test_clear_memory(self):
        """Verify that clear() empties working memory completely."""
        self.wm.add_original_fact("Earth", "orbits", "Sun")
        self.wm.add_inferred_fact("Sun", "orbited by", "Earth")
        self.assertEqual(self.wm.count_total(), 2)

        self.wm.clear()
        self.assertEqual(self.wm.count_total(), 0)
        self.assertEqual(self.wm.count_original(), 0)
        self.assertEqual(self.wm.count_inferred(), 0)
        self.assertEqual(len(self.wm.get_all_facts()), 0)

        # Re-adding a previously added fact succeeds because index was cleared
        self.assertTrue(self.wm.add_original_fact("Earth", "orbits", "Sun"))

    def test_independence_from_sqlite(self):
        """Verify that WorkingMemory operations do not import, open, or mutate SQLite."""
        # Check that wm instance has no sqlite references or database handles
        self.assertFalse(hasattr(self.wm, 'db'))
        self.assertFalse(hasattr(self.wm, 'conn'))
        self.assertFalse(hasattr(self.wm, 'cursor'))
        self.assertFalse(hasattr(self.wm, 'database_path'))

        # Adding facts does not touch any disk database
        self.wm.add_original_fact("Test", "is", "Isolated")
        self.wm.add_inferred_fact("Isolated", "is", "RAM-only")
        self.assertEqual(self.wm.count_total(), 2)

    def test_modifying_external_reference_blocked(self):
        """REGRESSION TEST: Modifying an externally held Fact reference is blocked by immutability."""
        f = Fact("Tokyo", "capital of", "Japan", fact_id=1)
        self.wm.add_fact(f)

        # Attempting to mutate the externally held reference raises FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            f.subject = "Kyoto"

        # Stored fact remains unaltered
        stored = self.wm.get_original_facts()[0]
        self.assertEqual(stored.subject, "Tokyo")

    def test_modifying_returned_fact_lists_does_not_affect_working_memory(self):
        """REGRESSION TEST: Appending or removing from returned fact lists cannot alter WorkingMemory."""
        self.wm.add_original_fact("Paris", "capital of", "France")
        self.wm.add_inferred_fact("Paris", "located in", "France")

        # Mutate list returned by get_original_facts
        orig_list = self.wm.get_original_facts()
        orig_list.append(Fact("Fake", "is", "Fact"))
        orig_list.clear()
        self.assertEqual(self.wm.count_original(), 1)
        self.assertEqual(len(self.wm.get_original_facts()), 1)

        # Mutate list returned by get_inferred_facts
        inf_list = self.wm.get_inferred_facts()
        inf_list.clear()
        self.assertEqual(self.wm.count_inferred(), 1)
        self.assertEqual(len(self.wm.get_inferred_facts()), 1)

        # Mutate list returned by get_all_facts
        all_list = self.wm.get_all_facts()
        all_list.pop()
        self.assertEqual(self.wm.count_total(), 2)
        self.assertEqual(len(self.wm.get_all_facts()), 2)

    def test_duplicate_index_consistency_preserved(self):
        """REGRESSION TEST: Normalized duplicate index remains 100% consistent with stored facts."""
        self.wm.add_original_fact("Madrid", "capital of", "Spain")
        self.assertTrue(self.wm.contains_triple("Madrid", "capital of", "Spain"))
        self.assertTrue(self.wm.contains_triple("madrid", "CAPITAL OF", "spain"))
        self.assertFalse(self.wm.contains_triple("Barcelona", "capital of", "Spain"))

        # Re-adding same fact as inferred is rejected
        rejected = self.wm.add_inferred_fact("madrid", "capital of", "spain")
        self.assertFalse(rejected)
        self.assertEqual(self.wm.count_total(), 1)


if __name__ == '__main__':
    unittest.main()
