"""
Unit tests for Module 5 Stage 1: Deterministic Answer Generator.
Verifies all 25 required capabilities:
1. Direct capital answer.
2. Direct location answer.
3. Establishment answer.
4. Creator answer.
5. Part-whole answer.
6. Border answer.
7. Production answer.
8. Identity answer.
9. Classification answer.
10. Boolean yes answer.
11. Boolean unknown answer.
12. Inferred answer using R1.
13. Inferred answer using R2.
14. Multiple matching facts.
15. Conflicting facts.
16. Unknown question.
17. Unsupported question.
18. Missing retrieval result.
19. Missing reasoning result.
20. Provenance preservation.
21. Rule ID preservation.
22. Original entity casing.
23. Empty supporting facts.
24. Error handling.
25. Deterministic repeated execution.
"""

import unittest

from modules.knowledge_base.models import Fact as KBF
from modules.question_processor.parser import QuestionParser
from modules.question_processor.query_models import StructuredQuery, RetrievalResult
from modules.reasoning_engine.reasoning_models import (
    Fact as ReasoningFact,
    ReasoningResult,
    InferenceTrace,
)
from modules.reasoning_engine.backward_chaining import ProofResult, ProofStatus
from modules.answer_generator.answer_models import AnswerResponse
from modules.answer_generator.generator import AnswerGenerator, generate_answer


class TestAnswerGenerator(unittest.TestCase):
    """Exhaustive test suite for Deterministic Answer Generator."""

    def setUp(self):
        self.generator = AnswerGenerator()
        self.parser = QuestionParser()

    # 1. Direct capital answer
    def test_01_direct_capital_answer(self):
        query = self.parser.parse("What is the capital of Karnataka?")
        fact = KBF(
            document_id=1,
            subject="Bangalore",
            predicate="capital of",
            object="Karnataka",
            source_sentence="Bangalore is the capital of Karnataka.",
            sentence_index=0,
            fact_id=101
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_type, "direct")
        self.assertEqual(ans.answer_text, "The capital of Karnataka is Bangalore.")
        self.assertEqual(ans.supporting_fact_ids, [101])
        self.assertEqual(ans.confidence, "verified")

    # 2. Direct location answer
    def test_02_direct_location_answer(self):
        query = self.parser.parse("Where is Tokyo located?")
        fact = KBF(
            document_id=1,
            subject="Tokyo",
            predicate="located in",
            object="Japan",
            source_sentence="Tokyo is located in Japan.",
            sentence_index=0,
            fact_id=102
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_type, "direct")
        self.assertEqual(ans.answer_text, "Tokyo is located in Japan.")
        self.assertEqual(ans.supporting_fact_ids, [102])

    # 3. Establishment answer
    def test_03_establishment_answer(self):
        query = self.parser.parse("When was Stanford University established?")
        fact = KBF(
            document_id=2,
            subject="Stanford University",
            predicate="established in",
            object="1885",
            source_sentence="Stanford University was established in 1885.",
            sentence_index=0,
            fact_id=103
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Stanford University was established in 1885.")

    # 4. Creator answer
    def test_04_creator_answer(self):
        query = self.parser.parse("Who developed Python?")
        fact = KBF(
            document_id=3,
            subject="Python",
            predicate="developed by",
            object="Guido van Rossum",
            source_sentence="Python was developed by Guido van Rossum.",
            sentence_index=0,
            fact_id=104
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Python was developed by Guido van Rossum.")

    # 5. Part-whole answer
    def test_05_part_whole_answer(self):
        query = self.parser.parse("What is part of Engine?")
        fact = KBF(
            document_id=4,
            subject="Piston",
            predicate="part of",
            object="Engine",
            source_sentence="Piston is part of Engine.",
            sentence_index=0,
            fact_id=105
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Piston is part of Engine.")

    # 6. Border answer
    def test_06_border_answer(self):
        query = self.parser.parse("What does Canada border?")
        fact = KBF(
            document_id=5,
            subject="Canada",
            predicate="borders",
            object="United States",
            source_sentence="Canada borders United States.",
            sentence_index=0,
            fact_id=106
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Canada borders United States.")

    # 7. Production answer
    def test_07_production_answer(self):
        query = self.parser.parse("What does Karnataka produce?")
        fact = KBF(
            document_id=6,
            subject="Karnataka",
            predicate="produces",
            object="Coffee",
            source_sentence="Karnataka produces Coffee.",
            sentence_index=0,
            fact_id=107
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Karnataka produces Coffee.")

    # 8. Identity answer
    def test_08_identity_answer(self):
        query = self.parser.parse("Who is Guido van Rossum?")
        fact = KBF(
            document_id=3,
            subject="Python",
            predicate="developed by",
            object="Guido van Rossum",
            source_sentence="Python was developed by Guido van Rossum.",
            sentence_index=0,
            fact_id=108
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Python was developed by Guido van Rossum.")

    # 9. Classification answer
    def test_09_classification_answer(self):
        query = self.parser.parse("What is Python?")
        fact = KBF(
            document_id=3,
            subject="Python",
            predicate="is a",
            object="programming language",
            source_sentence="Python is a programming language.",
            sentence_index=0,
            fact_id=109
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Python is a programming language.")

    # 10. Boolean yes answer
    def test_10_boolean_yes_answer(self):
        query = self.parser.parse("Does Canada border United States?")
        fact = KBF(
            document_id=5,
            subject="Canada",
            predicate="borders",
            object="United States",
            source_sentence="Canada borders United States.",
            sentence_index=0,
            fact_id=110
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_text, "Yes, Canada borders United States.")

    # 11. Boolean unknown answer (adhering to Open-World Assumption)
    def test_11_boolean_unknown_answer(self):
        query = self.parser.parse("Does Canada border Brazil?")
        retrieval = RetrievalResult(query=query, facts=[], status="no_match")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "unknown")
        self.assertEqual(ans.answer_type, "none")
        self.assertEqual(ans.answer_text, "I could not find sufficient information in the knowledge base to answer this question.")

    # 12. Inferred answer using R1
    def test_12_inferred_answer_using_r1(self):
        query = self.parser.parse("Where is Bangalore located?")
        # Ground reasoning result from backward chaining
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "India"),
            is_proven=True,
            directly_found=False,
            applied_rule_ids=["R1"],
            supporting_fact_ids=[1, 2]
        )

        ans = self.generator.generate_answer(query, reasoning_result=proof)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_type, "inferred")
        self.assertEqual(ans.answer_text, "Bangalore is located in India.")
        self.assertEqual(ans.applied_rule_ids, ["R1"])
        self.assertIn("Capital Location Rule", ans.explanation)
        self.assertEqual(ans.confidence, "deduced")

    # 13. Inferred answer using R2
    def test_13_inferred_answer_using_r2(self):
        query = self.parser.parse("Where is Bangalore located?")
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "Asia"),
            is_proven=True,
            directly_found=False,
            applied_rule_ids=["R2"],
            supporting_fact_ids=[1, 2, 3]
        )

        ans = self.generator.generate_answer(query, reasoning_result=proof)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_type, "inferred")
        self.assertEqual(ans.answer_text, "Bangalore is located in Asia.")
        self.assertEqual(ans.applied_rule_ids, ["R2"])
        self.assertIn("Location Transitivity Rule", ans.explanation)

    # 14. Multiple matching facts
    def test_14_multiple_matching_facts(self):
        query = self.parser.parse("What does Canada border?")
        facts = [
            KBF(document_id=1, subject="Canada", predicate="borders", object="United States", source_sentence="Canada borders United States.", sentence_index=0, fact_id=1),
            KBF(document_id=1, subject="Canada", predicate="borders", object="Greenland", source_sentence="Canada borders Greenland.", sentence_index=1, fact_id=2),
        ]
        retrieval = RetrievalResult(query=query, facts=facts, status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertIn("Canada borders:", ans.answer_text)
        self.assertIn("United States", ans.answer_text)
        self.assertIn("Greenland", ans.answer_text)

    # 15. Conflicting facts
    def test_15_conflicting_facts(self):
        query = self.parser.parse("When was Python established?")
        facts = [
            KBF(document_id=1, subject="Python", predicate="established in", object="1991", source_sentence="Python was established in 1991.", sentence_index=0, fact_id=1),
            KBF(document_id=2, subject="Python", predicate="established in", object="1989", source_sentence="Python was established in 1989.", sentence_index=0, fact_id=2),
        ]
        retrieval = RetrievalResult(query=query, facts=facts, status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "answered")
        self.assertIn("Conflicting information found in the knowledge base:", ans.answer_text)
        self.assertIn("Fact #1", ans.answer_text)
        self.assertIn("Fact #2", ans.answer_text)

    # 16. Unknown question
    def test_16_unknown_question(self):
        query = self.parser.parse("What is the capital of Atlantis?")
        retrieval = RetrievalResult(query=query, facts=[], status="no_match")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "unknown")
        self.assertEqual(ans.answer_type, "none")
        self.assertEqual(ans.answer_text, "I could not find sufficient information in the knowledge base to answer this question.")

    # 17. Unsupported question
    def test_17_unsupported_question(self):
        query = self.parser.parse("Why is the sky blue?")

        ans = self.generator.generate_answer(query)

        self.assertEqual(ans.answer_status, "unsupported")
        self.assertEqual(ans.answer_type, "none")
        self.assertEqual(ans.answer_text, "This question format is not currently supported by DocuMind AI.")

    # 18. Missing retrieval result
    def test_18_missing_retrieval_result(self):
        query = self.parser.parse("Where is Tokyo located?")
        # Passing None for retrieval_result and reasoning_result
        ans = self.generator.generate_answer(query, retrieval_result=None, reasoning_result=None)

        self.assertEqual(ans.answer_status, "unknown")
        self.assertEqual(ans.answer_type, "none")

    # 19. Missing reasoning result
    def test_19_missing_reasoning_result(self):
        query = self.parser.parse("Where is Tokyo located?")
        fact = KBF(document_id=1, subject="Tokyo", predicate="located in", object="Japan", source_sentence="Tokyo is located in Japan.", sentence_index=0, fact_id=1)
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        # reasoning_result is None
        ans = self.generator.generate_answer(query, retrieval_result=retrieval, reasoning_result=None)

        self.assertEqual(ans.answer_status, "answered")
        self.assertEqual(ans.answer_type, "direct")
        self.assertEqual(ans.answer_text, "Tokyo is located in Japan.")

    # 20. Provenance preservation
    def test_20_provenance_preservation(self):
        query = self.parser.parse("Where is Tokyo located?")
        fact = KBF(
            document_id=42,
            subject="Tokyo",
            predicate="located in",
            object="Japan",
            source_sentence="Tokyo is located in Japan.",
            sentence_index=3,
            fact_id=999
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.supporting_fact_ids, [999])
        self.assertEqual(len(ans.sources), 1)
        self.assertEqual(ans.sources[0]["document_id"], 42)
        self.assertEqual(ans.sources[0]["source_sentence"], "Tokyo is located in Japan.")

    # 21. Rule ID preservation
    def test_21_rule_id_preservation(self):
        query = self.parser.parse("Where is Bangalore located?")
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "India"),
            is_proven=True,
            applied_rule_ids=["R1", "R2"],
            supporting_fact_ids=[1, 2]
        )

        ans = self.generator.generate_answer(query, reasoning_result=proof)

        self.assertEqual(ans.applied_rule_ids, ["R1", "R2"])

    # 22. Original entity casing
    def test_22_original_entity_casing(self):
        query = self.parser.parse("where is tokyo located?")
        # Fact holds original casing
        fact = KBF(
            document_id=1,
            subject="Tokyo",
            predicate="located in",
            object="Japan",
            source_sentence="Tokyo is located in Japan.",
            sentence_index=0,
            fact_id=1
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        # Output preserves original capitalization
        self.assertEqual(ans.answer_text, "Tokyo is located in Japan.")

    # 23. Empty supporting facts
    def test_23_empty_supporting_facts(self):
        query = self.parser.parse("Where is Tokyo located?")
        retrieval = RetrievalResult(query=query, facts=[], status="success")

        ans = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans.answer_status, "unknown")
        self.assertEqual(ans.supporting_fact_ids, [])

    # 24. Error handling
    def test_24_error_handling(self):
        # Passing invalid input object
        ans = self.generator.generate_answer(12345)

        self.assertEqual(ans.answer_status, "error")
        self.assertEqual(ans.answer_type, "none")
        self.assertIn("error", ans.answer_text.lower())

    # 25. Deterministic repeated execution
    def test_25_deterministic_repeated_execution(self):
        query = self.parser.parse("What is the capital of Karnataka?")
        fact = KBF(document_id=1, subject="Bangalore", predicate="capital of", object="Karnataka", source_sentence="Bangalore is the capital of Karnataka.", sentence_index=0, fact_id=1)
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")

        ans1 = self.generator.generate_answer(query, retrieval_result=retrieval)
        ans2 = self.generator.generate_answer(query, retrieval_result=retrieval)

        self.assertEqual(ans1.answer_text, ans2.answer_text)
        self.assertEqual(ans1.to_dict(), ans2.to_dict())


if __name__ == '__main__':
    unittest.main()
