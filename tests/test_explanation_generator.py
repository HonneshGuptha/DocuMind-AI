"""
Unit tests for Module 5 Stage 2: Explanation Generator.
Verifies transparent, deterministic, evidence-backed explanations for:
1. Direct factual answer explanation.
2. Inferred answer explanation.
3. Correct identification of inference rule IDs.
4. Correct identification of supporting fact IDs.
5. Multi-step inference explanation.
6. Boolean affirmative explanation.
7. Unknown answer explanation.
8. Unsupported question explanation.
9. Multiple-result explanation.
10. Conflicting-facts explanation.
11. Missing provenance handling.
12. Missing reasoning trace handling.
13. No fabricated source references.
14. Deterministic output across repeated executions.
15. Graceful handling of incomplete input.
16. Boolean unknown answer explanation (Open-World Assumption).
17. Detailed explanation dictionary audit.
18. Convenience function generate_explanation.
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
from modules.answer_generator.generator import AnswerGenerator
from modules.answer_generator.explanation_generator import (
    ExplanationGenerator,
    generate_explanation,
)


class TestExplanationGenerator(unittest.TestCase):
    """Exhaustive test suite for Deterministic Explanation Generator."""

    def setUp(self):
        self.exp_gen = ExplanationGenerator()
        self.ans_gen = AnswerGenerator()
        self.parser = QuestionParser()

    # 1. Explanation for a direct factual answer
    def test_01_direct_factual_answer(self):
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
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans, retrieval_result=retrieval)

        self.assertIn("Direct Factual Retrieval", exp)
        self.assertIn("Fact #101", exp)
        self.assertIn("Bangalore", exp)
        self.assertIn("Karnataka", exp)
        self.assertIn("Document #1", exp)
        self.assertIn("Bangalore is the capital of Karnataka.", exp)

    # 2. Explanation for an inferred answer
    def test_02_inferred_answer(self):
        query = self.parser.parse("Where is Bangalore located?")
        trace = InferenceTrace(
            derived_fact=ReasoningFact("Bangalore", "located in", "India", fact_type="inferred", fact_id="inf_1"),
            applied_rule_id="R1",
            supporting_fact_ids=[1, 2],
            reasoning_method="backward_chaining"
        )
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "India"),
            is_proven=True,
            applied_rule_ids=["R1"],
            supporting_fact_ids=[1, 2],
            inference_traces=[trace]
        )
        ans = self.ans_gen.generate_answer(query, reasoning_result=proof)

        exp = self.exp_gen.generate_explanation(answer_response=ans, reasoning_result=proof)

        self.assertIn("Symbolic Deductive Inference", exp)
        self.assertIn("R1", exp)
        self.assertIn("Capital Location Rule", exp)
        self.assertIn("Derived Fact #inf_1", exp)

    # 3. Correct identification of inference rule IDs
    def test_03_correct_rule_id_identification(self):
        query = self.parser.parse("Where is Bangalore located?")
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "Asia"),
            is_proven=True,
            applied_rule_ids=["R1", "R2"],
            supporting_fact_ids=[1, 2, 3]
        )
        ans = self.ans_gen.generate_answer(query, reasoning_result=proof)

        exp = self.exp_gen.generate_explanation(answer_response=ans, reasoning_result=proof)

        self.assertIn("Rule R1", exp)
        self.assertIn("Capital Location Rule", exp)
        self.assertIn("Rule R2", exp)
        self.assertIn("Location Transitivity Rule", exp)

    # 4. Correct identification of supporting fact IDs
    def test_04_correct_supporting_fact_ids(self):
        query = self.parser.parse("Where is Bangalore located?")
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "India"),
            is_proven=True,
            applied_rule_ids=["R1"],
            supporting_fact_ids=[42, 43]
        )
        ans = self.ans_gen.generate_answer(query, reasoning_result=proof)

        exp = self.exp_gen.generate_explanation(answer_response=ans, reasoning_result=proof)

        self.assertIn("42", exp)
        self.assertIn("43", exp)
        self.assertIn("[42, 43]", exp)

    # 5. Multi-step inference explanation
    def test_05_multi_step_inference_explanation(self):
        query = self.parser.parse("Where is Bangalore located?")
        trace1 = InferenceTrace(
            derived_fact=ReasoningFact("Bangalore", "located in", "India", fact_type="inferred", fact_id="inf_1"),
            applied_rule_id="R1",
            supporting_fact_ids=[10, 11],
            reasoning_method="backward_chaining"
        )
        trace2 = InferenceTrace(
            derived_fact=ReasoningFact("Bangalore", "located in", "Asia", fact_type="inferred", fact_id="inf_2"),
            applied_rule_id="R2",
            supporting_fact_ids=["inf_1", 12],
            reasoning_method="backward_chaining"
        )
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "Asia"),
            is_proven=True,
            applied_rule_ids=["R1", "R2"],
            supporting_fact_ids=[10, 11, 12],
            inference_traces=[trace1, trace2]
        )
        ans = self.ans_gen.generate_answer(query, reasoning_result=proof)

        exp = self.exp_gen.generate_explanation(answer_response=ans, reasoning_result=proof)

        self.assertIn("Step 1: Applied Rule R1", exp)
        self.assertIn("Step 2: Applied Rule R2", exp)
        self.assertIn("Derived Fact #inf_1", exp)
        self.assertIn("Derived Fact #inf_2", exp)

    # 6. Boolean affirmative explanation
    def test_06_boolean_affirmative_explanation(self):
        query = self.parser.parse("Does Canada border United States?")
        fact = KBF(
            document_id=5,
            subject="Canada",
            predicate="borders",
            object="United States",
            source_sentence="Canada borders United States.",
            sentence_index=0,
            fact_id=501
        )
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans, query=query)

        self.assertIn("Boolean Verification confirmed", exp)
        self.assertIn("Fact #501", exp)

    # 7. Unknown answer explanation
    def test_07_unknown_answer_explanation(self):
        query = self.parser.parse("What is the capital of Atlantis?")
        retrieval = RetrievalResult(query=query, facts=[], status="no_match")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans, query=query)

        self.assertIn("Knowledge Base Insufficiency", exp)
        self.assertIn("Open-World Assumption", exp)
        self.assertIn("0 facts matched", exp)

    # 8. Unsupported question explanation
    def test_08_unsupported_question_explanation(self):
        query = self.parser.parse("Why is the sky blue?")
        ans = self.ans_gen.generate_answer(query)

        exp = self.exp_gen.generate_explanation(answer_response=ans, query=query)

        self.assertIn("Unsupported Question Format", exp)
        self.assertIn("Why is the sky blue?", exp)
        self.assertIn("no answer, rule, or source citation has been fabricated", exp)

    # 9. Multiple-result explanation
    def test_09_multiple_result_explanation(self):
        query = self.parser.parse("What does Canada border?")
        f1 = KBF(document_id=1, subject="Canada", predicate="borders", object="United States", source_sentence="Canada borders United States.", sentence_index=0, fact_id=1)
        f2 = KBF(document_id=1, subject="Canada", predicate="borders", object="Greenland", source_sentence="Canada borders Greenland.", sentence_index=1, fact_id=2)
        retrieval = RetrievalResult(query=query, facts=[f1, f2], status="success")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans, query=query)

        self.assertIn("Multiple Evidence Records (2 ground facts)", exp)
        self.assertIn("Fact #1: (Canada, borders, United States)", exp)
        self.assertIn("Fact #2: (Canada, borders, Greenland)", exp)

    # 10. Conflicting-facts explanation
    def test_10_conflicting_facts_explanation(self):
        query = self.parser.parse("When was Python established?")
        f1 = KBF(document_id=1, subject="Python", predicate="established in", object="1991", source_sentence="Python was established in 1991.", sentence_index=0, fact_id=1)
        f2 = KBF(document_id=2, subject="Python", predicate="established in", object="1989", source_sentence="Python was established in 1989.", sentence_index=0, fact_id=2)
        retrieval = RetrievalResult(query=query, facts=[f1, f2], status="success")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans, query=query)

        self.assertIn("Conflicting Information Detected", exp)
        self.assertIn("Fact #1", exp)
        self.assertIn("Fact #2", exp)
        self.assertIn("Resolution Policy", exp)

    # 11. Missing provenance handling
    def test_11_missing_provenance_handling(self):
        # Fact with document_id and source_sentence as None
        fact = KBF(
            document_id=None,
            subject="Earth",
            predicate="located in",
            object="Solar System",
            source_sentence=None,
            sentence_index=None,
            fact_id=777
        )
        query = self.parser.parse("Where is Earth located?")
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans)

        self.assertIn("Source document reference: Not recorded in knowledge base", exp)
        # Verify no fabricated document or sentence string appears
        self.assertNotIn("Document #None", exp)

    # 12. Missing reasoning trace handling
    def test_12_missing_reasoning_trace_handling(self):
        query = self.parser.parse("Where is Bangalore located?")
        # ProofResult with empty inference_traces list
        proof = ProofResult(
            status=ProofStatus("proven"),
            goal=("Bangalore", "located in", "India"),
            is_proven=True,
            applied_rule_ids=["R1"],
            supporting_fact_ids=[1, 2],
            inference_traces=[]
        )
        ans = self.ans_gen.generate_answer(query, reasoning_result=proof)

        exp = self.exp_gen.generate_explanation(answer_response=ans, reasoning_result=proof)

        self.assertIn("Symbolic Deductive Inference", exp)
        self.assertIn("Rule R1", exp)
        self.assertIn("Step 1:", exp)

    # 13. No fabricated source references
    def test_13_no_fabricated_source_references(self):
        query = self.parser.parse("Where is Mars located?")
        retrieval = RetrievalResult(query=query, facts=[], status="no_match")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans)

        self.assertNotIn("Document #", exp)
        self.assertIn("0 facts matched", exp)

    # 14. Deterministic output across repeated executions
    def test_14_deterministic_repeated_execution(self):
        query = self.parser.parse("What is the capital of Karnataka?")
        fact = KBF(document_id=1, subject="Bangalore", predicate="capital of", object="Karnataka", source_sentence="Bangalore is the capital of Karnataka.", sentence_index=0, fact_id=101)
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp1 = self.exp_gen.generate_explanation(answer_response=ans, retrieval_result=retrieval)
        exp2 = self.exp_gen.generate_explanation(answer_response=ans, retrieval_result=retrieval)

        self.assertEqual(exp1, exp2)

    # 15. Graceful handling of incomplete input
    def test_15_graceful_handling_of_incomplete_input(self):
        exp = self.exp_gen.generate_explanation(
            answer_response=None,
            query=None,
            retrieval_result=None,
            reasoning_result=None
        )
        self.assertIn("Incomplete input", exp)

    # 16. Boolean unknown answer explanation under Open-World Assumption
    def test_16_boolean_unknown_answer(self):
        query = self.parser.parse("Does Canada border Brazil?")
        retrieval = RetrievalResult(query=query, facts=[], status="no_match")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        exp = self.exp_gen.generate_explanation(answer_response=ans, query=query)

        self.assertIn("Boolean Verification", exp)
        self.assertIn("Open-World Assumption (OWA)", exp)
        self.assertIn("absence of proof is not proof of negation", exp)
        self.assertIn("unrecorded in the knowledge base", exp)

    # 17. Structured detailed explanation audit dictionary
    def test_17_structured_detailed_explanation(self):
        query = self.parser.parse("Where is Tokyo located?")
        fact = KBF(document_id=1, subject="Tokyo", predicate="located in", object="Japan", source_sentence="Tokyo is located in Japan.", sentence_index=0, fact_id=102)
        retrieval = RetrievalResult(query=query, facts=[fact], status="success")
        ans = self.ans_gen.generate_answer(query, retrieval_result=retrieval)

        audit = self.exp_gen.generate_detailed_explanation(answer_response=ans)

        self.assertIsInstance(audit, dict)
        self.assertEqual(audit['answer_status'], 'answered')
        self.assertEqual(audit['answer_type'], 'direct')
        self.assertEqual(audit['supporting_fact_ids'], [102])
        self.assertIn("explanation_text", audit)

    # 18. Convenience module-level function
    def test_18_convenience_function(self):
        query = "What is the capital of Karnataka?"
        fact = KBF(document_id=1, subject="Bangalore", predicate="capital of", object="Karnataka", source_sentence="Bangalore is the capital of Karnataka.", sentence_index=0, fact_id=101)
        retrieval = RetrievalResult(query=self.parser.parse(query), facts=[fact], status="success")

        exp = generate_explanation(query=query, retrieval_result=retrieval)

        self.assertIn("Direct Factual Retrieval", exp)
        self.assertIn("Bangalore", exp)


if __name__ == '__main__':
    unittest.main()
