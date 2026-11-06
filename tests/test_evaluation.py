"""
Unit tests for Module 5 Stage 3: Evaluation Framework.
Verifies all 18 required evaluation capabilities:
1. Exact answer matching.
2. Case and whitespace normalization.
3. Incorrect answer detection.
4. Unknown-answer evaluation.
5. Unsupported-question evaluation.
6. Retrieval precision.
7. Retrieval recall.
8. Retrieval F1.
9. Zero-denominator handling.
10. Correct inference evaluation.
11. Incorrect inference evaluation.
12. Rule ID validation.
13. Supporting fact validation.
14. Multi-step inference validation.
15. Explanation provenance validation.
16. Explanation completeness.
17. Deterministic repeated evaluation.
18. Full evaluation report generation.
"""

import unittest
from modules.evaluation.evaluation_models import EvaluationCase, CaseEvaluationResult, EvaluationSummary
from modules.evaluation.metrics import (
    exact_match,
    normalize_text,
    calculate_retrieval_metrics,
    safe_divide,
    aggregate_summary,
)
from modules.evaluation.evaluator import SystemEvaluator, run_evaluation
from modules.evaluation.test_dataset import (
    get_benchmark_dataset,
    BENCHMARK_DATASET,
    BENCHMARK_KB_FACTS,
)


class TestEvaluationFramework(unittest.TestCase):
    """Exhaustive test suite for the Deterministic Evaluation Framework."""

    @classmethod
    def setUpClass(cls):
        cls.evaluator = SystemEvaluator()

    @classmethod
    def tearDownClass(cls):
        cls.evaluator.cleanup()

    # 1. Exact answer matching
    def test_01_exact_answer_matching(self):
        s1 = "The capital of Karnataka is Bangalore."
        s2 = "The capital of Karnataka is Bangalore."
        self.assertTrue(exact_match(s1, s2))

        s3 = "Bangalore is the capital of Karnataka."
        self.assertFalse(exact_match(s1, s3))

    # 2. Case and whitespace normalization
    def test_02_case_and_whitespace_normalization(self):
        s1 = "  Tokyo   is located   in Japan.  "
        s2 = "tokyo is located in japan"
        self.assertTrue(exact_match(s1, s2))

        norm = normalize_text("  PYTHON  was   developed   by Guido!??  ")
        self.assertEqual(norm, "python was developed by guido")

    # 3. Incorrect answer detection
    def test_03_incorrect_answer_detection(self):
        case = EvaluationCase(
            case_id="TEST_WRONG_ANS",
            question="What is the capital of Karnataka?",
            category="capital",
            expected_answer_status="answered",
            expected_answer_text="The capital of Karnataka is Mysore."  # Wrong expectation
        )
        res = self.evaluator.evaluate_case(case)

        self.assertTrue(res.status_match)
        self.assertFalse(res.answer_match)
        self.assertFalse(res.is_overall_success)

    # 4. Unknown-answer evaluation
    def test_04_unknown_answer_evaluation(self):
        case = EvaluationCase(
            case_id="TEST_UNKNOWN",
            question="What is the capital of Atlantis?",
            category="capital",
            expected_answer_status="unknown",
            expected_answer_type="none",
            expected_answer_text="I could not find sufficient information in the knowledge base to answer this question."
        )
        res = self.evaluator.evaluate_case(case)

        self.assertEqual(res.actual_status, "unknown")
        self.assertTrue(res.status_match)
        self.assertTrue(res.answer_match)
        self.assertTrue(res.is_overall_success)

    # 5. Unsupported-question evaluation
    def test_05_unsupported_question_evaluation(self):
        case = EvaluationCase(
            case_id="TEST_UNSUPPORTED",
            question="Why is the sky blue?",
            category="unsupported",
            expected_answer_status="unsupported",
            expected_answer_type="none",
            expected_answer_text="This question format is not currently supported by DocuMind AI."
        )
        res = self.evaluator.evaluate_case(case)

        self.assertEqual(res.actual_status, "unsupported")
        self.assertTrue(res.status_match)
        self.assertTrue(res.answer_match)
        self.assertTrue(res.no_fabricated_provenance)
        self.assertTrue(res.is_overall_success)

    # 6. Retrieval precision
    def test_06_retrieval_precision(self):
        retrieved = [
            ("A", "borders", "B"),
            ("A", "borders", "C")
        ]
        expected = [
            ("A", "borders", "B")
        ]
        tp, fp, fn, p, r, f1 = calculate_retrieval_metrics(retrieved, expected)

        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)
        self.assertEqual(p, 0.5)

    # 7. Retrieval recall
    def test_07_retrieval_recall(self):
        retrieved = [
            ("A", "borders", "B")
        ]
        expected = [
            ("A", "borders", "B"),
            ("A", "borders", "C")
        ]
        tp, fp, fn, p, r, f1 = calculate_retrieval_metrics(retrieved, expected)

        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)
        self.assertEqual(r, 0.5)

    # 8. Retrieval F1
    def test_08_retrieval_f1(self):
        retrieved = [
            ("A", "borders", "B")
        ]
        expected = [
            ("A", "borders", "B")
        ]
        tp, fp, fn, p, r, f1 = calculate_retrieval_metrics(retrieved, expected)

        self.assertEqual(tp, 1)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f1, 1.0)

    # 9. Zero-denominator and N/A metric handling
    def test_09_zero_denominator_handling(self):
        self.assertEqual(safe_divide(10, 0), 0.0)
        self.assertEqual(safe_divide(10, 0, default=-1.0), -1.0)

        # 1. Both expected and retrieved sets empty -> all N/A (None)
        tp, fp, fn, p, r, f1 = calculate_retrieval_metrics([], [])
        self.assertIsNone(p)
        self.assertIsNone(r)
        self.assertIsNone(f1)

        # 2. Expected empty, retrieved non-empty -> P=0.0, R=None (N/A), F1=None (N/A)
        tp, fp, fn, p, r, f1 = calculate_retrieval_metrics([("A", "located in", "B")], [])
        self.assertEqual(p, 0.0)
        self.assertIsNone(r)
        self.assertIsNone(f1)

        # 3. Expected non-empty, retrieved empty -> P=None (N/A), R=0.0, F1=None (N/A)
        tp, fp, fn, p, r, f1 = calculate_retrieval_metrics([], [("A", "located in", "B")])
        self.assertIsNone(p)
        self.assertEqual(r, 0.0)
        self.assertIsNone(f1)

    # 9b. Aggregation excluding N/A cases
    def test_09b_aggregation_excluding_na_cases(self):
        c1 = CaseEvaluationResult(
            case_id="C1", category="direct", question="q1",
            actual_status="answered", expected_status="answered", status_match=True,
            actual_answer_text="a1", expected_answer_text="a1", answer_match=True,
            retrieval_precision=1.0, retrieval_recall=1.0, retrieval_f1=1.0,
            expected_fact_triples=[("A", "b", "C")], retrieved_fact_triples=[("A", "b", "C")]
        )
        c2 = CaseEvaluationResult(
            case_id="C2", category="direct", question="q2",
            actual_status="answered", expected_status="answered", status_match=True,
            actual_answer_text="a2", expected_answer_text="a2", answer_match=True,
            retrieval_precision=0.5, retrieval_recall=0.5, retrieval_f1=0.5,
            expected_fact_triples=[("D", "e", "F")], retrieved_fact_triples=[("D", "e", "F"), ("D", "e", "G")]
        )
        # c3 is an unknown case where retrieval is N/A
        c3 = CaseEvaluationResult(
            case_id="C3", category="unknown", question="q3",
            actual_status="unknown", expected_status="unknown", status_match=True,
            actual_answer_text="a3", expected_answer_text="a3", answer_match=True,
            retrieval_precision=None, retrieval_recall=None, retrieval_f1=None,
            expected_fact_triples=[], retrieved_fact_triples=[]
        )

        summary = aggregate_summary([c1, c2, c3])

        # (1.0 + 0.5) / 2 = 0.75; c3 is excluded from average rather than counting as 0.0
        self.assertEqual(summary.aggregate_precision, 0.75)
        self.assertEqual(summary.aggregate_recall, 0.75)
        self.assertEqual(summary.aggregate_f1, 0.75)
        self.assertEqual(summary.retrieval_evaluated_cases, 2)
        self.assertEqual(summary.retrieval_excluded_cases, 1)

    # 9c. Empty-retrieval handling accuracy
    def test_09c_empty_retrieval_handling_accuracy(self):
        # 2 cases where expected is empty: one retrieved empty (correct), one retrieved facts (incorrect)
        c1 = CaseEvaluationResult(
            case_id="E1", category="unknown", question="q1",
            actual_status="unknown", expected_status="unknown", status_match=True,
            actual_answer_text="a1", expected_answer_text="a1", answer_match=True,
            expected_fact_triples=[], retrieved_fact_triples=[]
        )
        c2 = CaseEvaluationResult(
            case_id="E2", category="unsupported", question="q2",
            actual_status="unsupported", expected_status="unsupported", status_match=True,
            actual_answer_text="a2", expected_answer_text="a2", answer_match=True,
            expected_fact_triples=[], retrieved_fact_triples=[("Erroneous", "fact", "triple")]
        )

        summary = aggregate_summary([c1, c2])

        self.assertEqual(summary.empty_retrieval_cases, 2)
        self.assertEqual(summary.empty_retrieval_correct, 1)
        self.assertEqual(summary.empty_retrieval_accuracy, 50.0)

    # 10. Correct inference evaluation
    def test_10_correct_inference_evaluation(self):
        case = EvaluationCase(
            case_id="TEST_INF_R1",
            question="Where is Bangalore located?",
            category="location",
            expected_answer_status="answered",
            expected_answer_type="inferred",
            expected_answer_text="Bangalore is located in India.",
            expected_facts=[("Bangalore", "located in", "India")],
            expected_fact_ids=[1, 2],
            expected_rule_ids=["R1"]
        )
        res = self.evaluator.evaluate_case(case)

        self.assertTrue(res.rules_match)
        self.assertIn("R1", res.applied_rule_ids)
        self.assertTrue(res.supporting_facts_match)
        self.assertTrue(res.is_overall_success)

    # 11. Incorrect inference evaluation
    def test_11_incorrect_inference_evaluation(self):
        case = EvaluationCase(
            case_id="TEST_INF_WRONG_RULE",
            question="Where is Bangalore located?",
            category="location",
            expected_answer_status="answered",
            expected_answer_type="inferred",
            expected_answer_text="Bangalore is located in India.",
            expected_facts=[("Bangalore", "located in", "India")],
            expected_fact_ids=[1, 2],
            expected_rule_ids=["R4"]  # Expected R4, but R1 was fired
        )
        res = self.evaluator.evaluate_case(case)

        self.assertFalse(res.rules_match)
        self.assertFalse(res.is_overall_success)

    # 12. Rule ID validation
    def test_12_rule_id_validation(self):
        case = EvaluationCase(
            case_id="TEST_R4_BORDER",
            question="Does United States border Canada?",
            category="border",
            expected_answer_status="answered",
            expected_answer_type="inferred",
            expected_answer_text="Yes, United States borders Canada.",
            expected_facts=[("United States", "borders", "Canada")],
            expected_fact_ids=[11],
            expected_rule_ids=["R4"],
            is_boolean=True
        )
        res = self.evaluator.evaluate_case(case)

        self.assertEqual(res.applied_rule_ids, ["R4"])
        self.assertTrue(res.rules_match)

    # 13. Supporting fact validation
    def test_13_supporting_fact_validation(self):
        case = EvaluationCase(
            case_id="TEST_SUPP_FACTS",
            question="What is the capital of Karnataka?",
            category="capital",
            expected_answer_status="answered",
            expected_facts=[("Bangalore", "capital of", "Karnataka")],
            expected_fact_ids=[1]
        )
        res = self.evaluator.evaluate_case(case)

        self.assertIn(1, res.supporting_fact_ids)
        self.assertTrue(res.supporting_facts_match)

    # 14. Multi-step inference validation
    def test_14_multi_step_inference_validation(self):
        case = EvaluationCase(
            case_id="TEST_MULTISTEP",
            question="Is Piston a part of Automobile?",
            category="part_whole",
            expected_answer_status="answered",
            expected_answer_type="inferred",
            expected_answer_text="Yes, Piston is part of Automobile.",
            expected_facts=[("Piston", "part of", "Automobile")],
            expected_fact_ids=[10, 11, 12],
            expected_rule_ids=["R3"],
            is_boolean=True
        )
        res = self.evaluator.evaluate_case(case)

        self.assertTrue(res.is_multi_step)
        self.assertTrue(res.multi_step_correct)
        self.assertIn("R3", res.applied_rule_ids)
        self.assertTrue(len(res.applied_rule_ids) >= 2)
        self.assertTrue(res.is_overall_success)

    # 15. Explanation provenance validation
    def test_15_explanation_provenance_validation(self):
        case = EvaluationCase(
            case_id="TEST_PROVENANCE",
            question="Where is Tokyo located?",
            category="location",
            expected_answer_status="answered",
            expected_facts=[("Tokyo", "located in", "Japan")],
            expected_fact_ids=[4]
        )
        res = self.evaluator.evaluate_case(case)

        self.assertTrue(res.explanation_has_provenance)
        self.assertIn("Document #2", res.explanation_text)
        self.assertIn("Tokyo is located in Japan.", res.explanation_text)

    # 16. Explanation completeness
    def test_16_explanation_completeness(self):
        case = EvaluationCase(
            case_id="TEST_COMPLETENESS",
            question="Where is Bangalore located?",
            category="location",
            expected_answer_status="answered",
            expected_answer_type="inferred",
            expected_rule_ids=["R1"],
            expected_fact_ids=[1, 2]
        )
        res = self.evaluator.evaluate_case(case)

        self.assertTrue(res.explanation_has_rules)
        self.assertIn("Capital Location Rule", res.explanation_text)
        self.assertIn("R1", res.explanation_text)

    # 17. Deterministic repeated evaluation
    def test_17_deterministic_repeated_evaluation(self):
        case = EvaluationCase(
            case_id="TEST_DETERMINISTIC",
            question="What is Python?",
            category="classification",
            expected_answer_status="answered",
            expected_answer_text="Python is a programming language."
        )
        res1 = self.evaluator.evaluate_case(case)
        res2 = self.evaluator.evaluate_case(case)

        self.assertEqual(res1.actual_answer_text, res2.actual_answer_text)
        self.assertEqual(res1.is_overall_success, res2.is_overall_success)
        self.assertEqual(res1.status_match, res2.status_match)
        self.assertEqual(res1.retrieval_f1, res2.retrieval_f1)

    # 18. Full evaluation report generation
    def test_18_full_evaluation_report_generation(self):
        summary, report = run_evaluation(BENCHMARK_DATASET)

        self.assertIsInstance(summary, EvaluationSummary)
        self.assertEqual(summary.total_cases, len(BENCHMARK_DATASET))
        self.assertTrue(summary.successful_cases > 0)
        self.assertGreaterEqual(summary.answer_accuracy, 95.0)

        # Check markdown report contents
        self.assertIn("# DOCUMIND AI – SYSTEM EVALUATION REPORT", report)
        self.assertIn("EXECUTIVE SUMMARY", report)
        self.assertIn("DETAILED METRIC BREAKDOWN", report)
        self.assertIn("INDIVIDUAL TEST CASE AUDIT LOG", report)
        self.assertIn("Mean Retrieval Precision", report)
        self.assertIn("Empty-Retrieval Handling Accuracy", report)
        self.assertIn("Retrieval Metric Coverage", report)
        self.assertIn("N/A", report)


if __name__ == '__main__':
    unittest.main()
