"""
Metrics Module
Implements deterministic, transparent evaluation metrics for DocuMind AI.

ACADEMIC RESTRICTION COMPLIANCE:
- Pure classical symbolic evaluation: strictly zero NLP similarity, zero embeddings, zero LLMs.
- Deterministic string normalization: case folding and whitespace stripping only.
- Strict division-by-zero safeguards and clear mathematical formulations.
"""

import re
from typing import List, Tuple, Dict, Any, Optional, Set, Union
from .evaluation_models import CaseEvaluationResult, EvaluationSummary


def normalize_text(text: Optional[str]) -> str:
    """
    Safely normalize natural language strings for exact-match comparison.
    - Strips leading and trailing whitespace.
    - Normalizes internal multiple whitespace sequences to a single space.
    - Removes trailing sentence punctuation (., ?, !).
    - Converts to lowercase.
    """
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'[\.\?!]+$', '', cleaned).strip()
    return cleaned


def exact_match(actual: Optional[str], expected: Optional[str]) -> bool:
    """
    Evaluate exact match after deterministic whitespace and case normalization.
    """
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    return normalize_text(actual) == normalize_text(expected)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely perform division with zero-denominator protection.
    """
    if denominator == 0 or denominator == 0.0:
        return default
    return float(numerator) / float(denominator)


def normalize_triple(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Normalize SPO triple terms for case-insensitive set matching."""
    s, p, o = triple
    return (str(s).strip().lower(), str(p).strip().lower(), str(o).strip().lower())


def calculate_retrieval_metrics(
    retrieved_triples: List[Tuple[str, str, str]],
    expected_triples: List[Tuple[str, str, str]]
) -> Tuple[int, int, int, Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Precision, Recall, and F1-Score for factual retrieval.

    Formulation:
        - True Positives (TP): Triples present in both retrieved and expected sets.
        - False Positives (FP): Triples present in retrieved but NOT in expected.
        - False Negatives (FN): Triples present in expected but NOT in retrieved.

    Academic N/A Conventions:
        - If both sets are empty: Precision = None (N/A), Recall = None (N/A), F1 = None (N/A).
        - If expected is empty and retrieved is non-empty:
          TP = 0, FP = len(retrieved), FN = 0.
          Precision = 0.0 (erroneous retrieval), Recall = None (N/A, zero denominator), F1 = None (N/A).
        - If expected is non-empty and retrieved is empty:
          TP = 0, FP = 0, FN = len(expected).
          Precision = None (N/A, zero denominator), Recall = 0.0 (missed retrieval), F1 = None (N/A).
        - If both sets are non-empty:
          Precision = TP / (TP + FP)
          Recall = TP / (TP + FN)
          F1 = (2 * P * R) / (P + R) if (P + R > 0) else 0.0.

    Returns:
        (tp, fp, fn, precision, recall, f1)
    """
    retrieved_set: Set[Tuple[str, str, str]] = {normalize_triple(t) for t in retrieved_triples}
    expected_set: Set[Tuple[str, str, str]] = {normalize_triple(t) for t in expected_triples}

    len_retrieved = len(retrieved_set)
    len_expected = len(expected_set)

    # Case 1: Both sets empty (N/A across all retrieval metrics)
    if len_expected == 0 and len_retrieved == 0:
        return 0, 0, 0, None, None, None

    # Case 2: Expected empty, retrieved non-empty
    if len_expected == 0 and len_retrieved > 0:
        return 0, len_retrieved, 0, 0.0, None, None

    # Case 3: Expected non-empty, retrieved empty
    if len_expected > 0 and len_retrieved == 0:
        return 0, 0, len_expected, None, 0.0, None

    # Case 4: Both non-empty
    tp = len(retrieved_set.intersection(expected_set))
    fp = len(retrieved_set.difference(expected_set))
    fn = len(expected_set.difference(retrieved_set))

    precision = round(float(tp) / float(tp + fp), 4)
    recall = round(float(tp) / float(tp + fn), 4)

    if precision + recall > 0.0:
        f1 = round((2.0 * precision * recall) / (precision + recall), 4)
    else:
        f1 = 0.0

    return tp, fp, fn, precision, recall, f1


def aggregate_summary(case_results: List[CaseEvaluationResult]) -> EvaluationSummary:
    """
    Aggregate individual case results into an empirical EvaluationSummary.
    """
    total_cases = len(case_results)
    if total_cases == 0:
        return EvaluationSummary()

    successful_cases = sum(1 for c in case_results if c.is_overall_success)
    failed_cases = total_cases - successful_cases

    # Answer Evaluation
    status_matches = sum(1 for c in case_results if c.status_match)
    exact_matches = sum(1 for c in case_results if c.answer_match)

    unknown_cases = [c for c in case_results if c.expected_status == 'unknown']
    unknown_correct = sum(1 for c in unknown_cases if c.actual_status == 'unknown')

    unsupported_cases = [c for c in case_results if c.expected_status == 'unsupported']
    unsupported_correct = sum(1 for c in unsupported_cases if c.actual_status == 'unsupported')

    # Retrieval Evaluation Aggregation (Strictly excluding N/A cases)
    valid_p_scores = [c.retrieval_precision for c in case_results if c.retrieval_precision is not None]
    valid_r_scores = [c.retrieval_recall for c in case_results if c.retrieval_recall is not None]
    valid_f1_scores = [c.retrieval_f1 for c in case_results if c.retrieval_f1 is not None]

    retrieval_evaluated_cases = len(set(
        i for i, c in enumerate(case_results)
        if c.retrieval_precision is not None or c.retrieval_recall is not None or c.retrieval_f1 is not None
    ))
    retrieval_excluded_cases = total_cases - retrieval_evaluated_cases

    agg_precision = round(sum(valid_p_scores) / len(valid_p_scores), 4) if valid_p_scores else None
    agg_recall = round(sum(valid_r_scores) / len(valid_r_scores), 4) if valid_r_scores else None
    agg_f1 = round(sum(valid_f1_scores) / len(valid_f1_scores), 4) if valid_f1_scores else None

    # Empty Retrieval Evaluation (Dedicated metric for unrecorded / unsupported queries)
    empty_expected_cases = [c for c in case_results if len(c.expected_fact_triples) == 0]
    empty_retrieval_cases_count = len(empty_expected_cases)
    empty_retrieval_correct_count = sum(1 for c in empty_expected_cases if len(c.retrieved_fact_triples) == 0)
    empty_retrieval_acc = round(
        safe_divide(empty_retrieval_correct_count, empty_retrieval_cases_count, 1.0) * 100.0, 2
    )

    # Reasoning Evaluation
    reasoning_cases = [c for c in case_results if c.expected_rule_ids or c.applied_rule_ids]
    total_reasoning = len(reasoning_cases)
    correct_inferences = sum(1 for c in reasoning_cases if c.rules_match and c.supporting_facts_match)
    incorrect_inferences = total_reasoning - correct_inferences
    rule_correct = sum(1 for c in reasoning_cases if c.rules_match)
    premise_correct = sum(1 for c in reasoning_cases if c.supporting_facts_match)

    multi_step_cases = [c for c in reasoning_cases if c.is_multi_step]
    multi_step_correct = sum(1 for c in multi_step_cases if c.multi_step_correct)

    # Explanation Evaluation
    rule_disclosure_ok = sum(1 for c in case_results if c.explanation_has_rules)
    fact_id_ok = sum(1 for c in case_results if c.supporting_facts_match)
    provenance_ok = sum(1 for c in case_results if c.explanation_has_provenance)
    no_fab_ok = sum(1 for c in case_results if c.no_fabricated_provenance)

    # Performance
    total_time = sum(c.execution_time_ms for c in case_results)
    avg_time = safe_divide(total_time, total_cases, 0.0)

    # Category Breakdown
    cat_breakdown: Dict[str, Dict[str, Any]] = {}
    for c in case_results:
        cat = c.category
        if cat not in cat_breakdown:
            cat_breakdown[cat] = {'total': 0, 'correct': 0, 'accuracy': 0.0}
        cat_breakdown[cat]['total'] += 1
        if c.is_overall_success:
            cat_breakdown[cat]['correct'] += 1

    for cat, data in cat_breakdown.items():
        data['accuracy'] = round(safe_divide(data['correct'], data['total'], 0.0) * 100.0, 2)

    return EvaluationSummary(
        total_cases=total_cases,
        successful_cases=successful_cases,
        failed_cases=failed_cases,
        correct_answers_count=status_matches,
        incorrect_answers_count=total_cases - status_matches,
        answer_accuracy=round(safe_divide(status_matches, total_cases, 0.0) * 100.0, 2),
        exact_match_accuracy=round(safe_divide(exact_matches, total_cases, 0.0) * 100.0, 2),
        unknown_detection_accuracy=round(safe_divide(unknown_correct, len(unknown_cases), 1.0) * 100.0, 2),
        unsupported_detection_accuracy=round(safe_divide(unsupported_correct, len(unsupported_cases), 1.0) * 100.0, 2),
        aggregate_precision=agg_precision,
        aggregate_recall=agg_recall,
        aggregate_f1=agg_f1,
        retrieval_evaluated_cases=retrieval_evaluated_cases,
        retrieval_excluded_cases=retrieval_excluded_cases,
        empty_retrieval_cases=empty_retrieval_cases_count,
        empty_retrieval_correct=empty_retrieval_correct_count,
        empty_retrieval_accuracy=empty_retrieval_acc,
        total_reasoning_cases=total_reasoning,
        correct_inferences_count=correct_inferences,
        incorrect_inferences_count=incorrect_inferences,
        inference_accuracy=round(safe_divide(correct_inferences, total_reasoning, 1.0) * 100.0, 2),
        rule_application_accuracy=round(safe_divide(rule_correct, total_reasoning, 1.0) * 100.0, 2),
        supporting_premise_accuracy=round(safe_divide(premise_correct, total_reasoning, 1.0) * 100.0, 2),
        multi_step_accuracy=round(safe_divide(multi_step_correct, len(multi_step_cases), 1.0) * 100.0, 2),
        rule_disclosure_accuracy=round(safe_divide(rule_disclosure_ok, total_cases, 0.0) * 100.0, 2),
        fact_id_accuracy=round(safe_divide(fact_id_ok, total_cases, 0.0) * 100.0, 2),
        source_provenance_accuracy=round(safe_divide(provenance_ok, total_cases, 0.0) * 100.0, 2),
        no_fabricated_provenance_rate=round(safe_divide(no_fab_ok, total_cases, 1.0), 4),
        average_execution_time_ms=round(avg_time, 2),
        total_execution_time_ms=round(total_time, 2),
        category_breakdown=cat_breakdown,
        case_results=case_results
    )
