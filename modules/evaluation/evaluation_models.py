"""
Evaluation Models Module
Defines data structures for evaluating the DocuMind AI symbolic question-answering pipeline.

ACADEMIC EXPLANATION:
In Classical Artificial Intelligence, rigorous empirical evaluation requires
comparing system outputs against ground-truth benchmarks across every stage of the pipeline:
1. Retrieval: Precision, Recall, and F1 over ground facts.
2. Deductive Reasoning: Rule application accuracy and premise correctness.
3. Answer Generation: Exact-match accuracy and status validation (e.g. OWA unknown detection).
4. Explanation Generation: Audit fidelity, provenance retention, and anti-fabrication verification.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union


@dataclass
class EvaluationCase:
    """
    Represents an atomic, manually verified evaluation test case.

    Attributes:
        case_id: Unique test identifier (e.g. 'CAPITAL_DIRECT_01').
        question: Verbatim natural language input query.
        category: Factual relation category (e.g. 'capital', 'location', 'border', 'unsupported').
        expected_answer_status: 'answered', 'unknown', 'unsupported', or 'error'.
        expected_answer_type: 'direct', 'inferred', 'mixed', or 'none'.
        expected_answer_text: Canonical answer string or None if unknown/unsupported.
        expected_facts: List of expected (subject, predicate, object) triples.
        expected_fact_ids: List of database Fact IDs expected to support the answer.
        expected_rule_ids: List of inference rule IDs expected to fire (e.g. ['R1']).
        description: Academic explanation of the target test scenario.
        is_boolean: True if query evaluates boolean truth value.
        is_conflicting: True if test involves contradictory ground facts.
        is_multiple: True if query expects multiple factual answers.
    """
    case_id: str
    question: str
    category: str
    expected_answer_status: str
    expected_answer_type: str = 'direct'
    expected_answer_text: Optional[str] = None
    expected_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    expected_fact_ids: List[Union[int, str]] = field(default_factory=list)
    expected_rule_ids: List[str] = field(default_factory=list)
    description: str = ""
    is_boolean: bool = False
    is_conflicting: bool = False
    is_multiple: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaseEvaluationResult:
    """
    Detailed evaluation outcome for a single EvaluationCase.
    """
    case_id: str
    category: str
    question: str

    # Answer Evaluation
    actual_status: str
    expected_status: str
    status_match: bool
    actual_answer_text: str
    expected_answer_text: Optional[str]
    answer_match: bool

    # Retrieval Evaluation
    retrieved_fact_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    expected_fact_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    retrieval_tp: int = 0
    retrieval_fp: int = 0
    retrieval_fn: int = 0
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    retrieval_f1: Optional[float] = None
    empty_retrieval_correct: Optional[bool] = None

    # Reasoning Evaluation
    applied_rule_ids: List[str] = field(default_factory=list)
    expected_rule_ids: List[str] = field(default_factory=list)
    rules_match: bool = True
    supporting_fact_ids: List[Union[int, str]] = field(default_factory=list)
    expected_fact_ids: List[Union[int, str]] = field(default_factory=list)
    supporting_facts_match: bool = True
    is_multi_step: bool = False
    multi_step_correct: bool = True

    # Explanation Evaluation
    explanation_text: str = ""
    explanation_has_rules: bool = True
    explanation_has_provenance: bool = True
    no_fabricated_provenance: bool = True

    # Overall Verdict
    is_overall_success: bool = True
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """
    Aggregated evaluation metrics across an entire evaluation dataset run.
    """
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0

    # Answer metrics
    correct_answers_count: int = 0
    incorrect_answers_count: int = 0
    answer_accuracy: float = 0.0
    exact_match_accuracy: float = 0.0
    unknown_detection_accuracy: float = 0.0
    unsupported_detection_accuracy: float = 0.0

    # Retrieval metrics
    aggregate_precision: Optional[float] = None
    aggregate_recall: Optional[float] = None
    aggregate_f1: Optional[float] = None
    retrieval_evaluated_cases: int = 0
    retrieval_excluded_cases: int = 0
    empty_retrieval_cases: int = 0
    empty_retrieval_correct: int = 0
    empty_retrieval_accuracy: float = 0.0

    # Reasoning metrics
    total_reasoning_cases: int = 0
    correct_inferences_count: int = 0
    incorrect_inferences_count: int = 0
    inference_accuracy: float = 0.0
    rule_application_accuracy: float = 0.0
    supporting_premise_accuracy: float = 0.0
    multi_step_accuracy: float = 0.0

    # Explanation metrics
    rule_disclosure_accuracy: float = 0.0
    fact_id_accuracy: float = 0.0
    source_provenance_accuracy: float = 0.0
    no_fabricated_provenance_rate: float = 1.0

    # Performance
    average_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0

    # Category breakdown: {category_name: {'total': int, 'correct': int, 'accuracy': float}}
    category_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Detailed test case results
    case_results: List[CaseEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res['case_results'] = [c.to_dict() for c in self.case_results]
        return res
