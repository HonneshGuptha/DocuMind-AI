"""
Evaluation Module
Provides deterministic performance evaluation, benchmarking datasets, and metric
calculations for DocuMind AI.
"""

from .evaluation_models import EvaluationCase, CaseEvaluationResult, EvaluationSummary
from .metrics import (
    exact_match,
    normalize_text,
    calculate_retrieval_metrics,
    aggregate_summary,
    safe_divide,
)
from .test_dataset import (
    BENCHMARK_KB_FACTS,
    BENCHMARK_DATASET,
    get_benchmark_dataset,
    get_benchmark_facts,
)
from .evaluator import SystemEvaluator, run_evaluation

__all__ = [
    'EvaluationCase',
    'CaseEvaluationResult',
    'EvaluationSummary',
    'exact_match',
    'normalize_text',
    'calculate_retrieval_metrics',
    'aggregate_summary',
    'safe_divide',
    'BENCHMARK_KB_FACTS',
    'BENCHMARK_DATASET',
    'get_benchmark_dataset',
    'get_benchmark_facts',
    'SystemEvaluator',
    'run_evaluation',
]
