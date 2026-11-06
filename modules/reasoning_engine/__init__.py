"""
Reasoning Engine Module
Implements classical rule-based reasoning, inference rules, and forward/backward chaining
over structured knowledge triples.
"""

from .reasoning_models import (
    Fact,
    InferenceRule,
    InferenceTrace,
    ReasoningResult,
)
from .working_memory import WorkingMemory
from .rule_base import (
    R1,
    R2,
    R3,
    R4,
    R1_CAPITAL_LOCATION,
    R2_LOCATION_TRANSITIVITY,
    R3_PART_WHOLE_TRANSITIVITY,
    R4_BORDER_SYMMETRY,
    ALL_RULES,
    get_all_rules,
    get_rule_by_id,
)
from .rule_matcher import (
    RuleMatcher,
    RuleMatch,
    match_rule,
    is_variable,
    get_variable_name,
)
from .forward_chaining import (
    ForwardChainer,
    forward_chain,
)
from .backward_chaining import (
    BackwardChainer,
    backward_chain,
    ProofResult,
    ProofStatus,
)

__all__ = [
    'Fact',
    'InferenceRule',
    'InferenceTrace',
    'ReasoningResult',
    'WorkingMemory',
    'R1',
    'R2',
    'R3',
    'R4',
    'R1_CAPITAL_LOCATION',
    'R2_LOCATION_TRANSITIVITY',
    'R3_PART_WHOLE_TRANSITIVITY',
    'R4_BORDER_SYMMETRY',
    'ALL_RULES',
    'get_all_rules',
    'get_rule_by_id',
    'RuleMatcher',
    'RuleMatch',
    'match_rule',
    'is_variable',
    'get_variable_name',
    'ForwardChainer',
    'forward_chain',
    'BackwardChainer',
    'backward_chain',
    'ProofResult',
    'ProofStatus',
]
