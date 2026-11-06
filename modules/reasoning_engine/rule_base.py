"""
Rule Base Module
Defines deterministic classical inference rules for symbolic reasoning in DocuMind AI.

ACADEMIC EXPLANATION:
In Symbolic Artificial Intelligence, production rules represent deductive knowledge
in the form of IF-THEN clauses (Horn clauses):
    Premise 1 AND Premise 2 AND ... -> Conclusion

These rules encode fundamental domain principles:
1. R1 (Capital Location): If X is the capital of Y and Y is a state in Z, then X is located in Z.
2. R2 (Location Transitivity): If X is located in Y and Y is located in Z, then X is located in Z.
3. R3 (Part-Whole Transitivity): If X is part of Y and Y is part of Z, then X is part of Z.
4. R4 (Border Symmetry): If X borders Y, then Y borders X.
"""

from typing import List, Optional, Dict
from .reasoning_models import InferenceRule


# ==============================================================================
# Rule Definitions
# ==============================================================================

R1_CAPITAL_LOCATION = InferenceRule(
    rule_id="R1",
    rule_name="Capital Location Rule",
    premises=[
        ("X", "capital of", "Y"),
        ("Y", "is a state in", "Z"),
    ],
    conclusion=("X", "located in", "Z"),
    description="If X is the capital of state Y, and Y is a state in country Z, then X is located in country Z."
)

R2_LOCATION_TRANSITIVITY = InferenceRule(
    rule_id="R2",
    rule_name="Location Transitivity Rule",
    premises=[
        ("X", "located in", "Y"),
        ("Y", "located in", "Z"),
    ],
    conclusion=("X", "located in", "Z"),
    description="If entity X is located in region Y, and Y is located in region Z, then X is located in Z."
)

R3_PART_WHOLE_TRANSITIVITY = InferenceRule(
    rule_id="R3",
    rule_name="Part-Whole Transitivity Rule",
    premises=[
        ("X", "part of", "Y"),
        ("Y", "part of", "Z"),
    ],
    conclusion=("X", "part of", "Z"),
    description="If component X is part of system Y, and Y is part of system Z, then X is part of Z."
)

R4_BORDER_SYMMETRY = InferenceRule(
    rule_id="R4",
    rule_name="Border Symmetry Rule",
    premises=[
        ("X", "borders", "Y"),
    ],
    conclusion=("Y", "borders", "X"),
    description="If entity X borders entity Y, then Y borders X (geographic adjacency is symmetric)."
)

# Canonical aliases
R1 = R1_CAPITAL_LOCATION
R2 = R2_LOCATION_TRANSITIVITY
R3 = R3_PART_WHOLE_TRANSITIVITY
R4 = R4_BORDER_SYMMETRY

# Master list of all available rules
ALL_RULES: List[InferenceRule] = [
    R1_CAPITAL_LOCATION,
    R2_LOCATION_TRANSITIVITY,
    R3_PART_WHOLE_TRANSITIVITY,
    R4_BORDER_SYMMETRY,
]

# Case-insensitive lookup dictionary supporting both short IDs ("R1") and full IDs ("R1_CAPITAL_LOCATION")
_RULE_LOOKUP: Dict[str, InferenceRule] = {
    "r1": R1_CAPITAL_LOCATION,
    "r1_capital_location": R1_CAPITAL_LOCATION,
    "capital_location": R1_CAPITAL_LOCATION,
    "r2": R2_LOCATION_TRANSITIVITY,
    "r2_location_transitivity": R2_LOCATION_TRANSITIVITY,
    "location_transitivity": R2_LOCATION_TRANSITIVITY,
    "r3": R3_PART_WHOLE_TRANSITIVITY,
    "r3_part_whole_transitivity": R3_PART_WHOLE_TRANSITIVITY,
    "part_whole_transitivity": R3_PART_WHOLE_TRANSITIVITY,
    "r4": R4_BORDER_SYMMETRY,
    "r4_border_symmetry": R4_BORDER_SYMMETRY,
    "border_symmetry": R4_BORDER_SYMMETRY,
}


# ==============================================================================
# Rule Retrieval Functions
# ==============================================================================

def get_all_rules() -> List[InferenceRule]:
    """
    Retrieve all deterministic inference rules defined in the Rule Base.
    Returns a new list containing the four canonical inference rules.
    """
    return list(ALL_RULES)


def get_rule_by_id(rule_id: str) -> Optional[InferenceRule]:
    """
    Retrieve a specific inference rule by its identifier.
    
    Supports:
    - Short IDs: 'R1', 'R2', 'R3', 'R4'
    - Full descriptive IDs: 'R1_CAPITAL_LOCATION', etc.
    - Case-insensitive lookups and trimmed whitespace.
    
    Returns:
        The matching InferenceRule, or None if no rule matches.
    """
    if not rule_id or not isinstance(rule_id, str):
        return None
    key = rule_id.strip().lower()
    return _RULE_LOOKUP.get(key)
