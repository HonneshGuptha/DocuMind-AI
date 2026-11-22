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

# Geographic Multi-Hop Containment Rules
R5_CITY_LOCATION = InferenceRule(
    rule_id="R5",
    rule_name="City Location Rule",
    premises=[
        ("X", "is a city in", "Y"),
    ],
    conclusion=("X", "located in", "Y"),
    description="If city X is in region Y, then X is located in Y."
)

R6_STATE_LOCATION = InferenceRule(
    rule_id="R6",
    rule_name="State Location Rule",
    premises=[
        ("X", "is a state in", "Y"),
    ],
    conclusion=("X", "located in", "Y"),
    description="If state X is in country Y, then X is located in Y."
)

R7_COUNTRY_LOCATION = InferenceRule(
    rule_id="R7",
    rule_name="Country Location Rule",
    premises=[
        ("X", "is a country in", "Y"),
    ],
    conclusion=("X", "located in", "Y"),
    description="If country X is in continent Y, then X is located in Y."
)

R8_CITY_STATE_LOCATION = InferenceRule(
    rule_id="R8",
    rule_name="City-State Location Rule",
    premises=[
        ("X", "is a city in", "Y"),
        ("Y", "is a state in", "Z"),
    ],
    conclusion=("X", "located in", "Z"),
    description="If city X is in state Y, and state Y is in country Z, then X is located in country Z."
)

R9_STATE_CONTINENT_LOCATION = InferenceRule(
    rule_id="R9",
    rule_name="State-Continent Location Rule",
    premises=[
        ("X", "is a state in", "Y"),
        ("Y", "located in", "Z"),
    ],
    conclusion=("X", "located in", "Z"),
    description="If state X is in country Y, and country Y is located in continent Z, then state X is located in continent Z."
)

R10_CITY_CONTINENT_LOCATION = InferenceRule(
    rule_id="R10",
    rule_name="City-Continent Location Rule",
    premises=[
        ("X", "is a city in", "Y"),
        ("Y", "is a state in", "Z"),
        ("Z", "located in", "W"),
    ],
    conclusion=("X", "located in", "W"),
    description="If city X is in state Y, state Y is in country Z, and country Z is located in continent W, then city X is located in continent W."
)

# Canonical aliases
R1 = R1_CAPITAL_LOCATION
R2 = R2_LOCATION_TRANSITIVITY
R3 = R3_PART_WHOLE_TRANSITIVITY
R4 = R4_BORDER_SYMMETRY
R5 = R5_CITY_LOCATION
R6 = R6_STATE_LOCATION
R7 = R7_COUNTRY_LOCATION
R8 = R8_CITY_STATE_LOCATION
R9 = R9_STATE_CONTINENT_LOCATION
R10 = R10_CITY_CONTINENT_LOCATION

# Canonical rules (Module 4 core baseline)
CANONICAL_RULES: List[InferenceRule] = [
    R1_CAPITAL_LOCATION,
    R2_LOCATION_TRANSITIVITY,
    R3_PART_WHOLE_TRANSITIVITY,
    R4_BORDER_SYMMETRY,
]

# Master list of all available rules (including extended multi-hop rules)
ALL_RULES: List[InferenceRule] = [
    R1_CAPITAL_LOCATION,
    R2_LOCATION_TRANSITIVITY,
    R3_PART_WHOLE_TRANSITIVITY,
    R4_BORDER_SYMMETRY,
    R5_CITY_LOCATION,
    R6_STATE_LOCATION,
    R7_COUNTRY_LOCATION,
    R8_CITY_STATE_LOCATION,
    R9_STATE_CONTINENT_LOCATION,
    R10_CITY_CONTINENT_LOCATION,
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
    "r5": R5_CITY_LOCATION,
    "r5_city_location": R5_CITY_LOCATION,
    "city_location": R5_CITY_LOCATION,
    "r6": R6_STATE_LOCATION,
    "r6_state_location": R6_STATE_LOCATION,
    "state_location": R6_STATE_LOCATION,
    "r7": R7_COUNTRY_LOCATION,
    "r7_country_location": R7_COUNTRY_LOCATION,
    "country_location": R7_COUNTRY_LOCATION,
    "r8": R8_CITY_STATE_LOCATION,
    "r8_city_state_location": R8_CITY_STATE_LOCATION,
    "city_state_location": R8_CITY_STATE_LOCATION,
    "r9": R9_STATE_CONTINENT_LOCATION,
    "r9_state_continent_location": R9_STATE_CONTINENT_LOCATION,
    "state_continent_location": R9_STATE_CONTINENT_LOCATION,
    "r10": R10_CITY_CONTINENT_LOCATION,
    "r10_city_continent_location": R10_CITY_CONTINENT_LOCATION,
    "city_continent_location": R10_CITY_CONTINENT_LOCATION,
}


# ==============================================================================
# Rule Retrieval Functions
# ==============================================================================

def get_all_rules(include_extended: bool = False) -> List[InferenceRule]:
    """
    Retrieve deterministic inference rules defined in the Rule Base.
    By default, returns the 4 canonical inference rules (R1-R4) to preserve
    Module 4 specifications and evaluation baselines.
    Pass include_extended=True to include multi-hop geographic rules (R5-R10).
    """
    if include_extended:
        return list(ALL_RULES)
    return list(CANONICAL_RULES)


def get_extended_rules() -> List[InferenceRule]:
    """Retrieve all rules including extended multi-hop inference rules (R1-R10)."""
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
