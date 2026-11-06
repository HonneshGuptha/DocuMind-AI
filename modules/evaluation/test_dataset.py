"""
Test Dataset Module
Provides a deterministic, curated evaluation benchmark dataset and controlled
in-memory test knowledge base for DocuMind AI.

ACADEMIC EXPLANATION:
In Symbolic AI evaluation, the test dataset must serve as a rigorous gold-standard
ground truth benchmark covering:
1. All nine canonical question categories (Capital, Location, Classification,
   Establishment, Creator, Part-Whole, Border, Production, Identity).
2. Multiple reasoning modalities (Direct lookup, Single-hop inference, Multi-hop chaining).
3. Boundary & epistemic edge cases (Open-World unknown, Syntax-unsupported, Multi-valued results, Conflicting evidence).
"""

from typing import List, Tuple, Dict, Any
from .evaluation_models import EvaluationCase
from modules.knowledge_base.models import Fact as KBF


# ==============================================================================
# 1. Controlled Benchmark Knowledge Base Facts
# ==============================================================================

BENCHMARK_KB_FACTS: List[KBF] = [
    # Document 1: Geography - India
    KBF(fact_id=1, document_id=1, subject="Bangalore", predicate="capital of", object="Karnataka",
        source_sentence="Bangalore is the capital of Karnataka.", sentence_index=0),
    KBF(fact_id=2, document_id=1, subject="Karnataka", predicate="is a state in", object="India",
        source_sentence="Karnataka is a state in India.", sentence_index=1),
    KBF(fact_id=3, document_id=1, subject="India", predicate="located in", object="Asia",
        source_sentence="India is located in Asia.", sentence_index=2),
    KBF(fact_id=4, document_id=1, subject="Karnataka", predicate="located in", object="India",
        source_sentence="Karnataka is located in India.", sentence_index=3),

    # Document 2: Geography - East Asia
    KBF(fact_id=5, document_id=2, subject="Tokyo", predicate="located in", object="Japan",
        source_sentence="Tokyo is located in Japan.", sentence_index=0),

    # Document 3: Academia
    KBF(fact_id=6, document_id=3, subject="Stanford University", predicate="established in", object="1885",
        source_sentence="Stanford University was established in 1885.", sentence_index=0),

    # Document 4: Computer Science
    KBF(fact_id=7, document_id=4, subject="Python", predicate="developed by", object="Guido van Rossum",
        source_sentence="Python was developed by Guido van Rossum.", sentence_index=0),
    KBF(fact_id=8, document_id=4, subject="Guido van Rossum", predicate="is a", object="Computer Programmer",
        source_sentence="Guido van Rossum is a Computer Programmer.", sentence_index=1),
    KBF(fact_id=9, document_id=4, subject="Python", predicate="is a", object="programming language",
        source_sentence="Python is a programming language.", sentence_index=2),

    # Document 5: Mechanical Engineering
    KBF(fact_id=10, document_id=5, subject="Piston", predicate="part of", object="Engine",
        source_sentence="Piston is part of Engine.", sentence_index=0),
    KBF(fact_id=11, document_id=5, subject="Engine", predicate="part of", object="Gearbox",
        source_sentence="Engine is part of Gearbox.", sentence_index=1),
    KBF(fact_id=12, document_id=5, subject="Gearbox", predicate="part of", object="Automobile",
        source_sentence="Gearbox is part of Automobile.", sentence_index=2),

    # Document 6: Geography - Borders (Multi-result)
    KBF(fact_id=13, document_id=6, subject="Canada", predicate="borders", object="United States",
        source_sentence="Canada borders United States.", sentence_index=0),
    KBF(fact_id=14, document_id=6, subject="Canada", predicate="borders", object="Greenland",
        source_sentence="Canada borders Greenland.", sentence_index=1),

    # Document 7: Agriculture
    KBF(fact_id=15, document_id=7, subject="Karnataka", predicate="produces", object="Coffee",
        source_sentence="Karnataka produces Coffee.", sentence_index=0),

    # Document 8: Conflicting Founding Years (Historical discrepancy test)
    KBF(fact_id=16, document_id=8, subject="C", predicate="established in", object="1972",
        source_sentence="C was established in 1972.", sentence_index=0),
    KBF(fact_id=17, document_id=8, subject="C", predicate="established in", object="1970",
        source_sentence="C was established in 1970.", sentence_index=1),
]


# ==============================================================================
# 2. Comprehensive Evaluation Benchmark Dataset
# ==============================================================================

BENCHMARK_DATASET: List[EvaluationCase] = [
    # --------------------------------------------------------------------------
    # Category 1: Capital (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_CAPITAL_DIRECT_01",
        question="What is the capital of Karnataka?",
        category="capital",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="The capital of Karnataka is Bangalore.",
        expected_facts=[("Bangalore", "capital of", "Karnataka")],
        expected_fact_ids=[1],
        expected_rule_ids=[],
        description="Verify direct subject-answer retrieval for capital city inquiry."
    ),

    # --------------------------------------------------------------------------
    # Category 2: Location (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_LOCATION_DIRECT_02",
        question="Where is Tokyo located?",
        category="location",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Tokyo is located in Japan.",
        expected_facts=[("Tokyo", "located in", "Japan")],
        expected_fact_ids=[5],
        expected_rule_ids=[],
        description="Verify direct location inquiry for geographic entity."
    ),

    # --------------------------------------------------------------------------
    # Category 3: Establishment (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_ESTABLISHMENT_DIRECT_03",
        question="When was Stanford University established?",
        category="establishment",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Stanford University was established in 1885.",
        expected_facts=[("Stanford University", "established in", "1885")],
        expected_fact_ids=[6],
        expected_rule_ids=[],
        description="Verify temporal establishment retrieval."
    ),

    # --------------------------------------------------------------------------
    # Category 4: Creator / Author (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_CREATOR_DIRECT_04",
        question="Who developed Python?",
        category="creator",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Python was developed by Guido van Rossum.",
        expected_facts=[("Python", "developed by", "Guido van Rossum")],
        expected_fact_ids=[7],
        expected_rule_ids=[],
        description="Verify developer/creator entity extraction."
    ),

    # --------------------------------------------------------------------------
    # Category 5: Part-Whole (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_PART_WHOLE_DIRECT_05",
        question="What is part of Engine?",
        category="part_whole",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Piston is part of Engine.",
        expected_facts=[("Piston", "part of", "Engine")],
        expected_fact_ids=[10],
        expected_rule_ids=[],
        description="Verify component part retrieval."
    ),

    # --------------------------------------------------------------------------
    # Category 6: Border (Multi-Result)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_BORDER_MULTI_06",
        question="What does Canada border?",
        category="border",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Canada borders: United States, Greenland.",
        expected_facts=[
            ("Canada", "borders", "United States"),
            ("Canada", "borders", "Greenland")
        ],
        expected_fact_ids=[13, 14],
        expected_rule_ids=[],
        description="Verify multiple-result listing for border relations.",
        is_multiple=True
    ),

    # --------------------------------------------------------------------------
    # Category 7: Production (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_PRODUCTION_DIRECT_07",
        question="What does Karnataka produce?",
        category="production",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Karnataka produces Coffee.",
        expected_facts=[("Karnataka", "produces", "Coffee")],
        expected_fact_ids=[15],
        expected_rule_ids=[],
        description="Verify production/goods retrieval."
    ),

    # --------------------------------------------------------------------------
    # Category 8: Identity (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_IDENTITY_DIRECT_08",
        question="Who is Guido van Rossum?",
        category="identity",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Guido van Rossum is a Computer Programmer.",
        expected_facts=[("Guido van Rossum", "is a", "Computer Programmer")],
        expected_fact_ids=[8],
        expected_rule_ids=[],
        description="Verify identity relation pattern priority."
    ),

    # --------------------------------------------------------------------------
    # Category 9: Classification (Direct)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_CLASSIFICATION_DIRECT_09",
        question="What is Python?",
        category="classification",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Python is a programming language.",
        expected_facts=[("Python", "is a", "programming language")],
        expected_fact_ids=[9],
        expected_rule_ids=[],
        description="Verify taxonomic classification retrieval."
    ),

    # --------------------------------------------------------------------------
    # Inferred Case 1: Rule R1 (Capital Location Rule)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_INFERENCE_R1_10",
        question="Where is Bangalore located?",
        category="location",
        expected_answer_status="answered",
        expected_answer_type="inferred",
        expected_answer_text="Bangalore is located in India.",
        expected_facts=[("Bangalore", "located in", "India")],
        expected_fact_ids=[1, 2],
        expected_rule_ids=["R1"],
        description="Verify deduction via R1: Bangalore capital of Karnataka AND Karnataka state in India -> Bangalore located in India."
    ),

    # --------------------------------------------------------------------------
    # Inferred Case 2: Rule R2 (Location Transitivity)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_INFERENCE_R2_11",
        question="Where is Karnataka located?",
        category="location",
        expected_answer_status="answered",
        expected_answer_type="inferred",
        expected_answer_text="Karnataka is located in India.",
        expected_facts=[("Karnataka", "located in", "Asia")],
        expected_fact_ids=[4, 3],
        expected_rule_ids=["R2"],
        description="Verify deduction via R2: Karnataka located in India AND India located in Asia -> Karnataka located in Asia."
    ),

    # --------------------------------------------------------------------------
    # Inferred Case 3: Rule R3 (Part-Whole Transitivity)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_INFERENCE_R3_12",
        question="Is Piston a part of Gearbox?",
        category="part_whole",
        expected_answer_status="answered",
        expected_answer_type="inferred",
        expected_answer_text="Yes, Piston is part of Gearbox.",
        expected_facts=[("Piston", "part of", "Gearbox")],
        expected_fact_ids=[10, 11],
        expected_rule_ids=["R3"],
        description="Verify deduction via R3: Piston part of Engine AND Engine part of Gearbox -> Piston part of Gearbox.",
        is_boolean=True
    ),

    # --------------------------------------------------------------------------
    # Inferred Case 4: Rule R4 (Border Symmetry)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_INFERENCE_R4_13",
        question="Does United States border Canada?",
        category="border",
        expected_answer_status="answered",
        expected_answer_type="inferred",
        expected_answer_text="Yes, United States borders Canada.",
        expected_facts=[("United States", "borders", "Canada")],
        expected_fact_ids=[13],
        expected_rule_ids=["R4"],
        description="Verify deduction via R4: Canada borders United States -> United States borders Canada.",
        is_boolean=True
    ),

    # --------------------------------------------------------------------------
    # Inferred Case 5: Multi-Step Chaining (Recursive R3 Transitivity)
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_INFERENCE_MULTISTEP_14",
        question="Is Piston a part of Automobile?",
        category="part_whole",
        expected_answer_status="answered",
        expected_answer_type="inferred",
        expected_answer_text="Yes, Piston is part of Automobile.",
        expected_facts=[("Piston", "part of", "Automobile")],
        expected_fact_ids=[10, 11, 12],
        expected_rule_ids=["R3"],
        description="Verify multi-step recursive proof chaining: R3 derives Piston part of Gearbox, then R3 derives Piston part of Automobile.",
        is_boolean=True
    ),

    # --------------------------------------------------------------------------
    # Boolean Case: Direct Affirmative
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_BOOLEAN_AFFIRMATIVE_15",
        question="Does Canada border United States?",
        category="border",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text="Yes, Canada borders United States.",
        expected_facts=[("Canada", "borders", "United States")],
        expected_fact_ids=[13],
        expected_rule_ids=[],
        description="Verify affirmative boolean confirmation directly from ground fact.",
        is_boolean=True
    ),

    # --------------------------------------------------------------------------
    # Boolean Case: Unknown under Open-World Assumption
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_BOOLEAN_UNKNOWN_16",
        question="Does Canada border Brazil?",
        category="border",
        expected_answer_status="unknown",
        expected_answer_type="none",
        expected_answer_text="I could not find sufficient information in the knowledge base to answer this question.",
        expected_facts=[],
        expected_fact_ids=[],
        expected_rule_ids=[],
        description="Verify OWA compliance: unrecorded boolean relation yields 'unknown', never 'No'.",
        is_boolean=True
    ),

    # --------------------------------------------------------------------------
    # Unknown Case: Wh- Question with Unrecorded Knowledge
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_UNKNOWN_WH_17",
        question="What is the capital of Atlantis?",
        category="capital",
        expected_answer_status="unknown",
        expected_answer_type="none",
        expected_answer_text="I could not find sufficient information in the knowledge base to answer this question.",
        expected_facts=[],
        expected_fact_ids=[],
        expected_rule_ids=[],
        description="Verify unrecorded wh-question returns unknown without hallucination."
    ),

    # --------------------------------------------------------------------------
    # Missing Evidence: Creator not in KB
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_UNKNOWN_CREATOR_18",
        question="Who invented Telescope?",
        category="creator",
        expected_answer_status="unknown",
        expected_answer_type="none",
        expected_answer_text="I could not find sufficient information in the knowledge base to answer this question.",
        expected_facts=[],
        expected_fact_ids=[],
        expected_rule_ids=[],
        description="Verify missing evidence question reports unknown."
    ),

    # --------------------------------------------------------------------------
    # Unsupported Case 1: Open-Domain Philosophical Query
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_UNSUPPORTED_01_19",
        question="Why is the sky blue?",
        category="unsupported",
        expected_answer_status="unsupported",
        expected_answer_type="none",
        expected_answer_text="This question format is not currently supported by DocuMind AI.",
        expected_facts=[],
        expected_fact_ids=[],
        expected_rule_ids=[],
        description="Verify out-of-domain open-world question is safely rejected as unsupported."
    ),

    # --------------------------------------------------------------------------
    # Unsupported Case 2: Conversational Query
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_UNSUPPORTED_02_20",
        question="Can you tell me a funny joke?",
        category="unsupported",
        expected_answer_status="unsupported",
        expected_answer_type="none",
        expected_answer_text="This question format is not currently supported by DocuMind AI.",
        expected_facts=[],
        expected_fact_ids=[],
        expected_rule_ids=[],
        description="Verify conversational command is safely identified as unsupported."
    ),

    # --------------------------------------------------------------------------
    # Conflicting Facts Case: Disputed Founding Year
    # --------------------------------------------------------------------------
    EvaluationCase(
        case_id="TC_CONFLICT_FACTS_21",
        question="When was C established?",
        category="establishment",
        expected_answer_status="answered",
        expected_answer_type="direct",
        expected_answer_text=None,  # Evaluated by conflict indicator
        expected_facts=[
            ("C", "established in", "1972"),
            ("C", "established in", "1970")
        ],
        expected_fact_ids=[16, 17],
        expected_rule_ids=[],
        description="Verify contradictory ground facts are disclosed with both Fact IDs rather than arbitrarily selecting one.",
        is_conflicting=True
    ),
]


def get_benchmark_dataset() -> List[EvaluationCase]:
    """Retrieve a fresh copy of the canonical evaluation benchmark dataset."""
    return list(BENCHMARK_DATASET)


def get_benchmark_facts() -> List[KBF]:
    """Retrieve a fresh copy of the benchmark knowledge base facts."""
    return list(BENCHMARK_KB_FACTS)
