"""
Answer Models Module
Defines structured response models for deterministic answer generation in DocuMind AI.

ACADEMIC EXPLANATION:
In Symbolic Artificial Intelligence, Answer Generation is the presentation phase
where formal query results (retrieved ground facts and derived inference chains)
are rendered into human-understandable natural language using predefined syntactic templates.

Every answer preserves:
1. Exact answer status ('answered', 'unknown', 'unsupported', 'error').
2. Deduction type ('direct', 'inferred', 'mixed', 'none').
3. Supporting facts, source sentences, and document references.
4. Explanations of any applied inference rules without probabilistic confidence scores.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union


@dataclass
class AnswerResponse:
    """
    Standardized, structured response produced by the Answer Generator.
    
    Attributes:
        original_question: Verbatim question string submitted by the user.
        answer_text: Human-readable answer synthesized from deterministic templates.
        answer_status: 'answered', 'unknown', 'unsupported', or 'error'.
        answer_type: 'direct' (ground KB fact), 'inferred' (deductive reasoning),
                     'mixed' (both direct & inferred), or 'none'.
        supporting_facts: Serialized dictionary representations of matching facts.
        supporting_fact_ids: List of database fact IDs or inferred fact IDs.
        applied_rule_ids: List of inference rules fired (e.g. ['R1', 'R2']).
        explanation: Clear provenance audit explaining how the answer was established.
        confidence: Evidence classification ('verified' for ground facts,
                    'deduced' for valid inferences, 'none' for unknown/unsupported).
        sources: List of source document IDs and sentences providing evidence.
    """
    original_question: str
    answer_text: str
    answer_status: str = 'answered'                         # 'answered', 'unknown', 'unsupported', 'error'
    answer_type: str = 'direct'                             # 'direct', 'inferred', 'mixed', 'none'
    supporting_facts: List[Dict[str, Any]] = field(default_factory=list)
    supporting_fact_ids: List[Union[int, str]] = field(default_factory=list)
    applied_rule_ids: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    confidence: str = 'verified'                            # 'verified', 'deduced', 'none'
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert answer response to a serializable dictionary."""
        return {
            'original_question': self.original_question,
            'answer_text': self.answer_text,
            'answer_status': self.answer_status,
            'answer_type': self.answer_type,
            'supporting_facts': list(self.supporting_facts),
            'supporting_fact_ids': list(self.supporting_fact_ids),
            'applied_rule_ids': list(self.applied_rule_ids),
            'explanation': self.explanation,
            'confidence': self.confidence,
            'sources': list(self.sources)
        }

    def __repr__(self) -> str:
        return (f"<AnswerResponse status='{self.answer_status}' type='{self.answer_type}' "
                f"text='{self.answer_text[:50]}...'>")
