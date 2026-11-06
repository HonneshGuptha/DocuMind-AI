"""
Structured Query Models for Question Processing
Defines formal representations for parsed deterministic questions and retrieved knowledge facts.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from modules.knowledge_base.models import Fact


@dataclass
class StructuredQuery:
    """
    Formal representation of a deterministically parsed question.
    Contains target entity, expected predicate, search constraints, and parsing status.
    """
    original_question: str
    question_type: str                          # e.g., 'capital', 'location', 'creator', etc.
    subject: Optional[str] = None               # Extracted subject (if known/queried)
    predicate: Optional[str] = None             # Extracted or mapped relation
    object: Optional[str] = None                # Extracted object (if known/queried)
    expected_answer_field: str = 'object'       # 'object', 'subject', or 'boolean'
    parsing_status: str = 'success'             # 'success', 'unsupported', 'error', 'empty'
    error_message: Optional[str] = None         # Reason if unsupported or error

    def to_dict(self) -> Dict[str, Any]:
        """Serialize query representation to dictionary."""
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        """Check if query was successfully parsed into a searchable specification."""
        return self.parsing_status == 'success' and (self.subject is not None or self.object is not None)


@dataclass
class RetrievalResult:
    """
    Structured outcome of querying the Knowledge Base using a StructuredQuery.
    Preserves all matching facts with full source sentence provenance.
    """
    query: StructuredQuery
    facts: List[Fact] = field(default_factory=list)
    status: str = 'success'                     # 'success', 'no_match', 'unsupported', 'error'
    total_matches: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        self.total_matches = len(self.facts)
        if self.status == 'success' and self.total_matches == 0:
            self.status = 'no_match'

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result and enclosed facts to a serializable dictionary."""
        return {
            'query': self.query.to_dict(),
            'facts': [f.to_dict() for f in self.facts],
            'status': self.status,
            'total_matches': self.total_matches,
            'error_message': self.error_message
        }
