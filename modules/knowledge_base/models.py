"""
Knowledge Base Models
Defines structured data classes for Document records and Subject-Predicate-Object (SPO) facts.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class DocumentRecord:
    """Represents a document tracked in the SQLite knowledge base."""
    filename: str
    file_format: str
    extraction_status: str
    document_id: Optional[int] = None
    upload_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert document record to dictionary."""
        return asdict(self)


@dataclass
class Fact:
    """
    Represents a structured factual assertion extracted from text.
    Modeled as a Subject-Predicate-Object (SPO) triple with source sentence attribution.
    """
    document_id: int
    subject: str
    predicate: str
    object: str
    source_sentence: str
    sentence_index: int
    fact_id: Optional[int] = None
    canonical_predicate: Optional[str] = None
    subject_entity_id: Optional[str] = None
    object_entity_id: Optional[str] = None
    fact_status: str = "asserted"

    def to_dict(self) -> Dict[str, Any]:
        """Convert fact to dictionary."""
        d = asdict(self)
        if not self.canonical_predicate:
            d['canonical_predicate'] = self.predicate
        return d

    @property
    def triple(self) -> tuple:
        """Return the (subject, predicate, object) tuple representation."""
        return (self.subject, self.predicate, self.object)

    def __repr__(self) -> str:
        return (f"<Fact #{self.fact_id or '?'} Doc:{self.document_id} "
                f"({self.subject} -> {self.predicate} -> {self.object})>")
