"""
Document Context and Pronoun Resolution Module
Provides deterministic reference resolution for first-person and third-person document statements.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly rule-based and deterministic.
No coreference neural models, no statistical NLP, no probabilistic guessing.
Uses explicit syntactic patterns, grammatical positions, and established document entities.
Preserves unresolved references rather than fabricating identities.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DocumentContext:
    """Tracks document-level contextual entities and speaker identity."""
    document_id: Optional[int] = None
    speaker_name: Optional[str] = None
    speaker_entity_id: Optional[str] = None
    established_entities: List[str] = field(default_factory=list)
    recent_gendered_entities: Dict[str, str] = field(default_factory=dict)  # 'male' -> Name, 'female' -> Name

    def set_speaker(self, name: str, entity_id: Optional[str] = None):
        """Establish the canonical speaker identity for this document."""
        clean_name = name.strip()
        if clean_name:
            self.speaker_name = clean_name
            self.speaker_entity_id = entity_id or f"person:{clean_name.lower().replace(' ', '_')}"
            if clean_name not in self.established_entities:
                self.established_entities.insert(0, clean_name)


class ContextResolver:
    """
    Deterministic rule-based resolver for document speaker identities and pronoun references.
    """

    # First-person speaker identification patterns
    SPEAKER_PATTERNS = [
        re.compile(r'^(?:my\s+name\s+is|i\s+am\s+called|i\s+go\s+by)\s+(?P<name>[A-Z][A-Za-z0-9_\s\.\-]+?)\.?$', re.IGNORECASE),
        re.compile(r'^i\s+am\s+(?P<name>[A-Z][A-Za-z0-9_\s\.\-]+?)\.?$', re.IGNORECASE),
    ]

    # First-person pronouns
    FIRST_PERSON_PRONOUNS = {'i', 'me', 'my', 'mine'}

    # Third-person pronouns
    THIRD_PERSON_MALE = {'he', 'him', 'his'}
    THIRD_PERSON_FEMALE = {'she', 'her', 'hers'}
    THIRD_PERSON_NEUTRAL = {'it', 'its', 'they', 'them', 'their'}

    # Common non-name words after "I am" to guard against misidentifying occupations as proper names
    NON_NAME_WORDS = {
        'a', 'an', 'the', 'studying', 'working', 'living', 'from', 'in', 'at', 'with', 'student',
        'engineer', 'developer', 'manager', 'professor', 'teacher', 'doctor', 'intern', 'happy', 'pleased'
    }

    def detect_speaker_identity(self, sentence: str) -> Optional[str]:
        """
        Check if a sentence explicitly establishes the document speaker's name.
        e.g., "My name is Honnesh." -> "Honnesh"
              "I am Honnesh." -> "Honnesh"
        """
        s = sentence.strip()
        for pat in self.SPEAKER_PATTERNS:
            m = pat.match(s)
            if m:
                cand = m.group('name').strip().rstrip('.!,')
                # Check that candidate is not an occupation phrase like "a student"
                words = cand.split()
                if words and words[0].lower() in self.NON_NAME_WORDS:
                    continue
                # Proper name should have at least one capitalized character
                if any(c.isupper() for c in cand):
                    # Clean trailing articles or common punctuation
                    cand = re.sub(r'^(?:The|A|An)\s+', '', cand, flags=re.IGNORECASE).strip()
                    if cand:
                        return cand
        return None

    def build_document_context(self, sentences: List[Tuple[int, str]], document_id: Optional[int] = None) -> DocumentContext:
        """
        Scan document sentences sequentially to establish document speaker and track entities.
        """
        ctx = DocumentContext(document_id=document_id)

        # Pass 1: Identify explicit speaker establishment
        for _, sent in sentences:
            speaker = self.detect_speaker_identity(sent)
            if speaker:
                ctx.set_speaker(speaker)
                break  # Earliest explicit speaker declaration binds document context

        return ctx

    def resolve_subject_reference(self, subject: str, context: Optional[DocumentContext]) -> Tuple[str, bool]:
        """
        Resolve a subject mention (including pronouns) using the document context.
        
        Returns:
            Tuple of (resolved_subject_name, is_resolved_from_context)
        """
        clean_subj = subject.strip()
        lower_subj = clean_subj.lower()

        # 1. First-person pronoun resolution ("I", "My name", etc.)
        if lower_subj in self.FIRST_PERSON_PRONOUNS:
            if context and context.speaker_name:
                return (context.speaker_name, True)
            # Default fallback when speaker name has not been explicitly declared
            return ("Document Speaker", True)

        if lower_subj.startswith("my "):
            # e.g., "my name", "my college"
            if context and context.speaker_name:
                return (context.speaker_name, True)
            return ("Document Speaker", True)

        # 2. Third-person pronoun resolution
        if lower_subj in self.THIRD_PERSON_MALE:
            if context and 'male' in context.recent_gendered_entities:
                return (context.recent_gendered_entities['male'], True)
            elif context and context.established_entities:
                return (context.established_entities[0], True)
            return (clean_subj, False)

        if lower_subj in self.THIRD_PERSON_FEMALE:
            if context and 'female' in context.recent_gendered_entities:
                return (context.recent_gendered_entities['female'], True)
            elif context and context.established_entities:
                return (context.established_entities[0], True)
            return (clean_subj, False)

        # Already a named entity or third-person subject
        if context and clean_subj not in context.established_entities:
            # Track newly encountered named entity
            if any(c.isupper() for c in clean_subj):
                context.established_entities.append(clean_subj)

        return (clean_subj, False)


GLOBAL_CONTEXT_RESOLVER = ContextResolver()
