"""
Deterministic Fact Extractor Module
Extracts Subject-Predicate-Object (SPO) triples from raw text using explicit,
rule-based regular expression patterns.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly rule-based and deterministic.
No NLP libraries (spaCy/NLTK), no pretrained language models, no transformers,
no embeddings, no semantic similarity, and no generative hallucination.
Unsupported sentences are strictly skipped without generating fabricated facts.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from .models import Fact


# Common English honorifics and abbreviations to protect during sentence splitting
ABBREVIATIONS = [
    (r'\bMr\.', '___MR_DOT___', 'Mr.'),
    (r'\bMrs\.', '___MRS_DOT___', 'Mrs.'),
    (r'\bMs\.', '___MS_DOT___', 'Ms.'),
    (r'\bDr\.', '___DR_DOT___', 'Dr.'),
    (r'\bProf\.', '___PROF_DOT___', 'Prof.'),
    (r'\be\.g\.', '___EG_DOT___', 'e.g.'),
    (r'\bi\.e\.', '___IE_DOT___', 'i.e.'),
    (r'\betc\.', '___ETC_DOT___', 'etc.'),
    (r'\bvs\.', '___VS_DOT___', 'vs.'),
    (r'\bU\.S\.', '___US_DOT___', 'U.S.'),
    (r'\bU\.K\.', '___UK_DOT___', 'U.K.'),
]


def split_sentences(text: str) -> List[Tuple[int, str]]:
    """
    Deterministically split raw text into sentences while tracking sentence indices.
    Protects common abbreviations and handles universal linebreaks.
    
    Returns:
        List of tuples: (sentence_index, clean_sentence)
    """
    if not text or not text.strip():
        return []

    processed = text
    # 1. Protect abbreviations
    for pattern, placeholder, _ in ABBREVIATIONS:
        processed = re.sub(pattern, placeholder, processed, flags=re.IGNORECASE)

    # 2. Split on sentence terminal punctuation (. ? !) followed by whitespace or linebreaks
    raw_sentences = re.split(r'(?<=[.!?])(?:\s+|\n+)|(?:\r?\n)+', processed)

    sentences = []
    sentence_idx = 0

    for s in raw_sentences:
        clean = s.strip()
        if not clean:
            continue
        # Restore abbreviation original text
        for _, placeholder, original in ABBREVIATIONS:
            clean = clean.replace(placeholder, original)

        sentences.append((sentence_idx, clean))
        sentence_idx += 1

    return sentences


class FactPattern:
    """Represents a deterministic extraction rule with documentation and regex."""

    def __init__(self, name: str, regex: str, default_predicate: Optional[str] = None, description: str = ""):
        self.name = name
        self.regex = re.compile(regex, re.IGNORECASE)
        self.default_predicate = default_predicate
        self.description = description

    def match(self, sentence: str) -> Optional[Tuple[str, str, str]]:
        """
        Attempt to match a sentence against this pattern.
        Returns (subject, predicate, object) or None if unsupported.
        """
        m = self.regex.match(sentence.strip())
        if not m:
            return None

        groups = m.groupdict()
        subject = groups.get('subject', '').strip()
        obj = groups.get('object', '').strip().rstrip('.!,')

        # Clean leading articles from subject if present
        cleaned_subject = re.sub(r'^(?:The|A|An)\s+', '', subject, flags=re.IGNORECASE).strip()
        if not cleaned_subject:
            cleaned_subject = subject

        # Determine predicate
        if self.default_predicate:
            predicate = self.default_predicate
        elif 'type' in groups:
            predicate = f"is a {groups['type'].strip().lower()} in"
        elif 'action' in groups:
            predicate = f"{groups['action'].strip().lower()} in"
        elif 'by_action' in groups:
            predicate = f"{groups['by_action'].strip().lower()} by"
        else:
            predicate = "related to"

        if not cleaned_subject or not obj:
            return None

        return (cleaned_subject, predicate, obj)


# ==============================================================================
# Canonical Deterministic Factual Patterns
# ==============================================================================
FACTUAL_PATTERNS = [
    # 1. Capital of: "X is the capital of Y."
    FactPattern(
        name="CAPITAL_OF",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+is\s+the\s+capital\s+of\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\.?$',
        default_predicate="capital of",
        description="Captures capital city relationships (e.g. Bangalore is the capital of Karnataka)"
    ),

    # 2. Located in / Situated in: "X is located in Y."
    FactPattern(
        name="LOCATED_IN",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+is\s+(?:located|situated)\s+in\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\.?$',
        default_predicate="located in",
        description="Captures geographic containment (e.g. Tokyo is located in Japan)"
    ),

    # 3. State / City / Country in: "X is a state in Y."
    FactPattern(
        name="REGION_IN",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+is\s+a\s+(?P<type>state|city|country|province|district|territory|continent)\s+in\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\.?$',
        default_predicate=None,  # Dynamic: "is a {type} in"
        description="Captures administrative subdivisions (e.g. Karnataka is a state in India)"
    ),

    # 4. Established / Founded in: "X was established in YEAR."
    FactPattern(
        name="ESTABLISHED_IN",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+was\s+(?P<action>established|founded|created|built)\s+in\s+(?P<object>\d{1,4}(?:\s*(?:BC|BCE|AD|CE))?|[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\.?$',
        default_predicate=None,  # Dynamic: "{action} in"
        description="Captures founding year or institution inception (e.g. Harvard University was founded in 1636)"
    ),

    # 5. Invented / Discovered / Developed by: "X was invented by Y."
    FactPattern(
        name="CREATED_BY",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+was\s+(?P<by_action>invented|discovered|developed|authored|written)\s+by\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\.?$',
        default_predicate=None,  # Dynamic: "{by_action} by"
        description="Captures creator or discoverer relationships (e.g. Python was developed by Guido van Rossum)"
    ),

    # 6. Part of / Component of: "X is part of Y."
    FactPattern(
        name="PART_OF",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+is\s+(?:a\s+)?part\s+of\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\.?$',
        default_predicate="part of",
        description="Captures meronymic part-whole relations (e.g. Engine is part of Car)"
    ),

    # 7. Borders / Neighbors: "X borders Y."
    FactPattern(
        name="BORDERS",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+(?:borders|neighbors)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\.?$',
        default_predicate="borders",
        description="Captures geographic adjacency (e.g. Canada borders United States)"
    ),

    # 8. Produces / Manufactures: "X produces Y."
    FactPattern(
        name="PRODUCES",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+(?:produces|manufactures)\s+(?P<object>[a-zA-Z0-9\s\.\-]*?)\.?$',
        default_predicate="produces",
        description="Captures production relationships (e.g. Brazil produces coffee)"
    ),

    # 9. Is a / Is an (Classification/Type): "X is a TYPE."
    FactPattern(
        name="IS_A",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-]*?)\s+is\s+(?:a|an)\s+(?P<object>[a-zA-Z0-9\s\.\-]*?)\.?$',
        default_predicate="is a",
        description="Captures taxonomic / class membership (e.g. Python is a programming language)"
    ),
]


class FactExtractor:
    """
    Deterministic rule-based extractor that scans raw text, splits it into
    sentences, tests each sentence against supported patterns, and produces Fact instances.
    """

    def __init__(self, patterns: Optional[List[FactPattern]] = None):
        self.patterns = patterns or FACTUAL_PATTERNS

    def extract_facts(self, raw_text: str, document_id: int) -> List[Fact]:
        """
        Extract structured SPO facts from raw text for a specific document.
        
        Args:
            raw_text: Raw character text extracted from document.
            document_id: Foreign key reference to documents table.
            
        Returns:
            List of Fact instances matching supported deterministic patterns.
        """
        if not raw_text or not raw_text.strip():
            return []

        sentences = split_sentences(raw_text)
        facts: List[Fact] = []
        seen_triples = set()

        for sent_idx, sentence in sentences:
            # Test each pattern deterministically
            matched_triple = None
            for pattern in self.patterns:
                match_result = pattern.match(sentence)
                if match_result:
                    matched_triple = match_result
                    break

            # If unsupported sentence, strictly skip
            if not matched_triple:
                continue

            subj, pred, obj = matched_triple
            triple_key = (subj.lower(), pred.lower(), obj.lower())

            # Avoid duplicate facts from the same sentence/document
            if triple_key in seen_triples:
                continue

            seen_triples.add(triple_key)
            facts.append(
                Fact(
                    document_id=document_id,
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    source_sentence=sentence,
                    sentence_index=sent_idx
                )
            )

        return facts
