"""
Deterministic Fact Extractor Module
Extracts Subject-Predicate-Object (SPO) triples from raw text using an extensible,
rule-based pattern and relationship registry.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly rule-based and deterministic.
No NLP libraries (spaCy/NLTK), no pretrained language models, no transformers,
no embeddings, no semantic similarity, and no generative hallucination.
Unsupported sentences are strictly skipped without generating fabricated facts.
"""

import re
from typing import List, Tuple, Optional, Dict, Any, Set
from .models import Fact
from .relationship_registry import GLOBAL_RELATIONSHIP_REGISTRY, RelationshipRegistry
from .entity_registry import GLOBAL_ENTITY_REGISTRY, EntityRegistry
from .context_resolver import GLOBAL_CONTEXT_RESOLVER, ContextResolver, DocumentContext


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

# Connecting words that indicate intra-sentence line continuation in wrapped documents (e.g., PyMuPDF output)
CONNECTING_WORDS: Set[str] = {
    'in', 'at', 'on', 'of', 'to', 'for', 'with', 'by', 'from',
    'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'i'
}

# Known initial words of multi-word named entities frequently split across lines
MULTIWORD_FIRST_TOKENS: Set[str] = {
    'tamil', 'new', 'national', 'tech', 'research', 'city', 'abc'
}


def normalize_document_linebreaks(text: str) -> str:
    """
    Normalizes intra-sentence visual line breaks produced by document extractors (such as PyMuPDF)
    while preserving genuine sentence boundaries, paragraph breaks, and numbered section headers.

    Deterministic heuristic:
    1. Unifies CRLF/CR to LF.
    2. Identifies line continuations:
       - If current line ends with a hyphen (word split across lines), merge without space.
       - If current line does not end with terminal punctuation (. ! ?):
         * Next line starts with a lowercase letter.
         * Next line starts with standalone pronoun 'I' followed by a verb (e.g., 'I work').
         * Current line ends with a connecting word (preposition, conjunction, auxiliary).
         * Current line ends with a known multi-word entity token (e.g. 'Tamil', 'New').
       Merge these continuations with a single space.
    3. Preserves terminal punctuation boundaries, blank line paragraph separators, and numbered headers.
    """
    if not text:
        return ""

    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = normalized.split('\n')
    merged_lines: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            merged_lines.append("")
            i += 1
            continue

        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line:
                break

            # Preserve explicit numbered section headers e.g. "1. Personal and Academic Profile"
            if re.match(r'^\d+\.\s+[A-Z]', next_line):
                break

            line_tokens = line.split()
            last_token = line_tokens[-1].lower().rstrip('.,;:!?') if line_tokens else ''
            first_next_char = next_line[0] if next_line else ''

            is_continuation = False

            # Case A: Current line ends with hyphen
            if line.endswith('-'):
                line = line[:-1] + next_line
                i += 1
                continue

            # Case B: Current line does NOT end with sentence terminal punctuation (. ! ?)
            if not line.endswith(('.', '!', '?')):
                if first_next_char.islower():
                    is_continuation = True
                elif last_token in CONNECTING_WORDS:
                    is_continuation = True
                elif last_token in MULTIWORD_FIRST_TOKENS:
                    is_continuation = True

            if is_continuation:
                line = line + " " + next_line
                i += 1
            else:
                break

        merged_lines.append(line)
        i += 1

    return '\n'.join(merged_lines)


def split_sentences(text: str) -> List[Tuple[int, str]]:
    """
    Deterministically split raw text into sentences while tracking sentence indices.
    Normalizes intra-sentence visual line breaks, protects common abbreviations and
    numbered section headings, and splits on genuine sentence terminals.
    
    Returns:
        List of tuples: (sentence_index, clean_sentence)
    """
    if not text or not text.strip():
        return []

    # 1. Normalize linebreaks from visual wrapping
    processed = normalize_document_linebreaks(text)

    # 2. Protect abbreviations
    for pattern, placeholder, _ in ABBREVIATIONS:
        processed = re.sub(pattern, placeholder, processed, flags=re.IGNORECASE)

    # 3. Protect numbered headers at start of lines e.g. "1. Personal and Academic Profile"
    processed = re.sub(r'(?m)^(\d+)\.\s+', r'\1___DOT_NUM___ ', processed)

    # 4. Split on sentence terminal punctuation (. ? !) followed by whitespace or linebreaks
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

        # Restore numbered headers
        clean = clean.replace('___DOT_NUM___', '.')

        sentences.append((sentence_idx, clean))
        sentence_idx += 1

    return sentences


class FactPattern:
    """Represents a deterministic extraction rule with documentation and regex."""

    def __init__(self,
                 name: str,
                 regex: str,
                 default_predicate: Optional[str] = None,
                 description: str = "",
                 is_speaker_declaration: bool = False):
        self.name = name
        self.regex = re.compile(regex, re.IGNORECASE)
        self.default_predicate = default_predicate
        self.description = description
        self.is_speaker_declaration = is_speaker_declaration

    def match(self, sentence: str) -> Optional[Tuple[str, str, str]]:
        """
        Attempt to match a sentence against this pattern.
        Returns (subject, predicate, object) or None if unsupported.
        """
        m = self.regex.match(sentence.strip())
        if not m:
            return None

        groups = m.groupdict()
        raw_subject = (groups.get('subject') or '').strip()
        obj = (groups.get('object') or '').strip().rstrip('.!,')

        # Clean leading articles from subject if present (except pronouns like 'I')
        if raw_subject and raw_subject.lower() != 'i':
            cleaned_subject = re.sub(r'^(?:The|A|An)\s+', '', raw_subject, flags=re.IGNORECASE).strip()
            if not cleaned_subject:
                cleaned_subject = raw_subject
        else:
            cleaned_subject = raw_subject

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

        # Handle special speaker intro pattern without explicit subject group
        if self.is_speaker_declaration and not cleaned_subject and obj:
            cleaned_subject = obj  # Subject is the speaker's name, e.g. (Honnesh, has name, Honnesh)

        if not cleaned_subject or not obj:
            return None

        # Guard against incomplete trailing preposition fragments (e.g., from unmerged visual line breaks)
        trailing_preps = {'in', 'at', 'on', 'of', 'to', 'for', 'with', 'by', 'from'}
        obj_tokens = obj.lower().split()
        if obj_tokens and obj_tokens[-1] in trailing_preps:
            return None

        return (cleaned_subject, predicate, obj)


# ==============================================================================
# Canonical Deterministic Factual Patterns Registry
# ==============================================================================
FACTUAL_PATTERNS = [
    # 1. Capital of: "X is the capital of Y."
    FactPattern(
        name="CAPITAL_OF",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+is\s+the\s+capital\s+of\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="capital of",
        description="Captures capital city relationships (e.g. Bangalore is the capital of Karnataka)"
    ),

    # 2. Located in / Situated in / Is in: "X is located in Y." or "X is in Y."
    FactPattern(
        name="LOCATED_IN",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+is\s+(?:located\s+in|situated\s+in|in)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="located in",
        description="Captures geographic containment or presence (e.g. Tokyo is located in Japan / Ayyappa is in Sai's dream)"
    ),

    # 3. State / City / Country in: "X is a state in Y."
    FactPattern(
        name="REGION_IN",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+is\s+a\s+(?P<type>state|city|country|province|district|territory|continent)\s+in\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate=None,  # Dynamic: "is a {type} in"
        description="Captures administrative subdivisions (e.g. Karnataka is a state in India)"
    ),

    # 4. Established / Founded in: "X was established in YEAR."
    FactPattern(
        name="ESTABLISHED_IN",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+was\s+(?P<action>established|founded|created|built)\s+in\s+(?P<object>\d{1,4}(?:\s*(?:BC|BCE|AD|CE))?|[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate=None,  # Dynamic: "{action} in"
        description="Captures founding year or institution inception (e.g. Harvard University was founded in 1636)"
    ),

    # 5. Invented / Discovered / Developed by: "X was invented by Y."
    FactPattern(
        name="CREATED_BY",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+was\s+(?P<by_action>invented|discovered|developed|authored|written)\s+by\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate=None,  # Dynamic: "{by_action} by"
        description="Captures creator or discoverer relationships (e.g. Python was developed by Guido van Rossum)"
    ),

    # 6. Part of / Component of: "X is part of Y."
    FactPattern(
        name="PART_OF",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+is\s+(?:a\s+)?part\s+of\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="part of",
        description="Captures meronymic part-whole relations (e.g. Engine is part of Car)"
    ),

    # 7. Borders / Neighbors: "X borders Y."
    FactPattern(
        name="BORDERS",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+(?:borders|neighbors)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="borders",
        description="Captures geographic adjacency (e.g. Canada borders United States)"
    ),

    # 8. Produces / Manufactures: "X produces Y."
    FactPattern(
        name="PRODUCES",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+(?:produces|manufactures)\s+(?P<object>[a-zA-Z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="produces",
        description="Captures production relationships (e.g. Brazil produces coffee)"
    ),

    # 9. Speaker Name Declaration: "My name is X."
    FactPattern(
        name="NAME_DECLARATION",
        regex=r'^(?:My\s+name\s+is|I\s+am\s+called|I\s+go\s+by)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="has name",
        description="Captures speaker name declaration (e.g. My name is Honnesh)",
        is_speaker_declaration=True
    ),

    # 10. Living / Residence: "I live in X." or "X lives in Y."
    FactPattern(
        name="LIVES_IN",
        regex=r'^(?:The\s+)?(?P<subject>I|[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+(?:lives?|resides?|am\s+living|is\s+living|am\s+residing|is\s+residing)\s+in\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="lives in",
        description="Captures residence (e.g. I live in Bangalore / Prakruthi lives in Bengaluru)"
    ),

    # 11. Employment / Work: "I work at X." or "X works at Y."
    FactPattern(
        name="WORKS_AT",
        regex=r'^(?:The\s+)?(?P<subject>I|[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+(?:works?|is\s+employed|am\s+employed)\s+(?:at|for|in)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="works at",
        description="Captures workplace or employer (e.g. I work at KPMG / Honnesh works at KPMG)"
    ),

    # 12. Studies at / Education: "I study at X." or "X studies at Y."
    FactPattern(
        name="STUDIES_AT",
        regex=r'^(?:The\s+)?(?P<subject>I|[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+(?:stud(?:y|ies)|attends?)\s+at\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="studies at",
        description="Captures education institution (e.g. I study at RUAS / Honnesh studies at RUAS)"
    ),

    # 13. College / Institution specification: "My college is X."
    FactPattern(
        name="COLLEGE_IS",
        regex=r'^(?:My|(?P<subject>[A-Za-z0-9\s\.\-\'’]+?)[\'’]s)\s+(?:college|university|school|institution)\s+is\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="studies at",
        description="Captures college association (e.g. My college is RUAS)"
    ),

    # 14. Student / Occupational Role: "I am an AI and ML engineering student."
    FactPattern(
        name="STUDENT_ROLE",
        regex=r'^(?:(?P<subject>I|[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+(?:am|is|are|\'m))\s+(?:an?|the)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?\s+student)\.?$',
        default_predicate="is a",
        description="Captures student classification (e.g. I am an AI and ML engineering student)"
    ),

    # 15. Belongs To: "X belongs to Y."
    FactPattern(
        name="BELONGS_TO",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+belongs?\s+to\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="belongs to",
        description="Captures geopolitical or organizational belonging (e.g. Bangalore belongs to Karnataka)"
    ),

    # 16. Generic Is a / Is an (Classification/Type): "X is a TYPE."
    FactPattern(
        name="IS_A",
        regex=r'^(?:The\s+)?(?P<subject>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+is\s+(?:a|an)\s+(?P<object>[a-zA-Z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="is a",
        description="Captures taxonomic / class membership (e.g. Python is a programming language)"
    ),

    # 17. Possession / Ownership: "X has Y."
    FactPattern(
        name="POSSESSION",
        regex=r'^(?:The\s+)?(?P<subject>I|[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\s+(?:has|have|owns?|possesses?|holds?)\s+(?P<object>[A-Za-z0-9][A-Za-z0-9\s\.\-\'’]*?)\.?$',
        default_predicate="has",
        description="Captures possession or ownership (e.g. Kishore has rolex watch / Akash has kishore)"
    ),
]


class FactExtractor:
    """
    Deterministic rule-based extractor that scans raw text, splits it into
    sentences, tests each sentence against supported patterns, resolves pronouns
    via DocumentContext, and produces structured Fact instances.
    """

    def __init__(self,
                 patterns: Optional[List[FactPattern]] = None,
                 relationship_registry: Optional[RelationshipRegistry] = None,
                 entity_registry: Optional[EntityRegistry] = None,
                 context_resolver: Optional[ContextResolver] = None):
        self.patterns = patterns or FACTUAL_PATTERNS
        self.rel_registry = relationship_registry or GLOBAL_RELATIONSHIP_REGISTRY
        self.entity_registry = entity_registry or GLOBAL_ENTITY_REGISTRY
        self.context_resolver = context_resolver or GLOBAL_CONTEXT_RESOLVER

    def extract_facts(self,
                      raw_text: str,
                      document_id: int,
                      context: Optional[DocumentContext] = None) -> List[Fact]:
        """
        Extract structured SPO facts from raw text for a specific document with
        speaker context and deterministic pronoun resolution.
        
        Args:
            raw_text: Raw character text extracted from document.
            document_id: Foreign key reference to documents table.
            context: Optional pre-constructed DocumentContext.
            
        Returns:
            List of Fact instances matching supported deterministic patterns.
        """
        if not raw_text or not raw_text.strip():
            return []

        sentences = split_sentences(raw_text)
        facts: List[Fact] = []
        seen_triples = set()

        # Build or use existing document context (Pass 1)
        doc_ctx = context or self.context_resolver.build_document_context(sentences, document_id)

        # Pass 2: Extract factual statements with reference resolution
        for sent_idx, sentence in sentences:
            matched_triple = None
            matched_pattern = None

            for pattern in self.patterns:
                match_result = pattern.match(sentence)
                if match_result:
                    matched_triple = match_result
                    matched_pattern = pattern
                    break

            # If unsupported sentence, strictly skip
            if not matched_triple or not matched_pattern:
                continue

            subj, pred, obj = matched_triple

            # Perform reference & pronoun resolution for first-person / speaker statements
            if matched_pattern.is_speaker_declaration:
                # E.g. "My name is Honnesh"
                speaker_name = obj
                doc_ctx.set_speaker(speaker_name)
                subj = speaker_name
                self.entity_registry.register_entity(canonical_name=speaker_name, entity_type="Person")
            elif subj.lower() in ('i', 'my') or subj.lower().startswith('my '):
                resolved_subj, is_res = self.context_resolver.resolve_subject_reference(subj, doc_ctx)
                subj = resolved_subj
            else:
                resolved_subj, _ = self.context_resolver.resolve_subject_reference(subj, doc_ctx)
                subj = resolved_subj

            # Clean and canonicalize predicate
            canon_pred = self.rel_registry.get_canonical_predicate(pred)

            # Auto-register recognized entities in entity registry
            subj_entity = self.entity_registry.register_entity(canonical_name=subj)
            obj_entity = self.entity_registry.register_entity(canonical_name=obj)

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
                    sentence_index=sent_idx,
                    canonical_predicate=canon_pred,
                    subject_entity_id=subj_entity.entity_id if subj_entity else None,
                    object_entity_id=obj_entity.entity_id if obj_entity else None,
                    fact_status="asserted"
                )
            )

        return facts
