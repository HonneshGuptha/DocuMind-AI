"""
Deterministic Question Parser Module
Converts classified natural language questions into formal StructuredQuery objects
by extracting entities, mapping predicates, and defining expected answer targets.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly rule-based and deterministic.
No NLP libraries, no POS tagging, no NER models, no semantic parsing.
Uses regular expressions and deterministic mappings. Unsupported questions are
explicitly flagged with status='unsupported'.
"""

import re
from typing import Optional, Dict, Any, Tuple
from .query_models import StructuredQuery
from .classifier import QuestionClassifier


def clean_entity(entity: Optional[str]) -> Optional[str]:
    """
    Clean extracted entity string:
    Strips leading articles ('the', 'a', 'an'), collapses internal whitespace,
    and strips trailing punctuation while preserving internal casing and hyphenation.
    """
    if not entity:
        return None
    cleaned = entity.strip().rstrip("?.!,")
    # Collapse multiple consecutive spaces within multi-word entity
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    # Strip leading articles
    cleaned = re.sub(r'^(?:the|a|an)\s+', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else None


class QuestionParser:
    """
    Deterministic rule-based parser that translates user questions into StructuredQuery
    specifications.
    """

    def __init__(self):
        self.classifier = QuestionClassifier()

    def parse(self, question: Any) -> StructuredQuery:
        """
        Parse a natural language question into a StructuredQuery.
        
        Args:
            question: User input (string).
            
        Returns:
            StructuredQuery populated with extracted entity, predicate, answer target,
            and parsing status.
        """
        # Preserve original verbatim question string
        original_q = str(question) if question is not None else ""

        # 1. Validation for malformed input (non-string or empty)
        if not isinstance(question, str):
            return StructuredQuery(
                original_question=original_q,
                question_type="malformed",
                parsing_status="error",
                error_message="Question must be a non-empty string."
            )

        raw_trimmed = question.strip()
        if not raw_trimmed:
            return StructuredQuery(
                original_question="",
                question_type="empty",
                parsing_status="empty",
                error_message="Question is empty."
            )

        # 2. Check for single punctuation or gibberish symbols
        if len(re.sub(r'[^\w\s]', '', raw_trimmed).strip()) == 0:
            return StructuredQuery(
                original_question=original_q,
                question_type="malformed",
                parsing_status="error",
                error_message="Question contains only punctuation or invalid characters."
            )

        # 3. Normalize whitespace and trailing punctuation for classification & pattern matching
        # Collapse internal multiple spaces to single space and normalize trailing punctuation
        normalized_q = re.sub(r'[ \t]+', ' ', raw_trimmed)
        normalized_q = re.sub(r'\s+(\?+)$', r'\1', normalized_q)

        # 4. Classify question type
        q_type = self.classifier.classify(normalized_q)
        if not q_type:
            return StructuredQuery(
                original_question=original_q,
                question_type="unsupported",
                parsing_status="unsupported",
                error_message="Question pattern is unsupported or unrecognized."
            )

        # 5. Dispatch to type-specific entity extraction rule
        parse_method = getattr(self, f"_parse_{q_type}", None)
        if not parse_method:
            return StructuredQuery(
                original_question=original_q,
                question_type=q_type,
                parsing_status="unsupported",
                error_message=f"Parser for question type '{q_type}' is not implemented."
            )

        return parse_method(normalized_q, original_q)

    # --------------------------------------------------------------------------
    # Type-Specific Deterministic Parsing Methods
    # --------------------------------------------------------------------------

    def _parse_capital(self, q: str, orig_q: str) -> StructuredQuery:
        # Pattern A: "What is Bangalore the capital of?" -> Subject: Bangalore, Pred: capital of, Target: object
        m_rev = re.match(r'^(?:what|which)\s+is\s+(?P<entity>.+?)\s+the\s+capital\s+of\s*\?*$', q, re.IGNORECASE)
        if m_rev:
            subj = clean_entity(m_rev.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='capital',
                subject=subj,
                predicate='capital of',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # Pattern B: "What is the capital of Karnataka?" / "Which city is the capital of Karnataka?"
        m = re.match(r'^(?:what|which(?:\s+city)?|name)\s+(?:is\s+the|the)\s+capital\s+(?:city\s+)?of\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^(?:what|which)\s+is\s+(?P<entity>[^?]+?)(?:\'s|s\')\s+capital\s*\?*$', q, re.IGNORECASE)

        if m:
            obj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='capital',
                subject=None,
                predicate='capital of',
                object=obj,
                expected_answer_field='subject',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'capital')

    def _parse_location(self, q: str, orig_q: str) -> StructuredQuery:
        m = re.match(r'^where\s+is\s+(?P<entity>.+?)\s+(?:located|situated)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^in\s+which\s+(?:country|state|continent|region|place)\s+is\s+(?P<entity>.+?)(?:\s+(?:located|situated))?\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^where\s+is\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)

        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='location',
                subject=subj,
                predicate='located in',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'location')

    def _parse_establishment(self, q: str, orig_q: str) -> StructuredQuery:
        m = re.match(r'^(?:when|(?:in\s+)?(?:what|which)\s+year)\s+was\s+(?P<entity>.+?)\s+(?P<action>founded|established|created|built)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^when\s+was\s+(?P<entity>.+?)\s+started\s*\?*$', q, re.IGNORECASE)

        if m:
            subj = clean_entity(m.group('entity'))
            action = m.groupdict().get('action', 'established').lower()
            return StructuredQuery(
                original_question=orig_q,
                question_type='establishment',
                subject=subj,
                predicate=f"{action} in" if action in ['founded', 'established', 'created', 'built'] else 'established in',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'establishment')

    def _parse_creator(self, q: str, orig_q: str) -> StructuredQuery:
        m = re.match(r'^who\s+(?P<action>developed|invented|authored|created|wrote|discovered)\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^who\s+was\s+(?P<entity>.+?)\s+(?P<action>developed|invented|authored|created|written|discovered)\s+by\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^who\s+is\s+the\s+(?P<role>creator|author|developer|inventor|discoverer)\s+of\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)

        if m:
            subj = clean_entity(m.group('entity'))
            groups = m.groupdict()
            if 'action' in groups and groups['action']:
                act = groups['action'].lower()
                pred_map = {
                    'developed': 'developed by',
                    'invented': 'invented by',
                    'authored': 'authored by',
                    'written': 'written by',
                    'wrote': 'written by',
                    'discovered': 'discovered by',
                    'created': 'developed by',
                }
                pred = pred_map.get(act, f"{act} by")
            elif 'role' in groups and groups['role']:
                role = groups['role'].lower()
                role_map = {
                    'developer': 'developed by',
                    'creator': 'developed by',
                    'inventor': 'invented by',
                    'author': 'authored by',
                    'discoverer': 'discovered by',
                }
                pred = role_map.get(role, 'developed by')
            else:
                pred = 'developed by'

            return StructuredQuery(
                original_question=orig_q,
                question_type='creator',
                subject=subj,
                predicate=pred,
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'creator')

    def _parse_part_whole(self, q: str, orig_q: str) -> StructuredQuery:
        m = re.match(r'^(?:what|which(?:\s+component)?)\s+is\s+(?:a\s+)?part\s+of\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^what\s+does\s+(?P<entity>.+?)\s+consist\s+of\s*\?*$', q, re.IGNORECASE)
        if m:
            obj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='part_whole',
                subject=None,
                predicate='part of',
                object=obj,
                expected_answer_field='subject',
                parsing_status='success'
            )

        m_bool = re.match(r'^is\s+(?P<sub_entity>.+?)\s+(?:a\s+)?part\s+of\s+(?P<main_entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_bool:
            subj = clean_entity(m_bool.group('sub_entity'))
            obj = clean_entity(m_bool.group('main_entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='part_whole',
                subject=subj,
                predicate='part of',
                object=obj,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'part_whole')

    def _parse_border(self, q: str, orig_q: str) -> StructuredQuery:
        m = re.match(r'^(?:what|which(?:\s+country|\s+state|\s+place)?)\s+does\s+(?P<entity>.+?)\s+(?:border|neighbor)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^(?:what|which(?:\s+countries|\s+states)?)\s+border\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='border',
                subject=subj,
                predicate='borders',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        m_bool = re.match(r'^does\s+(?P<e1>.+?)\s+(?:border|neighbor)\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_bool:
            subj = clean_entity(m_bool.group('e1'))
            obj = clean_entity(m_bool.group('e2'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='border',
                subject=subj,
                predicate='borders',
                object=obj,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'border')

    def _parse_production(self, q: str, orig_q: str) -> StructuredQuery:
        m = re.match(r'^(?:what|which\s+goods?|which\s+products?)\s+does\s+(?P<entity>.+?)\s+(?:produce|manufacture)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^what\s+is\s+produced\s+by\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)

        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='production',
                subject=subj,
                predicate='produces',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'production')

    def _parse_identity(self, q: str, orig_q: str) -> StructuredQuery:
        # Pattern A: "Who is Guido van Rossum?" / "Who was Dennis Ritchie?"
        m = re.match(r'^who\s+(?:is|was)\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        # Pattern B: "What is the identity of Guido van Rossum?"
        if not m:
            m = re.match(r'^what\s+is\s+the\s+identity\s+of\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        # Pattern C: "What is Guido van Rossum's identity?"
        if not m:
            m = re.match(r'^what\s+is\s+(?P<entity>.+?)(?:\'s|s\')\s+identity\s*\?*$', q, re.IGNORECASE)

        if m:
            entity = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='identity',
                subject=entity,
                predicate=None,
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'identity')

    def _parse_classification(self, q: str, orig_q: str) -> StructuredQuery:
        m = re.match(r'^what\s+(?:kind|type|sort)\s+of\s+(?:entity|thing|language|system)\s+is\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^what\s+is\s+(?:a|an)\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^what\s+is\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)

        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='classification',
                subject=subj,
                predicate='is a',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        m_bool = re.match(r'^is\s+(?P<e1>.+?)\s+(?:a|an)\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_bool:
            subj = clean_entity(m_bool.group('e1'))
            obj = clean_entity(m_bool.group('e2'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='classification',
                subject=subj,
                predicate='is a',
                object=obj,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'classification')

    def _unsupported_query(self, orig_q: str, q_type: str) -> StructuredQuery:
        return StructuredQuery(
            original_question=orig_q,
            question_type=q_type,
            parsing_status='unsupported',
            error_message=f"Ambiguous or unsupported syntax for {q_type} question."
        )


def parse_question(question: str) -> StructuredQuery:
    """Helper function to parse a question in a single call."""
    parser = QuestionParser()
    return parser.parse(question)
