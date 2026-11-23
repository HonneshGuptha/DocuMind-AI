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
from modules.knowledge_base.relationship_registry import GLOBAL_RELATIONSHIP_REGISTRY
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

        result = parse_method(normalized_q, original_q)
        if result.parsing_status == 'unsupported':
            # Check if generic reverse relationship provides a more specific query
            rev = self._parse_reverse_relationship(normalized_q, original_q)
            if rev:
                return rev
        return result

    def _parse_reverse_relationship(self, q: str, orig_q: str) -> Optional[StructuredQuery]:
        """
        Generic reverse relationship parser driven by RelationshipRegistry.
        Detects 'Who <verb_phrase> <entity>?' and extracts predicate & object.
        """
        m = re.match(r'^(?:who|which\s+(?:person|entity|individual))\s+(?:was\s+|is\s+)?(?P<remainder>.+?)\s*\?*$', q, re.IGNORECASE)
        if not m:
            return None

        remainder = m.group('remainder')
        match = GLOBAL_RELATIONSHIP_REGISTRY.match_reverse_phrase(remainder)
        if not match:
            return None

        var, entity_part, defn = match
        obj = clean_entity(entity_part)
        if not obj:
            return None

        if not defn.supports_reverse:
            return StructuredQuery(
                original_question=orig_q,
                question_type=defn.question_type,
                subject=None,
                predicate=defn.canonical_predicate,
                object=obj,
                expected_answer_field='subject',
                parsing_status='unsupported',
                error_message=f"Reverse relationship query for '{defn.canonical_predicate}' is unsupported."
            )

        return StructuredQuery(
            original_question=orig_q,
            question_type=defn.question_type,
            subject=None,
            predicate=defn.canonical_predicate,
            object=obj,
            expected_answer_field='subject',
            parsing_status='success'
        )

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

    def _parse_name(self, q: str, orig_q: str) -> StructuredQuery:
        # "Who am I?" / "What is my name?" / "What is my identity?" -> subject: 'Speaker', predicate: 'has name', target: 'object'
        if re.match(r'^who\s+am\s+i\s*\?*$', q, re.IGNORECASE) or \
           re.match(r'^(?:what\s+is|what\'s)\s+my\s+(?:name|identity)\s*\?*$', q, re.IGNORECASE) or \
           re.match(r'^(?:what\s+is|what\'s)\s+the\s+identity\s+of\s+me\s*\?*$', q, re.IGNORECASE):
            return StructuredQuery(
                original_question=orig_q,
                question_type='name',
                subject='Speaker',
                predicate='has name',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        m = re.match(r'^(?:what\s+is|what\'s)\s+(?P<entity>[^?]+?)(?:\'s|s\')\s+name\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^name\s+of\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='name',
                subject=subj,
                predicate='has name',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'name')

    def _parse_residence(self, q: str, orig_q: str) -> StructuredQuery:
        # Check reverse query first (e.g. "Who lives in Bangalore?")
        rev = self._parse_reverse_relationship(q, orig_q)
        if rev and rev.question_type == 'residence':
            return rev

        # "Where do I live?"
        if re.match(r'^(?:where|in\s+which\s+(?:city|place))\s+do\s+i\s+(?:live|reside|stay)\s*\?*$', q, re.IGNORECASE):
            return StructuredQuery(
                original_question=orig_q,
                question_type='residence',
                subject='Speaker',
                predicate='lives in',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # "Where does Honnesh live?"
        m = re.match(r'^(?:where|in\s+which\s+(?:city|place))\s+(?:does|is)\s+(?P<entity>.+?)\s+(?:live|reside|residing|stay|staying)\s*\?*$', q, re.IGNORECASE)
        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='residence',
                subject=subj,
                predicate='lives in',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'residence')

    def _parse_employment(self, q: str, orig_q: str) -> StructuredQuery:
        # Check reverse query first (e.g. "Who works at KPMG?")
        rev = self._parse_reverse_relationship(q, orig_q)
        if rev and rev.question_type == 'employment':
            return rev

        # "Where do I work?"
        if re.match(r'^(?:where|which\s+company)\s+do\s+i\s+work(?:\s+(?:at|for))?\s*\?*$', q, re.IGNORECASE):
            return StructuredQuery(
                original_question=orig_q,
                question_type='employment',
                subject='Speaker',
                predicate='works at',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # "Where does Honnesh work?"
        m = re.match(r'^(?:where|which\s+company)\s+(?:does|is)\s+(?P<entity>.+?)\s+(?:work|working|employed)(?:\s+(?:at|for))?\s*\?*$', q, re.IGNORECASE)
        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='employment',
                subject=subj,
                predicate='works at',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'employment')

    def _parse_education(self, q: str, orig_q: str) -> StructuredQuery:
        # Check reverse query first (e.g. "Who studies at RUAS?")
        rev = self._parse_reverse_relationship(q, orig_q)
        if rev and rev.question_type == 'education':
            return rev

        # "Where do I study?" / "What is my college?"
        if re.match(r'^(?:where|which\s+college|which\s+university)\s+do\s+i\s+(?:study|attend|go\s+to)(?:\s+at)?\s*\?*$', q, re.IGNORECASE) or \
           re.match(r'^(?:what|which)\s+is\s+my\s+(?:college|university|school|institution)\s*\?*$', q, re.IGNORECASE):
            return StructuredQuery(
                original_question=orig_q,
                question_type='education',
                subject='Speaker',
                predicate='studies at',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # "What am I studying?"
        if re.match(r'^what\s+am\s+i\s+studying\s*\?*$', q, re.IGNORECASE):
            return StructuredQuery(
                original_question=orig_q,
                question_type='education',
                subject='Speaker',
                predicate='is a',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # "What is Honnesh studying?"
        m_stud = re.match(r'^what\s+is\s+(?P<entity>[^?]+?)\s+studying\s*\?*$', q, re.IGNORECASE)
        if m_stud:
            subj = clean_entity(m_stud.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='education',
                subject=subj,
                predicate='is a',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # "Where does Honnesh study?"
        m = re.match(r'^(?:where|which\s+college|which\s+university)\s+does\s+(?P<entity>.+?)\s+(?:study|attend|go\s+to)(?:\s+at)?\s*\?*$', q, re.IGNORECASE) or \
            re.match(r'^(?:what|which)\s+is\s+(?P<entity>[^?]+?)(?:\'s|s\')\s+(?:college|university|school)\s*\?*$', q, re.IGNORECASE)
        if m:
            subj = clean_entity(m.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='education',
                subject=subj,
                predicate='studies at',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'education')

    def _parse_location(self, q: str, orig_q: str) -> StructuredQuery:
        # Reverse location query: "Who is in Sai's dream?" / "What is in Sai's dream?"
        m_rev = re.match(r'^(?:who|what)\s+is\s+(?:located\s+in|situated\s+in|in)\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_rev:
            obj = clean_entity(m_rev.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='location',
                subject=None,
                predicate='located in',
                object=obj,
                expected_answer_field='subject',
                parsing_status='success'
            )

        m = re.match(r'^where\s+is\s+(?P<entity>.+?)\s+(?:located|situated)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^which\s+(?:country|state|continent|region)\s+(?:is|does)\s+(?P<entity>.+?)\s+(?:located\s+in|situated\s+in|belong\s+to|in)\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^(?:what|which)\s+(?:continent|country)\s+is\s+(?P<entity>.+?)\s+(?:located|situated)\s+in\s*\?*$', q, re.IGNORECASE)
        if not m:
            m = re.match(r'^(?:what|which)\s+(?:country|state|continent)\s+does\s+(?P<entity>.+?)\s+belong\s+to\s*\?*$', q, re.IGNORECASE)
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

    def _parse_possession(self, q: str, orig_q: str) -> StructuredQuery:
        # "Who has rolex watch?" / "Who owns rolex watch?"
        m_who = re.match(r'^who\s+(?:has|owns|possesses|holds)\s+(?P<entity>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_who:
            obj = clean_entity(m_who.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='possession',
                subject=None,
                predicate='has',
                object=obj,
                expected_answer_field='subject',
                parsing_status='success'
            )

        # "What does Kishore have?" / "What does Kishore own?"
        m_what = re.match(r'^what\s+does\s+(?P<entity>.+?)\s+(?:have|own|possess|hold)\s*\?*$', q, re.IGNORECASE)
        if m_what:
            subj = clean_entity(m_what.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='possession',
                subject=subj,
                predicate='has',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # "What do I have?"
        if re.match(r'^what\s+do\s+i\s+(?:have|own|possess|hold)\s*\?*$', q, re.IGNORECASE):
            return StructuredQuery(
                original_question=orig_q,
                question_type='possession',
                subject='Speaker',
                predicate='has',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # "Who is <entity> owned by?"
        m_by = re.match(r'^who\s+is\s+(?P<entity>.+?)\s+(?:owned|possessed|held)\s+by\s*\?*$', q, re.IGNORECASE)
        if m_by:
            obj = clean_entity(m_by.group('entity'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='possession',
                subject=None,
                predicate='has',
                object=obj,
                expected_answer_field='subject',
                parsing_status='success'
            )

        # Boolean: "Does Kishore have rolex watch?" / "Does Akash have kishore?"
        m_bool = re.match(r'^does\s+(?P<e1>.+?)\s+(?:have|own|possess|hold)\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_bool:
            s = clean_entity(m_bool.group('e1'))
            o = clean_entity(m_bool.group('e2'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='possession',
                subject=s,
                predicate='has',
                object=o,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        # Boolean: "Do I have rolex watch?"
        m_bool_i = re.match(r'^do\s+i\s+(?:have|own|possess|hold)\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_bool_i:
            o = clean_entity(m_bool_i.group('e2'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='possession',
                subject='Speaker',
                predicate='has',
                object=o,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        return self._unsupported_query(orig_q, 'possession')

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
        # Pattern A0: First-person role queries: "What am I?" / "What student am I?" / "Which student am I?"
        if re.match(r'^what\s+am\s+i\s*\?*$', q, re.IGNORECASE) or \
           re.match(r'^(?:what|which)\s+(?:student|kind\s+of\s+student|type\s+of\s+student)\s+am\s+i\s*\?*$', q, re.IGNORECASE):
            return StructuredQuery(
                original_question=orig_q,
                question_type='classification',
                subject='Speaker',
                predicate='is a',
                object=None,
                expected_answer_field='object',
                parsing_status='success'
            )

        # Pattern A: "Am I a student?" / "Am I an AI student?"
        m_ami = re.match(r'^am\s+i\s+(?:a|an)\s+(?P<role>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_ami:
            role = clean_entity(m_ami.group('role'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='classification',
                subject='Speaker',
                predicate='is a',
                object=role,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        # Pattern B: "Is Karnataka a state in India?"
        m_region = re.match(r'^is\s+(?P<e1>.+?)\s+a\s+(?P<type>state|city|country|province|continent)\s+in\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_region:
            subj = clean_entity(m_region.group('e1'))
            obj = clean_entity(m_region.group('e2'))
            rtype = m_region.group('type').lower()
            return StructuredQuery(
                original_question=orig_q,
                question_type='classification',
                subject=subj,
                predicate=f"is a {rtype} in",
                object=obj,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        # Pattern C: "Is Bangalore located in India?"
        m_loc = re.match(r'^is\s+(?P<e1>.+?)\s+(?:located|situated)\s+in\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_loc:
            subj = clean_entity(m_loc.group('e1'))
            obj = clean_entity(m_loc.group('e2'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='location',
                subject=subj,
                predicate='located in',
                object=obj,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        # Pattern C2: "Is Ayyappa in Sai's dream?"
        m_in = re.match(r'^is\s+(?P<e1>.+?)\s+in\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_in:
            subj = clean_entity(m_in.group('e1'))
            obj = clean_entity(m_in.group('e2'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='location',
                subject=subj,
                predicate='located in',
                object=obj,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        # Pattern D: "Does Bangalore belong to Karnataka?"
        m_bel = re.match(r'^does\s+(?P<e1>.+?)\s+belong\s+to\s+(?P<e2>.+?)\s*\?*$', q, re.IGNORECASE)
        if m_bel:
            subj = clean_entity(m_bel.group('e1'))
            obj = clean_entity(m_bel.group('e2'))
            return StructuredQuery(
                original_question=orig_q,
                question_type='classification',
                subject=subj,
                predicate='belongs to',
                object=obj,
                expected_answer_field='boolean',
                parsing_status='success'
            )

        # Pattern E: WH-Classification "What is Python?" / "What kind of language is Python?"
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

        # Pattern F: Boolean Classification "Is Honnesh a student?" / "Is Python a programming language?"
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
