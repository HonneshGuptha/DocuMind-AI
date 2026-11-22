"""
Deterministic Question Classifier Module
Categorizes user natural language questions into predefined structural types
using declarative regular expression patterns.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly rule-based and deterministic.
No NLP libraries (spaCy/NLTK), no POS tagging, no dependency parsing,
no pretrained language models, and no semantic embedding classification.
"""

import re
from typing import Optional, List, Tuple
from modules.knowledge_base.relationship_registry import GLOBAL_RELATIONSHIP_REGISTRY


class QuestionClassifier:
    """
    Deterministic rule-based classifier that maps questions to one of nine
    canonical factual question types.
    """

    # Predefined question classification patterns
    # NOTE: Identity rules are placed before generic classification to avoid keyword collisions.
    CLASSIFICATION_RULES: List[Tuple[str, List[re.Pattern]]] = [
        # 1. Capital Question
        (
            'capital',
            [
                re.compile(r'^(?:what|which(?:\s+city)?)\s+is\s+the\s+capital\s+(?:city\s+)?of\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^name\s+the\s+capital\s+(?:city\s+)?of\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which)\s+is\s+(?P<entity>[^?]+?)(?:\'s|s\')\s+capital\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which)\s+is\s+(?P<entity>.+?)\s+the\s+capital\s+of\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 2. Name & Personal Identity Question
        (
            'name',
            [
                re.compile(r'^who\s+am\s+i\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what\s+is|what\'s)\s+my\s+name\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what\s+is|what\'s)\s+my\s+identity\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what\s+is|what\'s)\s+the\s+identity\s+of\s+me\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what\s+is|what\'s)\s+(?P<entity>[^?]+?)(?:\'s|s\')\s+name\s*\?*$', re.IGNORECASE),
                re.compile(r'^name\s+of\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 3. Residence / Living Location Question
        (
            'residence',
            [
                re.compile(r'^where\s+do\s+i\s+(?:live|reside|stay)\s*\?*$', re.IGNORECASE),
                re.compile(r'^where\s+(?:does|is)\s+(?P<entity>.+?)\s+(?:live|reside|residing|stay|staying)\s*\?*$', re.IGNORECASE),
                re.compile(r'^in\s+which\s+(?:city|place)\s+(?:do\s+i|does\s+(?P<entity>.+?))\s+(?:live|reside)\s*\?*$', re.IGNORECASE),
                re.compile(r'^who\s+(?:lives|resides|stays)\s+in\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 4. Employment / Workplace Question
        (
            'employment',
            [
                re.compile(r'^where\s+do\s+i\s+work\s*\?*$', re.IGNORECASE),
                re.compile(r'^where\s+(?:does|is)\s+(?P<entity>.+?)\s+(?:work|working|employed)\s*\?*$', re.IGNORECASE),
                re.compile(r'^who\s+(?:works|is\s+employed)\s+(?:at|for|in)\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^which\s+company\s+(?:do\s+i|does\s+(?P<entity>.+?))\s+work\s+(?:at|for)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 5. Education / Studies Question
        (
            'education',
            [
                re.compile(r'^where\s+do\s+i\s+study\s*\?*$', re.IGNORECASE),
                re.compile(r'^where\s+does\s+(?P<entity>.+?)\s+study\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+am\s+i\s+studying\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+(?P<entity>.+?)\s+studying\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which)\s+is\s+my\s+(?:college|university|school|institution)\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which)\s+is\s+(?P<entity>[^?]+?)(?:\'s|s\')\s+(?:college|university|school)\s*\?*$', re.IGNORECASE),
                re.compile(r'^who\s+(?:studies|is\s+enrolled)\s+(?:at|in)\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^who\s+attends\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 6. Location Question (Expanded for containment & belonging)
        (
            'location',
            [
                re.compile(r'^where\s+is\s+(?P<entity>.+?)\s+(?:located|situated)\s*\?*$', re.IGNORECASE),
                re.compile(r'^which\s+(?:country|state|continent|region)\s+(?:is|does)\s+(?P<entity>.+?)\s+(?:located\s+in|situated\s+in|belong\s+to|in)\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which)\s+(?:continent|country)\s+is\s+(?P<entity>.+?)\s+(?:located|situated)\s+in\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which)\s+(?:country|state|continent)\s+does\s+(?P<entity>.+?)\s+belong\s+to\s*\?*$', re.IGNORECASE),
                re.compile(r'^in\s+which\s+(?:country|state|continent|region|place)\s+is\s+(?P<entity>.+?)(?:\s+(?:located|situated))?\s*\?*$', re.IGNORECASE),
                re.compile(r'^where\s+is\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 3. Establishment / Founding Question
        (
            'establishment',
            [
                re.compile(r'^(?:when|(?:in\s+)?(?:what|which)\s+year)\s+was\s+(?P<entity>.+?)\s+(?:founded|established|created|built)\s*\?*$', re.IGNORECASE),
                re.compile(r'^when\s+was\s+(?P<entity>.+?)\s+started\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 4. Creator / Inventor / Author Question
        (
            'creator',
            [
                re.compile(r'^who\s+(?:developed|invented|authored|created|wrote|discovered)\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^who\s+was\s+(?P<entity>.+?)\s+(?:developed|invented|authored|created|written|discovered)\s+by\s*\?*$', re.IGNORECASE),
                re.compile(r'^who\s+is\s+the\s+(?:creator|author|developer|inventor|discoverer)\s+of\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 5. Part-Whole Question
        (
            'part_whole',
            [
                re.compile(r'^(?:what|which(?:\s+component)?)\s+is\s+(?:a\s+)?part\s+of\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^is\s+(?P<sub_entity>.+?)\s+(?:a\s+)?part\s+of\s+(?P<main_entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+does\s+(?P<entity>.+?)\s+consist\s+of\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 6. Border / Adjacency Question
        (
            'border',
            [
                re.compile(r'^(?:what|which(?:\s+country|\s+state|\s+place)?)\s+does\s+(?P<entity>.+?)\s+(?:border|neighbor)\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which(?:\s+countries|\s+states)?)\s+border\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^does\s+(?P<e1>.+?)\s+border\s+(?P<e2>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 7. Production Question
        (
            'production',
            [
                re.compile(r'^(?:what|which\s+goods?|which\s+products?)\s+does\s+(?P<entity>.+?)\s+(?:produce|manufacture)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+produced\s+by\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 8. Identity Question (Evaluated before generic classification)
        (
            'identity',
            [
                re.compile(r'^who\s+(?:is|was)\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+the\s+identity\s+of\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+(?P<entity>.+?)(?:\'s|s\')\s+identity\s*\?*$', re.IGNORECASE),
            ]
        ),

        # 9. Classification & Boolean Verification Question
        (
            'classification',
            [
                re.compile(r'^what\s+am\s+i\s*\?*$', re.IGNORECASE),
                re.compile(r'^(?:what|which)\s+(?:student|kind\s+of\s+student|type\s+of\s+student)\s+am\s+i\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+(?:kind|type|sort)\s+of\s+(?:entity|thing|language|system)\s+is\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+(?:a|an)\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^am\s+i\s+(?:a|an)\s+(?P<e2>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^is\s+(?P<e1>.+?)\s+a\s+(?P<e2>(?:state|city|country|province|continent)\s+in\s+.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^is\s+(?P<e1>.+?)\s+(?:located|situated)\s+in\s+(?P<e2>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^does\s+(?P<e1>.+?)\s+belong\s+to\s+(?P<e2>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^is\s+(?P<e1>.+?)\s+(?:a|an)\s+(?P<e2>.+?)\s*\?*$', re.IGNORECASE),
            ]
        ),
    ]

    def classify(self, question: str) -> Optional[str]:
        """
        Deterministically classify a question string into one of the 9 supported types.
        
        Returns:
            Type name string ('capital', 'location', etc.) or None if unsupported or empty.
        """
        if not question or not question.strip():
            return None

        # Normalize leading/trailing whitespace, collapse internal spaces, and handle trailing punctuation
        clean_question = re.sub(r'[ \t]+', ' ', question.strip())
        clean_question = re.sub(r'\s+(\?+)$', r'\1', clean_question)

        # Generic reverse relationship resolution driven by RelationshipRegistry
        m_rev = re.match(r'^(?:who|which\s+(?:person|entity|individual))\s+(?:was\s+|is\s+)?(?P<remainder>.+?)\s*\?*$', clean_question, re.IGNORECASE)
        if m_rev:
            match = GLOBAL_RELATIONSHIP_REGISTRY.match_reverse_phrase(m_rev.group('remainder'))
            if match:
                var, entity_part, defn = match
                if entity_part:
                    return defn.question_type

        for q_type, patterns in self.CLASSIFICATION_RULES:
            for pattern in patterns:
                if pattern.match(clean_question):
                    return q_type

        return None
