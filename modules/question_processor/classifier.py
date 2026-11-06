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

        # 2. Location Question
        (
            'location',
            [
                re.compile(r'^where\s+is\s+(?P<entity>.+?)\s+(?:located|situated)\s*\?*$', re.IGNORECASE),
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

        # 9. Classification Question (What is X? What kind of X is Y?)
        (
            'classification',
            [
                re.compile(r'^what\s+(?:kind|type|sort)\s+of\s+(?:entity|thing|language|system)\s+is\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+(?:a|an)\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
                re.compile(r'^what\s+is\s+(?P<entity>.+?)\s*\?*$', re.IGNORECASE),
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

        for q_type, patterns in self.CLASSIFICATION_RULES:
            for pattern in patterns:
                if pattern.match(clean_question):
                    return q_type

        return None
