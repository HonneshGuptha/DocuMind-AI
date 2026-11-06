"""
Answer Generator Module
Produces deterministic, coherent natural language answers synthesized exclusively
from matched facts and inference chains using predefined templates.
"""

from .answer_models import AnswerResponse
from .generator import AnswerGenerator, generate_answer
from .explanation_generator import ExplanationGenerator, generate_explanation

__all__ = [
    'AnswerResponse',
    'AnswerGenerator',
    'generate_answer',
    'ExplanationGenerator',
    'generate_explanation',
]
