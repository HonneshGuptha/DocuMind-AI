"""
Unit tests for Module 3: Question Processing and Information Retrieval Subsystem.
Tests deterministic classification, parsing across 9 question types, multiple phrasings,
entity/predicate extraction, case-insensitivity, empty/malformed/unsupported queries,
and parameterized retrieval from KnowledgeBaseRepository with provenance preservation.
"""

import os
import shutil
import tempfile
import unittest

from modules.knowledge_base.models import Fact
from modules.knowledge_base.repository import KnowledgeBaseRepository
from modules.question_processor import (
    StructuredQuery,
    RetrievalResult,
    QuestionClassifier,
    QuestionParser,
    parse_question,
    QuestionRetriever,
    retrieve_facts,
    process_and_retrieve,
)


class TestQuestionProcessor(unittest.TestCase):
    """Test suite for deterministic question classification, parsing, and relational retrieval."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary sandbox directory for test SQLite databases."""
        cls.test_dir = tempfile.mkdtemp(prefix="documind_qp_test_")

    @classmethod
    def tearDownClass(cls):
        """Remove temporary directory after all tests complete."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        """Initialize parser, classifier, and clean isolated repository for each test."""
        self.classifier = QuestionClassifier()
        self.parser = QuestionParser()
        self.db_path = os.path.join(self.test_dir, f"test_qp_{self._testMethodName}.db")
        self.repo = KnowledgeBaseRepository(self.db_path)
        self.retriever = QuestionRetriever(self.repo)

    # --------------------------------------------------------------------------
    # 1. Classification & Parsing Across All 9 Supported Question Types
    # --------------------------------------------------------------------------

    def test_type_1_capital_questions(self):
        """Test parsing of 'capital' questions across multiple phrasings."""
        # Phrasing A: "What is the capital of Karnataka?"
        q1 = self.parser.parse("What is the capital of Karnataka?")
        self.assertEqual(q1.question_type, "capital")
        self.assertEqual(q1.predicate, "capital of")
        self.assertEqual(q1.object, "Karnataka")
        self.assertEqual(q1.expected_answer_field, "subject")
        self.assertEqual(q1.parsing_status, "success")

        # Phrasing B: "Which city is the capital of France?"
        q2 = self.parser.parse("Which city is the capital of France?")
        self.assertEqual(q2.question_type, "capital")
        self.assertEqual(q2.object, "France")
        self.assertEqual(q2.expected_answer_field, "subject")

        # Phrasing C: "Name the capital of India"
        q3 = self.parser.parse("Name the capital of India")
        self.assertEqual(q3.question_type, "capital")
        self.assertEqual(q3.object, "India")

        # Phrasing D: "What is Bangalore the capital of?"
        q4 = self.parser.parse("What is Bangalore the capital of?")
        self.assertEqual(q4.question_type, "capital")
        self.assertEqual(q4.subject, "Bangalore")
        self.assertEqual(q4.predicate, "capital of")
        self.assertEqual(q4.expected_answer_field, "object")

    def test_type_2_location_questions(self):
        """Test parsing of 'location' questions across multiple phrasings."""
        # Phrasing A: "Where is Tokyo located?"
        q1 = self.parser.parse("Where is Tokyo located?")
        self.assertEqual(q1.question_type, "location")
        self.assertEqual(q1.subject, "Tokyo")
        self.assertEqual(q1.predicate, "located in")
        self.assertEqual(q1.expected_answer_field, "object")

        # Phrasing B: "Where is the Eiffel Tower situated?"
        q2 = self.parser.parse("Where is the Eiffel Tower situated?")
        self.assertEqual(q2.subject, "Eiffel Tower")
        self.assertEqual(q2.predicate, "located in")

        # Phrasing C: "In which country is Tokyo located?"
        q3 = self.parser.parse("In which country is Tokyo located?")
        self.assertEqual(q3.subject, "Tokyo")
        self.assertEqual(q3.predicate, "located in")

    def test_type_3_classification_questions(self):
        """Test parsing of 'classification' (is-a / taxonomic) questions."""
        # Phrasing A: "What is Python?"
        q1 = self.parser.parse("What is Python?")
        self.assertEqual(q1.question_type, "classification")
        self.assertEqual(q1.subject, "Python")
        self.assertEqual(q1.predicate, "is a")
        self.assertEqual(q1.expected_answer_field, "object")

        # Phrasing B: "What kind of language is Python?"
        q2 = self.parser.parse("What kind of language is Python?")
        self.assertEqual(q2.subject, "Python")
        self.assertEqual(q2.predicate, "is a")

        # Phrasing C: Boolean verification: "Is Python a programming language?"
        q3 = self.parser.parse("Is Python a programming language?")
        self.assertEqual(q3.subject, "Python")
        self.assertEqual(q3.object, "programming language")
        self.assertEqual(q3.predicate, "is a")
        self.assertEqual(q3.expected_answer_field, "boolean")

    def test_type_4_establishment_questions(self):
        """Test parsing of 'establishment' questions."""
        # Phrasing A: "When was Harvard University founded?"
        q1 = self.parser.parse("When was Harvard University founded?")
        self.assertEqual(q1.question_type, "establishment")
        self.assertEqual(q1.subject, "Harvard University")
        self.assertEqual(q1.predicate, "founded in")
        self.assertEqual(q1.expected_answer_field, "object")

        # Phrasing B: "In what year was Stanford established?"
        q2 = self.parser.parse("In what year was Stanford established?")
        self.assertEqual(q2.subject, "Stanford")
        self.assertEqual(q2.predicate, "established in")

    def test_type_5_creator_questions(self):
        """Test parsing of 'creator / inventor / author' questions."""
        # Phrasing A: "Who developed Python?"
        q1 = self.parser.parse("Who developed Python?")
        self.assertEqual(q1.question_type, "creator")
        self.assertEqual(q1.subject, "Python")
        self.assertEqual(q1.predicate, "developed by")
        self.assertEqual(q1.expected_answer_field, "object")

        # Phrasing B: "Who invented Telescope?"
        q2 = self.parser.parse("Who invented Telescope?")
        self.assertEqual(q2.subject, "Telescope")
        self.assertEqual(q2.predicate, "invented by")

        # Phrasing C: "Who was C written by?"
        q3 = self.parser.parse("Who was C written by?")
        self.assertEqual(q3.subject, "C")
        self.assertEqual(q3.predicate, "written by")

        # Phrasing D: "Who is the author of C?"
        q4 = self.parser.parse("Who is the author of C?")
        self.assertEqual(q4.subject, "C")
        self.assertEqual(q4.predicate, "authored by")

    def test_type_6_part_whole_questions(self):
        """Test parsing of 'part-whole' questions."""
        # Phrasing A: "What is part of Computer?"
        q1 = self.parser.parse("What is part of Computer?")
        self.assertEqual(q1.question_type, "part_whole")
        self.assertEqual(q1.object, "Computer")
        self.assertEqual(q1.predicate, "part of")
        self.assertEqual(q1.expected_answer_field, "subject")

        # Phrasing B: Boolean: "Is CPU a part of Computer?"
        q2 = self.parser.parse("Is CPU a part of Computer?")
        self.assertEqual(q2.subject, "CPU")
        self.assertEqual(q2.object, "Computer")
        self.assertEqual(q2.predicate, "part of")
        self.assertEqual(q2.expected_answer_field, "boolean")

    def test_type_7_border_questions(self):
        """Test parsing of 'border / adjacency' questions."""
        # Phrasing A: "What does Canada border?"
        q1 = self.parser.parse("What does Canada border?")
        self.assertEqual(q1.question_type, "border")
        self.assertEqual(q1.subject, "Canada")
        self.assertEqual(q1.predicate, "borders")
        self.assertEqual(q1.expected_answer_field, "object")

        # Phrasing B: "Which countries border Canada?"
        q2 = self.parser.parse("Which countries border Canada?")
        self.assertEqual(q2.subject, "Canada")
        self.assertEqual(q2.predicate, "borders")

        # Phrasing C: Boolean: "Does Canada border United States?"
        q3 = self.parser.parse("Does Canada border United States?")
        self.assertEqual(q3.subject, "Canada")
        self.assertEqual(q3.object, "United States")
        self.assertEqual(q3.predicate, "borders")
        self.assertEqual(q3.expected_answer_field, "boolean")

    def test_type_8_production_questions(self):
        """Test parsing of 'production' questions."""
        # Phrasing A: "What does Brazil produce?"
        q1 = self.parser.parse("What does Brazil produce?")
        self.assertEqual(q1.question_type, "production")
        self.assertEqual(q1.subject, "Brazil")
        self.assertEqual(q1.predicate, "produces")
        self.assertEqual(q1.expected_answer_field, "object")

        # Phrasing B: "What is produced by Brazil?"
        q2 = self.parser.parse("What is produced by Brazil?")
        self.assertEqual(q2.subject, "Brazil")
        self.assertEqual(q2.predicate, "produces")

    def test_type_9_identity_questions(self):
        """Test parsing of 'identity' questions."""
        q1 = self.parser.parse("Who is Guido van Rossum?")
        self.assertEqual(q1.question_type, "identity")
        self.assertEqual(q1.subject, "Guido van Rossum")
        self.assertIsNone(q1.predicate)
        self.assertEqual(q1.expected_answer_field, "object")

    # --------------------------------------------------------------------------
    # 2. Case-Insensitive Parsing & Article Cleaning
    # --------------------------------------------------------------------------

    def test_case_insensitive_question_parsing(self):
        """Verify that questions in uppercase or mixed case parse consistently."""
        q_upper = self.parser.parse("WHERE IS TOKYO LOCATED?")
        self.assertEqual(q_upper.question_type, "location")
        self.assertEqual(q_upper.subject, "TOKYO")
        self.assertEqual(q_upper.predicate, "located in")

        q_lower = self.parser.parse("what is the capital of karnataka?")
        self.assertEqual(q_lower.question_type, "capital")
        self.assertEqual(q_lower.object, "karnataka")

    def test_article_stripping_from_entities(self):
        """Verify leading 'the', 'a', 'an' are stripped from extracted entities."""
        q = self.parser.parse("Where is the Eiffel Tower situated?")
        self.assertEqual(q.subject, "Eiffel Tower")

    # --------------------------------------------------------------------------
    # 3. Empty, Malformed, and Unsupported Inputs
    # --------------------------------------------------------------------------

    def test_empty_questions(self):
        """Test empty string and whitespace-only questions."""
        q1 = self.parser.parse("")
        self.assertEqual(q1.parsing_status, "empty")
        self.assertEqual(q1.question_type, "empty")
        self.assertFalse(q1.is_valid)

        q2 = self.parser.parse("    ")
        self.assertEqual(q2.parsing_status, "empty")

    def test_malformed_inputs(self):
        """Test non-string inputs and pure punctuation."""
        q_none = self.parser.parse(None)
        self.assertEqual(q_none.parsing_status, "error")

        q_int = self.parser.parse(12345)
        self.assertEqual(q_int.parsing_status, "error")

        q_punct = self.parser.parse("??? ! .")
        self.assertEqual(q_punct.parsing_status, "error")

    def test_unsupported_questions(self):
        """
        ACADEMIC INTEGRITY CHECK:
        Ensure that subjective, generative, or unmodeled questions are classified
        as unsupported and NOT converted into arbitrary queries.
        """
        unsupported = [
            "Why is the sky blue?",
            "Tell me a story about a dragon.",
            "What will happen to the economy next year?",
            "How do I cook pasta?",
            "Should we invest in quantum computing?",
        ]
        for u in unsupported:
            q = self.parser.parse(u)
            self.assertEqual(q.parsing_status, "unsupported")
            self.assertEqual(q.question_type, "unsupported")
            self.assertFalse(q.is_valid)

    # --------------------------------------------------------------------------
    # 4. Information Retrieval from Knowledge Base Repository
    # --------------------------------------------------------------------------

    def test_retrieval_with_matching_facts(self):
        """Test successful relational retrieval of stored facts."""
        # Seed Knowledge Base
        doc_id = self.repo.add_document("geography.txt", "txt", "success")
        fact = Fact(
            document_id=doc_id,
            subject="Bangalore",
            predicate="capital of",
            object="Karnataka",
            source_sentence="Bangalore is the capital of Karnataka.",
            sentence_index=0
        )
        self.repo.add_fact(fact)

        # Parse question and retrieve
        query = parse_question("What is the capital of Karnataka?")
        result = self.retriever.retrieve(query)

        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.total_matches, 1)

        # Verify provenance preservation
        retrieved_fact = result.facts[0]
        self.assertEqual(retrieved_fact.subject, "Bangalore")
        self.assertEqual(retrieved_fact.predicate, "capital of")
        self.assertEqual(retrieved_fact.object, "Karnataka")
        self.assertEqual(retrieved_fact.document_id, doc_id)
        self.assertEqual(retrieved_fact.source_sentence, "Bangalore is the capital of Karnataka.")
        self.assertEqual(retrieved_fact.sentence_index, 0)

    def test_retrieval_with_no_matching_facts(self):
        """
        Distinguish between an unsupported question and a valid question
        that simply has no matching data in the knowledge base.
        """
        query = parse_question("Where is Honolulu located?")
        self.assertEqual(query.parsing_status, "success")

        # Knowledge Base does not have Honolulu
        result = self.retriever.retrieve(query)
        self.assertEqual(result.status, "no_match")
        self.assertEqual(result.total_matches, 0)
        self.assertEqual(len(result.facts), 0)
        self.assertIn("No matching facts", result.error_message)

    def test_retrieval_with_multiple_matching_facts(self):
        """Test retrieving multiple matching facts (e.g. multiple borders)."""
        doc_id = self.repo.add_document("borders.txt", "txt", "success")
        self.repo.add_facts([
            Fact(doc_id, "Germany", "borders", "France", "Germany borders France.", 0),
            Fact(doc_id, "Germany", "borders", "Poland", "Germany borders Poland.", 1),
            Fact(doc_id, "Germany", "borders", "Austria", "Germany borders Austria.", 2),
        ])

        query = parse_question("What does Germany border?")
        result = self.retriever.retrieve(query)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.total_matches, 3)
        objects = [f.object for f in result.facts]
        self.assertIn("France", objects)
        self.assertIn("Poland", objects)
        self.assertIn("Austria", objects)

    def test_retrieval_on_unsupported_question(self):
        """Verify retriever immediately returns 'unsupported' without executing SQL."""
        query = parse_question("Tell me a funny joke?")
        result = self.retriever.retrieve(query)

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.total_matches, 0)
        self.assertEqual(len(result.facts), 0)

    def test_process_and_retrieve_convenience_function(self):
        """Test the end-to-end convenience pipeline process_and_retrieve."""
        doc_id = self.repo.add_document("tech.txt", "txt", "success")
        self.repo.add_fact(
            Fact(doc_id, "Python", "developed by", "Guido van Rossum", "Python was developed by Guido van Rossum.", 0)
        )

        result = process_and_retrieve("Who developed Python?", repository=self.repo)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.total_matches, 1)
        self.assertEqual(result.facts[0].object, "Guido van Rossum")

    def test_identity_retrieval(self):
        """Test retrieving facts about an entity in an open identity query."""
        doc_id = self.repo.add_document("people.txt", "txt", "success")
        self.repo.add_fact(
            Fact(doc_id, "Python", "developed by", "Guido van Rossum", "Python was developed by Guido van Rossum.", 0)
        )

        result = process_and_retrieve("Who is Guido van Rossum?", repository=self.repo)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.total_matches, 1)
        self.assertEqual(result.facts[0].subject, "Python")
        self.assertEqual(result.facts[0].object, "Guido van Rossum")

    def test_serialization_to_dict(self):
        """Test serialization of StructuredQuery and RetrievalResult."""
        query = parse_question("Where is Tokyo located?")
        d_q = query.to_dict()
        self.assertIsInstance(d_q, dict)
        self.assertEqual(d_q["question_type"], "location")
        self.assertEqual(d_q["subject"], "Tokyo")

        result = RetrievalResult(query=query, facts=[], status="no_match")
        d_r = result.to_dict()
        self.assertIsInstance(d_r, dict)
        self.assertEqual(d_r["status"], "no_match")
        self.assertEqual(d_r["total_matches"], 0)

    # --------------------------------------------------------------------------
    # 5. Regression Tests for Audit Corrections (Bugs 1, 2, 3)
    # --------------------------------------------------------------------------

    def test_regression_bug_1_whitespace_normalization(self):
        """
        REGRESSION TEST FOR BUG 1:
        Verify that excessive internal/leading/trailing whitespace and space before
        punctuation are normalized cleanly without corrupting extracted entity names,
        while StructuredQuery.original_question preserves the exact input.
        """
        raw_input = "  Where   is   Tokyo   located  ?  "
        q = self.parser.parse(raw_input)

        self.assertEqual(q.original_question, raw_input)
        self.assertEqual(q.question_type, "location")
        self.assertEqual(q.subject, "Tokyo")
        self.assertEqual(q.predicate, "located in")
        self.assertEqual(q.expected_answer_field, "object")
        self.assertEqual(q.parsing_status, "success")

    def test_regression_bug_2_repeated_question_marks(self):
        """
        REGRESSION TEST FOR BUG 2:
        Verify that multiple trailing question marks ('???') are supported and
        do not cause specific patterns to fall through to generic patterns.
        """
        raw_input = "WHERE IS TOKYO LOCATED???"
        q = self.parser.parse(raw_input)

        self.assertEqual(q.original_question, raw_input)
        self.assertEqual(q.question_type, "location")
        self.assertEqual(q.subject, "TOKYO")
        self.assertEqual(q.predicate, "located in")
        self.assertEqual(q.expected_answer_field, "object")
        self.assertEqual(q.parsing_status, "success")

    def test_regression_bug_3_identity_priority_over_classification(self):
        """
        REGRESSION TEST FOR BUG 3:
        Verify that identity patterns take priority over generic classification:
        - 'What is Guido van Rossum's identity?' must parse as 'identity'
        - 'What is Python?' must remain 'classification'
        """
        # Identity query with possessive
        q_identity = self.parser.parse("What is Guido van Rossum's identity?")
        self.assertEqual(q_identity.question_type, "identity")
        self.assertEqual(q_identity.subject, "Guido van Rossum")
        self.assertIsNone(q_identity.predicate)
        self.assertEqual(q_identity.expected_answer_field, "object")
        self.assertEqual(q_identity.parsing_status, "success")

        # Identity query with 'identity of'
        q_identity_of = self.parser.parse("What is the identity of Dennis Ritchie?")
        self.assertEqual(q_identity_of.question_type, "identity")
        self.assertEqual(q_identity_of.subject, "Dennis Ritchie")

        # Classification query
        q_class = self.parser.parse("What is Python?")
        self.assertEqual(q_class.question_type, "classification")
        self.assertEqual(q_class.subject, "Python")
        self.assertEqual(q_class.predicate, "is a")
        self.assertEqual(q_class.expected_answer_field, "object")
        self.assertEqual(q_class.parsing_status, "success")

    def test_regression_multi_word_entities_preserved(self):
        """Verify that multi-word entity names with multiple spaces are preserved."""
        q1 = self.parser.parse("What is the capital of New York State?")
        self.assertEqual(q1.object, "New York State")

        q2 = self.parser.parse("When was Massachusetts Institute of Technology founded?")
        self.assertEqual(q2.subject, "Massachusetts Institute of Technology")


if __name__ == "__main__":
    unittest.main()
