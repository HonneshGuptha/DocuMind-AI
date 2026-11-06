"""
Unit tests for Module 2: Knowledge Base Subsystem.
Tests deterministic fact extraction, Subject-Predicate-Object (SPO) representation,
SQLite persistence, parameterized search, duplicate handling, and source attribution.
"""

import os
import shutil
import tempfile
import unittest

from modules.knowledge_base import (
    Fact,
    DocumentRecord,
    FactExtractor,
    FactPattern,
    FACTUAL_PATTERNS,
    split_sentences,
    KnowledgeBaseRepository,
    init_db
)
from modules.document_processor import DocumentContent


class TestKnowledgeBase(unittest.TestCase):
    """Test suite for Knowledge Base models, deterministic extraction, and SQLite repository."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary sandbox directory for test SQLite database."""
        cls.test_dir = tempfile.mkdtemp(prefix="documind_kb_test_")

    @classmethod
    def tearDownClass(cls):
        """Remove temporary directory after all tests complete."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        """Initialize a fresh database for each test."""
        self.db_path = os.path.join(self.test_dir, f"test_kb_{self._testMethodName}.db")
        self.repo = KnowledgeBaseRepository(self.db_path)
        self.extractor = FactExtractor()

    # --------------------------------------------------------------------------
    # 1. Deterministic Pattern Matching & SPO Extraction Tests
    # --------------------------------------------------------------------------

    def test_pattern_capital_of(self):
        """Test extraction of 'capital of' relationship."""
        text = "Bangalore is the capital of Karnataka. Paris is the capital of France."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        f1 = facts[0]
        self.assertEqual(f1.subject, "Bangalore")
        self.assertEqual(f1.predicate, "capital of")
        self.assertEqual(f1.object, "Karnataka")
        self.assertEqual(f1.source_sentence, "Bangalore is the capital of Karnataka.")
        self.assertEqual(f1.sentence_index, 0)

        f2 = facts[1]
        self.assertEqual(f2.subject, "Paris")
        self.assertEqual(f2.predicate, "capital of")
        self.assertEqual(f2.object, "France")

    def test_pattern_located_in(self):
        """Test extraction of 'located in' and 'situated in' relationships."""
        text = "Tokyo is located in Japan. The Eiffel Tower is situated in Paris."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].triple, ("Tokyo", "located in", "Japan"))
        self.assertEqual(facts[1].triple, ("Eiffel Tower", "located in", "Paris"))

    def test_pattern_region_in(self):
        """Test extraction of administrative subdivisions ('is a state in', 'is a city in')."""
        text = "Karnataka is a state in India. Austin is a city in Texas."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].triple, ("Karnataka", "is a state in", "India"))
        self.assertEqual(facts[1].triple, ("Austin", "is a city in", "Texas"))

    def test_pattern_established_in(self):
        """Test extraction of founding year / inception facts."""
        text = "Harvard University was founded in 1636. Stanford was established in 1885."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].triple, ("Harvard University", "founded in", "1636"))
        self.assertEqual(facts[1].triple, ("Stanford", "established in", "1885"))

    def test_pattern_created_by(self):
        """Test extraction of creator/developer relationships."""
        text = "Python was developed by Guido van Rossum. C was authored by Dennis Ritchie."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].triple, ("Python", "developed by", "Guido van Rossum"))
        self.assertEqual(facts[1].triple, ("C", "authored by", "Dennis Ritchie"))

    def test_pattern_is_a(self):
        """Test extraction of taxonomic classification facts."""
        text = "Python is a programming language. Oxygen is an element."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].triple, ("Python", "is a", "programming language"))
        self.assertEqual(facts[1].triple, ("Oxygen", "is a", "element"))

    def test_pattern_part_of(self):
        """Test extraction of part-whole relationships."""
        text = "CPU is part of Computer. Engine is a part of Vehicle."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].triple, ("CPU", "part of", "Computer"))
        self.assertEqual(facts[1].triple, ("Engine", "part of", "Vehicle"))

    def test_pattern_borders_and_produces(self):
        """Test extraction of borders and produces relationships."""
        text = "Canada borders United States. Brazil produces coffee."
        facts = self.extractor.extract_facts(text, document_id=1)

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].triple, ("Canada", "borders", "United States"))
        self.assertEqual(facts[1].triple, ("Brazil", "produces", "coffee"))

    # --------------------------------------------------------------------------
    # 2. Academic Restriction Compliance: Strict Filtering of Unsupported Text
    # --------------------------------------------------------------------------

    def test_unsupported_sentences_strictly_skipped(self):
        """
        ACADEMIC INTEGRITY CHECK:
        Ensure that unsupported sentences, opinions, or general text are NOT converted
        into false/hallucinated facts.
        """
        unsupported_text = (
            "The weather in California is very pleasant today. "
            "Tomorrow might be rainy or windy. "
            "I believe that classical AI is fascinating. "
            "Neural networks often require large quantities of data."
        )
        facts = self.extractor.extract_facts(unsupported_text, document_id=1)

        # Must skip all 4 unsupported sentences and return empty list
        self.assertEqual(len(facts), 0)

    def test_empty_and_whitespace_documents(self):
        """Test extraction from empty or whitespace-only inputs."""
        self.assertEqual(self.extractor.extract_facts("", document_id=1), [])
        self.assertEqual(self.extractor.extract_facts("   \n\t  ", document_id=1), [])

    def test_sentence_splitter_preserves_abbreviations(self):
        """Test that common abbreviations do not trigger premature sentence splits."""
        text = "Dr. Smith visited the U.S. in 2020. Python is a programming language."
        sentences = split_sentences(text)

        self.assertEqual(len(sentences), 2)
        self.assertIn("Dr. Smith visited the U.S. in 2020.", sentences[0][1])
        self.assertIn("Python is a programming language.", sentences[1][1])

    # --------------------------------------------------------------------------
    # 3. SQLite Repository & Persistence Operations
    # --------------------------------------------------------------------------

    def test_add_document_and_counts(self):
        """Test recording document metadata and counting stored documents."""
        doc_id1 = self.repo.add_document("doc1.pdf", "pdf", "success")
        doc_id2 = self.repo.add_document("doc2.docx", "docx", "success")

        self.assertEqual(doc_id1, 1)
        self.assertEqual(doc_id2, 2)
        self.assertEqual(self.repo.count_documents(), 2)

        doc_record = self.repo.get_document(doc_id1)
        self.assertIsNotNone(doc_record)
        self.assertEqual(doc_record.filename, "doc1.pdf")
        self.assertEqual(doc_record.file_format, "pdf")
        self.assertEqual(doc_record.extraction_status, "success")

    def test_duplicate_facts_within_same_document_ignored(self):
        """
        CANONICAL FIRST-MENTION TEST (Direct insertion):
        Test that direct repository insertion of identical SPO facts with different sentence indices
        within the same document safely retains only the first occurrence.
        """
        doc_id = self.repo.add_document("state_info.txt", "txt", "success")
        fact1 = Fact(
            document_id=doc_id,
            subject="Bangalore",
            predicate="capital of",
            object="Karnataka",
            source_sentence="Bangalore is the capital of Karnataka.",
            sentence_index=0
        )
        # Same SPO in same document at sentence index 5 with different sentence phrasing
        fact2 = Fact(
            document_id=doc_id,
            subject="Bangalore",
            predicate="capital of",
            object="Karnataka",
            source_sentence="As stated earlier, Bangalore is the capital of Karnataka.",
            sentence_index=5
        )

        id1 = self.repo.add_fact(fact1)
        id2 = self.repo.add_fact(fact2)

        self.assertIsNotNone(id1)
        self.assertIsNone(id2)  # Duplicate within same document must be ignored
        self.assertEqual(self.repo.count_facts(), 1)

        # Confirm provenance preserved from the first mention
        stored_facts = self.repo.get_facts_by_document(doc_id)
        self.assertEqual(len(stored_facts), 1)
        self.assertEqual(stored_facts[0].sentence_index, 0)
        self.assertEqual(stored_facts[0].source_sentence, "Bangalore is the capital of Karnataka.")

    def test_repeated_spo_in_one_document_via_extractor(self):
        """
        CANONICAL FIRST-MENTION TEST (Text Extraction):
        Test that when the exact same SPO proposition appears multiple times in raw text
        at different sentence positions, only the first occurrence is extracted and stored.
        """
        text = (
            "Bangalore is the capital of Karnataka. "
            "Python is a programming language. "
            "In summary, Bangalore is the capital of Karnataka."
        )
        doc_id, facts = self.repo.store_document_content(
            DocumentContent("city.txt", "txt", text, None, len(text), 3, "success")
        )

        # Exactly 2 unique facts: (Bangalore, capital of, Karnataka) and (Python, is a, programming language)
        self.assertEqual(len(facts), 2)
        self.assertEqual(self.repo.count_facts(), 2)

        # Check Bangalore provenance
        b_facts = self.repo.search_by_subject("Bangalore")
        self.assertEqual(len(b_facts), 1)
        self.assertEqual(b_facts[0].sentence_index, 0)
        self.assertEqual(b_facts[0].source_sentence, "Bangalore is the capital of Karnataka.")

    def test_case_insensitive_duplicate_handling_in_repository(self):
        """
        Test that direct repository insertion enforces case-insensitive uniqueness
        for subject, predicate, and object within the same document.
        """
        doc_id = self.repo.add_document("case_test.txt", "txt", "success")
        fact_upper = Fact(doc_id, "TOKYO", "LOCATED IN", "JAPAN", "TOKYO is LOCATED IN JAPAN.", 0)
        fact_lower = Fact(doc_id, "tokyo", "located in", "japan", "tokyo is located in japan.", 2)

        id1 = self.repo.add_fact(fact_upper)
        id2 = self.repo.add_fact(fact_lower)

        self.assertIsNotNone(id1)
        self.assertIsNone(id2)  # Case-insensitive duplicate must be rejected
        self.assertEqual(self.repo.count_facts(), 1)

        stored = self.repo.get_facts_by_document(doc_id)
        self.assertEqual(stored[0].subject, "TOKYO")
        self.assertEqual(stored[0].source_sentence, "TOKYO is LOCATED IN JAPAN.")

    def test_batch_insertion_deduplication(self):
        """Test that add_facts deduplicates repeated items in the list."""
        doc_id = self.repo.add_document("batch.txt", "txt", "success")
        facts = [
            Fact(doc_id, "Paris", "capital of", "France", "Sentence 1", 0),
            Fact(doc_id, "Paris", "capital of", "France", "Sentence 2", 3),
            Fact(doc_id, "France", "is a", "country", "Sentence 3", 4),
        ]
        inserted = self.repo.add_facts(facts)
        self.assertEqual(inserted, 2)
        self.assertEqual(self.repo.count_facts(), 2)

    def test_schema_safety_on_existing_database(self):
        """
        Test that init_db handles pre-existing database tables gracefully,
        ensuring the unique index uq_facts_doc_spo is safely established.
        """
        # Call init_db again on the existing database
        init_db(self.db_path)
        # Should still work seamlessly without exceptions
        self.assertEqual(self.repo.count_documents(), 0)

    def test_same_fact_from_multiple_documents_retained(self):
        """
        Verify that the exact same factual assertion appearing in two different documents
        retains separate records with its own document_id and source_sentence attribution.
        """
        doc_id1 = self.repo.add_document("source_alpha.txt", "txt", "success")
        doc_id2 = self.repo.add_document("source_beta.pdf", "pdf", "success")

        fact_doc1 = Fact(
            document_id=doc_id1,
            subject="Bangalore",
            predicate="capital of",
            object="Karnataka",
            source_sentence="Bangalore is the capital of Karnataka.",
            sentence_index=0
        )
        fact_doc2 = Fact(
            document_id=doc_id2,
            subject="Bangalore",
            predicate="capital of",
            object="Karnataka",
            source_sentence="Bangalore is the capital of Karnataka.",
            sentence_index=4
        )

        id1 = self.repo.add_fact(fact_doc1)
        id2 = self.repo.add_fact(fact_doc2)

        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        self.assertNotEqual(id1, id2)
        self.assertEqual(self.repo.count_facts(), 2)

        # Confirm distinct document associations
        facts_doc1 = self.repo.get_facts_by_document(doc_id1)
        facts_doc2 = self.repo.get_facts_by_document(doc_id2)
        self.assertEqual(len(facts_doc1), 1)
        self.assertEqual(len(facts_doc2), 1)
        self.assertEqual(facts_doc1[0].document_id, doc_id1)
        self.assertEqual(facts_doc2[0].document_id, doc_id2)

    def test_searches_by_subject_predicate_object(self):
        """Test searching facts by subject, predicate, and object (exact and partial)."""
        doc_id = self.repo.add_document("knowledge.txt", "txt", "success")
        facts = [
            Fact(doc_id, "Bangalore", "capital of", "Karnataka", "Bangalore is the capital of Karnataka.", 0),
            Fact(doc_id, "Karnataka", "is a state in", "India", "Karnataka is a state in India.", 1),
            Fact(doc_id, "Tokyo", "located in", "Japan", "Tokyo is located in Japan.", 2),
            Fact(doc_id, "Japan", "borders", "Russia", "Japan borders Russia.", 3),
        ]
        self.repo.add_facts(facts)

        # Subject search (case-insensitive substring)
        subj_results = self.repo.search_by_subject("bangalore")
        self.assertEqual(len(subj_results), 1)
        self.assertEqual(subj_results[0].subject, "Bangalore")

        # Predicate search
        pred_results = self.repo.search_by_predicate("capital of")
        self.assertEqual(len(pred_results), 1)
        self.assertEqual(pred_results[0].object, "Karnataka")

        # Object search
        obj_results = self.repo.search_by_object("India")
        self.assertEqual(len(obj_results), 1)
        self.assertEqual(obj_results[0].subject, "Karnataka")

        # Non-matching search
        empty_results = self.repo.search_by_subject("NonExistentCity")
        self.assertEqual(len(empty_results), 0)

    def test_combined_multi_field_search(self):
        """Test searching facts by combination of subject and predicate."""
        doc_id = self.repo.add_document("multi.txt", "txt", "success")
        self.repo.add_facts([
            Fact(doc_id, "Python", "is a", "programming language", "Python is a programming language.", 0),
            Fact(doc_id, "Python", "developed by", "Guido van Rossum", "Python was developed by Guido van Rossum.", 1),
            Fact(doc_id, "Java", "is a", "programming language", "Java is a programming language.", 2),
        ])

        # Find what Python is a
        results = self.repo.search_facts(subject="Python", predicate="is a")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].object, "programming language")

        # Find all programming languages
        results = self.repo.search_facts(predicate="is a", object="programming language")
        self.assertEqual(len(results), 2)
        subjects = [r.subject for r in results]
        self.assertIn("Python", subjects)
        self.assertIn("Java", subjects)

    def test_database_persistence_across_connections(self):
        """Test that data persisted in SQLite survives closing and reopening connection."""
        doc_id = self.repo.add_document("persisted.txt", "txt", "success")
        self.repo.add_fact(
            Fact(doc_id, "Rome", "capital of", "Italy", "Rome is the capital of Italy.", 0)
        )

        # Create a fresh repository instance pointing to the exact same file
        reopened_repo = KnowledgeBaseRepository(self.db_path)
        self.assertEqual(reopened_repo.count_documents(), 1)
        self.assertEqual(reopened_repo.count_facts(), 1)

        facts = reopened_repo.search_by_subject("Rome")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].triple, ("Rome", "capital of", "Italy"))
        self.assertEqual(facts[0].source_sentence, "Rome is the capital of Italy.")

    def test_integration_with_document_content(self):
        """
        Verify seamless end-to-end integration between Module 1 DocumentContent
        and Module 2 KnowledgeBaseRepository.
        """
        content = DocumentContent(
            filename="geography.txt",
            format="txt",
            raw_text="Bangalore is the capital of Karnataka.\nKarnataka is a state in India.\nUnsupported sentence.",
            page_count=None,
            char_count=87,
            line_count=3,
            status="success"
        )

        doc_id, facts = self.repo.store_document_content(content)

        self.assertEqual(doc_id, 1)
        # Should extract 2 valid facts, skipping the unsupported sentence
        self.assertEqual(len(facts), 2)
        self.assertEqual(self.repo.count_facts(), 2)

        stored_facts = self.repo.get_facts_by_document(doc_id)
        self.assertEqual(len(stored_facts), 2)
        self.assertEqual(stored_facts[0].subject, "Bangalore")
        self.assertEqual(stored_facts[1].subject, "Karnataka")

    def test_model_to_dict_and_repr(self):
        """Test serialization of Fact and DocumentRecord models."""
        fact = Fact(
            document_id=1,
            subject="A",
            predicate="B",
            object="C",
            source_sentence="A is B of C.",
            sentence_index=0,
            fact_id=42
        )
        d = fact.to_dict()
        self.assertEqual(d["fact_id"], 42)
        self.assertEqual(d["subject"], "A")
        self.assertEqual(fact.triple, ("A", "B", "C"))
        self.assertIn("Fact #42", repr(fact))

    def test_invalid_or_incomplete_fact_handling(self):
        """Test rejection of invalid facts (e.g., non-existent foreign key document_id)."""
        import sqlite3
        invalid_fact = Fact(
            document_id=99999,  # Non-existent foreign key document_id
            subject="Mars",
            predicate="is a",
            object="planet",
            source_sentence="Mars is a planet.",
            sentence_index=0
        )
        # SQLite foreign key constraint will raise IntegrityError
        with self.assertRaises(sqlite3.IntegrityError):
            self.repo.add_fact(invalid_fact)

    def test_pagination_and_all_documents(self):
        """Test retrieving all documents and fact pagination."""
        doc_id = self.repo.add_document("sample_doc.txt", "txt", "success")
        for i in range(5):
            self.repo.add_fact(Fact(
                document_id=doc_id,
                subject=f"Entity{i}",
                predicate="is a",
                object="test",
                source_sentence=f"Entity{i} is a test.",
                sentence_index=i
            ))

        self.assertEqual(len(self.repo.get_all_documents()), 1)
        self.assertEqual(len(self.repo.get_all_facts(limit=3, offset=0)), 3)
        self.assertEqual(len(self.repo.get_all_facts(limit=3, offset=3)), 2)


if __name__ == "__main__":
    unittest.main()
