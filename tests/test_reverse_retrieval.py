"""
Unit and Integration Tests for Generic Reverse Relationship Retrieval
Validates reverse lookup functionality across canonical relationships,
variations, alias resolution, multiple matching entities, conflicts, and OWA.

ACADEMIC RESTRICTION COMPLIANCE:
Strictly deterministic and symbolic.
No NLP, no embeddings, no probabilistic models, no vector search.
"""

import os
import shutil
import tempfile
import unittest

from modules.document_processor import DocumentContent
from modules.knowledge_base.entity_registry import EntityRegistry
from modules.knowledge_base.relationship_registry import RelationshipRegistry, GLOBAL_RELATIONSHIP_REGISTRY
from modules.knowledge_base.models import Fact
from modules.knowledge_base.repository import KnowledgeBaseRepository
from modules.question_processor.classifier import QuestionClassifier
from modules.question_processor.parser import QuestionParser
from modules.question_processor.retrieval import QuestionRetriever
from modules.answer_generator.generator import AnswerGenerator
from modules.app_service import DocuMindService


class TestReverseRelationshipRetrieval(unittest.TestCase):
    """Comprehensive test suite for generic reverse relationship retrieval."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_reverse.db")
        self.upload_dir = os.path.join(self.test_dir, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.service = DocuMindService(db_path=self.db_path, upload_folder=self.upload_dir)

        # Ingest foundational document
        doc_text = """My name is Honnesh.
I live in Bangalore.
I work at KPMG.
I study at RUAS.
Bangalore is the capital of Karnataka."""

        content = DocumentContent(
            filename="honnesh_profile.txt",
            format="txt",
            raw_text=doc_text,
            status="success",
            page_count=1,
            char_count=len(doc_text),
            line_count=5
        )
        self.doc_id, self.facts = self.service.repo.store_document_content(content)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Single match tests for core reverse relationships
    def test_single_match_residence(self):
        res = self.service.answer_question("Who lives in Bangalore?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh lives in Bangalore.")
        self.assertGreater(len(res['supporting_fact_ids']), 0)
        self.assertGreater(len(res['sources']), 0)

    def test_single_match_employment(self):
        res = self.service.answer_question("Who works at KPMG?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh works at KPMG.")
        self.assertGreater(len(res['supporting_fact_ids']), 0)
        self.assertGreater(len(res['sources']), 0)

    def test_single_match_education(self):
        res = self.service.answer_question("Who studies at RUAS?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh studies at RUAS.")
        self.assertGreater(len(res['supporting_fact_ids']), 0)
        self.assertGreater(len(res['sources']), 0)

    def test_single_match_capital(self):
        res = self.service.answer_question("What is the capital of Karnataka?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "The capital of Karnataka is Bangalore.")
        self.assertGreater(len(res['supporting_fact_ids']), 0)

    # 2. Grammatical variations
    def test_grammatical_variation_residence(self):
        res = self.service.answer_question("Who resides in Bangalore?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh lives in Bangalore.")

    def test_grammatical_variation_education(self):
        res = self.service.answer_question("Who attends RUAS?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh studies at RUAS.")

    def test_grammatical_variation_employment(self):
        res = self.service.answer_question("Who is employed at KPMG?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh works at KPMG.")

    # 3. Alias and Ambiguity resolution
    def test_alias_resolution_residence(self):
        # "Bengaluru" is registered as an alias for "Bangalore"
        res = self.service.answer_question("Who lives in Bengaluru?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertIn("Honnesh", res['answer_text'])

    def test_alias_resolution_education(self):
        # "Ramaiah University" is registered as an alias for "RUAS"
        res = self.service.answer_question("Who studies at Ramaiah University?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertIn("Honnesh", res['answer_text'])

    # 4. Multiple matching results
    def test_multiple_matches_residence(self):
        prak_content = DocumentContent(
            filename="prakruthi.txt",
            format="txt",
            raw_text="Prakruthi lives in Bangalore.",
            status="success",
            page_count=1,
            char_count=29,
            line_count=1
        )
        self.service.repo.store_document_content(prak_content)

        res = self.service.answer_question("Who lives in Bangalore?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh and Prakruthi live in Bangalore.")
        self.assertEqual(len(res['supporting_fact_ids']), 2)

    def test_multiple_matches_employment(self):
        prak_content = DocumentContent(
            filename="prakruthi_work.txt",
            format="txt",
            raw_text="Prakruthi works at KPMG.",
            status="success",
            page_count=1,
            char_count=24,
            line_count=1
        )
        self.service.repo.store_document_content(prak_content)

        res = self.service.answer_question("Who works at KPMG?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh and Prakruthi work at KPMG.")
        self.assertEqual(len(res['supporting_fact_ids']), 2)

    def test_multiple_matches_education(self):
        prak_content = DocumentContent(
            filename="prakruthi_study.txt",
            format="txt",
            raw_text="Prakruthi studies at RUAS.",
            status="success",
            page_count=1,
            char_count=26,
            line_count=1
        )
        self.service.repo.store_document_content(prak_content)

        res = self.service.answer_question("Who studies at RUAS?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_text'], "Honnesh and Prakruthi study at RUAS.")
        self.assertEqual(len(res['supporting_fact_ids']), 2)

    # 5. Open-World Assumption (Unknown answer)
    def test_unknown_location_residence(self):
        res = self.service.answer_question("Who lives in Atlantis?")
        self.assertEqual(res['answer_status'], "unknown")
        self.assertEqual(res['answer_text'], "I could not find sufficient information in the knowledge base to answer this question.")

    def test_unknown_institution_education(self):
        res = self.service.answer_question("Who studies at Hogwarts?")
        self.assertEqual(res['answer_status'], "unknown")
        self.assertEqual(res['answer_status'], "unknown")

    # 6. Unsupported reverse relationship
    def test_unsupported_reverse_relationship_established(self):
        res = self.service.answer_question("Who was established in 1885?")
        self.assertEqual(res['answer_status'], "unsupported")
        self.assertIn("not currently supported", res['answer_text'])

    def test_unsupported_reverse_relationship_active_form(self):
        res = self.service.answer_question("Who established in 1885?")
        self.assertEqual(res['answer_status'], "unsupported")
        self.assertIn("not currently supported", res['answer_text'])

    def test_unsupported_unrecognized_predicate(self):
        res = self.service.answer_question("Who dances at KPMG?")
        self.assertEqual(res['answer_status'], "unsupported")
        self.assertIn("not currently supported", res['answer_text'])

    # 7. Conflict handling for functional reverse relationships
    def test_conflict_handling_capital(self):
        # Insert conflicting capital fact
        conflict_content = DocumentContent(
            filename="conflict_capital.txt",
            format="txt",
            raw_text="Mysore is the capital of Karnataka.",
            status="success",
            page_count=1,
            char_count=35,
            line_count=1
        )
        self.service.repo.store_document_content(conflict_content)

        res = self.service.answer_question("What is the capital of Karnataka?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertIn("Conflicting information found in the knowledge base:", res['answer_text'])
        self.assertIn("Bangalore", res['answer_text'])
        self.assertIn("Mysore", res['answer_text'])

    # 8. Provenance and trace integrity
    def test_reverse_provenance_and_sources(self):
        res = self.service.answer_question("Who lives in Bangalore?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertIn("Supporting Ground Fact:", res['explanation'])
        self.assertTrue(len(res['sources']) > 0)
        self.assertEqual(res['sources'][0]['document_id'], self.doc_id)
        self.assertEqual(res['sources'][0]['source_sentence'], "I live in Bangalore.")


if __name__ == "__main__":
    unittest.main()
