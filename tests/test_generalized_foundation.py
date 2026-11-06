"""
Comprehensive Automated Test Suite: Generalized Knowledge-Based Document Question Answering
Validates the foundational capabilities introduced in the Master Foundation Engineering phase:
1. Entity Registry and Alias Resolution
2. Relationship Registry and Logical Properties
3. Document Context and Deterministic Pronoun Resolution
4. Extensible Pattern and Fact Extraction
5. Generalized Question Classification, Parsing, and Reverse Retrieval
6. End-to-End Acceptance Tests on the Mandatory Foundational Dataset (12 Questions)
7. Document Reset, Deletion, and Truth Maintenance
"""

import os
import shutil
import tempfile
import unittest

from modules.document_processor import DocumentContent
from modules.knowledge_base.entity_registry import EntityRegistry, Entity
from modules.knowledge_base.relationship_registry import RelationshipRegistry, RelationshipDefinition
from modules.knowledge_base.context_resolver import ContextResolver, DocumentContext
from modules.knowledge_base.fact_extractor import FactExtractor
from modules.knowledge_base.repository import KnowledgeBaseRepository
from modules.question_processor.classifier import QuestionClassifier
from modules.question_processor.parser import QuestionParser
from modules.question_processor.retrieval import QuestionRetriever
from modules.app_service import DocuMindService


class TestEntityRegistry(unittest.TestCase):
    """Test Entity Registry, Alias Resolution, and Type Tracking."""

    def setUp(self):
        self.registry = EntityRegistry()

    def test_canonical_registration_and_lookup(self):
        ent = self.registry.register_entity("Honnesh", entity_type="Person")
        self.assertEqual(ent.canonical_name, "Honnesh")
        self.assertEqual(ent.entity_type, "Person")
        resolved = self.registry.resolve_entity("honnesh")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.canonical_name, "Honnesh")

    def test_alias_resolution(self):
        # Initial well-known aliases include Bangalore <-> Bengaluru
        self.assertTrue(self.registry.are_aliases("Bangalore", "Bengaluru"))
        self.assertTrue(self.registry.are_aliases("USA", "United States"))
        self.assertEqual(self.registry.get_canonical_name("Bengaluru"), "Bangalore")

    def test_alias_addition_and_normalization(self):
        self.registry.register_entity("KPMG", entity_type="Organization", aliases=["KPMG LLP", "KPMG India"])
        self.assertTrue(self.registry.are_aliases("KPMG", "KPMG India"))
        self.assertTrue(self.registry.are_aliases("kpmg llp", "KPMG"))


class TestRelationshipRegistry(unittest.TestCase):
    """Test Relationship Registry, Predicate Variations, and Logical Properties."""

    def setUp(self):
        self.registry = RelationshipRegistry()

    def test_canonical_predicate_mapping(self):
        self.assertEqual(self.registry.get_canonical_predicate("resides in"), "lives in")
        self.assertEqual(self.registry.get_canonical_predicate("works for"), "works at")
        self.assertEqual(self.registry.get_canonical_predicate("student at"), "studies at")
        self.assertEqual(self.registry.get_canonical_predicate("name is"), "has name")

    def test_transitive_and_symmetric_flags(self):
        self.assertTrue(self.registry.is_transitive("located in"))
        self.assertTrue(self.registry.is_transitive("part of"))
        self.assertFalse(self.registry.is_transitive("lives in"))
        self.assertTrue(self.registry.is_symmetric("borders"))
        self.assertFalse(self.registry.is_symmetric("capital of"))

    def test_inverse_predicates(self):
        self.assertEqual(self.registry.get_inverse("works at"), "employs")
        self.assertEqual(self.registry.get_inverse("capital of"), "has capital")


class TestContextResolver(unittest.TestCase):
    """Test Document Context and Deterministic Pronoun Resolution."""

    def setUp(self):
        self.resolver = ContextResolver()

    def test_speaker_detection(self):
        name1 = self.resolver.detect_speaker_identity("My name is Honnesh.")
        self.assertEqual(name1, "Honnesh")

        name2 = self.resolver.detect_speaker_identity("I am Honnesh.")
        self.assertEqual(name2, "Honnesh")

        # Guard against non-names
        non_name = self.resolver.detect_speaker_identity("I am a student.")
        self.assertIsNone(non_name)

    def test_pronoun_resolution_with_established_speaker(self):
        ctx = DocumentContext()
        ctx.set_speaker("Honnesh")

        resolved, is_ctx = self.resolver.resolve_subject_reference("I", ctx)
        self.assertEqual(resolved, "Honnesh")
        self.assertTrue(is_ctx)

        resolved_my, _ = self.resolver.resolve_subject_reference("My college", ctx)
        self.assertEqual(resolved_my, "Honnesh")

    def test_fallback_without_speaker(self):
        ctx = DocumentContext()
        resolved, is_ctx = self.resolver.resolve_subject_reference("I", ctx)
        self.assertEqual(resolved, "Document Speaker")


class TestExtensibleFactExtractor(unittest.TestCase):
    """Test FactExtractor on the foundational dataset."""

    def setUp(self):
        self.extractor = FactExtractor()

    def test_foundational_document_extraction(self):
        doc_text = """My name is Honnesh.
I live in Bangalore.
I study at RUAS.
I am an AI and ML engineering student.
I work at KPMG.
Bangalore is the capital of Karnataka.
Karnataka is a state in India.
India is located in Asia."""

        facts = self.extractor.extract_facts(doc_text, document_id=1)
        self.assertEqual(len(facts), 8)

        triples = [f.triple for f in facts]
        self.assertIn(("Honnesh", "has name", "Honnesh"), triples)
        self.assertIn(("Honnesh", "lives in", "Bangalore"), triples)
        self.assertIn(("Honnesh", "studies at", "RUAS"), triples)
        self.assertIn(("Honnesh", "is a", "AI and ML engineering student"), triples)
        self.assertIn(("Honnesh", "works at", "KPMG"), triples)
        self.assertIn(("Bangalore", "capital of", "Karnataka"), triples)
        self.assertIn(("Karnataka", "is a state in", "India"), triples)
        self.assertIn(("India", "located in", "Asia"), triples)


class TestQuestionUnderstanding(unittest.TestCase):
    """Test Question Classifier and Parser on diverse grammatical inquiries."""

    def setUp(self):
        self.parser = QuestionParser()

    def test_parse_first_person_name_query(self):
        q = self.parser.parse("Who am I?")
        self.assertEqual(q.parsing_status, "success")
        self.assertEqual(q.question_type, "name")
        self.assertEqual(q.subject, "Speaker")
        self.assertEqual(q.predicate, "has name")
        self.assertEqual(q.expected_answer_field, "object")

    def test_parse_residence_queries(self):
        q1 = self.parser.parse("Where do I live?")
        self.assertEqual(q1.question_type, "residence")
        self.assertEqual(q1.subject, "Speaker")
        self.assertEqual(q1.predicate, "lives in")

        q2 = self.parser.parse("Where does Honnesh live?")
        self.assertEqual(q2.question_type, "residence")
        self.assertEqual(q2.subject, "Honnesh")
        self.assertEqual(q2.predicate, "lives in")

    def test_parse_reverse_employment_query(self):
        q = self.parser.parse("Who works at KPMG?")
        self.assertEqual(q.question_type, "employment")
        self.assertIsNone(q.subject)
        self.assertEqual(q.predicate, "works at")
        self.assertEqual(q.object, "KPMG")
        self.assertEqual(q.expected_answer_field, "subject")

    def test_parse_boolean_role_query(self):
        q = self.parser.parse("Am I a student?")
        self.assertEqual(q.question_type, "classification")
        self.assertEqual(q.subject, "Speaker")
        self.assertEqual(q.predicate, "is a")
        self.assertEqual(q.object, "student")
        self.assertEqual(q.expected_answer_field, "boolean")


class TestEndToEndFoundationalDataset(unittest.TestCase):
    """Full end-to-end integration test across all 12 mandatory acceptance questions."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="documind_foundation_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.db_path = os.path.join(self.test_dir, f"{self._testMethodName}.db")
        self.upload_dir = os.path.join(self.test_dir, "uploads")
        self.service = DocuMindService(db_path=self.db_path, upload_folder=self.upload_dir)

        # Ingest mandatory foundational document
        doc_text = """My name is Honnesh.
I live in Bangalore.
I study at RUAS.
I am an AI and ML engineering student.
I work at KPMG.
Bangalore is the capital of Karnataka.
Karnataka is a state in India.
India is located in Asia."""

        content = DocumentContent(
            filename="honnesh.txt",
            format="txt",
            raw_text=doc_text,
            status="success",
            page_count=1,
            char_count=len(doc_text),
            line_count=len(doc_text.splitlines())
        )
        self.doc_id, self.facts = self.service.repo.store_document_content(content)

    def test_01_who_am_i(self):
        res = self.service.answer_question("Who am I?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "direct")
        self.assertIn("Honnesh", res['answer_text'])

    def test_02_what_is_my_name(self):
        res = self.service.answer_question("What is my name?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "direct")
        self.assertEqual(res['answer_text'], "Your name is Honnesh.")

    def test_03_where_do_i_live(self):
        res = self.service.answer_question("Where do I live?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "direct")
        self.assertEqual(res['answer_text'], "You live in Bangalore.")

    def test_04_where_do_i_study(self):
        res = self.service.answer_question("Where do I study?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "direct")
        self.assertEqual(res['answer_text'], "You study at RUAS.")

    def test_05_what_am_i_studying(self):
        res = self.service.answer_question("What am I studying?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertIn("AI and ML engineering student", res['answer_text'])

    def test_06_am_i_a_student(self):
        res = self.service.answer_question("Am I a student?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertTrue(res['answer_text'].startswith("Yes,"))
        self.assertIn("AI and ML engineering student", res['answer_text'])

    def test_07_where_do_i_work(self):
        res = self.service.answer_question("Where do I work?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "direct")
        self.assertEqual(res['answer_text'], "You work at KPMG.")

    def test_08_who_works_at_kpmg(self):
        res = self.service.answer_question("Who works at KPMG?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "direct")
        self.assertEqual(res['answer_text'], "Honnesh works at KPMG.")

    def test_09_where_is_bangalore_located(self):
        res = self.service.answer_question("Where is Bangalore located?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "inferred")
        self.assertIn("R1", res['applied_rule_ids'])
        self.assertEqual(res['answer_text'], "Bangalore is located in India.")

    def test_10_which_country_does_bangalore_belong_to(self):
        res = self.service.answer_question("Which country does Bangalore belong to?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "inferred")
        self.assertEqual(res['answer_text'], "Bangalore is located in India.")

    def test_11_is_karnataka_a_state_in_india(self):
        res = self.service.answer_question("Is Karnataka a state in India?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertTrue(res['answer_text'].startswith("Yes,"))
        self.assertIn("Karnataka is a state in India", res['answer_text'])

    def test_12_which_continent_is_india_located_in(self):
        res = self.service.answer_question("Which continent is India located in?")
        self.assertEqual(res['answer_status'], "answered")
        self.assertEqual(res['answer_type'], "direct")
        self.assertEqual(res['answer_text'], "India is located in Asia.")

    def test_13_third_person_variations(self):
        res1 = self.service.answer_question("Where does Honnesh live?")
        self.assertEqual(res1['answer_text'], "Honnesh lives in Bangalore.")

        res2 = self.service.answer_question("Where does Honnesh work?")
        self.assertEqual(res2['answer_text'], "Honnesh works at KPMG.")

        res3 = self.service.answer_question("Is Honnesh a student?")
        self.assertTrue(res3['answer_text'].startswith("Yes,"))

    def test_14_unknown_and_unsupported(self):
        res_unk = self.service.answer_question("Where is Atlantis located?")
        self.assertEqual(res_unk['answer_status'], "unknown")

        res_unsupp = self.service.answer_question("Why is the sky blue?")
        self.assertEqual(res_unsupp['answer_status'], "unsupported")

    def test_15_workspace_reset_clears_contexts_and_facts(self):
        reset_res = self.service.reset_workspace(confirmation="DELETE")
        self.assertEqual(reset_res['status'], "success")
        self.assertEqual(self.service.repo.count_facts(), 0)
        self.assertIsNone(self.service.repo.get_primary_speaker())


if __name__ == '__main__':
    unittest.main()
