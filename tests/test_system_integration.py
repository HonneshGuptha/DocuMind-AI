"""
End-to-End System Integration Tests for DocuMind AI.
Verifies cross-module workflows across Modules 1, 2, 3, and 4:
- Scenario A: PDF processing (PDF -> DocumentContent -> Knowledge Base -> Fact retrieval)
- Scenario B: DOCX processing (DOCX -> DocumentContent -> Knowledge Base -> Fact retrieval)
- Scenario C: TXT processing (TXT -> DocumentContent -> Knowledge Base -> Fact retrieval)
- Scenario D: Direct question answering pipeline (Question -> Classification -> Parsing -> Fact retrieval)
- Scenario E: Forward reasoning (R1: Capital Location)
- Scenario F: Multi-step forward reasoning (R2: Transitive containment chain)
- Scenario G: Backward reasoning (Goal proof via R1)
- Scenario H: Unsupported question handling (Open-domain questions safely rejected)
- Scenario I: Missing knowledge handling (Unknown/unproven rather than false)
- Scenario J: Data isolation (DB immutability, transient WorkingMemory, zero leakage)
"""

import os
import shutil
import tempfile
import unittest

import fitz  # PyMuPDF
import docx  # python-docx

from modules.document_processor.extractor import (
    PDFExtractor,
    DOCXExtractor,
    TXTExtractor,
    extract_document,
)
from modules.knowledge_base.database import init_db
from modules.knowledge_base.fact_extractor import FactExtractor
from modules.knowledge_base.repository import KnowledgeBaseRepository
from modules.question_processor.classifier import QuestionClassifier
from modules.question_processor.parser import QuestionParser
from modules.question_processor.retrieval import QuestionRetriever
from modules.reasoning_engine.reasoning_models import Fact as ReasoningFact
from modules.reasoning_engine.working_memory import WorkingMemory
from modules.reasoning_engine.rule_base import R1, R2, R3, R4, get_all_rules
from modules.reasoning_engine.forward_chaining import ForwardChainer, forward_chain
from modules.reasoning_engine.backward_chaining import BackwardChainer, backward_chain


class TestSystemIntegration(unittest.TestCase):
    """Full cross-module end-to-end integration test suite."""

    def setUp(self):
        # Create an isolated temporary directory for documents and database
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_documind.db")
        init_db(self.db_path)
        self.repo = KnowledgeBaseRepository(self.db_path)
        self.fact_extractor = FactExtractor()
        self.classifier = QuestionClassifier()
        self.parser = QuestionParser()
        self.retriever = QuestionRetriever(self.repo)

    def tearDown(self):
        # Clean up temporary test files and database
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Scenario A: PDF Processing -> KB -> Retrieval
    # --------------------------------------------------------------------------
    def test_scenario_a_pdf_pipeline(self):
        """PDF -> DocumentContent -> Knowledge Base -> Fact retrieval."""
        pdf_path = os.path.join(self.test_dir, "geography.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "Tokyo is the capital of Japan. Japan borders South Korea.")
        doc.save(pdf_path)
        doc.close()

        # 1. Module 1: Extract
        extracted = extract_document(pdf_path)
        self.assertEqual(extracted.status, "success")
        self.assertIn("Tokyo is the capital of Japan.", extracted.raw_text)

        # 2. Module 2: Ingest via store_document_content
        doc_id, facts = self.repo.store_document_content(extracted)
        self.assertIsNotNone(doc_id)
        self.assertTrue(len(facts) >= 1)

        # 3. Module 3: Retrieve
        query = self.parser.parse("What is the capital of Japan?")
        self.assertEqual(query.parsing_status, "success")
        retrieval = self.retriever.retrieve(query)

        self.assertEqual(retrieval.status, "success")
        self.assertEqual(len(retrieval.facts), 1)
        self.assertEqual(retrieval.facts[0].subject, "Tokyo")
        self.assertEqual(retrieval.facts[0].object, "Japan")

    # --------------------------------------------------------------------------
    # Scenario B: DOCX Processing -> KB -> Retrieval
    # --------------------------------------------------------------------------
    def test_scenario_b_docx_pipeline(self):
        """DOCX -> DocumentContent -> Knowledge Base -> Fact retrieval."""
        docx_path = os.path.join(self.test_dir, "history.docx")
        doc = docx.Document()
        doc.add_paragraph("Stanford University was established in 1885.")
        doc.save(docx_path)

        # 1. Module 1: Extract
        extracted = extract_document(docx_path)
        self.assertEqual(extracted.status, "success")

        # 2. Module 2: Ingest
        doc_id, facts = self.repo.store_document_content(extracted)
        self.assertIsNotNone(doc_id)
        self.assertEqual(len(facts), 1)

        # 3. Module 3: Retrieve
        query = self.parser.parse("When was Stanford University established?")
        retrieval = self.retriever.retrieve(query)

        self.assertEqual(retrieval.status, "success")
        self.assertEqual(len(retrieval.facts), 1)
        self.assertEqual(retrieval.facts[0].subject, "Stanford University")
        self.assertEqual(retrieval.facts[0].object, "1885")

    # --------------------------------------------------------------------------
    # Scenario C: TXT Processing -> KB -> Retrieval
    # --------------------------------------------------------------------------
    def test_scenario_c_txt_pipeline(self):
        """TXT -> DocumentContent -> Knowledge Base -> Fact retrieval."""
        txt_path = os.path.join(self.test_dir, "tech.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Python was developed by Guido van Rossum.\n")

        # 1. Module 1: Extract
        extracted = extract_document(txt_path)
        self.assertEqual(extracted.status, "success")

        # 2. Module 2: Ingest
        doc_id, facts = self.repo.store_document_content(extracted)
        self.assertIsNotNone(doc_id)
        self.assertEqual(len(facts), 1)

        # 3. Module 3: Retrieve
        query = self.parser.parse("Who developed Python?")
        retrieval = self.retriever.retrieve(query)

        self.assertEqual(retrieval.status, "success")
        self.assertEqual(len(retrieval.facts), 1)
        self.assertEqual(retrieval.facts[0].subject, "Python")
        self.assertEqual(retrieval.facts[0].object, "Guido van Rossum")

    # --------------------------------------------------------------------------
    # Scenario D: Direct Question Answering Pipeline
    # --------------------------------------------------------------------------
    def test_scenario_d_question_processing_pipeline(self):
        """Question -> Classification -> Parsing -> Fact retrieval."""
        # Insert known fact into KB
        txt_path = os.path.join(self.test_dir, "data.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Canada borders United States.\n")

        extracted = extract_document(txt_path)
        self.repo.store_document_content(extracted)

        # Classification check
        q_type = self.classifier.classify("Does Canada border United States?")
        self.assertEqual(q_type, "border")

        # Parsing check
        parsed_query = self.parser.parse("Does Canada border United States?")
        self.assertEqual(parsed_query.parsing_status, "success")
        self.assertEqual(parsed_query.subject, "Canada")
        self.assertEqual(parsed_query.object, "United States")
        self.assertEqual(parsed_query.expected_answer_field, "boolean")

        # Retrieval check
        result = self.retriever.retrieve(parsed_query)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.facts), 1)

    # --------------------------------------------------------------------------
    # Scenario E: Forward Reasoning (R1: Capital Location)
    # --------------------------------------------------------------------------
    def test_scenario_e_forward_reasoning_r1(self):
        """Input: Bangalore is capital of Karnataka. Karnataka is a state in India. -> Bangalore located in India."""
        txt_path = os.path.join(self.test_dir, "india.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Bangalore is the capital of Karnataka. Karnataka is a state in India.\n")

        extracted = extract_document(txt_path)
        self.repo.store_document_content(extracted)

        # Load from KB into Module 4 WorkingMemory
        wm = WorkingMemory()
        kb_facts = self.repo.get_all_facts()
        wm.load_from_kb_facts(kb_facts)

        # Forward Chain
        chainer = ForwardChainer(rules=[R1])
        res = chainer.run(wm)

        self.assertEqual(res.status, "success")
        triples = [f.triple for f in res.inferred_facts]
        self.assertIn(("Bangalore", "located in", "India"), triples)

    # --------------------------------------------------------------------------
    # Scenario F: Multi-step Forward Reasoning (R2: Transitivity)
    # --------------------------------------------------------------------------
    def test_scenario_f_multi_step_forward_reasoning(self):
        """Input: Bangalore in Karnataka, Karnataka in India, India in Asia."""
        wm = WorkingMemory()
        wm.add_original_fact("Bangalore", "located in", "Karnataka", fact_id=1)
        wm.add_original_fact("Karnataka", "located in", "India", fact_id=2)
        wm.add_original_fact("India", "located in", "Asia", fact_id=3)

        res = forward_chain(wm, rules=[R2])

        self.assertEqual(res.status, "success")
        triples = [f.triple for f in res.inferred_facts]
        self.assertIn(("Bangalore", "located in", "India"), triples)
        self.assertIn(("Bangalore", "located in", "Asia"), triples)
        self.assertIn(("Karnataka", "located in", "Asia"), triples)

    # --------------------------------------------------------------------------
    # Scenario G: Backward Reasoning (Goal Proof via R1)
    # --------------------------------------------------------------------------
    def test_scenario_g_backward_reasoning_proof(self):
        """Goal: Bangalore located in India -> Proved via R1."""
        wm = WorkingMemory()
        wm.add_original_fact("Bangalore", "capital of", "Karnataka", fact_id=1)
        wm.add_original_fact("Karnataka", "is a state in", "India", fact_id=2)

        proof = backward_chain(("Bangalore", "located in", "India"), wm, rules=[R1])

        self.assertTrue(proof.is_proven)
        self.assertEqual(proof.status, "proven")
        self.assertIn("R1", proof.applied_rule_ids)
        self.assertEqual(set(proof.supporting_fact_ids), {1, 2})

    # --------------------------------------------------------------------------
    # Scenario H: Unsupported Question Handling
    # --------------------------------------------------------------------------
    def test_scenario_h_unsupported_question(self):
        """Open-domain question 'Why is the sky blue?' must report unsupported without hallucinating."""
        q_type = self.classifier.classify("Why is the sky blue?")
        self.assertIsNone(q_type)

        parsed = self.parser.parse("Why is the sky blue?")
        self.assertEqual(parsed.parsing_status, "unsupported")

        retrieval = self.retriever.retrieve(parsed)
        self.assertEqual(retrieval.status, "unsupported")
        self.assertEqual(len(retrieval.facts), 0)

    # --------------------------------------------------------------------------
    # Scenario I: Missing Knowledge Handling (Open-World Assumption)
    # --------------------------------------------------------------------------
    def test_scenario_i_missing_knowledge_reports_unknown(self):
        """Question about unrecorded knowledge reports no_match / unproven, never false."""
        # 1. Retrieval
        query = self.parser.parse("Where is Atlantis located?")
        self.assertEqual(query.parsing_status, "success")
        retrieval = self.retriever.retrieve(query)
        self.assertEqual(retrieval.status, "no_match")
        self.assertEqual(len(retrieval.facts), 0)

        # 2. Backward reasoning
        wm = WorkingMemory()
        proof = backward_chain(("Atlantis", "located in", "Ocean"), wm, rules=get_all_rules())
        self.assertFalse(proof.is_proven)
        self.assertEqual(proof.status, "unproven")

    # --------------------------------------------------------------------------
    # Scenario J: Data Isolation and Database Immutability
    # --------------------------------------------------------------------------
    def test_scenario_j_data_isolation(self):
        """Verify reasoning does not mutate SQLite DB, and backward chaining does not mutate WorkingMemory."""
        txt_path = os.path.join(self.test_dir, "isolation.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("France borders Germany.\n")

        extracted = extract_document(txt_path)
        self.repo.store_document_content(extracted)

        db_facts_before = self.repo.count_facts()
        self.assertEqual(db_facts_before, 1)

        # Load into WorkingMemory
        wm = WorkingMemory()
        wm.load_from_kb_facts(self.repo.get_all_facts())
        self.assertEqual(wm.count_original(), 1)
        self.assertEqual(wm.count_inferred(), 0)

        # 1. Run Backward Chaining
        backward_chain(("Germany", "borders", "France"), wm, rules=[R4])

        # Verify WorkingMemory was NOT modified by backward chaining
        self.assertEqual(wm.count_original(), 1)
        self.assertEqual(wm.count_inferred(), 0)

        # 2. Run Forward Chaining
        fwd_res = forward_chain(wm, rules=[R4])
        self.assertEqual(fwd_res.status, "success")
        self.assertEqual(wm.count_inferred(), 1)

        # 3. Verify SQLite DB was NOT written to by forward chaining
        db_facts_after = self.repo.count_facts()
        self.assertEqual(db_facts_before, db_facts_after)
        self.assertEqual(db_facts_after, 1)


if __name__ == '__main__':
    unittest.main()
