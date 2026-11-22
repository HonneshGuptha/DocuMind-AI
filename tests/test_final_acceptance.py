"""
Final Acceptance Tests for DocuMind AI
Validates question grammar generalization, multi-hop location reasoning,
and the complete set of 15 acceptance questions against the test document.
"""

import unittest
import tempfile
import os
import io
from werkzeug.datastructures import FileStorage

from modules.app_service import DocuMindService
from modules.reasoning_engine.working_memory import WorkingMemory
from modules.reasoning_engine.forward_chaining import forward_chain
from modules.reasoning_engine.backward_chaining import backward_chain
from modules.reasoning_engine.rule_base import get_extended_rules, R5, R6, R8, R10
from modules.knowledge_base.fact_extractor import FactExtractor


class TestFinalAcceptance(unittest.TestCase):
    """End-to-end acceptance tests for the final project freeze."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_acceptance.db")
        self.svc = DocuMindService(db_path=self.db_path, upload_folder=self.tmpdir)

        self.doc_text = (
            "My name is Arjun.\n"
            "I live in Mysore.\n"
            "I study at ABC University.\n"
            "I am a computer science student.\n"
            "I work at Infosys.\n"
            "Mysore is a city in Karnataka.\n"
            "Karnataka is a state in India.\n"
            "India is located in Asia."
        )

        fs = FileStorage(
            stream=io.BytesIO(self.doc_text.encode("utf-8")),
            filename="profile.txt",
            content_type="text/plain"
        )
        res = self.svc.process_uploaded_file(fs)
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res.get("facts_extracted"), 8)

    def test_01_question_grammar_generalisation(self):
        """Verify natural question structures retrieve facts correctly."""
        # 1. What am I?
        ans1 = self.svc.answer_question("What am I?")
        self.assertEqual(ans1["answer_status"], "answered")
        self.assertEqual(ans1["answer_text"], "You are a computer science student.")

        # 2. What student am I?
        ans2 = self.svc.answer_question("What student am I?")
        self.assertEqual(ans2["answer_status"], "answered")
        self.assertEqual(ans2["answer_text"], "You are a computer science student.")

        # 3. What is my identity?
        ans3 = self.svc.answer_question("What is my identity?")
        self.assertEqual(ans3["answer_status"], "answered")
        self.assertEqual(ans3["answer_text"], "Your name is Arjun.")

        # 4. What is Arjun?
        ans4 = self.svc.answer_question("What is Arjun?")
        self.assertEqual(ans4["answer_status"], "answered")
        self.assertEqual(ans4["answer_text"], "Arjun is a computer science student.")

    def test_02_multihop_location_reasoning(self):
        """Verify transitive multi-hop geographic queries at immediate, country, and continent levels."""
        # 1. Where is Mysore located? (Immediate)
        ans1 = self.svc.answer_question("Where is Mysore located?")
        self.assertEqual(ans1["answer_status"], "answered")
        self.assertEqual(ans1["answer_text"], "Mysore is located in Karnataka.")
        self.assertIn("R5", ans1["applied_rule_ids"])

        # 2. Which country does Mysore belong to? (Country level)
        ans2 = self.svc.answer_question("Which country does Mysore belong to?")
        self.assertEqual(ans2["answer_status"], "answered")
        self.assertEqual(ans2["answer_text"], "Mysore is located in India.")
        self.assertIn("R8", ans2["applied_rule_ids"])

        # 3. Which continent is Mysore located in? (Continent level)
        ans3 = self.svc.answer_question("Which continent is Mysore located in?")
        self.assertEqual(ans3["answer_status"], "answered")
        self.assertEqual(ans3["answer_text"], "Mysore is located in Asia.")
        self.assertIn("R10", ans3["applied_rule_ids"])

        # 4. Where is Karnataka located? (State location)
        ans4 = self.svc.answer_question("Where is Karnataka located?")
        self.assertEqual(ans4["answer_status"], "answered")
        self.assertEqual(ans4["answer_text"], "Karnataka is located in India.")
        self.assertIn("R6", ans4["applied_rule_ids"])

    def test_03_profile_and_reverse_retrieval(self):
        """Verify first-person profile and reverse relational lookups."""
        # Profile questions
        self.assertEqual(self.svc.answer_question("Who am I?")["answer_text"], "Your name is Arjun.")
        self.assertEqual(self.svc.answer_question("Where do I live?")["answer_text"], "You live in Mysore.")
        self.assertEqual(self.svc.answer_question("Where do I study?")["answer_text"], "You study at ABC University.")
        self.assertEqual(self.svc.answer_question("Where do I work?")["answer_text"], "You work at Infosys.")

        # Reverse retrieval questions
        self.assertEqual(self.svc.answer_question("Who lives in Mysore?")["answer_text"], "Arjun lives in Mysore.")
        self.assertEqual(self.svc.answer_question("Who works at Infosys?")["answer_text"], "Arjun works at Infosys.")
        self.assertEqual(self.svc.answer_question("Who studies at ABC University?")["answer_text"], "Arjun studies at ABC University.")

    def test_04_direct_backward_chaining_proofs(self):
        """Verify backward chaining theorem proving on multi-hop goals."""
        fe = FactExtractor()
        facts = fe.extract_facts(self.doc_text, 1)

        wm = WorkingMemory()
        wm.load_from_kb_facts(facts)

        rules = get_extended_rules()

        # Goal 1: Mysore in Karnataka via R5
        p1 = backward_chain(("Mysore", "located in", "Karnataka"), wm, rules=rules)
        self.assertTrue(p1.is_proven)
        self.assertIn("R5", p1.applied_rule_ids)

        # Goal 2: Mysore in India via R8
        p2 = backward_chain(("Mysore", "located in", "India"), wm, rules=rules)
        self.assertTrue(p2.is_proven)
        self.assertIn("R8", p2.applied_rule_ids)

        # Goal 3: Mysore in Asia via R10
        p3 = backward_chain(("Mysore", "located in", "Asia"), wm, rules=rules)
        self.assertTrue(p3.is_proven)
        self.assertIn("R10", p3.applied_rule_ids)

        # Goal 4: Karnataka in India via R6
        p4 = backward_chain(("Karnataka", "located in", "India"), wm, rules=rules)
        self.assertTrue(p4.is_proven)
        self.assertIn("R6", p4.applied_rule_ids)

    def test_05_pdf_geographical_answer_scopes_and_reverse_integrity(self):
        """Verify 8 geographical questions with answer scopes and reverse retrieval integrity on PDF."""
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', 'ef5faeb6_DocuMind_Expanded_PDF_Acceptance_Test.pdf')
        if not os.path.exists(pdf_path):
            self.skipTest("PDF acceptance test document not found.")

        svc = DocuMindService(db_path=os.path.join(self.tmpdir, "test_pdf.db"), upload_folder=self.tmpdir)
        with open(pdf_path, 'rb') as f:
            fs = FileStorage(stream=f, filename="acceptance.pdf", content_type="application/pdf")
            res = svc.process_uploaded_file(fs)
        self.assertEqual(res.get("status"), "success")

        # 8 Geographical Questions
        self.assertEqual(svc.answer_question("Where is Mysore located?")["answer_text"], "Mysore is located in Karnataka.")
        self.assertEqual(svc.answer_question("Which country does Mysore belong to?")["answer_text"], "Mysore is located in India.")
        self.assertEqual(svc.answer_question("Which continent is Mysore located in?")["answer_text"], "Mysore is located in Asia.")
        self.assertEqual(svc.answer_question("Which continent is Bengaluru located in?")["answer_text"], "Bengaluru is located in Asia.")
        self.assertEqual(svc.answer_question("Where is Tokyo located?")["answer_text"], "Tokyo is located in Japan.")
        self.assertEqual(svc.answer_question("Which continent is Tokyo located in?")["answer_text"], "Tokyo is located in Asia.")
        self.assertEqual(svc.answer_question("Where is Karnataka located?")["answer_text"], "Karnataka is located in India.")
        self.assertEqual(svc.answer_question("Where is Tamil Nadu located?")["answer_text"], "Tamil Nadu is located in India.")

        # Reverse retrieval integrity: now that line-wrap segmentation is normalized,
        # both Arjun and Ravi are extracted and retrieved accurately.
        self.assertEqual(svc.answer_question("Who works at Infosys?")["answer_text"], "Arjun works at Infosys.")
        self.assertEqual(svc.answer_question("Who works at Research Labs?")["answer_text"], "Ravi works at Research Labs.")
        self.assertEqual(svc.answer_question("Who works at Tech Solutions?")["answer_text"], "Meera works at Tech Solutions.")

    def test_06_pdf_line_wrap_normalization_and_recovery(self):
        """Verify intra-sentence line wrap normalization and the 9 recovered facts from PDF."""
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', 'ef5faeb6_DocuMind_Expanded_PDF_Acceptance_Test.pdf')
        if not os.path.exists(pdf_path):
            self.skipTest("PDF acceptance test document not found.")

        svc = DocuMindService(db_path=os.path.join(self.tmpdir, "test_recovery.db"), upload_folder=self.tmpdir)
        with open(pdf_path, 'rb') as f:
            fs = FileStorage(stream=f, filename="acceptance.pdf", content_type="application/pdf")
            res = svc.process_uploaded_file(fs)
        self.assertEqual(res.get("status"), "success")

        # Verify all 9 recovered facts are stored in knowledge base
        stored_facts = {(f.subject, f.predicate, f.object) for f in svc.repo.get_all_facts()}

        expected_facts = [
            ("Arjun", "works at", "Infosys"),
            ("Meera", "works at", "Tech Solutions"),
            ("Ravi", "works at", "Research Labs"),
            ("New Delhi", "capital of", "India"),
            ("Infosys", "located in", "Bengaluru"),
            ("Tamil Nadu", "located in", "India"),
            ("Japan", "is a country in", "Asia"),
            ("Chennai", "located in", "Tamil Nadu"),
            ("Karnataka", "is a state in", "India"),
        ]

        for subj, pred, obj in expected_facts:
            self.assertIn((subj, pred, obj), stored_facts, f"Missing recovered fact: ({subj}, {pred}, {obj})")

        # Verify no corrupted preposition fragments exist
        for subj, pred, obj in stored_facts:
            self.assertFalse(obj.endswith((" in", " at", " on", " of")), f"Dangling preposition in object: {obj}")


if __name__ == "__main__":
    unittest.main()
