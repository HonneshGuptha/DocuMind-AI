"""
Automated Test Suite for india.txt Testing Document
Validates complete end-to-end ingestion, fact extraction, question parsing,
symbolic reasoning (including possession transitivity R11), and answer generation.
"""

import os
import shutil
import tempfile
import unittest
import io

from werkzeug.datastructures import FileStorage
from modules.app_service import DocuMindService


class TestIndiaAcceptance(unittest.TestCase):
    """End-to-end validation of india.txt document ingestion and QA."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.upload_dir = os.path.join(self.test_dir, 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)
        self.service = DocuMindService(db_path=self.db_path, upload_folder=self.upload_dir)

        self.india_text = """My name is Honnesh.
I live in Bangalore.
I study at RUAS.
I am an AI and ML engineering student.
I work at KPMG.
Bangalore is the capital of Karnataka.
Karnataka is a state in India.
India is located in Asia.
Ayyappa is in Sai's dream 
Kishore has rolex watch
Akash has kishore"""

        f = FileStorage(
            stream=io.BytesIO(self.india_text.encode('utf-8')),
            filename='india.txt',
            content_type='text/plain'
        )
        self.upload_result = self.service.process_uploaded_file(f, force=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_all_11_facts_extracted(self):
        self.assertEqual(self.upload_result['status'], 'success')
        self.assertEqual(self.upload_result['facts_extracted'], 11)
        self.assertEqual(self.service.repo.count_facts(), 11)

    def test_02_speaker_identity_and_profile(self):
        res1 = self.service.answer_question("What is my name?")
        self.assertEqual(res1['answer_status'], 'answered')
        self.assertEqual(res1['answer_text'], "Your name is Honnesh.")

        res2 = self.service.answer_question("Where do I live?")
        self.assertEqual(res2['answer_status'], 'answered')
        self.assertEqual(res2['answer_text'], "You live in Bangalore.")

        res3 = self.service.answer_question("Where do I study?")
        self.assertEqual(res3['answer_status'], 'answered')
        self.assertEqual(res3['answer_text'], "You study at RUAS.")

        res4 = self.service.answer_question("Where do I work?")
        self.assertEqual(res4['answer_status'], 'answered')
        self.assertEqual(res4['answer_text'], "You work at KPMG.")

    def test_03_geographic_and_multi_hop(self):
        res_cap = self.service.answer_question("What is the capital of Karnataka?")
        self.assertEqual(res_cap['answer_status'], 'answered')
        self.assertEqual(res_cap['answer_text'], "The capital of Karnataka is Bangalore.")

        res_state = self.service.answer_question("Is Karnataka a state in India?")
        self.assertEqual(res_state['answer_status'], 'answered')
        self.assertTrue(res_state['answer_text'].startswith("Yes,"))

        res_blr = self.service.answer_question("Where is Bangalore located?")
        self.assertEqual(res_blr['answer_status'], 'answered')
        self.assertIn("India", res_blr['answer_text'])

        res_ind = self.service.answer_question("Where is India located?")
        self.assertEqual(res_ind['answer_status'], 'answered')
        self.assertEqual(res_ind['answer_text'], "India is located in Asia.")

    def test_04_ayyappa_location_and_reverse_query(self):
        # Forward location: "Where is Ayyappa?"
        res1 = self.service.answer_question("Where is Ayyappa?")
        self.assertEqual(res1['answer_status'], 'answered')
        self.assertIn("Sai's dream", res1['answer_text'])

        # Reverse location: "Who is in Sai's dream?"
        res2 = self.service.answer_question("Who is in Sai's dream?")
        self.assertEqual(res2['answer_status'], 'answered')
        self.assertIn("Ayyappa", res2['answer_text'])

        # Boolean location: "Is Ayyappa in Sai's dream?"
        res3 = self.service.answer_question("Is Ayyappa in Sai's dream?")
        self.assertEqual(res3['answer_status'], 'answered')
        self.assertTrue(res3['answer_text'].startswith("Yes,"))

        # Open-World Assumption check: "Where is Sai?"
        res4 = self.service.answer_question("Where is Sai?")
        self.assertEqual(res4['answer_status'], 'unknown')

    def test_05_possession_and_deductive_transitivity(self):
        # Direct retrieval: "What does Kishore have?"
        res1 = self.service.answer_question("What does Kishore have?")
        self.assertEqual(res1['answer_status'], 'answered')
        self.assertIn("rolex watch", res1['answer_text'])

        # Direct retrieval: "Who has rolex watch?"
        res2 = self.service.answer_question("Who has rolex watch?")
        self.assertEqual(res2['answer_status'], 'answered')
        self.assertIn("Kishore", res2['answer_text'])

        # Direct retrieval: "What does Akash have?"
        res3 = self.service.answer_question("What does Akash have?")
        self.assertEqual(res3['answer_status'], 'answered')
        self.assertIn("kishore", res3['answer_text'])

        # Direct retrieval: "Who has kishore?"
        res4 = self.service.answer_question("Who has kishore?")
        self.assertEqual(res4['answer_status'], 'answered')
        self.assertIn("Akash", res4['answer_text'])

        # Boolean direct: "Does Kishore have rolex watch?"
        res5 = self.service.answer_question("Does Kishore have rolex watch?")
        self.assertEqual(res5['answer_status'], 'answered')
        self.assertTrue(res5['answer_text'].startswith("Yes,"))

        # Boolean direct: "Does Akash have kishore?"
        res6 = self.service.answer_question("Does Akash have kishore?")
        self.assertEqual(res6['answer_status'], 'answered')
        self.assertTrue(res6['answer_text'].startswith("Yes,"))

        # Deductive Inference (Transitive Possession R11): "Does Akash have rolex watch?"
        res7 = self.service.answer_question("Does Akash have rolex watch?")
        self.assertEqual(res7['answer_status'], 'answered')
        self.assertEqual(res7['answer_type'], 'inferred')
        self.assertIn("R11", res7['applied_rule_ids'])
        self.assertEqual(res7['answer_text'], "Yes, Akash has rolex watch.")


if __name__ == '__main__':
    unittest.main()
