"""
Unit & Integration Tests for DocuMind-AI New Features:
- Nuke Data / Workspace Reset with Safety Constraints
- Question History Persistence & Audit Trail
- Structured JSON Analysis Export
- Academic PDF Report Generation (PyMuPDF)
- Workflow: Export before Reset -> Clean Empty State -> Post-Reset Functional Ingestion
"""

import os
import json
import io
import tempfile
import unittest

from app import create_app
from config import TestingConfig
from modules.app_service import DocuMindService
from modules.export_service import generate_workspace_json, generate_workspace_pdf
import fitz  # PyMuPDF


class TestFeaturesResetAndExport(unittest.TestCase):
    """Test suite verifying Nuke Data, Question History, and JSON/PDF Exports."""

    def setUp(self):
        # Create unique temporary workspace directory
        self.test_dir = tempfile.mkdtemp(prefix="documind_test_ws_")
        self.db_path = os.path.join(self.test_dir, "test_documind.db")
        self.upload_folder = os.path.join(self.test_dir, "uploads")
        os.makedirs(self.upload_folder, exist_ok=True)

        class CustomTestConfig(TestingConfig):
            DATABASE_PATH = self.db_path
            UPLOAD_FOLDER = self.upload_folder
            TESTING = True

        self.config = CustomTestConfig
        self.app = create_app(CustomTestConfig)
        self.client = self.app.test_client()
        self.service = self.app.service

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _seed_sample_data(self):
        """Helper to seed a document and an interaction."""
        sample_txt = b"Tokyo is the capital of Japan. Japan is an island nation in Asia."
        response = self.client.post(
            '/upload',
            data={'file': (io.BytesIO(sample_txt), 'japan_info.txt')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)

        # Ask a direct question
        q_res = self.client.post('/ask', json={'question': 'What is the capital of Japan?'})
        self.assertEqual(q_res.status_code, 200)

    # =========================================================================
    # 1. Question History Persistence
    # =========================================================================

    def test_01_question_history_persistence(self):
        """Verify question answering queries are recorded and retrievable."""
        self._seed_sample_data()

        # Query history endpoint
        res = self.client.get('/api/history')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertGreaterEqual(data['total_queries'], 1)

        first = data['history'][0]
        self.assertIn("capital of Japan", first['question_text'])
        self.assertEqual(first['answer_status'], 'answered')
        self.assertEqual(first['answer_type'], 'direct')
        self.assertIn("Tokyo", first['answer_text'])

    def test_02_unsupported_and_unknown_history(self):
        """Verify unsupported and unknown questions are recorded accurately."""
        self.client.post('/ask', json={'question': 'Why is the sky blue?'})
        self.client.post('/ask', json={'question': 'Where is Atlantis located?'})

        res = self.client.get('/api/history')
        data = res.get_json()
        self.assertGreaterEqual(data['total_queries'], 2)

        history_texts = [h['question_text'] for h in data['history']]
        self.assertTrue(any("Why is the sky blue?" in t for t in history_texts))
        self.assertTrue(any("Atlantis" in t for t in history_texts))

    # =========================================================================
    # 2. Structured JSON Export
    # =========================================================================

    def test_03_json_export_structure_and_completeness(self):
        """Verify JSON export contains all required top-level keys, metadata, and facts."""
        self._seed_sample_data()

        res = self.client.get('/api/export/json')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'application/json')
        self.assertIn('attachment; filename="documind_workspace_report.json"', res.headers.get('Content-Disposition', ''))

        data = json.loads(res.data.decode('utf-8'))
        required_keys = ['report_metadata', 'documents', 'knowledge_base_facts', 'query_history', 'inference_rules', 'evaluation']
        for k in required_keys:
            self.assertIn(k, data, f"Missing key in JSON export: {k}")

        meta = data['report_metadata']
        self.assertEqual(meta['application'], 'DocuMind-AI')
        self.assertGreaterEqual(meta['total_documents'], 1)
        self.assertGreaterEqual(meta['total_facts'], 1)
        self.assertGreaterEqual(meta['total_queries'], 1)

        # Check facts structure
        facts = data['knowledge_base_facts']
        self.assertTrue(any(f['subject'] == 'Tokyo' and f['predicate'] == 'capital of' for f in facts))

        # Check rules
        rules = data['inference_rules']
        self.assertGreaterEqual(len(rules), 4)

    # =========================================================================
    # 3. Academic PDF Export
    # =========================================================================

    def test_04_pdf_export_generation_and_validity(self):
        """Verify PDF export produces valid PyMuPDF document with multiple pages & header."""
        self._seed_sample_data()

        res = self.client.get('/api/export/pdf')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'application/pdf')
        self.assertIn('documind_workspace_report.pdf', res.headers.get('Content-Disposition', ''))

        pdf_bytes = res.data
        self.assertGreater(len(pdf_bytes), 1000)

        # Inspect generated PDF via fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.assertGreaterEqual(len(doc), 1)

        # Verify cover text presence
        first_page_text = doc[0].get_text()
        self.assertIn("DOCUMIND AI", first_page_text)
        self.assertIn("WORKSPACE ANALYSIS REPORT", first_page_text)
        self.assertIn("AID201A", first_page_text)

        # Verify no double percentage conversion (e.g. 10000.0%) anywhere across pages
        full_text = "".join(page.get_text() for page in doc)
        self.assertNotIn("10000.0%", full_text)
        self.assertNotIn("10000%", full_text)
        doc.close()

    def test_04b_pdf_evaluation_metric_percentage_formatting(self):
        """Verify empirical evaluation benchmark metrics in PDF are formatted with single conversion."""
        self._seed_sample_data()

        res = self.client.get('/api/export/pdf')
        self.assertEqual(res.status_code, 200)

        doc = fitz.open(stream=res.data, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        doc.close()

        # Check that 100.0% is rendered for 100% metrics and 10000.0% is not rendered
        self.assertIn("100.0%", full_text)
        self.assertNotIn("10000.0%", full_text)

    def test_04c_pdf_formatting_proportions_and_na(self):
        """Verify metric values 1.0 -> 100.0%, 0.938 -> 93.8%, 0.5 -> 50.0%, and N/A -> N/A."""
        from modules.export_service import PDFReportBuilder

        custom_data = {
            'report_metadata': {'course': 'AID201A', 'institution': 'RUAS', 'generated_at': '2026-11-07'},
            'documents': [],
            'knowledge_base_facts': [],
            'query_history': [],
            'evaluation': {
                'summary': {
                    'total_cases': 10,
                    'successful_cases': 10,
                    'answer_accuracy': 1.0,         # 1.0 -> 100.0%
                    'retrieval_precision': 0.938,   # 0.938 -> 93.8%
                    'retrieval_recall': 0.5,        # 0.5 -> 50.0%
                    'retrieval_f1': 'N/A',          # N/A -> N/A
                    'reasoning_accuracy': 100.0,    # 100.0 -> 100.0%
                    'rule_precision': 0.938,        # 0.938 -> 93.8%
                    'supporting_fact_accuracy': 0.5,# 0.5 -> 50.0%
                    'provenance_accuracy': None,    # None -> N/A
                    'average_execution_time_ms': 5.25
                }
            }
        }

        builder = PDFReportBuilder(custom_data)
        pdf_bytes = builder.build_pdf_bytes()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        doc.close()

        self.assertIn("100.0%", full_text)
        self.assertIn("93.8%", full_text)
        self.assertIn("50.0%", full_text)
        self.assertIn("N/A", full_text)
        self.assertNotIn("10000.0%", full_text)

    # =========================================================================
    # 4. Nuke Data / Workspace Reset Safety & Mechanics
    # =========================================================================

    def test_05_reset_rejection_without_delete_confirmation(self):
        """Verify reset fails if confirmation is missing or not 'DELETE'."""
        self._seed_sample_data()

        # Try with no data
        res1 = self.client.post('/api/workspace/reset', json={})
        self.assertEqual(res1.status_code, 400)
        self.assertIn("Confirmation mismatch", res1.get_json()['message'])

        # Try with wrong word
        res2 = self.client.post('/api/workspace/reset', json={'confirmation': 'CONFIRM'})
        self.assertEqual(res2.status_code, 400)

        # Verify nothing was deleted
        stats = self.service.get_dashboard_stats()
        self.assertEqual(stats['total_documents'], 1)
        self.assertGreaterEqual(stats['total_facts'], 1)

    def test_06_reset_clears_files_and_db_safely(self):
        """Verify successful reset clears files, documents, facts, and queries."""
        self._seed_sample_data()

        # Confirm files exist on disk
        files_before = [f for f in os.listdir(self.upload_folder) if f != '.gitkeep']
        self.assertGreaterEqual(len(files_before), 1)

        # Execute confirmed reset
        res = self.client.post('/api/workspace/reset', json={'confirmation': 'DELETE'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertGreaterEqual(data['deleted_files'], 1)
        self.assertGreaterEqual(data['deleted_documents'], 1)
        self.assertGreaterEqual(data['deleted_facts'], 1)
        self.assertGreaterEqual(data['deleted_queries'], 1)

        # Verify upload folder has no leftover document files
        files_after = [f for f in os.listdir(self.upload_folder) if f != '.gitkeep']
        self.assertEqual(len(files_after), 0)

        # Verify DB counts are 0
        stats = self.service.get_dashboard_stats()
        self.assertEqual(stats['total_documents'], 0)
        self.assertEqual(stats['successful_documents'], 0)
        self.assertEqual(stats['total_facts'], 0)
        self.assertEqual(stats['total_queries'], 0)

    def test_07_upload_directory_safety(self):
        """Verify reset function never deletes files outside the upload folder."""
        # Create a sibling file outside upload folder
        outside_file = os.path.join(self.test_dir, "safe_outside_file.txt")
        with open(outside_file, "w") as f:
            f.write("Do not delete this outside file")

        # Execute reset
        res = self.service.reset_workspace("DELETE")
        self.assertEqual(res['status'], 'success')

        # Outside file must still exist
        self.assertTrue(os.path.exists(outside_file))

    # =========================================================================
    # 5. Export -> Reset -> Fresh Start Lifecycle
    # =========================================================================

    def test_08_export_before_reset_workflow_and_fresh_post_reset_ingestion(self):
        """
        Verify the recommended workflow:
        1. Ingest document & answer question
        2. Export workspace JSON & PDF
        3. Nuke data
        4. Confirm workspace is completely empty
        5. Ingest new document and verify it processes cleanly from scratch
        """
        self._seed_sample_data()

        # Step 2: Export
        json_res = self.client.get('/api/export/json')
        self.assertEqual(json_res.status_code, 200)
        pdf_res = self.client.get('/api/export/pdf')
        self.assertEqual(pdf_res.status_code, 200)

        # Step 3: Nuke Data
        reset_res = self.client.post('/api/workspace/reset', json={'confirmation': 'DELETE'})
        self.assertEqual(reset_res.status_code, 200)

        # Step 4: Empty workspace verification
        empty_stats = self.client.get('/api/stats').get_json()
        self.assertEqual(empty_stats['total_documents'], 0)
        self.assertEqual(empty_stats['total_facts'], 0)

        # Step 5: Fresh Ingestion after reset
        fresh_txt = b"Paris is the capital of France. France is located in Europe."
        up_res = self.client.post(
            '/upload',
            data={'file': (io.BytesIO(fresh_txt), 'france_info.txt')},
            content_type='multipart/form-data'
        )
        self.assertEqual(up_res.status_code, 200)
        fresh_data = up_res.get_json()
        self.assertEqual(fresh_data['status'], 'success')
        self.assertEqual(fresh_data['document_id'], 1)  # Sequence restart or clean ID

        # Query fresh document
        q_fresh = self.client.post('/ask', json={'question': 'What is the capital of France?'})
        self.assertEqual(q_fresh.status_code, 200)
        self.assertIn('Paris', q_fresh.get_json()['answer_text'])


if __name__ == '__main__':
    unittest.main()
