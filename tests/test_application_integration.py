"""
DocuMind-AI: Module 5, Stage 4 End-to-End Application Integration Test Suite
Verifies the complete Flask application workflows, REST endpoints, security safeguards,
file ingestion, Q&A pipeline, symbolic deduction, and evaluation framework.

Course: AID201A – Artificial Intelligence
Institution: M S Ramaiah University of Applied Sciences
"""

import io
import os
import shutil
import tempfile
import unittest

import docx
import fitz  # PyMuPDF

from app import create_app
from config import Config
from modules.knowledge_base.database import init_db


class TestApplicationIntegration(unittest.TestCase):
    """
    Comprehensive end-to-end integration tests using Flask's test client
    in an isolated temporary filesystem and database environment.
    """

    def setUp(self):
        # Create isolated temporary workspace for tests
        self.test_dir = tempfile.mkdtemp(prefix="documind_test_")
        self.test_upload_folder = os.path.join(self.test_dir, "uploads")
        self.test_db_path = os.path.join(self.test_dir, "test_documind.db")

        class IsolatedTestConfig(Config):
            TESTING = True
            UPLOAD_FOLDER = self.test_upload_folder
            DATABASE_PATH = self.test_db_path
            MAX_CONTENT_LENGTH = 16 * 1024 * 1024

        self.app = create_app(IsolatedTestConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        # Clean up temporary test files
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Application Startup & Health Check
    # --------------------------------------------------------------------------
    def test_01_application_startup(self):
        """Verify app factory, routes registered, and /api/status health check."""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'online')
        self.assertEqual(data.get('app_name'), 'DocuMind-AI')
        self.assertTrue(data.get('nlp_rag_disabled'))
        self.assertEqual(sorted(data.get('supported_formats')), ['docx', 'pdf', 'txt'])
        self.assertEqual(data.get('max_upload_size_mb'), 16)

    # --------------------------------------------------------------------------
    # 2. Dashboard Route
    # --------------------------------------------------------------------------
    def test_02_dashboard_route(self):
        """Verify GET / returns 200 and loads HTML template with UI sections."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("DocuMind AI", html)
        self.assertIn("Classical Symbolic AI Architecture", html)
        self.assertIn("tab-dashboard", html)
        self.assertIn("tab-documents", html)
        self.assertIn("tab-kb", html)
        self.assertIn("tab-qa", html)
        self.assertIn("tab-evaluation", html)

    # --------------------------------------------------------------------------
    # 3. Valid TXT Document Upload
    # --------------------------------------------------------------------------
    def test_03_valid_txt_upload(self):
        """Upload valid .txt file, verify extraction succeeds and facts are stored."""
        content = b"Tokyo is the capital of Japan. Japan borders South Korea.\n"
        data = {'file': (io.BytesIO(content), 'japan.txt')}

        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()

        self.assertEqual(res_data.get('status'), 'success')
        self.assertEqual(res_data.get('filename'), 'japan.txt')
        self.assertEqual(res_data.get('file_format'), 'txt')
        self.assertGreaterEqual(res_data.get('facts_extracted'), 1)

    # --------------------------------------------------------------------------
    # 4. Valid DOCX Document Upload
    # --------------------------------------------------------------------------
    def test_04_valid_docx_upload(self):
        """Upload valid .docx file, verify extraction succeeds and facts are stored."""
        doc = docx.Document()
        doc.add_paragraph("Stanford University was established in 1885.")
        docx_io = io.BytesIO()
        doc.save(docx_io)
        docx_io.seek(0)

        data = {'file': (docx_io, 'stanford.docx')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()

        self.assertEqual(res_data.get('status'), 'success')
        self.assertEqual(res_data.get('file_format'), 'docx')
        self.assertGreaterEqual(res_data.get('facts_extracted'), 1)

    # --------------------------------------------------------------------------
    # 5. Valid PDF Document Upload
    # --------------------------------------------------------------------------
    def test_05_valid_pdf_upload(self):
        """Upload valid .pdf file, verify extraction succeeds and facts are stored."""
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        page.insert_text((50, 72), "Python was developed by Guido van Rossum.")
        pdf_io = io.BytesIO(pdf_doc.write())
        pdf_doc.close()

        data = {'file': (pdf_io, 'python.pdf')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()

        self.assertEqual(res_data.get('status'), 'success')
        self.assertEqual(res_data.get('file_format'), 'pdf')
        self.assertGreaterEqual(res_data.get('facts_extracted'), 1)

    # --------------------------------------------------------------------------
    # 6. Unsupported File Format Rejection
    # --------------------------------------------------------------------------
    def test_06_unsupported_file_rejection(self):
        """Uploading unsupported format (.exe, .jpg, etc.) returns 400 Bad Request."""
        data = {'file': (io.BytesIO(b'Binary executable content'), 'script.exe')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_data = response.get_json()
        self.assertEqual(res_data.get('status'), 'error')
        self.assertIn("Unsupported file extension", res_data.get('message'))

    # --------------------------------------------------------------------------
    # 7. Empty File Handling
    # --------------------------------------------------------------------------
    def test_07_empty_file_handling(self):
        """Uploading an empty (0 bytes) file returns 400 Bad Request."""
        data = {'file': (io.BytesIO(b''), 'empty.txt')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_data = response.get_json()
        self.assertEqual(res_data.get('status'), 'error')
        self.assertIn("empty", res_data.get('message').lower())

    # --------------------------------------------------------------------------
    # 8. Corrupt Document Handling
    # --------------------------------------------------------------------------
    def test_08_corrupt_document_handling(self):
        """Uploading a corrupt file (e.g. non-zip DOCX or broken PDF) is handled safely."""
        corrupt_docx = io.BytesIO(b"This is not a zip archive for docx")
        data = {'file': (corrupt_docx, 'corrupt.docx')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_data = response.get_json()
        self.assertEqual(res_data.get('status'), 'error')
        self.assertIn("failed", res_data.get('message').lower())

    # --------------------------------------------------------------------------
    # 9. Oversized Upload Rejection
    # --------------------------------------------------------------------------
    def test_09_oversized_upload_rejection(self):
        """Uploading a file exceeding 16 MB is rejected safely."""
        # Test DocuMindService directly with a 17 MB file wrapper
        class LargeFileMock:
            filename = 'large.txt'
            content_length = 17 * 1024 * 1024
            def seek(self, pos, whence=0): pass
            def tell(self): return 17 * 1024 * 1024

        res = self.app.service.process_uploaded_file(LargeFileMock())
        self.assertEqual(res['status'], 'error')
        self.assertIn('16 MB', res['message'])

    # --------------------------------------------------------------------------
    # 10. Unsafe Filename / Directory Traversal Handling
    # --------------------------------------------------------------------------
    def test_10_unsafe_filename_path_handling(self):
        """Verify directory traversal filenames are sanitized and confined to upload folder."""
        data = {'file': (io.BytesIO(b'Tokyo is the capital of Japan.\n'), '../../malicious.txt')}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()
        # Werkzeug sanitizes '../../malicious.txt' to 'malicious.txt'
        self.assertEqual(res_data.get('filename'), 'malicious.txt')
        # Ensure no file was created outside upload folder
        parent_dir = os.path.dirname(self.test_upload_folder)
        self.assertFalse(os.path.exists(os.path.join(parent_dir, 'malicious.txt')))

    # --------------------------------------------------------------------------
    # 11. Knowledge Base Persistence
    # --------------------------------------------------------------------------
    def test_11_knowledge_base_persistence(self):
        """Uploaded facts persist in SQLite database and can be queried."""
        content = b"Tokyo is the capital of Japan.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'doc.txt')}, content_type='multipart/form-data')

        facts = self.app.service.repo.get_all_facts()
        self.assertGreaterEqual(len(facts), 1)
        self.assertEqual(facts[0].subject, "Tokyo")
        self.assertEqual(facts[0].predicate, "capital of")
        self.assertEqual(facts[0].object, "Japan")

    # --------------------------------------------------------------------------
    # 12. Duplicate Upload Behavior
    # --------------------------------------------------------------------------
    def test_12_duplicate_upload_behavior(self):
        """Re-uploading the exact same document detects duplicate and avoids redundancy."""
        content = b"Tokyo is the capital of Japan.\n"
        res1 = self.client.post('/upload', data={'file': (io.BytesIO(content), 'same.txt')}, content_type='multipart/form-data')
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.get_json().get('status'), 'success')

        res2 = self.client.post('/upload', data={'file': (io.BytesIO(content), 'same.txt')}, content_type='multipart/form-data')
        self.assertEqual(res2.status_code, 200)
        res2_data = res2.get_json()
        self.assertEqual(res2_data.get('status'), 'duplicate')
        self.assertTrue(res2_data.get('is_duplicate'))

    # --------------------------------------------------------------------------
    # 13. Direct Factual Question
    # --------------------------------------------------------------------------
    def test_13_direct_factual_question(self):
        """Direct retrieval question returns verified direct answer."""
        content = b"Tokyo is the capital of Japan.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'geo.txt')}, content_type='multipart/form-data')

        response = self.client.post('/ask', json={'question': 'What is the capital of Japan?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('answer_status'), 'answered')
        self.assertEqual(data.get('answer_type'), 'direct')
        self.assertEqual(data.get('answer_text'), 'The capital of Japan is Tokyo.')
        self.assertEqual(data.get('confidence'), 'verified')
        self.assertGreaterEqual(len(data.get('supporting_fact_ids')), 1)

    # --------------------------------------------------------------------------
    # 14. Inferred Factual Question (Rule R1)
    # --------------------------------------------------------------------------
    def test_14_inferred_factual_question(self):
        """Question requiring deduction applies inference rule R1 and returns inferred answer."""
        content = b"Bangalore is the capital of Karnataka. Karnataka is a state in India.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'india.txt')}, content_type='multipart/form-data')

        response = self.client.post('/ask', json={'question': 'Where is Bangalore located?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('answer_status'), 'answered')
        self.assertEqual(data.get('answer_type'), 'inferred')
        self.assertEqual(data.get('answer_text'), 'Bangalore is located in India.')
        self.assertIn('R1', data.get('applied_rule_ids'))
        self.assertEqual(data.get('confidence'), 'deduced')

    # --------------------------------------------------------------------------
    # 15. Boolean Affirmative Question
    # --------------------------------------------------------------------------
    def test_15_boolean_affirmative_question(self):
        """Boolean question with matching ground fact returns affirmative answer."""
        content = b"Canada borders United States.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'borders.txt')}, content_type='multipart/form-data')

        response = self.client.post('/ask', json={'question': 'Does Canada border United States?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('answer_status'), 'answered')
        self.assertEqual(data.get('answer_text'), 'Yes, Canada borders United States.')

    # --------------------------------------------------------------------------
    # 16. Unknown Question (Open-World Assumption)
    # --------------------------------------------------------------------------
    def test_16_unknown_question(self):
        """Querying missing information returns unknown status, never false."""
        response = self.client.post('/ask', json={'question': 'Where is Atlantis located?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('answer_status'), 'unknown')
        self.assertEqual(data.get('answer_type'), 'none')
        self.assertIn("could not find sufficient information", data.get('answer_text'))

    # --------------------------------------------------------------------------
    # 17. Unsupported Question Handling
    # --------------------------------------------------------------------------
    def test_17_unsupported_question(self):
        """Open-domain question outside supported grammar returns unsupported status without hallucinating."""
        response = self.client.post('/ask', json={'question': 'Why is the sky blue?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('answer_status'), 'unsupported')
        self.assertIn("not currently supported", data.get('answer_text'))

    # --------------------------------------------------------------------------
    # 18. Explanation and Provenance Display
    # --------------------------------------------------------------------------
    def test_18_explanation_and_provenance_display(self):
        """Verify response contains transparent explanation and source citations."""
        content = b"Python was developed by Guido van Rossum.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'python.txt')}, content_type='multipart/form-data')

        response = self.client.post('/ask', json={'question': 'Who developed Python?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertTrue(len(data.get('explanation')) > 0)
        self.assertIn('Python', data.get('explanation'))
        self.assertGreaterEqual(len(data.get('sources')), 1)
        self.assertEqual(data.get('sources')[0]['filename'], 'python.txt')

    # --------------------------------------------------------------------------
    # 19. Evaluation Benchmark Execution
    # --------------------------------------------------------------------------
    def test_19_evaluation_execution(self):
        """Verify /api/evaluate executes the 21 benchmark cases with 100% accuracy."""
        response = self.client.post('/api/evaluate')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('status'), 'success')
        summary = data.get('summary')
        self.assertEqual(summary.get('total_cases'), 21)
        self.assertEqual(summary.get('successful_cases'), 21)
        self.assertEqual(summary.get('failed_cases'), 0)
        self.assertEqual(summary.get('answer_accuracy'), 100.0)

    # --------------------------------------------------------------------------
    # 20. Evaluation Report Rendering
    # --------------------------------------------------------------------------
    def test_20_evaluation_report_rendering(self):
        """Verify evaluation returns publication-ready academic Markdown report."""
        response = self.client.get('/api/evaluate')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        report_md = data.get('report_markdown')
        self.assertIsNotNone(report_md)
        self.assertIn("DOCUMIND AI – SYSTEM EVALUATION REPORT", report_md)
        self.assertIn("EXECUTIVE SUMMARY", report_md)
        self.assertIn("ACADEMIC COMPLIANCE & LIMITATIONS", report_md)

    # --------------------------------------------------------------------------
    # 21. Graceful Handling of Missing or Invalid Input
    # --------------------------------------------------------------------------
    def test_21_graceful_handling_missing_invalid_input(self):
        """Verify 400 Bad Request on empty or missing question payloads."""
        # Empty string
        res1 = self.client.post('/ask', json={'question': ''})
        self.assertEqual(res1.status_code, 400)

        # Whitespace-only string
        res2 = self.client.post('/ask', json={'question': '   \n  \t'})
        self.assertEqual(res2.status_code, 400)

        # Missing question key
        res3 = self.client.post('/ask', json={})
        self.assertEqual(res3.status_code, 400)

    # --------------------------------------------------------------------------
    # 22. Database and Temporary File Isolation
    # --------------------------------------------------------------------------
    def test_22_database_and_temporary_file_isolation(self):
        """Verify test operations remain isolated in temporary directory without touching production DB."""
        # Production database should not contain test document 'isolated_test.txt'
        content = b"Tokyo is the capital of Japan.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'isolated_test.txt')}, content_type='multipart/form-data')

        # Verify it exists in test_db_path
        self.assertTrue(os.path.exists(self.test_db_path))

        # Check production db path
        prod_db = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'database', 'documind.db'))
        if os.path.exists(prod_db):
            from modules.knowledge_base.repository import KnowledgeBaseRepository
            prod_repo = KnowledgeBaseRepository(prod_db)
            prod_docs = [d.filename for d in prod_repo.get_all_documents()]
            self.assertNotIn('isolated_test.txt', prod_docs)

    # --------------------------------------------------------------------------
    # 23. API Stats Endpoint
    # --------------------------------------------------------------------------
    def test_23_api_stats_endpoint(self):
        """Verify GET /api/stats returns document and fact counters."""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn('total_documents', data)
        self.assertIn('successful_documents', data)
        self.assertIn('failed_documents', data)
        self.assertIn('total_facts', data)
        self.assertIn('recent_documents', data)

    # --------------------------------------------------------------------------
    # 24. API Facts Endpoint (Pagination & Retrieval)
    # --------------------------------------------------------------------------
    def test_24_api_facts_endpoint(self):
        """Verify GET /api/facts returns stored facts with provenance and limit/offset."""
        content = b"Tokyo is the capital of Japan. Japan borders South Korea.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'geography.txt')}, content_type='multipart/form-data')

        response = self.client.get('/api/facts?limit=1&offset=0')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(len(data.get('facts')), 1)
        self.assertEqual(data.get('limit'), 1)
        self.assertEqual(data.get('offset'), 0)
        self.assertGreaterEqual(data.get('total_facts'), 1)

    # --------------------------------------------------------------------------
    # 25. Defect 1: Proper Bullet Separators Without Literal Entity Text
    # --------------------------------------------------------------------------
    def test_25_html_entity_rendering_bullet_separator(self):
        """Verify frontend main.js does not render &bull; as literal text in parsed bar."""
        with open('static/js/main.js', 'r', encoding='utf-8') as f:
            js_content = f.read()

        # Ensure literal '&bull;' is not assigned to textContent
        self.assertNotIn("textContent = `Category: [${catStr.toUpperCase()}] &bull;", js_content)
        self.assertIn("•", js_content)

    # --------------------------------------------------------------------------
    # 26. Defect 2: Duplicate File Extension Normalization
    # --------------------------------------------------------------------------
    def test_26_duplicate_file_extension_normalization(self):
        """Verify uploaded documents with duplicate extensions (e.g. india.txt.txt) are displayed with single extension."""
        from modules.app_service import normalize_filename
        self.assertEqual(normalize_filename("india.txt.txt"), "india.txt")
        self.assertEqual(normalize_filename("report.pdf.pdf"), "report.pdf")
        self.assertEqual(normalize_filename("document.docx.docx"), "document.docx")

        # Test upload endpoint with duplicate extension
        content = b"Tokyo is the capital of Japan.\n"
        res = self.client.post('/upload', data={'file': (io.BytesIO(content), 'sample.txt.txt')}, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('filename'), 'sample.txt')

        # Check stats endpoint formatting
        stats_res = self.client.get('/api/stats')
        recent = stats_res.get_json().get('recent_documents', [])
        sample_doc = next((d for d in recent if 'sample' in d['filename']), None)
        self.assertIsNotNone(sample_doc)
        self.assertEqual(sample_doc['filename'], 'sample.txt')

    # --------------------------------------------------------------------------
    # 27. Defect 3: Provenance Consistency in Inferred Explanations
    # --------------------------------------------------------------------------
    def test_27_provenance_consistency_in_inferred_explanation(self):
        """
        Verify inferred answers retain full source provenance across both supporting ground facts,
        display Rule R1, include original source sentences, and agree with visual timeline.
        """
        content = b"Bangalore is the capital of Karnataka. Karnataka is a state in India.\n"
        self.client.post('/upload', data={'file': (io.BytesIO(content), 'india.txt')}, content_type='multipart/form-data')

        response = self.client.post('/ask', json={'question': 'Where is Bangalore located?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data.get('answer_status'), 'answered')
        self.assertEqual(data.get('answer_type'), 'inferred')
        self.assertIn('R1', data.get('applied_rule_ids'))
        self.assertEqual(data.get('answer_text'), 'Bangalore is located in India.')

        # Verify sources list is populated with both supporting ground facts
        sources = data.get('sources', [])
        self.assertGreaterEqual(len(sources), 2)
        source_sentences = [s.get('source_sentence') for s in sources]
        self.assertIn('Bangalore is the capital of Karnataka.', source_sentences)
        self.assertIn('Karnataka is a state in India.', source_sentences)
        for s in sources:
            self.assertEqual(s.get('filename'), 'india.txt')

        # Verify explanation text contains Rule R1, source document, and original sentences
        exp = data.get('explanation', '')
        self.assertIn('R1', exp)
        self.assertIn('Capital Location Rule', exp)
        self.assertIn('india.txt', exp)
        self.assertIn('Bangalore is the capital of Karnataka.', exp)
        self.assertIn('Karnataka is a state in India.', exp)

        # Ensure missing provenance is NOT falsely reported when provenance exists
        self.assertNotIn('Source document reference: Not recorded in knowledge base.', exp)


if __name__ == '__main__':
    unittest.main()
