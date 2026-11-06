"""
Basic environment and structure verification test for DocuMind-AI.
"""

import os
import unittest
from config import Config, TestingConfig
from app import create_app


class TestInitialization(unittest.TestCase):
    """Verifies that project directories and basic configuration are set up."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()

    def test_paths_configured(self):
        """Check that configured directory paths exist or are valid."""
        self.assertTrue(os.path.isabs(Config.BASE_DIR))
        self.assertTrue(os.path.isabs(Config.UPLOAD_FOLDER))
        self.assertTrue(os.path.isabs(Config.DATABASE_PATH))
        self.assertIn('pdf', Config.ALLOWED_EXTENSIONS)
        self.assertIn('docx', Config.ALLOWED_EXTENSIONS)
        self.assertIn('txt', Config.ALLOWED_EXTENSIONS)

    def test_home_page(self):
        """Verify the Flask home page route renders successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'DocuMind AI', response.data)

    def test_api_status(self):
        """Verify the health check endpoint returns JSON status."""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data.get('status'), 'online')
        self.assertEqual(json_data.get('nlp_rag_disabled'), True)


if __name__ == '__main__':
    unittest.main()
