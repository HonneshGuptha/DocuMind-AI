"""
Unit tests for Module 1: Document Processing Subsystem.
Tests PDF, DOCX, and TXT extraction, metadata preservation, empty documents,
corrupted files, and unsupported formats.
"""

import os
import codecs
import shutil
import tempfile
import unittest

import fitz  # PyMuPDF
import docx  # python-docx

from modules.document_processor import (
    DocumentProcessor,
    DocumentContent,
    extract_document,
    PDFExtractor,
    DOCXExtractor,
    TXTExtractor,
    SUPPORTED_EXTENSIONS
)


class TestDocumentProcessor(unittest.TestCase):
    """Test suite for document extraction, format validation, and error handling."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary sandbox directory for test documents."""
        cls.test_dir = tempfile.mkdtemp(prefix="documind_test_")

    @classmethod
    def tearDownClass(cls):
        """Remove temporary directory after all tests complete."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.processor = DocumentProcessor()

    # --------------------------------------------------------------------------
    # 1. TXT Extraction Tests
    # --------------------------------------------------------------------------

    def test_txt_extraction_utf8(self):
        """Test extraction from a valid UTF-8 text file."""
        file_path = os.path.join(self.test_dir, "sample.txt")
        content = "DocuMind-AI is a knowledge-based QA system.\nIt uses rule-based reasoning."
        with open(file_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

        result = self.processor.process(file_path)

        self.assertIsInstance(result, DocumentContent)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "txt")
        self.assertEqual(result.filename, "sample.txt")
        self.assertEqual(result.raw_text, content)
        self.assertEqual(result.char_count, len(content))
        self.assertEqual(result.line_count, 2)
        self.assertIsNone(result.page_count)
        self.assertIsNone(result.error_message)

    def test_txt_line_endings_preserved(self):
        """
        Verify that CRLF, LF, and CR newline formats are preserved verbatim
        without universal newline conversion or transformation.
        """
        # 1. CRLF line endings (Windows standard)
        crlf_path = os.path.join(self.test_dir, "sample_crlf.txt")
        crlf_content = "First line\r\nSecond line\r\nThird line"
        with open(crlf_path, "wb") as f:
            f.write(crlf_content.encode("utf-8"))

        crlf_result = self.processor.process(crlf_path)
        self.assertEqual(crlf_result.status, "success")
        self.assertEqual(crlf_result.raw_text, crlf_content)
        self.assertIn("\r\n", crlf_result.raw_text)

        # 2. LF line endings (Unix/Linux standard)
        lf_path = os.path.join(self.test_dir, "sample_lf.txt")
        lf_content = "First line\nSecond line\nThird line"
        with open(lf_path, "wb") as f:
            f.write(lf_content.encode("utf-8"))

        lf_result = self.processor.process(lf_path)
        self.assertEqual(lf_result.status, "success")
        self.assertEqual(lf_result.raw_text, lf_content)
        self.assertIn("\n", lf_result.raw_text)
        self.assertNotIn("\r", lf_result.raw_text)

        # 3. CR line endings (Classic Mac standard)
        cr_path = os.path.join(self.test_dir, "sample_cr.txt")
        cr_content = "First line\rSecond line\rThird line"
        with open(cr_path, "wb") as f:
            f.write(cr_content.encode("utf-8"))

        cr_result = self.processor.process(cr_path)
        self.assertEqual(cr_result.status, "success")
        self.assertEqual(cr_result.raw_text, cr_content)
        self.assertIn("\r", cr_result.raw_text)
        self.assertNotIn("\n", cr_result.raw_text)

    def test_txt_extraction_latin1(self):
        """Test extraction from a text file encoded with Latin-1 characters."""
        file_path = os.path.join(self.test_dir, "sample_latin1.txt")
        content = "Café au lait and résumé processing in DocuMind-AI."
        with open(file_path, "w", encoding="latin-1") as f:
            f.write(content)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "txt")
        self.assertEqual(result.raw_text, content)

    def test_txt_extraction_utf8_bom(self):
        """Test extraction from a UTF-8 file with Byte Order Mark (BOM)."""
        file_path = os.path.join(self.test_dir, "sample_utf8_bom.txt")
        content = "UTF-8 with BOM: Rule-Based Reasoning Engine."
        raw_bytes = codecs.BOM_UTF8 + content.encode("utf-8")
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "txt")
        self.assertEqual(result.raw_text, content)
        # Ensure BOM character is not part of the raw_text content
        self.assertFalse(result.raw_text.startswith("\ufeff"))

    def test_txt_extraction_utf16_le(self):
        """Test extraction from a UTF-16 Little Endian encoded text file."""
        file_path = os.path.join(self.test_dir, "sample_utf16_le.txt")
        content = "UTF-16 LE: Subject-Predicate-Object triples."
        raw_bytes = codecs.BOM_UTF16_LE + content.encode("utf-16-le")
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "txt")
        self.assertEqual(result.raw_text, content)

    def test_txt_extraction_utf16_be(self):
        """Test extraction from a UTF-16 Big Endian encoded text file."""
        file_path = os.path.join(self.test_dir, "sample_utf16_be.txt")
        content = "UTF-16 BE: Backward chaining and goal verification."
        raw_bytes = codecs.BOM_UTF16_BE + content.encode("utf-16-be")
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "txt")
        self.assertEqual(result.raw_text, content)

    def test_txt_extraction_utf16_no_bom(self):
        """Test extraction from a UTF-16 LE file without BOM (properly detected before Latin-1)."""
        file_path = os.path.join(self.test_dir, "sample_utf16_nobom.txt")
        content = "UTF-16 without BOM must not degrade into Latin-1 mojibake."
        raw_bytes = content.encode("utf-16-le")  # No BOM prepended
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "txt")
        self.assertEqual(result.raw_text, content)

    def test_txt_latin1_with_high_bytes(self):
        """Test extraction of pure Latin-1 binary byte sequence that is invalid UTF-8."""
        file_path = os.path.join(self.test_dir, "sample_latin1_bytes.txt")
        # 0xE9 is 'é', 0xFC is 'ü' in Latin-1; invalid standalone in UTF-8
        raw_bytes = b"DocuMind-AI: Caf\xe9 and M\xfcnchen knowledge base."
        with open(file_path, "wb") as f:
            f.write(raw_bytes)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "txt")
        self.assertEqual(result.raw_text, "DocuMind-AI: Café and München knowledge base.")

    def test_txt_empty_file(self):
        """Test handling of an empty text file (0 bytes)."""
        file_path = os.path.join(self.test_dir, "empty.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.char_count, 0)
        self.assertIn("empty", result.error_message.lower())

    # --------------------------------------------------------------------------
    # 2. PDF Extraction Tests (PyMuPDF)
    # --------------------------------------------------------------------------

    def test_pdf_extraction_single_page(self):
        """Test raw text extraction from a valid single-page PDF."""
        file_path = os.path.join(self.test_dir, "sample_single.pdf")
        doc = fitz.open()
        page = doc.new_page()
        text_content = "Antigravity Knowledge Representation and Reasoning."
        page.insert_text((72, 72), text_content)
        doc.save(file_path)
        doc.close()

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "pdf")
        self.assertEqual(result.page_count, 1)
        self.assertIn(text_content, result.raw_text)
        self.assertGreater(result.char_count, 0)
        self.assertIsNone(result.error_message)

    def test_pdf_extraction_multi_page(self):
        """Test extraction and page count preservation on multi-page PDF."""
        file_path = os.path.join(self.test_dir, "sample_multi.pdf")
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((72, 72), "First page: Subject-Predicate-Object.")
        p2 = doc.new_page()
        p2.insert_text((72, 72), "Second page: Forward Chaining Inference.")
        doc.save(file_path)
        doc.close()

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "pdf")
        self.assertEqual(result.page_count, 2)
        self.assertIn("First page", result.raw_text)
        self.assertIn("Second page", result.raw_text)

    def test_pdf_empty_document(self):
        """Test handling of a PDF with no extractable text."""
        file_path = os.path.join(self.test_dir, "blank.pdf")
        doc = fitz.open()
        doc.new_page()  # Blank page with no text inserted
        doc.save(file_path)
        doc.close()

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.page_count, 1)

    def test_pdf_corrupted_file(self):
        """Test handling of a corrupted PDF file."""
        file_path = os.path.join(self.test_dir, "corrupted.pdf")
        with open(file_path, "wb") as f:
            f.write(b"%PDF-1.4 [corrupted invalid binary stream]")

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "error")
        self.assertIn("PDF extraction error", result.error_message)

    # --------------------------------------------------------------------------
    # 3. DOCX Extraction Tests (python-docx)
    # --------------------------------------------------------------------------

    def test_docx_extraction_paragraphs_and_table(self):
        """Test extraction of both paragraphs and table cells from a DOCX."""
        file_path = os.path.join(self.test_dir, "sample.docx")
        doc = docx.Document()
        doc.add_paragraph("DocuMind-AI Knowledge Representation.")
        doc.add_paragraph("Deterministic query classification.")

        # Add a table
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Entity"
        table.cell(0, 1).text = "Type"
        table.cell(1, 0).text = "DocuMind"
        table.cell(1, 1).text = "System"

        doc.save(file_path)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.format, "docx")
        self.assertIn("DocuMind-AI Knowledge Representation.", result.raw_text)
        self.assertIn("Deterministic query classification.", result.raw_text)
        self.assertIn("Entity | Type", result.raw_text)
        self.assertIn("DocuMind | System", result.raw_text)
        self.assertIsNone(result.page_count)

    def test_docx_empty_document(self):
        """Test handling of an empty DOCX file."""
        file_path = os.path.join(self.test_dir, "empty.docx")
        doc = docx.Document()
        doc.save(file_path)

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.raw_text, "")

    def test_docx_corrupted_file(self):
        """Test handling of a corrupted DOCX file (non-zip content)."""
        file_path = os.path.join(self.test_dir, "corrupted.docx")
        with open(file_path, "wb") as f:
            f.write(b"NOT_A_VALID_DOCX_ZIP_ARCHIVE")

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "error")
        self.assertIn("DOCX extraction error", result.error_message)

    # --------------------------------------------------------------------------
    # 4. Format Validation, Edge Cases & Serialisation
    # --------------------------------------------------------------------------

    def test_unsupported_format(self):
        """Test rejection of unsupported file extensions."""
        file_path = os.path.join(self.test_dir, "data.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("a,b,c\n1,2,3")

        result = self.processor.process(file_path)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.format, "csv")
        self.assertIn("Unsupported format", result.error_message)

    def test_nonexistent_file(self):
        """Test handling of non-existent file path."""
        file_path = os.path.join(self.test_dir, "does_not_exist.pdf")
        result = self.processor.process(file_path)

        self.assertEqual(result.status, "error")
        self.assertIn("File not found", result.error_message)

    def test_directory_path_rejected(self):
        """Test rejection when a directory path is supplied."""
        result = self.processor.process(self.test_dir)
        self.assertEqual(result.status, "error")
        self.assertIn("Expected a file but received directory", result.error_message)

    def test_zero_byte_empty_file(self):
        """Test zero-byte file detection."""
        file_path = os.path.join(self.test_dir, "zero.pdf")
        with open(file_path, "wb") as f:
            pass  # 0 bytes

        result = self.processor.process(file_path)
        self.assertEqual(result.status, "empty")
        self.assertIn("0 bytes", result.error_message)

    def test_raw_text_preserved_verbatim(self):
        """
        ACADEMIC INTEGRITY CHECK:
        Verify that punctuation, casing, spacing, and capitalization are preserved
        without tokenization, stemming, lemmatization, or stopword removal.
        """
        file_path = os.path.join(self.test_dir, "verbatim.txt")
        raw_input = "  The quick BROWN fox Jumps... over 123 lazy dogs!?  "
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(raw_input)

        result = self.processor.process(file_path)
        self.assertEqual(result.raw_text, raw_input.strip())
        self.assertIn("BROWN", result.raw_text)
        self.assertIn("...", result.raw_text)
        self.assertIn("over", result.raw_text)  # Stopwords NOT removed

    def test_to_dict_method(self):
        """Test that DocumentContent converts cleanly to a Python dictionary."""
        content = DocumentContent(
            filename="test.txt",
            format="txt",
            raw_text="Hello World",
            page_count=None,
            char_count=11,
            line_count=1,
            status="success"
        )
        d = content.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["filename"], "test.txt")
        self.assertEqual(d["format"], "txt")
        self.assertEqual(d["raw_text"], "Hello World")
        self.assertEqual(d["status"], "success")

    def test_extract_document_convenience_function(self):
        """Test the extract_document convenience helper function."""
        file_path = os.path.join(self.test_dir, "helper.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Helper test")

        result = extract_document(file_path)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.raw_text, "Helper test")


if __name__ == "__main__":
    unittest.main()
