"""
Document Processor Subsystem
Responsible strictly for raw text extraction and metadata preservation from
PDF, DOCX, and TXT files.

ACADEMIC RESTRICTION NOTICE:
No NLP, tokenization, stemming, lemmatization, semantic embeddings,
or generative summarisation are applied here. Text is preserved in its
verbatim raw extracted form for subsequent rule-based and symbolic parsing.
"""

import os
import codecs
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
import docx  # python-docx


SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}


@dataclass
class DocumentContent:
    """
    Standardized result structure returned by all document extractors.
    Preserves raw extracted text alongside source metadata and execution status.
    """
    filename: str
    format: str                     # 'pdf', 'docx', 'txt', or 'unknown'
    raw_text: str                   # Verbatim extracted character sequence
    page_count: Optional[int]       # Number of pages (PDF) or None (DOCX/TXT)
    char_count: int                 # Total raw character count
    line_count: int                 # Line count based on newline breaks
    status: str                     # 'success', 'empty', or 'error'
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert document content object to a serializable dictionary."""
        return asdict(self)


class BaseExtractor:
    """Abstract base extractor interface for format-specific handlers."""

    def extract(self, file_path: str) -> DocumentContent:
        """
        Extract raw text and metadata from the specified file.
        Must be implemented by concrete format extractors.
        """
        raise NotImplementedError("Subclasses must implement extract().")


class PDFExtractor(BaseExtractor):
    """
    PDF text extractor using PyMuPDF (fitz).
    Iterates through document pages and extracts raw character streams.
    """

    def extract(self, file_path: str) -> DocumentContent:
        filename = os.path.basename(file_path)
        doc = None
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)

            if page_count == 0:
                return DocumentContent(
                    filename=filename,
                    format='pdf',
                    raw_text="",
                    page_count=0,
                    char_count=0,
                    line_count=0,
                    status='empty',
                    error_message="PDF contains no pages."
                )

            extracted_pages = []
            for page in doc:
                text = page.get_text()
                if text:
                    extracted_pages.append(text)

            raw_text = "\n".join(extracted_pages).strip()

            if not raw_text:
                return DocumentContent(
                    filename=filename,
                    format='pdf',
                    raw_text="",
                    page_count=page_count,
                    char_count=0,
                    line_count=0,
                    status='empty',
                    error_message="Document contains no extractable text."
                )

            return DocumentContent(
                filename=filename,
                format='pdf',
                raw_text=raw_text,
                page_count=page_count,
                char_count=len(raw_text),
                line_count=len(raw_text.splitlines()),
                status='success',
                error_message=None
            )

        except Exception as e:
            return DocumentContent(
                filename=filename,
                format='pdf',
                raw_text="",
                page_count=None,
                char_count=0,
                line_count=0,
                status='error',
                error_message=f"PDF extraction error: {str(e)}"
            )
        finally:
            if doc is not None:
                doc.close()


class DOCXExtractor(BaseExtractor):
    """
    DOCX text extractor using python-docx.
    Extracts text from paragraphs and document tables without semantic modification.
    """

    def extract(self, file_path: str) -> DocumentContent:
        filename = os.path.basename(file_path)
        try:
            doc = docx.Document(file_path)
            extracted_elements = []

            # 1. Extract paragraphs
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    extracted_elements.append(text)

            # 2. Extract tables (preserving row-level association)
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        extracted_elements.append(" | ".join(row_cells))

            raw_text = "\n".join(extracted_elements).strip()

            if not raw_text:
                return DocumentContent(
                    filename=filename,
                    format='docx',
                    raw_text="",
                    page_count=None,
                    char_count=0,
                    line_count=0,
                    status='empty',
                    error_message="Document contains no extractable text."
                )

            return DocumentContent(
                filename=filename,
                format='docx',
                raw_text=raw_text,
                page_count=None,
                char_count=len(raw_text),
                line_count=len(raw_text.splitlines()),
                status='success',
                error_message=None
            )

        except Exception as e:
            return DocumentContent(
                filename=filename,
                format='docx',
                raw_text="",
                page_count=None,
                char_count=0,
                line_count=0,
                status='error',
                error_message=f"DOCX extraction error: {str(e)}"
            )


class TXTExtractor(BaseExtractor):
    """
    Plain text extractor using standard Python file handling.
    Implements reliable BOM detection for UTF-8, UTF-16 LE, and UTF-16 BE,
    followed by sequential decoding fallbacks: UTF-8 -> UTF-16 -> Latin-1.
    """

    def extract(self, file_path: str) -> DocumentContent:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()

            if not raw_bytes:
                return DocumentContent(
                    filename=filename,
                    format='txt',
                    raw_text="",
                    page_count=None,
                    char_count=0,
                    line_count=0,
                    status='empty',
                    error_message="Document contains no extractable text."
                )

            raw_text = None

            # 1. Reliable BOM Detection
            if raw_bytes.startswith(codecs.BOM_UTF8):  # b'\xef\xbb\xbf'
                raw_text = raw_bytes.decode('utf-8-sig')
            elif raw_bytes.startswith(codecs.BOM_UTF16_LE) or raw_bytes.startswith(codecs.BOM_UTF16_BE):
                raw_text = raw_bytes.decode('utf-16')
            else:
                # 2. Heuristic check: UTF-16 without BOM (detect alternating null bytes in ASCII/Latin text)
                if b'\x00' in raw_bytes and len(raw_bytes) % 2 == 0:
                    for utf16_variant in ('utf-16-le', 'utf-16-be', 'utf-16'):
                        try:
                            candidate = raw_bytes.decode(utf16_variant)
                            if '\x00' not in candidate:
                                raw_text = candidate
                                break
                        except UnicodeError:
                            pass

                # 3. Fallback: UTF-8 strict decoding
                if raw_text is None:
                    try:
                        raw_text = raw_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        pass

                # 4. Fallback: UTF-16 decoding (attempted before Latin-1)
                if raw_text is None:
                    try:
                        candidate = raw_bytes.decode('utf-16')
                        # Check candidate doesn't contain unexpected private use characters from 8-bit bytes
                        if candidate and not any(0xE000 <= ord(c) <= 0xF8FF for c in candidate):
                            raw_text = candidate
                    except UnicodeError:
                        pass

                # 5. Final fallback: Latin-1 decoding
                if raw_text is None:
                    try:
                        raw_text = raw_bytes.decode('latin-1')
                    except UnicodeError as err:
                        return DocumentContent(
                            filename=filename,
                            format='txt',
                            raw_text="",
                            page_count=None,
                            char_count=0,
                            line_count=0,
                            status='error',
                            error_message=f"TXT decoding error: {str(err)}"
                        )

            # Preserve decoded source text without newline transformations
            raw_text = raw_text.strip()

            if not raw_text:
                return DocumentContent(
                    filename=filename,
                    format='txt',
                    raw_text="",
                    page_count=None,
                    char_count=0,
                    line_count=0,
                    status='empty',
                    error_message="Document contains no extractable text."
                )

            return DocumentContent(
                filename=filename,
                format='txt',
                raw_text=raw_text,
                page_count=None,
                char_count=len(raw_text),
                line_count=len(raw_text.splitlines()),
                status='success',
                error_message=None
            )

        except Exception as err:
            return DocumentContent(
                filename=filename,
                format='txt',
                raw_text="",
                page_count=None,
                char_count=0,
                line_count=0,
                status='error',
                error_message=f"TXT read error: {str(err)}"
            )


class DocumentProcessor:
    """
    Main entry point for document text extraction.
    Provides format routing, file validation, and error management.
    """

    def __init__(self):
        self._extractors = {
            '.pdf': PDFExtractor(),
            '.docx': DOCXExtractor(),
            '.txt': TXTExtractor()
        }

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check whether the given file extension is supported."""
        suffix = Path(file_path).suffix.lower()
        return suffix in SUPPORTED_EXTENSIONS

    def process(self, file_path: str) -> DocumentContent:
        """
        Validate and extract text from the specified file path.
        Returns a standardized DocumentContent instance.
        """
        filename = os.path.basename(file_path)

        # 1. Check file existence
        if not os.path.exists(file_path):
            return DocumentContent(
                filename=filename,
                format='unknown',
                raw_text="",
                page_count=None,
                char_count=0,
                line_count=0,
                status='error',
                error_message=f"File not found: {file_path}"
            )

        # 2. Check if it's a directory
        if os.path.isdir(file_path):
            return DocumentContent(
                filename=filename,
                format='unknown',
                raw_text="",
                page_count=None,
                char_count=0,
                line_count=0,
                status='error',
                error_message=f"Expected a file but received directory: {file_path}"
            )

        # 3. Validate file extension
        suffix = Path(file_path).suffix.lower()
        if suffix not in self._extractors:
            valid_formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            return DocumentContent(
                filename=filename,
                format=suffix.lstrip('.') if suffix else 'unknown',
                raw_text="",
                page_count=None,
                char_count=0,
                line_count=0,
                status='error',
                error_message=f"Unsupported format '{suffix}'. Supported formats are: {valid_formats}"
            )

        # 4. Check file size (0-byte file check)
        try:
            if os.path.getsize(file_path) == 0:
                format_name = suffix.lstrip('.')
                return DocumentContent(
                    filename=filename,
                    format=format_name,
                    raw_text="",
                    page_count=0 if format_name == 'pdf' else None,
                    char_count=0,
                    line_count=0,
                    status='empty',
                    error_message="Document is empty (0 bytes)."
                )
        except OSError as e:
            return DocumentContent(
                filename=filename,
                format=suffix.lstrip('.'),
                raw_text="",
                page_count=None,
                char_count=0,
                line_count=0,
                status='error',
                error_message=f"OS file access error: {str(e)}"
            )

        # 5. Delegate extraction to format-specific extractor
        extractor = self._extractors[suffix]
        return extractor.extract(file_path)


# Convenience function for quick extraction
def extract_document(file_path: str) -> DocumentContent:
    """Helper function to extract document content in a single call."""
    processor = DocumentProcessor()
    return processor.process(file_path)
