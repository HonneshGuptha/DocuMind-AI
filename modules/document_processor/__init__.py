"""
Document Processor Subsystem
Provides raw text extraction and metadata preservation for PDF, DOCX, and TXT files.

Exports:
- DocumentProcessor: Main processor with validation and format routing.
- DocumentContent: Dataclass holding extracted raw text and metadata.
- extract_document: Quick helper function for single-call extraction.
- PDFExtractor: PyMuPDF-based PDF text extractor.
- DOCXExtractor: python-docx-based DOCX text extractor.
- TXTExtractor: Standard library TXT text extractor.
- SUPPORTED_EXTENSIONS: Set of supported extensions ('.pdf', '.docx', '.txt').
"""

from .extractor import (
    DocumentProcessor,
    DocumentContent,
    extract_document,
    BaseExtractor,
    PDFExtractor,
    DOCXExtractor,
    TXTExtractor,
    SUPPORTED_EXTENSIONS,
)

__all__ = [
    'DocumentProcessor',
    'DocumentContent',
    'extract_document',
    'BaseExtractor',
    'PDFExtractor',
    'DOCXExtractor',
    'TXTExtractor',
    'SUPPORTED_EXTENSIONS',
]
