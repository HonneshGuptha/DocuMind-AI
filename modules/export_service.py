"""
DocuMind-AI Export Service Module
Provides structured JSON and academic PDF report generation for the workspace.

ACADEMIC COMPLIANCE:
Classical Symbolic AI (AID201A).
Deterministic provenance preservation, anti-fabrication integrity,
and structured verifiable output.
"""

import io
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF


def get_current_timestamp_str() -> str:
    """Return formatted UTC/local ISO timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_workspace_json(service) -> Dict[str, Any]:
    """
    Compile full structured workspace analysis data as a valid JSON-serializable dictionary.
    Includes metadata, documents, SPO facts, question history, reasoning rules,
    and evaluation benchmark metrics.
    """
    from modules.app_service import normalize_filename
    from modules.reasoning_engine.rule_base import get_all_rules

    # 1. Workspace Documents
    docs = service.repo.get_all_documents()
    documents_data = [
        {
            'document_id': d.document_id,
            'filename': normalize_filename(d.filename),
            'file_format': d.file_format,
            'upload_date': str(d.upload_date),
            'extraction_status': d.extraction_status
        }
        for d in docs
    ]

    # 2. Extracted SPO Facts
    facts = service.repo.get_all_facts(limit=None)
    doc_map = {d.document_id: normalize_filename(d.filename) for d in docs}
    facts_data = [
        {
            'fact_id': f.fact_id,
            'document_id': f.document_id,
            'filename': doc_map.get(f.document_id, f"Document #{f.document_id}"),
            'subject': f.subject,
            'predicate': f.predicate,
            'object': f.object,
            'source_sentence': f.source_sentence,
            'sentence_index': f.sentence_index
        }
        for f in facts
    ]

    # 3. Question-Answering History
    history = service.repo.get_query_history(limit=200)

    # 4. Inference Rules
    rules = [
        {
            'rule_id': r.rule_id,
            'name': getattr(r, 'rule_name', r.rule_id),
            'description': r.description,
            'premises': [(p[0], p[1], p[2]) if isinstance(p, (tuple, list)) else getattr(p, 'triple', str(p)) for p in r.premises],
            'conclusion': (r.conclusion[0], r.conclusion[1], r.conclusion[2]) if isinstance(r.conclusion, (tuple, list)) else getattr(r.conclusion, 'triple', str(r.conclusion))
        }
        for r in get_all_rules()
    ]

    # 5. Live / Cached Evaluation Benchmark
    try:
        eval_result = service.run_system_evaluation()
        eval_summary = eval_result.get('summary', {})
        eval_cases = eval_result.get('cases', [])
    except Exception as e:
        eval_summary = {'error': f"Evaluation could not be completed: {str(e)}"}
        eval_cases = []

    return {
        'report_metadata': {
            'title': 'DocuMind AI — Workspace Analysis Report',
            'generated_at': get_current_timestamp_str(),
            'application': 'DocuMind-AI',
            'version': '0.1.0',
            'course': 'AID201A – Artificial Intelligence',
            'institution': 'M S Ramaiah University of Applied Sciences',
            'paradigm': 'Classical Symbolic Knowledge-Based AI (Strictly No NLP/LLMs/Embeddings)',
            'total_documents': len(documents_data),
            'total_facts': len(facts_data),
            'total_queries': len(history),
            'active_rules': len(rules)
        },
        'documents': documents_data,
        'knowledge_base_facts': facts_data,
        'query_history': history,
        'inference_rules': rules,
        'evaluation': {
            'summary': eval_summary,
            'cases': eval_cases
        }
    }


class PDFReportBuilder:
    """
    Constructs an academic PDF analysis report using PyMuPDF (fitz).
    Enforces multi-page flow, headers, footers with page numbers,
    and structured visual hierarchy.
    """

    # Palette
    COLOR_NAVY = (15 / 255, 23 / 255, 42 / 255)       # #0F172A
    COLOR_DARK = (23 / 255, 34 / 255, 56 / 255)       # #172238
    COLOR_INDIGO = (83 / 255, 105 / 255, 216 / 255)   # #5369D8
    COLOR_CYAN = (34 / 255, 184 / 255, 207 / 255)     # #22B8CF
    COLOR_BG = (245 / 255, 247 / 255, 251 / 255)      # #F5F7FB
    COLOR_BORDER = (226 / 255, 232 / 255, 240 / 255)  # #E2E8F0
    COLOR_TEXT = (23 / 255, 32 / 255, 51 / 255)       # #172033
    COLOR_MUTED = (100 / 255, 116 / 255, 139 / 255)   # #64748B
    COLOR_WHITE = (1.0, 1.0, 1.0)
    COLOR_GREEN = (22 / 255, 163 / 255, 74 / 255)     # #16A34A

    PAGE_WIDTH = 595.32   # A4 Width
    PAGE_HEIGHT = 841.92  # A4 Height
    MARGIN_LEFT = 45.0
    MARGIN_RIGHT = 550.0
    USABLE_WIDTH = MARGIN_RIGHT - MARGIN_LEFT
    MARGIN_TOP = 50.0
    MARGIN_BOTTOM = 790.0

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.doc = fitz.open()
        self.current_page = None
        self.y = self.MARGIN_TOP

    def new_page(self):
        """Add a new page and set header."""
        self.current_page = self.doc.new_page(width=self.PAGE_WIDTH, height=self.PAGE_HEIGHT)
        self.y = self.MARGIN_TOP

    def check_space(self, height_needed: float):
        """Ensure current page has sufficient space, else advance to a fresh page."""
        if self.current_page is None or (self.y + height_needed > self.MARGIN_BOTTOM):
            self.new_page()

    def draw_cover_header(self):
        """Draw top branding bar and document title."""
        self.check_space(110)
        p = self.current_page

        # Top dark header box
        header_rect = fitz.Rect(self.MARGIN_LEFT, self.y, self.MARGIN_RIGHT, self.y + 75)
        p.draw_rect(header_rect, fill=self.COLOR_NAVY, color=None)

        # Title text
        p.insert_text(
            (self.MARGIN_LEFT + 15, self.y + 26),
            "DOCUMIND AI — WORKSPACE ANALYSIS REPORT",
            fontsize=13,
            fontname="helv",
            color=self.COLOR_WHITE
        )

        p.insert_text(
            (self.MARGIN_LEFT + 15, self.y + 44),
            "Knowledge-Based Document Question Answering System",
            fontsize=9.5,
            fontname="helv",
            color=self.COLOR_CYAN
        )

        meta = self.data.get('report_metadata', {})
        p.insert_text(
            (self.MARGIN_LEFT + 15, self.y + 60),
            f"Course: {meta.get('course')}  •  Institution: {meta.get('institution')}  •  Generated: {meta.get('generated_at')}",
            fontsize=7.5,
            fontname="helv",
            color=self.COLOR_MUTED
        )

        self.y += 90

    def draw_section_heading(self, title: str):
        """Draw prominent section header."""
        self.check_space(32)
        p = self.current_page
        
        # Indigo accent bar
        p.draw_rect(fitz.Rect(self.MARGIN_LEFT, self.y + 2, self.MARGIN_LEFT + 4, self.y + 16), fill=self.COLOR_INDIGO, color=None)
        p.insert_text((self.MARGIN_LEFT + 10, self.y + 14), title, fontsize=11, fontname="helv", color=self.COLOR_NAVY)
        p.draw_line(
            fitz.Point(self.MARGIN_LEFT + 10, self.y + 18),
            fitz.Point(self.MARGIN_RIGHT, self.y + 18),
            color=self.COLOR_BORDER,
            width=0.75
        )
        self.y += 26

    def draw_metrics_summary(self):
        """Draw 4 summary metric boxes."""
        self.check_space(60)
        p = self.current_page
        meta = self.data.get('report_metadata', {})
        
        box_w = (self.USABLE_WIDTH - 30) / 4.0
        box_h = 45.0

        metrics = [
            ("DOCUMENTS", str(meta.get('total_documents', 0)), "Registered & parsed"),
            ("KB FACTS", str(meta.get('total_facts', 0)), "SPO Triples stored"),
            ("QUERIES ASKED", str(meta.get('total_queries', 0)), "Persisted interactions"),
            ("ACTIVE RULES", str(meta.get('active_rules', 4)), "Deductive rules (R1-R4)")
        ]

        for i, (label, val, desc) in enumerate(metrics):
            bx = self.MARGIN_LEFT + i * (box_w + 10)
            rect = fitz.Rect(bx, self.y, bx + box_w, self.y + box_h)
            p.draw_rect(rect, fill=self.COLOR_BG, color=self.COLOR_BORDER, width=0.75)
            p.insert_text((bx + 10, self.y + 13), label, fontsize=6.5, fontname="helv", color=self.COLOR_MUTED)
            p.insert_text((bx + 10, self.y + 29), val, fontsize=14, fontname="helv", color=self.COLOR_INDIGO)
            p.insert_text((bx + 10, self.y + 39), desc, fontsize=6, fontname="helv", color=self.COLOR_MUTED)

        self.y += 58

    def draw_documents_table(self):
        """Render uploaded documents."""
        self.draw_section_heading("1. WORKSPACE DOCUMENTS & INGESTION STATUS")
        docs = self.data.get('documents', [])
        p = self.current_page

        if not docs:
            self.check_space(20)
            self.current_page.insert_text((self.MARGIN_LEFT + 5, self.y + 10), "No documents ingested in this workspace.", fontsize=8, fontname="helv", color=self.COLOR_MUTED)
            self.y += 22
            return

        # Table header
        self.check_space(35)
        p = self.current_page
        p.draw_rect(fitz.Rect(self.MARGIN_LEFT, self.y, self.MARGIN_RIGHT, self.y + 16), fill=self.COLOR_BG, color=self.COLOR_BORDER, width=0.5)
        p.insert_text((self.MARGIN_LEFT + 8, self.y + 11), "Doc ID", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 60, self.y + 11), "Filename", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 260, self.y + 11), "Format", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 320, self.y + 11), "Ingested Timestamp", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 440, self.y + 11), "Extraction Status", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        self.y += 18

        for d in docs:
            self.check_space(16)
            p = self.current_page
            p.insert_text((self.MARGIN_LEFT + 8, self.y + 10), f"#{d['document_id']}", fontsize=7.5, fontname="helv", color=self.COLOR_INDIGO)
            p.insert_text((self.MARGIN_LEFT + 60, self.y + 10), str(d['filename'])[:36], fontsize=7.5, fontname="helv", color=self.COLOR_TEXT)
            p.insert_text((self.MARGIN_LEFT + 260, self.y + 10), str(d['file_format']).upper(), fontsize=7.5, fontname="helv", color=self.COLOR_MUTED)
            p.insert_text((self.MARGIN_LEFT + 320, self.y + 10), str(d['upload_date'])[:19], fontsize=7, fontname="helv", color=self.COLOR_MUTED)
            status_color = self.COLOR_GREEN if d['extraction_status'] == 'success' else self.COLOR_INDIGO
            p.insert_text((self.MARGIN_LEFT + 440, self.y + 10), str(d['extraction_status']), fontsize=7.5, fontname="helv", color=status_color)
            p.draw_line(
                fitz.Point(self.MARGIN_LEFT, self.y + 13),
                fitz.Point(self.MARGIN_RIGHT, self.y + 13),
                color=self.COLOR_BORDER,
                width=0.4
            )
            self.y += 15

        self.y += 10

    def draw_facts_summary(self):
        """Render knowledge base SPO facts."""
        self.draw_section_heading("2. KNOWLEDGE BASE — RELATIONAL SPO TRIPLES")
        facts = self.data.get('knowledge_base_facts', [])
        p = self.current_page

        if not facts:
            self.check_space(20)
            self.current_page.insert_text((self.MARGIN_LEFT + 5, self.y + 10), "Knowledge base is empty.", fontsize=8, fontname="helv", color=self.COLOR_MUTED)
            self.y += 22
            return

        # Display up to first 25 facts in PDF with note
        display_facts = facts[:25]
        self.check_space(35)
        p = self.current_page
        p.draw_rect(fitz.Rect(self.MARGIN_LEFT, self.y, self.MARGIN_RIGHT, self.y + 16), fill=self.COLOR_BG, color=self.COLOR_BORDER, width=0.5)
        p.insert_text((self.MARGIN_LEFT + 6, self.y + 11), "ID", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 35, self.y + 11), "Subject", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 155, self.y + 11), "Predicate", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 255, self.y + 11), "Object", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        p.insert_text((self.MARGIN_LEFT + 375, self.y + 11), "Source Sentence / Provenance", fontsize=7.5, fontname="helv", color=self.COLOR_NAVY)
        self.y += 18

        for f in display_facts:
            self.check_space(16)
            p = self.current_page
            p.insert_text((self.MARGIN_LEFT + 6, self.y + 10), f"#{f['fact_id']}", fontsize=7, fontname="helv", color=self.COLOR_INDIGO)
            p.insert_text((self.MARGIN_LEFT + 35, self.y + 10), str(f['subject'])[:22], fontsize=7.5, fontname="helv", color=self.COLOR_TEXT)
            p.insert_text((self.MARGIN_LEFT + 155, self.y + 10), str(f['predicate'])[:18], fontsize=7, fontname="helv", color=self.COLOR_MUTED)
            p.insert_text((self.MARGIN_LEFT + 255, self.y + 10), str(f['object'])[:22], fontsize=7.5, fontname="helv", color=self.COLOR_TEXT)
            sent_str = f"\"{f.get('source_sentence', '')}\""
            p.insert_text((self.MARGIN_LEFT + 375, self.y + 10), sent_str[:32], fontsize=6.5, fontname="helv", color=self.COLOR_MUTED)
            p.draw_line(fitz.Point(self.MARGIN_LEFT, self.y + 13), fitz.Point(self.MARGIN_RIGHT, self.y + 13), color=self.COLOR_BORDER, width=0.4)
            self.y += 15

        if len(facts) > 25:
            self.check_space(18)
            self.current_page.insert_text(
                (self.MARGIN_LEFT + 5, self.y + 12),
                f"... and {len(facts) - 25} more facts stored in SQLite (all exported in full JSON export).",
                fontsize=7.5,
                fontname="helv",
                color=self.COLOR_MUTED
            )
            self.y += 20

        self.y += 10

    def draw_queries_history(self):
        """Render question answering history."""
        self.draw_section_heading("3. QUESTION ANSWERING & REASONING INTERACTIONS")
        queries = self.data.get('query_history', [])

        if not queries:
            self.check_space(20)
            self.current_page.insert_text((self.MARGIN_LEFT + 5, self.y + 10), "No questions recorded in this workspace session yet.", fontsize=8, fontname="helv", color=self.COLOR_MUTED)
            self.y += 22
            return

        for q in queries[:15]:
            self.check_space(55)
            p = self.current_page

            # Card background
            card_rect = fitz.Rect(self.MARGIN_LEFT, self.y, self.MARGIN_RIGHT, self.y + 48)
            p.draw_rect(card_rect, fill=self.COLOR_BG, color=self.COLOR_BORDER, width=0.5)

            # Question and Status
            p.insert_text((self.MARGIN_LEFT + 10, self.y + 13), f"Q: {q.get('question_text', '')}", fontsize=8.5, fontname="helv", color=self.COLOR_NAVY)
            status_str = f"Status: {q.get('answer_status', '').upper()} ({q.get('answer_type', '')})"
            p.insert_text((self.MARGIN_RIGHT - 160, self.y + 13), status_str, fontsize=7, fontname="helv", color=self.COLOR_INDIGO)

            # Answer
            ans_str = f"Answer: {q.get('answer_text', '')}"
            p.insert_text((self.MARGIN_LEFT + 10, self.y + 27), ans_str[:90], fontsize=8, fontname="helv", color=self.COLOR_TEXT)

            # Provenance & Rules
            supp = q.get('supporting_fact_ids', [])
            rules = q.get('applied_rule_ids', [])
            prov_str = f"Supporting Facts: {supp if supp else 'None'}  •  Rules Applied: {rules if rules else 'None'}  •  Latency: {q.get('execution_time_ms', 0)} ms"
            p.insert_text((self.MARGIN_LEFT + 10, self.y + 40), prov_str, fontsize=6.8, fontname="helv", color=self.COLOR_MUTED)

            self.y += 54

        if len(queries) > 15:
            self.check_space(18)
            self.current_page.insert_text(
                (self.MARGIN_LEFT + 5, self.y + 12),
                f"... and {len(queries) - 15} more queries recorded (full history available in JSON export).",
                fontsize=7.5,
                fontname="helv",
                color=self.COLOR_MUTED
            )
            self.y += 20

        self.y += 10

    def draw_evaluation_summary(self):
        """Render empirical evaluation benchmark metrics."""
        self.draw_section_heading("4. EMPIRICAL EVALUATION BENCHMARK (21 TEST CASES)")
        eval_data = self.data.get('evaluation', {})
        summary = eval_data.get('summary', {})

        if not summary or 'error' in summary:
            self.check_space(20)
            self.current_page.insert_text((self.MARGIN_LEFT + 5, self.y + 10), "Evaluation not executed or unavailable.", fontsize=8, fontname="helv", color=self.COLOR_MUTED)
            self.y += 22
            return

        self.check_space(95)
        p = self.current_page

        def fmt_pct(val):
            if val is None or val == 'N/A':
                return "N/A"
            try:
                return f"{float(val) * 100:.1f}%"
            except Exception:
                return str(val)

        # 2-column key metrics display
        col_w = (self.USABLE_WIDTH - 20) / 2.0
        metrics_left = [
            ("Overall Pipeline Success", f"{summary.get('successful_cases', 0)}/{summary.get('total_cases', 0)} ({fmt_pct(summary.get('answer_accuracy'))})"),
            ("Answer Status Accuracy", fmt_pct(summary.get('answer_accuracy'))),
            ("Retrieval Precision", fmt_pct(summary.get('retrieval_precision'))),
            ("Retrieval Recall", fmt_pct(summary.get('retrieval_recall'))),
            ("Retrieval F1 Score", fmt_pct(summary.get('retrieval_f1'))),
        ]
        metrics_right = [
            ("Reasoning / Inference Correctness", fmt_pct(summary.get('reasoning_accuracy'))),
            ("Rule Application Accuracy", fmt_pct(summary.get('rule_precision'))),
            ("Supporting Ground Premise Fidelity", fmt_pct(summary.get('supporting_fact_accuracy'))),
            ("Source Provenance Accuracy", fmt_pct(summary.get('provenance_accuracy'))),
            ("Average Query Latency", f"{summary.get('average_execution_time_ms', 0):.2f} ms"),
        ]

        # Draw left table box
        p.draw_rect(fitz.Rect(self.MARGIN_LEFT, self.y, self.MARGIN_LEFT + col_w, self.y + 85), fill=self.COLOR_BG, color=self.COLOR_BORDER, width=0.5)
        for i, (k, v) in enumerate(metrics_left):
            py = self.y + 14 + i * 15
            p.insert_text((self.MARGIN_LEFT + 10, py), k, fontsize=7, fontname="helv", color=self.COLOR_TEXT)
            p.insert_text((self.MARGIN_LEFT + col_w - 65, py), v, fontsize=7.5, fontname="helv", color=self.COLOR_INDIGO)

        # Draw right table box
        rx = self.MARGIN_LEFT + col_w + 20
        p.draw_rect(fitz.Rect(rx, self.y, rx + col_w, self.y + 85), fill=self.COLOR_BG, color=self.COLOR_BORDER, width=0.5)
        for i, (k, v) in enumerate(metrics_right):
            py = self.y + 14 + i * 15
            p.insert_text((rx + 10, py), k, fontsize=7, fontname="helv", color=self.COLOR_TEXT)
            p.insert_text((rx + col_w - 65, py), v, fontsize=7.5, fontname="helv", color=self.COLOR_INDIGO)

        self.y += 98

    def apply_footers(self):
        """Add academic footer and page numbers to all pages."""
        total_pages = len(self.doc)
        for idx, page in enumerate(self.doc):
            # Thin divider
            page.draw_line(
                fitz.Point(self.MARGIN_LEFT, self.PAGE_HEIGHT - 35),
                fitz.Point(self.MARGIN_RIGHT, self.PAGE_HEIGHT - 35),
                color=self.COLOR_BORDER,
                width=0.5
            )
            # Left note
            page.insert_text(
                (self.MARGIN_LEFT, self.PAGE_HEIGHT - 22),
                "DocuMind AI  •  Classical Symbolic Knowledge-Based AI  •  AID201A MSRUAS",
                fontsize=7,
                fontname="helv",
                color=self.COLOR_MUTED
            )
            # Right page number
            page_text = f"Page {idx + 1} of {total_pages}"
            page.insert_text(
                (self.MARGIN_RIGHT - 50, self.PAGE_HEIGHT - 22),
                page_text,
                fontsize=7,
                fontname="helv",
                color=self.COLOR_MUTED
            )

    def build_pdf_bytes(self) -> bytes:
        """Assemble all sections and return PDF bytes."""
        self.new_page()
        self.draw_cover_header()
        self.draw_metrics_summary()
        self.draw_documents_table()
        self.draw_facts_summary()
        self.draw_queries_history()
        self.draw_evaluation_summary()
        self.apply_footers()
        
        pdf_bytes = self.doc.write()
        self.doc.close()
        return pdf_bytes


def generate_workspace_pdf(service) -> bytes:
    """Generate professional academic PDF report bytes for the current workspace."""
    workspace_data = generate_workspace_json(service)
    builder = PDFReportBuilder(workspace_data)
    return builder.build_pdf_bytes()
