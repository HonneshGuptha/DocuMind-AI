"""
DocuMind-AI Application Service Layer
Orchestration service coordinating:
- Document Ingestion & Text Extraction (Module 1)
- Knowledge Base Repository & Fact Storage (Module 2)
- Question Classification, Parsing & Retrieval (Module 3)
- Symbolic Reasoning Engine: Forward & Backward Chaining (Module 4)
- Deterministic Answer & Explanation Generation (Module 5 Stages 1 & 2)
- Performance & Reliability Evaluation Framework (Module 5 Stage 3)

ACADEMIC COMPLIANCE:
Strictly adheres to classical symbolic AI principles.
Zero NLP, zero LLMs, zero vector embeddings, zero probabilistic hallucinations.
"""

import os
import re
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from werkzeug.utils import secure_filename

from modules.document_processor import extract_document, DocumentContent
from modules.knowledge_base.database import init_db
from modules.knowledge_base.models import Fact as KBF, DocumentRecord
from modules.knowledge_base.repository import KnowledgeBaseRepository
from modules.knowledge_base.entity_registry import GLOBAL_ENTITY_REGISTRY
from modules.question_processor.classifier import QuestionClassifier
from modules.question_processor.parser import QuestionParser
from modules.question_processor.query_models import StructuredQuery, RetrievalResult
from modules.question_processor.retrieval import QuestionRetriever
from modules.reasoning_engine.reasoning_models import Fact as ReasoningFact, ReasoningResult
from modules.reasoning_engine.working_memory import WorkingMemory
from modules.reasoning_engine.rule_base import get_all_rules, ALL_RULES, get_rule_by_id
from modules.reasoning_engine.forward_chaining import forward_chain, ForwardChainer
from modules.reasoning_engine.backward_chaining import backward_chain, BackwardChainer, ProofResult, ProofStatus
from modules.answer_generator.answer_models import AnswerResponse
from modules.answer_generator.generator import AnswerGenerator, RULE_DISPLAY_NAMES
from modules.answer_generator.explanation_generator import ExplanationGenerator
from modules.evaluation.evaluator import run_evaluation, SystemEvaluator
from modules.evaluation.evaluation_models import EvaluationSummary


def normalize_filename(filename: Optional[str]) -> str:
    """
    Ensure recognized extensions (.txt, .pdf, .docx) appear exactly once at the end.
    e.g. 'india.txt.txt' -> 'india.txt'
         'report.pdf.pdf' -> 'report.pdf'
         'document.docx.docx' -> 'document.docx'
    """
    if not filename:
        return ""
    name = str(filename).strip()
    valid_exts = ('.txt', '.pdf', '.docx')
    while True:
        lower_name = name.lower()
        cleaned = False
        for ext in valid_exts:
            for prev_ext in valid_exts:
                stacked = prev_ext + ext
                if lower_name.endswith(stacked):
                    name = name[:-len(ext)]
                    cleaned = True
                    break
            if cleaned:
                break
        if not cleaned:
            break
    return name


def entity_matches_scope(entity_name: str, scope: str, kb_facts: List[Any], entity_reg) -> bool:
    """
    Deterministically check whether an entity matches a requested geographical scope
    (country, continent, state, city) using canonical registry types, standard ontology,
    and relational deduction from KB facts.
    """
    if not entity_name or not scope:
        return False
    norm = entity_reg.normalize_name(entity_name)
    scope = scope.lower().strip()

    # 1. Canonical entity registry lookup
    if norm in entity_reg._name_to_id:
        eid = entity_reg._name_to_id[norm]
        etype = entity_reg._entities[eid].entity_type.lower()
        if etype == scope:
            return True

    # 2. Standard geographic continents ontology
    if scope == 'continent':
        if norm in {'asia', 'africa', 'europe', 'north america', 'south america', 'australia', 'antarctica'}:
            return True

    # 3. Dynamic relational deduction from knowledge base facts
    norm_ent = entity_name.strip().lower()
    for f in kb_facts:
        f_s = f.subject.strip().lower()
        f_p = f.predicate.strip().lower()
        f_o = f.object.strip().lower()

        if scope == 'country':
            # "X is a state/province in Y" -> Y is a Country
            if f_p in ('is a state in', 'is a province in') and f_o == norm_ent:
                return True
            # "Y is a country in Z" -> Y is a Country
            if f_p == 'is a country in' and f_s == norm_ent:
                return True
            if f_p == 'is a' and f_s == norm_ent and f_o == 'country':
                return True

        elif scope == 'continent':
            # "Y is a country in Z" -> Z is a Continent
            if f_p == 'is a country in' and f_o == norm_ent:
                return True
            # "Z is a continent in W" -> Z is a Continent
            if f_p == 'is a continent in' and f_s == norm_ent:
                return True
            if f_p == 'is a' and f_s == norm_ent and f_o == 'continent':
                return True

        elif scope == 'state':
            # "X is a city in Y" -> Y is a State
            if f_p == 'is a city in' and f_o == norm_ent:
                return True
            # "Y is a state in Z" -> Y is a State
            if f_p == 'is a state in' and f_s == norm_ent:
                return True
            if f_p == 'is a' and f_s == norm_ent and f_o == 'state':
                return True

        elif scope == 'city':
            # "X is a city in Y" -> X is a City
            if f_p == 'is a city in' and f_s == norm_ent:
                return True
            if f_p == 'is a' and f_s == norm_ent and f_o == 'city':
                return True

    return False


class DocuMindService:
    """
    Unified application orchestration service for DocuMind AI.
    Coordinates operations across all 5 modules with end-to-end provenance.
    """

    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

    def __init__(self,
                 db_path: Optional[str] = None,
                 upload_folder: Optional[str] = None):
        """
        Initialize the DocuMind service.
        
        Args:
            db_path: Path to the SQLite database file.
            upload_folder: Path to directory for persisting uploaded documents.
        """
        self.db_path = db_path
        self.upload_folder = os.path.abspath(upload_folder or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads'))
        os.makedirs(self.upload_folder, exist_ok=True)

        # Initialize SQLite Knowledge Base
        init_db(self.db_path)
        self.repo = KnowledgeBaseRepository(self.db_path)

        # Initialize Question Processing Components
        self.classifier = QuestionClassifier()
        self.parser = QuestionParser()
        self.retriever = QuestionRetriever(self.repo)

        # Initialize Answer & Explanation Generators
        self.answer_generator = AnswerGenerator()
        self.explanation_generator = ExplanationGenerator()

    # ==========================================================================
    # 1. Fact & Document Lookup Helpers
    # ==========================================================================

    def get_fact_by_id(self, fact_id: int) -> Optional[KBF]:
        """Fetch a specific Fact by fact_id from SQLite."""
        conn = self.repo._get_connection()
        try:
            cursor = conn.execute(
                "SELECT fact_id, document_id, subject, predicate, object, source_sentence, sentence_index "
                "FROM facts WHERE fact_id = ?",
                (fact_id,)
            )
            row = cursor.fetchone()
            if row:
                return self.repo._row_to_fact(row)
            return None
        finally:
            conn.close()

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Aggregate live system metrics for the dashboard:
        - Total documents registered.
        - Successful vs failed extractions.
        - Total extracted facts.
        - Recent documents list.
        """
        docs = self.repo.get_all_documents()
        total_docs = len(docs)
        successful_docs = sum(1 for d in docs if d.extraction_status == 'success')
        failed_docs = total_docs - successful_docs
        total_facts = self.repo.count_facts()

        # Build recent documents summary
        recent_docs = []
        for d in reversed(docs[-10:]):
            doc_facts = self.repo.get_facts_by_document(d.document_id)
            recent_docs.append({
                'document_id': d.document_id,
                'filename': normalize_filename(d.filename),
                'file_format': d.file_format.upper() if d.file_format else 'UNKNOWN',
                'upload_date': d.upload_date,
                'extraction_status': d.extraction_status,
                'fact_count': len(doc_facts)
            })

        return {
            'total_documents': total_docs,
            'successful_documents': successful_docs,
            'failed_documents': failed_docs,
            'total_facts': total_facts,
            'total_queries': self.repo.count_queries(),
            'active_rules_count': len(get_all_rules()),
            'recent_documents': recent_docs
        }

    def get_all_facts(self, limit: Optional[int] = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve a paginated list of all stored SPO facts with document provenance.
        """
        facts = self.repo.get_all_facts(limit=limit, offset=offset)
        docs = {d.document_id: normalize_filename(d.filename) for d in self.repo.get_all_documents()}

        fact_list = []
        for f in facts:
            fact_list.append({
                'fact_id': f.fact_id,
                'document_id': f.document_id,
                'filename': docs.get(f.document_id, f"Document #{f.document_id}"),
                'subject': f.subject,
                'predicate': f.predicate,
                'object': f.object,
                'source_sentence': f.source_sentence,
                'sentence_index': f.sentence_index
            })
        return fact_list

    # ==========================================================================
    # 2. Document Processing & Ingestion
    # ==========================================================================

    def process_uploaded_file(self, file_storage, force: bool = False) -> Dict[str, Any]:
        """
        Validate, securely store, extract, and ingest an uploaded document.
        
        Args:
            file_storage: A Werkzeug FileStorage object or compatible file wrapper.
            force: If True, bypass duplicate document detection.
            
        Returns:
            Dictionary detailing extraction status, document_id, and extracted facts.
        """
        # 1. Validate file presence
        if not file_storage or not getattr(file_storage, 'filename', None):
            return {
                'status': 'error',
                'message': 'No file was provided for upload.',
                'facts_extracted': 0
            }

        original_filename = file_storage.filename.strip()
        if not original_filename:
            return {
                'status': 'error',
                'message': 'Empty filename provided.',
                'facts_extracted': 0
            }

        # 2. Validate file extension
        ext = ''
        if '.' in original_filename:
            ext = original_filename.rsplit('.', 1)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return {
                'status': 'error',
                'message': f"Unsupported file extension '{ext}'. Only PDF, DOCX, and TXT files are permitted.",
                'facts_extracted': 0
            }

        # 3. Sanitize filename to prevent directory traversal & ensure single extension
        sanitized_name = secure_filename(original_filename)
        sanitized_name = normalize_filename(sanitized_name)
        if not sanitized_name:
            sanitized_name = f"document.{ext}"

        # 4. Measure file size and check bounds
        try:
            file_storage.seek(0, os.SEEK_END)
            size = file_storage.tell()
            file_storage.seek(0)
        except Exception:
            size = getattr(file_storage, 'content_length', 0)

        if size == 0:
            doc_id = self.repo.add_document(
                filename=sanitized_name,
                file_format=ext,
                extraction_status='empty_file'
            )
            return {
                'status': 'error',
                'message': 'Uploaded file is empty (0 bytes).',
                'document_id': doc_id,
                'facts_extracted': 0
            }

        if size > self.MAX_FILE_SIZE:
            return {
                'status': 'error',
                'message': f'File exceeds the maximum permitted size of 16 MB ({round(size / (1024*1024), 2)} MB).',
                'facts_extracted': 0
            }

        # 5. Prevent path traversal with safe destination path
        unique_prefix = uuid.uuid4().hex[:8]
        safe_stored_name = f"{unique_prefix}_{sanitized_name}"
        destination_path = os.path.abspath(os.path.join(self.upload_folder, safe_stored_name))

        # Check path stays strictly inside upload folder
        if os.path.commonpath([self.upload_folder, destination_path]) != self.upload_folder:
            return {
                'status': 'error',
                'message': 'Unsafe file path or directory traversal attempt detected.',
                'facts_extracted': 0
            }

        # 6. Check for duplicate upload
        if not force:
            all_docs = self.repo.get_all_documents()
            existing_doc = next(
                (d for d in all_docs if d.filename == sanitized_name and d.extraction_status == 'success'),
                None
            )
            if existing_doc:
                existing_facts = self.repo.get_facts_by_document(existing_doc.document_id)
                return {
                    'status': 'duplicate',
                    'is_duplicate': True,
                    'message': f"Document '{sanitized_name}' has already been processed (Document #{existing_doc.document_id}).",
                    'document_id': existing_doc.document_id,
                    'filename': sanitized_name,
                    'file_format': ext,
                    'char_count': size,
                    'word_count': 0,
                    'facts_extracted': len(existing_facts),
                    'extracted_facts': [
                        {
                            'fact_id': f.fact_id,
                            'subject': f.subject,
                            'predicate': f.predicate,
                            'object': f.object,
                            'source_sentence': f.source_sentence,
                            'sentence_index': f.sentence_index
                        }
                        for f in existing_facts
                    ]
                }

        # 7. Persist file to disk
        try:
            file_storage.save(destination_path)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to write file to storage: {str(e)}',
                'facts_extracted': 0
            }

        # 8. Extract Document Text via Module 1
        try:
            extracted = extract_document(destination_path)
        except Exception as e:
            doc_id = self.repo.add_document(
                filename=sanitized_name,
                file_format=ext,
                extraction_status='extraction_error'
            )
            return {
                'status': 'error',
                'message': f'Text extraction encountered an unhandled error: {str(e)}',
                'document_id': doc_id,
                'facts_extracted': 0
            }

        # 9. Handle extraction failures & scanned PDFs
        if extracted.status != 'success':
            doc_id = self.repo.add_document(
                filename=sanitized_name,
                file_format=extracted.format or ext,
                extraction_status=extracted.status
            )
            return {
                'status': 'error',
                'message': f"Document extraction failed: {extracted.status.replace('_', ' ').capitalize()}.",
                'document_id': doc_id,
                'facts_extracted': 0
            }

        # Detect scanned/image-only PDF with zero text
        if extracted.format == 'pdf' and (not extracted.raw_text or not extracted.raw_text.strip()):
            doc_id = self.repo.add_document(
                filename=sanitized_name,
                file_format='pdf',
                extraction_status='scanned_pdf_no_text'
            )
            return {
                'status': 'error',
                'message': 'Scanned or image-only PDF detected with no extractable text. (OCR is not supported under symbolic AI principles).',
                'document_id': doc_id,
                'facts_extracted': 0
            }

        if not extracted.raw_text or not extracted.raw_text.strip():
            doc_id = self.repo.add_document(
                filename=sanitized_name,
                file_format=extracted.format,
                extraction_status='empty_text'
            )
            return {
                'status': 'error',
                'message': 'Document contains no extractable text content.',
                'document_id': doc_id,
                'facts_extracted': 0
            }

        # 10. Store Document & Extracted Facts in Knowledge Base
        # Use clean sanitized filename in database records
        extracted.filename = sanitized_name
        doc_id, facts = self.repo.store_document_content(extracted)
        word_count = len(extracted.raw_text.split()) if extracted.raw_text else 0

        return {
            'status': 'success',
            'message': f"Successfully processed '{sanitized_name}' ({extracted.char_count} chars, {word_count} words). Extracted {len(facts)} knowledge facts.",
            'document_id': doc_id,
            'filename': sanitized_name,
            'file_format': extracted.format,
            'char_count': extracted.char_count,
            'word_count': word_count,
            'facts_extracted': len(facts),
            'extracted_facts': [
                {
                    'fact_id': f.fact_id,
                    'subject': f.subject,
                    'predicate': f.predicate,
                    'object': f.object,
                    'source_sentence': f.source_sentence
                }
                for f in facts[:10]
            ]
        }

    # ==========================================================================
    # 3. Question Answering & Reasoning Pipeline
    # ==========================================================================

    def answer_question(self, question_text: str) -> Dict[str, Any]:
        """
        Execute the end-to-end question answering pipeline:
        1. Question Classification & Parsing (Module 3)
        2. Knowledge Base Retrieval (Module 3)
        3. Backward / Forward Reasoning when needed (Module 4)
        4. Deterministic Answer Generation (Module 5 Stage 1)
        5. Explainable Provenance Generation (Module 5 Stage 2)
        
        Args:
            question_text: Raw natural-language question string.
            
        Returns:
            Structured answer response dictionary with complete provenance.
        """
        start_time = time.perf_counter()

        if not question_text or not question_text.strip():
            return {
                'status': 'error',
                'answer_status': 'error',
                'answer_type': 'none',
                'answer_text': 'Please enter a valid question.',
                'explanation': 'Empty or whitespace-only query string received.',
                'confidence': 'none',
                'supporting_fact_ids': [],
                'applied_rule_ids': [],
                'applied_rule_names': [],
                'supporting_facts': [],
                'sources': [],
                'execution_time_ms': 0.0
            }

        cleaned_question = question_text.strip()

        # Step 1: Parse the Question
        query = self.parser.parse(cleaned_question)

        # Step 2: Handle Unsupported Question Category
        if query.parsing_status != 'success':
            ans = self.answer_generator.generate_answer(query)
            explanation = self.explanation_generator.generate_explanation(
                answer_response=ans,
                query=query
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            
            # Persist query interaction
            try:
                self.repo.record_query(
                    question_text=cleaned_question,
                    category=query.question_type,
                    expected_answer_field=query.expected_answer_field,
                    parsing_status=query.parsing_status,
                    answer_status=ans.answer_status,
                    answer_type=ans.answer_type,
                    answer_text=ans.answer_text,
                    confidence=ans.confidence,
                    supporting_fact_ids=[],
                    applied_rule_ids=[],
                    explanation=explanation,
                    execution_time_ms=elapsed_ms
                )
            except Exception:
                pass

            return {
                'status': 'success',
                'question': cleaned_question,
                'answer_status': ans.answer_status,
                'answer_type': ans.answer_type,
                'answer_text': ans.answer_text,
                'confidence': ans.confidence,
                'explanation': explanation,
                'supporting_fact_ids': [],
                'applied_rule_ids': [],
                'applied_rule_names': [],
                'supporting_facts': [],
                'sources': [],
                'query_info': {
                    'category': query.question_type,
                    'expected_answer_field': query.expected_answer_field,
                    'subject': query.subject,
                    'predicate': query.predicate,
                    'object': query.object,
                    'parsing_status': query.parsing_status
                },
                'execution_time_ms': elapsed_ms
            }

        # Step 3: Direct Fact Retrieval
        retrieval = self.retriever.retrieve(query)
        direct_facts = retrieval.facts if retrieval and retrieval.status == 'success' else []
        reasoning_result: Optional[Union[ReasoningResult, ProofResult]] = None

        # Step 4: Symbolic Reasoning Integration
        # Only invoke deduction when direct facts are insufficient or query is boolean
        all_kb_facts = self.repo.get_all_facts()
        eff_subject = query.subject
        if eff_subject in ('Speaker', 'I', 'me'):
            speaker = self.repo.get_primary_speaker()
            if speaker:
                eff_subject = speaker

        # Determine explicit geographical query scope (country / continent / state / city)
        orig_q = (query.original_question or "").lower()
        target_scope = None
        if query.question_type in ('location', 'classification') or 'where' in orig_q or 'which' in orig_q:
            if re.search(r'\bcountry\b', orig_q):
                target_scope = 'country'
            elif re.search(r'\bcontinent\b', orig_q):
                target_scope = 'continent'
            elif re.search(r'\bstate\b|\bprovince\b', orig_q):
                target_scope = 'state'
            elif re.search(r'\bcity\b', orig_q):
                target_scope = 'city'

        # If a specific geographical scope was explicitly requested, direct facts must satisfy it.
        # Otherwise, direct facts are deferred in favor of hierarchical deduction.
        if target_scope and direct_facts:
            scoped_direct = [
                f for f in direct_facts
                if entity_matches_scope(f.object, target_scope, all_kb_facts, GLOBAL_ENTITY_REGISTRY)
            ]
            direct_facts = scoped_direct
            retrieval = RetrievalResult(
                query=query,
                facts=direct_facts,
                status='success' if direct_facts else 'empty'
            )

        if query.expected_answer_field == 'boolean':
            # Boolean queries: verify if proposition holds directly or deduce via backward chaining
            if not direct_facts and eff_subject and query.predicate and query.object:
                goal = (eff_subject, query.predicate, query.object)
                wm = WorkingMemory()
                wm.load_from_kb_facts(all_kb_facts)
                proof = backward_chain(goal, wm, rules=get_all_rules(include_extended=True))
                if proof.is_proven:
                    reasoning_result = proof
        elif len(direct_facts) == 0:
            # WH-questions: no direct facts found; discover derived conclusions via forward chaining
            # on an isolated temporary working memory
            temp_wm = WorkingMemory()
            temp_wm.load_from_kb_facts(all_kb_facts)
            fc_res = forward_chain(temp_wm, rules=get_all_rules(include_extended=True))

            # Find matching inferred facts matching the query constraints
            matching_inferred = []
            for inf in fc_res.inferred_facts:
                inf_s = inf.subject.strip().lower()
                inf_p = inf.predicate.strip().lower()
                inf_o = inf.object.strip().lower()

                subj_match = False
                if eff_subject:
                    subj_match = (inf_s == eff_subject.strip().lower())

                if eff_subject and query.predicate:
                    if subj_match and (inf_p == query.predicate.strip().lower() or inf_p == 'located in'):
                        matching_inferred.append(inf)
                elif query.predicate and query.object:
                    if (inf_p == query.predicate.strip().lower() and
                        inf_o == query.object.strip().lower()):
                        matching_inferred.append(inf)

            goal: Optional[Tuple[str, str, str]] = None
            if matching_inferred:
                selected_inf = None

                # Check if question explicitly targets country or continent or state or city
                if target_scope:
                    for inf in matching_inferred:
                        if entity_matches_scope(inf.object, target_scope, all_kb_facts, GLOBAL_ENTITY_REGISTRY):
                            selected_inf = inf
                            break

                if not selected_inf:
                    selected_inf = matching_inferred[0]

                goal = (selected_inf.subject, selected_inf.predicate, selected_inf.object)

            # If an inferred fact was discovered, prove it formally via backward chaining
            if goal:
                wm = WorkingMemory()
                wm.load_from_kb_facts(all_kb_facts)
                proof = backward_chain(goal, wm, rules=get_all_rules(include_extended=True))
                if proof.is_proven:
                    reasoning_result = proof

        # Step 5: Answer Generation (Module 5 Stage 1)
        ans = self.answer_generator.generate_answer(
            query=query,
            retrieval_result=retrieval,
            reasoning_result=reasoning_result
        )

        # Step 5b: Enrich Supporting Facts & Provenance Citations from Repository
        doc_map = {d.document_id: normalize_filename(d.filename) for d in self.repo.get_all_documents()}
        enriched_supporting_facts = []
        enriched_sources = []
        seen_sources = set()

        # For all supporting_fact_ids (ground facts supporting direct or inferred answers):
        for fid in ans.supporting_fact_ids:
            f = self.get_fact_by_id(fid)
            if f:
                doc_name = doc_map.get(f.document_id, f"Document #{f.document_id}" if f.document_id else "Unknown Document")
                enriched_supporting_facts.append({
                    'fact_id': f.fact_id,
                    'document_id': f.document_id,
                    'filename': doc_name,
                    'subject': f.subject,
                    'predicate': f.predicate,
                    'object': f.object,
                    'source_sentence': f.source_sentence,
                    'sentence_index': f.sentence_index
                })
                key = (f.document_id, f.source_sentence, f.fact_id)
                if key not in seen_sources:
                    seen_sources.add(key)
                    enriched_sources.append({
                        'document_id': f.document_id,
                        'filename': doc_name,
                        'source_sentence': f.source_sentence,
                        'fact_id': f.fact_id
                    })

        # Also include any direct fact sources from ans.sources if not already in enriched_sources
        for s in ans.sources:
            doc_id = s.get('document_id')
            fid = s.get('fact_id')
            sent = s.get('source_sentence')
            doc_name = doc_map.get(doc_id, f"Document #{doc_id}" if doc_id else "Unknown Document")
            key = (doc_id, sent, fid)
            if key not in seen_sources:
                seen_sources.add(key)
                enriched_sources.append({
                    'document_id': doc_id,
                    'filename': doc_name,
                    'source_sentence': sent,
                    'fact_id': fid
                })

        # Attach enriched facts and sources BEFORE generating explanation
        ans.supporting_facts = enriched_supporting_facts
        ans.sources = enriched_sources

        # Step 6: Explanation Generation (Module 5 Stage 2)
        explanation = self.explanation_generator.generate_explanation(
            answer_response=ans,
            query=query,
            retrieval_result=retrieval,
            reasoning_result=reasoning_result
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        applied_rule_names = [RULE_DISPLAY_NAMES.get(r, r) for r in ans.applied_rule_ids]

        # Persist query interaction
        try:
            self.repo.record_query(
                question_text=cleaned_question,
                category=query.question_type,
                expected_answer_field=query.expected_answer_field,
                parsing_status=query.parsing_status,
                answer_status=ans.answer_status,
                answer_type=ans.answer_type,
                answer_text=ans.answer_text,
                confidence=ans.confidence,
                supporting_fact_ids=ans.supporting_fact_ids,
                applied_rule_ids=ans.applied_rule_ids,
                explanation=explanation,
                execution_time_ms=elapsed_ms
            )
        except Exception:
            pass

        return {
            'status': 'success',
            'question': cleaned_question,
            'answer_status': ans.answer_status,
            'answer_type': ans.answer_type,
            'answer_text': ans.answer_text,
            'confidence': ans.confidence,
            'explanation': explanation,
            'supporting_fact_ids': ans.supporting_fact_ids,
            'applied_rule_ids': ans.applied_rule_ids,
            'applied_rule_names': applied_rule_names,
            'supporting_facts': enriched_supporting_facts,
            'sources': enriched_sources,
            'query_info': {
                'category': query.question_type,
                'expected_answer_field': query.expected_answer_field,
                'subject': query.subject,
                'predicate': query.predicate,
                'object': query.object,
                'parsing_status': query.parsing_status
            },
            'execution_time_ms': elapsed_ms
        }

    # ==========================================================================
    # 4. Interactive System Evaluation Execution
    # ==========================================================================

    def run_system_evaluation(self) -> Dict[str, Any]:
        """
        Execute the deterministic 21-case evaluation benchmark.
        Returns aggregated performance metrics and academic Markdown report.
        """
        summary, report_md = run_evaluation()

        cases_data = []
        for c in summary.case_results:
            cases_data.append({
                'case_id': c.case_id,
                'category': c.category,
                'question': c.question,
                'actual_status': c.actual_status,
                'expected_status': c.expected_status,
                'status_match': c.status_match,
                'actual_answer_text': c.actual_answer_text,
                'expected_answer_text': c.expected_answer_text,
                'answer_match': c.answer_match,
                'retrieval_precision': c.retrieval_precision,
                'retrieval_recall': c.retrieval_recall,
                'retrieval_f1': c.retrieval_f1,
                'applied_rule_ids': c.applied_rule_ids,
                'expected_rule_ids': c.expected_rule_ids,
                'rules_match': c.rules_match,
                'supporting_fact_ids': c.supporting_fact_ids,
                'expected_fact_ids': c.expected_fact_ids,
                'supporting_facts_match': c.supporting_facts_match,
                'is_overall_success': c.is_overall_success,
                'execution_time_ms': c.execution_time_ms
            })

        return {
            'status': 'success',
            'summary': {
                'total_cases': summary.total_cases,
                'successful_cases': summary.successful_cases,
                'failed_cases': summary.failed_cases,
                'answer_accuracy': summary.answer_accuracy,
                'exact_match_accuracy': summary.exact_match_accuracy,
                'unknown_detection_accuracy': summary.unknown_detection_accuracy,
                'unsupported_detection_accuracy': summary.unsupported_detection_accuracy,
                'retrieval_precision': summary.aggregate_precision,
                'retrieval_recall': summary.aggregate_recall,
                'retrieval_f1': summary.aggregate_f1,
                'empty_retrieval_accuracy': summary.empty_retrieval_accuracy,
                'reasoning_accuracy': summary.inference_accuracy,
                'rule_precision': summary.rule_application_accuracy,
                'supporting_fact_accuracy': summary.supporting_premise_accuracy,
                'explanation_accuracy': summary.rule_disclosure_accuracy,
                'provenance_accuracy': summary.source_provenance_accuracy,
                'average_execution_time_ms': summary.average_execution_time_ms,
                'total_execution_time_ms': summary.total_execution_time_ms
            },
            'cases': cases_data,
            'report_markdown': report_md
        }

    # ==========================================================================
    # 5. Question History, Export & Workspace Reset
    # ==========================================================================

    def get_query_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve recent question answering history."""
        return self.repo.get_query_history(limit=limit, offset=offset)

    def reset_workspace(self, confirmation: str) -> Dict[str, Any]:
        """
        Nuke Data / Workspace Reset:
        Safely clears uploaded document files in the uploads directory,
        purges SQLite documents, facts, and query_history records inside an atomic transaction.
        
        Requires explicit confirmation: "DELETE".
        Strictly validates that deletions occur ONLY inside the workspace uploads directory.
        Preserves database schema, application code, inference rules, and evaluation benchmarks.
        """
        if not confirmation or confirmation.strip() != "DELETE":
            return {
                'status': 'error',
                'message': "Reset rejected: Confirmation mismatch. You must explicitly type 'DELETE' to confirm.",
                'deleted_files': 0,
                'deleted_documents': 0,
                'deleted_facts': 0,
                'deleted_queries': 0
            }

        # 1. Safely remove physical files inside upload_folder
        deleted_files = 0
        failed_files = 0
        if os.path.exists(self.upload_folder):
            for entry in os.listdir(self.upload_folder):
                if entry == '.gitkeep':
                    continue
                file_path = os.path.abspath(os.path.join(self.upload_folder, entry))
                # Security constraint: never delete outside upload_folder
                if os.path.commonpath([self.upload_folder, file_path]) == self.upload_folder and os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files += 1
                    except Exception:
                        failed_files += 1

        # 2. Transactional database reset
        try:
            db_counts = self.repo.clear_all_workspace_data()
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Database reset failed: {str(e)}",
                'deleted_files': deleted_files,
                'deleted_documents': 0,
                'deleted_facts': 0,
                'deleted_queries': 0
            }

        return {
            'status': 'success',
            'message': (
                f"Workspace successfully reset. Purged {deleted_files} document files, "
                f"{db_counts['deleted_documents']} document records, "
                f"{db_counts['deleted_facts']} facts, and {db_counts['deleted_queries']} queries."
            ),
            'deleted_files': deleted_files,
            'deleted_documents': db_counts['deleted_documents'],
            'deleted_facts': db_counts['deleted_facts'],
            'deleted_queries': db_counts['deleted_queries'],
            'failed_files': failed_files
        }

    def export_workspace_json(self) -> Dict[str, Any]:
        """Compile and return full structured workspace analysis data."""
        from modules.export_service import generate_workspace_json
        return generate_workspace_json(self)

    def export_workspace_pdf(self) -> bytes:
        """Generate and return binary PDF report bytes."""
        from modules.export_service import generate_workspace_pdf
        return generate_workspace_pdf(self)

