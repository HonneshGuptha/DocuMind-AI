"""
Knowledge Base Repository
Provides structured CRUD operations, deterministic search, and relational storage
for documents and SPO facts using SQLite.

ACADEMIC RESTRICTION COMPLIANCE:
All database operations use strict relational queries with parameterized inputs.
No vector embeddings, no semantic vector similarity, no vector indexes.
"""

import sqlite3
from typing import List, Optional, Tuple, Union, Dict, Any
from .models import Fact, DocumentRecord
from .database import get_connection, init_db
from .fact_extractor import FactExtractor
from modules.document_processor import DocumentContent


class KnowledgeBaseRepository:
    """
    Relational repository for storing and querying extracted knowledge facts
    and source document metadata.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        # Ensure database tables and indexes exist
        init_db(self.db_path)
        self.fact_extractor = FactExtractor()

    def _get_connection(self) -> sqlite3.Connection:
        """Helper to get a database connection."""
        return get_connection(self.db_path)

    # ==========================================================================
    # 1. Document Operations
    # ==========================================================================

    def add_document(self,
                     filename: str,
                     file_format: str,
                     extraction_status: str) -> int:
        """
        Record document metadata in the documents table.
        Returns newly generated document_id.
        """
        query = """
        INSERT INTO documents (filename, file_format, extraction_status)
        VALUES (?, ?, ?);
        """
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute(query, (filename, file_format, extraction_status))
                return cursor.lastrowid
        finally:
            conn.close()

    def get_document(self, document_id: int) -> Optional[DocumentRecord]:
        """Retrieve document metadata by document_id."""
        query = """
        SELECT document_id, filename, file_format, upload_date, extraction_status
        FROM documents
        WHERE document_id = ?;
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(query, (document_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return DocumentRecord(
                document_id=row['document_id'],
                filename=row['filename'],
                file_format=row['file_format'],
                upload_date=str(row['upload_date']),
                extraction_status=row['extraction_status']
            )
        finally:
            conn.close()

    def get_all_documents(self) -> List[DocumentRecord]:
        """Retrieve all registered documents."""
        query = """
        SELECT document_id, filename, file_format, upload_date, extraction_status
        FROM documents
        ORDER BY document_id ASC;
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(query)
            return [
                DocumentRecord(
                    document_id=row['document_id'],
                    filename=row['filename'],
                    file_format=row['file_format'],
                    upload_date=str(row['upload_date']),
                    extraction_status=row['extraction_status']
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def count_documents(self) -> int:
        """Count total stored documents."""
        query = "SELECT COUNT(*) as total FROM documents;"
        conn = self._get_connection()
        try:
            cursor = conn.execute(query)
            return cursor.fetchone()['total']
        finally:
            conn.close()

    # ==========================================================================
    # 2. Fact Operations
    # ==========================================================================

    def add_fact(self, fact: Fact) -> Optional[int]:
        """
        Insert a single fact into the database.
        Safely handles duplicates within the same document using INSERT OR IGNORE.
        Returns fact_id if inserted, or None if skipped as duplicate.
        """
        query = """
        INSERT OR IGNORE INTO facts (
            document_id, subject, predicate, object, source_sentence, sentence_index,
            canonical_predicate, subject_entity_id, object_entity_id, fact_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute(query, (
                    fact.document_id,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.source_sentence,
                    fact.sentence_index,
                    fact.canonical_predicate or fact.predicate,
                    fact.subject_entity_id,
                    fact.object_entity_id,
                    fact.fact_status or 'asserted'
                ))
                if cursor.rowcount > 0:
                    fact.fact_id = cursor.lastrowid
                    return cursor.lastrowid
                return None
        finally:
            conn.close()

    def add_facts(self, facts: List[Fact]) -> int:
        """
        Batch insert a list of facts within a single transaction.
        Returns total count of successfully inserted new facts.
        """
        if not facts:
            return 0

        query = """
        INSERT OR IGNORE INTO facts (
            document_id, subject, predicate, object, source_sentence, sentence_index,
            canonical_predicate, subject_entity_id, object_entity_id, fact_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        inserted_count = 0
        conn = self._get_connection()
        try:
            with conn:
                for fact in facts:
                    cursor = conn.execute(query, (
                        fact.document_id,
                        fact.subject,
                        fact.predicate,
                        fact.object,
                        fact.source_sentence,
                        fact.sentence_index,
                        fact.canonical_predicate or fact.predicate,
                        fact.subject_entity_id,
                        fact.object_entity_id,
                        fact.fact_status or 'asserted'
                    ))
                    if cursor.rowcount > 0:
                        fact.fact_id = cursor.lastrowid
                        inserted_count += 1
            return inserted_count
        finally:
            conn.close()

    def count_facts(self) -> int:
        """Count total stored facts."""
        query = "SELECT COUNT(*) as total FROM facts;"
        conn = self._get_connection()
        try:
            cursor = conn.execute(query)
            return cursor.fetchone()['total']
        finally:
            conn.close()

    # ==========================================================================
    # 3. Fact Retrieval & Search Operations
    # ==========================================================================

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        """Convert a database row into a Fact model."""
        keys = row.keys()
        return Fact(
            fact_id=row['fact_id'],
            document_id=row['document_id'],
            subject=row['subject'],
            predicate=row['predicate'],
            object=row['object'],
            source_sentence=row['source_sentence'],
            sentence_index=row['sentence_index'],
            canonical_predicate=row['canonical_predicate'] if 'canonical_predicate' in keys and row['canonical_predicate'] else row['predicate'],
            subject_entity_id=row['subject_entity_id'] if 'subject_entity_id' in keys else None,
            object_entity_id=row['object_entity_id'] if 'object_entity_id' in keys else None,
            fact_status=row['fact_status'] if 'fact_status' in keys and row['fact_status'] else 'asserted'
        )

    def get_all_facts(self, limit: Optional[int] = None, offset: int = 0) -> List[Fact]:
        """Retrieve all facts with optional pagination."""
        query = """
        SELECT fact_id, document_id, subject, predicate, object, source_sentence, sentence_index
        FROM facts
        ORDER BY fact_id ASC
        """
        params = []
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        conn = self._get_connection()
        try:
            cursor = conn.execute(query, params)
            return [self._row_to_fact(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def get_facts_by_document(self, document_id: int) -> List[Fact]:
        """Retrieve all facts originating from a specific document ID."""
        query = """
        SELECT fact_id, document_id, subject, predicate, object, source_sentence, sentence_index
        FROM facts
        WHERE document_id = ?
        ORDER BY sentence_index ASC;
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(query, (document_id,))
            return [self._row_to_fact(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def search_by_subject(self, subject: str, exact: bool = False) -> List[Fact]:
        """Search facts by subject using parameterized relational query."""
        return self.search_facts(subject=subject, exact=exact)

    def search_by_predicate(self, predicate: str, exact: bool = False) -> List[Fact]:
        """Search facts by predicate using parameterized relational query."""
        return self.search_facts(predicate=predicate, exact=exact)

    def search_by_object(self, object_val: str, exact: bool = False) -> List[Fact]:
        """Search facts by object using parameterized relational query."""
        return self.search_facts(object=object_val, exact=exact)

    def search_facts(self,
                     subject: Optional[str] = None,
                     predicate: Optional[str] = None,
                     object: Optional[str] = None,
                     document_id: Optional[int] = None,
                     exact: bool = False) -> List[Fact]:
        """
        Search facts across combinations of subject, predicate, object, and document_id.
        Supports case-insensitive exact matching or substring matching.
        """
        conditions = []
        params = []

        if subject:
            if exact:
                conditions.append("LOWER(subject) = LOWER(?)")
                params.append(subject.strip())
            else:
                conditions.append("LOWER(subject) LIKE LOWER(?)")
                params.append(f"%{subject.strip()}%")

        if predicate:
            if exact:
                conditions.append("LOWER(predicate) = LOWER(?)")
                params.append(predicate.strip())
            else:
                conditions.append("LOWER(predicate) LIKE LOWER(?)")
                params.append(f"%{predicate.strip()}%")

        if object:
            if exact:
                conditions.append("LOWER(object) = LOWER(?)")
                params.append(object.strip())
            else:
                conditions.append("LOWER(object) LIKE LOWER(?)")
                params.append(f"%{object.strip()}%")

        if document_id is not None:
            conditions.append("document_id = ?")
            params.append(document_id)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        SELECT fact_id, document_id, subject, predicate, object, source_sentence, sentence_index
        FROM facts
        {where_clause}
        ORDER BY fact_id ASC;
        """

        conn = self._get_connection()
        try:
            cursor = conn.execute(query, params)
            return [self._row_to_fact(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    # ==========================================================================
    # 4. End-to-End Ingestion Integration
    # ==========================================================================

    def store_document_content(self, content: DocumentContent) -> Tuple[int, List[Fact]]:
        """
        Ingest extracted DocumentContent from Module 1 into the knowledge base:
        1. Adds document metadata.
        2. Extracts facts if document extraction was successful.
        3. Inserts facts into SQLite.
        
        Returns:
            Tuple of (document_id, list_of_extracted_facts)
        """
        # 1. Record document
        doc_id = self.add_document(
            filename=content.filename,
            file_format=content.format,
            extraction_status=content.status
        )

        # 2. Extract facts if extraction succeeded
        if content.status != 'success' or not content.raw_text:
            return (doc_id, [])

        facts = self.fact_extractor.extract_facts(content.raw_text, doc_id)

        # 3. Persist facts
        if facts:
            self.add_facts(facts)

            # Check for speaker declaration to establish document context
            for f in facts:
                if f.predicate == 'has name' or f.canonical_predicate == 'has name':
                    self.set_document_context(doc_id, speaker_name=f.object)
                    break

        return (doc_id, facts)

    def set_document_context(self, document_id: int, speaker_name: str, speaker_entity_id: Optional[str] = None) -> None:
        """Record or update document speaker context in document_contexts table."""
        if not speaker_name:
            return
        eid = speaker_entity_id or f"person:{speaker_name.lower().replace(' ', '_')}"
        query = """
        INSERT OR REPLACE INTO document_contexts (document_id, speaker_name, speaker_entity_id)
        VALUES (?, ?, ?);
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(query, (document_id, speaker_name.strip(), eid))
        finally:
            conn.close()

    def get_document_context(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve speaker context for a specific document."""
        query = "SELECT document_id, speaker_name, speaker_entity_id FROM document_contexts WHERE document_id = ?;"
        conn = self._get_connection()
        try:
            cursor = conn.execute(query, (document_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_primary_speaker(self) -> Optional[str]:
        """
        Retrieve primary speaker across documents (latest established speaker).
        Used to resolve first-person user queries ('Who am I?', 'Where do I live?').
        """
        query = "SELECT speaker_name FROM document_contexts ORDER BY context_id DESC LIMIT 1;"
        conn = self._get_connection()
        try:
            cursor = conn.execute(query)
            row = cursor.fetchone()
            if row and row['speaker_name']:
                return row['speaker_name']
            
            # Fallback: check facts directly for 'has name' predicate
            fallback_query = "SELECT object FROM facts WHERE LOWER(predicate) = 'has name' ORDER BY fact_id DESC LIMIT 1;"
            cursor = conn.execute(fallback_query)
            fb_row = cursor.fetchone()
            if fb_row and fb_row['object']:
                return fb_row['object']
            return None
        finally:
            conn.close()

    # ==========================================================================
    # 5. Question History & Workspace Management
    # ==========================================================================

    def record_query(self,
                     question_text: str,
                     category: Optional[str] = None,
                     expected_answer_field: Optional[str] = None,
                     parsing_status: Optional[str] = None,
                     answer_status: Optional[str] = None,
                     answer_type: Optional[str] = None,
                     answer_text: Optional[str] = None,
                     confidence: Optional[str] = None,
                     supporting_fact_ids: Optional[List[int]] = None,
                     applied_rule_ids: Optional[List[str]] = None,
                     explanation: Optional[str] = None,
                     execution_time_ms: Optional[float] = None) -> int:
        """
        Record a question-answering interaction for workspace audit and export.
        Returns the generated query_id.
        """
        import json
        query = """
        INSERT INTO query_history (
            question_text, category, expected_answer_field, parsing_status,
            answer_status, answer_type, answer_text, confidence,
            supporting_fact_ids, applied_rule_ids, explanation, execution_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        supp_json = json.dumps(supporting_fact_ids or [])
        rules_json = json.dumps(applied_rule_ids or [])

        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute(query, (
                    question_text, category, expected_answer_field, parsing_status,
                    answer_status, answer_type, answer_text, confidence,
                    supp_json, rules_json, explanation, execution_time_ms
                ))
                return cursor.lastrowid
        finally:
            conn.close()

    def get_query_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve recent question-answering history ordered chronologically descending."""
        import json
        query = """
        SELECT query_id, timestamp, question_text, category, expected_answer_field,
               parsing_status, answer_status, answer_type, answer_text, confidence,
               supporting_fact_ids, applied_rule_ids, explanation, execution_time_ms
        FROM query_history
        ORDER BY query_id DESC
        LIMIT ? OFFSET ?;
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(query, (limit, offset))
            history = []
            for row in cursor.fetchall():
                try:
                    supp_ids = json.loads(row['supporting_fact_ids'] or '[]')
                except Exception:
                    supp_ids = []
                try:
                    rule_ids = json.loads(row['applied_rule_ids'] or '[]')
                except Exception:
                    rule_ids = []

                history.append({
                    'query_id': row['query_id'],
                    'timestamp': str(row['timestamp']),
                    'question_text': row['question_text'],
                    'category': row['category'],
                    'expected_answer_field': row['expected_answer_field'],
                    'parsing_status': row['parsing_status'],
                    'answer_status': row['answer_status'],
                    'answer_type': row['answer_type'],
                    'answer_text': row['answer_text'],
                    'confidence': row['confidence'],
                    'supporting_fact_ids': supp_ids,
                    'applied_rule_ids': rule_ids,
                    'explanation': row['explanation'],
                    'execution_time_ms': row['execution_time_ms']
                })
            return history
        finally:
            conn.close()

    def count_queries(self) -> int:
        """Count total recorded queries in the workspace."""
        query = "SELECT COUNT(*) FROM query_history;"
        conn = self._get_connection()
        try:
            cursor = conn.execute(query)
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def clear_all_workspace_data(self) -> Dict[str, int]:
        """
        Perform transactional deletion of all workspace records from SQLite:
        - facts
        - documents
        - query_history
        Preserves database schema and indexes.
        
        Returns:
            Dictionary containing count of cleared entities.
        """
        conn = self._get_connection()
        try:
            with conn:
                f_count = conn.execute("SELECT COUNT(*) FROM facts;").fetchone()[0]
                d_count = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
                q_count = conn.execute("SELECT COUNT(*) FROM query_history;").fetchone()[0]

                # Clear tables within atomic transaction
                conn.execute("DELETE FROM facts;")
                conn.execute("DELETE FROM documents;")
                conn.execute("DELETE FROM query_history;")
                try:
                    conn.execute("DELETE FROM document_contexts;")
                except sqlite3.OperationalError:
                    pass

                # Optionally reset autoincrement sequence if sqlite_sequence exists
                try:
                    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('facts', 'documents', 'query_history', 'document_contexts');")
                except sqlite3.OperationalError:
                    pass

                return {
                    'deleted_facts': f_count,
                    'deleted_documents': d_count,
                    'deleted_queries': q_count
                }
        finally:
            conn.close()
