"""
DocuMind-AI: Knowledge-Based Document Question Answering System
Flask Application Entry Point and Web Interface Routes

ACADEMIC COMPLIANCE:
Classical Symbolic Artificial Intelligence (AID201A).
Deterministic NLP-free rule-based extraction, relational KB storage,
forward and backward chaining inference, and verifiable provenance tracking.
"""

import os
import json
import io
from flask import Flask, render_template, request, jsonify, Response, send_file
from config import Config
from modules.knowledge_base.database import init_db
from modules.app_service import DocuMindService


def create_app(config_class=Config):
    """
    Application factory pattern to configure and instantiate the Flask app.
    Initializes upload folders, SQLite database, and the DocuMindService.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure runtime directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config['DATABASE_PATH']), exist_ok=True)

    # Initialize SQLite database schema
    init_db(app.config['DATABASE_PATH'])

    # Instantiate Application Service Orchestration Layer
    app.service = DocuMindService(
        db_path=app.config['DATABASE_PATH'],
        upload_folder=app.config['UPLOAD_FOLDER']
    )

    # ==========================================================================
    # Web Interface Routes
    # ==========================================================================

    @app.route('/')
    def index():
        """Home page route displaying the dashboard, document manager, QA, and evaluation."""
        stats = app.service.get_dashboard_stats()
        facts = app.service.get_all_facts(limit=25)
        return render_template('index.html', stats=stats, facts=facts)

    # ==========================================================================
    # REST API & Interaction Endpoints
    # ==========================================================================

    @app.route('/api/status', methods=['GET'])
    def api_status():
        """Health check endpoint confirming application initialization."""
        return jsonify({
            'status': 'online',
            'app_name': 'DocuMind-AI',
            'version': '0.1.0',
            'paradigm': 'Classical Knowledge-Based AI',
            'nlp_rag_disabled': True,
            'supported_formats': sorted(list(app.config['ALLOWED_EXTENSIONS'])),
            'max_upload_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        }), 200

    @app.route('/api/stats', methods=['GET'])
    def api_stats():
        """Return live knowledge base and document ingestion statistics."""
        stats = app.service.get_dashboard_stats()
        return jsonify(stats), 200

    @app.route('/api/facts', methods=['GET'])
    def api_facts():
        """Return stored SPO facts with pagination support."""
        try:
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
        except ValueError:
            limit = 50
            offset = 0

        facts = app.service.get_all_facts(limit=limit, offset=offset)
        total_facts = app.service.repo.count_facts()
        return jsonify({
            'status': 'success',
            'total_facts': total_facts,
            'limit': limit,
            'offset': offset,
            'facts': facts
        }), 200

    @app.route('/upload', methods=['POST'])
    def upload_document():
        """
        Handle document upload (PDF, DOCX, TXT), perform validation,
        extract text, and ingest facts into the relational knowledge base.
        """
        if 'file' not in request.files and 'document' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request. Use key "file" or "document".'
            }), 400

        file = request.files.get('file') or request.files.get('document')
        if not file or not file.filename:
            return jsonify({
                'status': 'error',
                'message': 'No selected file.'
            }), 400

        # Check file extension
        filename = file.filename
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({
                'status': 'error',
                'message': f"Unsupported file extension '{ext}'. Only {sorted(list(app.config['ALLOWED_EXTENSIONS']))} are supported."
            }), 400

        # Process upload via service
        force = request.form.get('force', 'false').lower() in ('true', '1')
        result = app.service.process_uploaded_file(file, force=force)

        status_code = 200
        if result.get('status') == 'error':
            status_code = 400

        return jsonify(result), status_code

    @app.route('/api/load-sample', methods=['POST'])
    def load_sample_document():
        """
        Load and ingest a pre-configured canonical academic knowledge document
        for immediate testing and interactive demonstration.
        """
        sample_filename = 'DocuMind_AI_Academic_Sample.txt'
        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)
        sample_content = (
            "Tokyo is the capital of Japan.\n"
            "Japan is located in Asia.\n"
            "Bangalore is the capital of Karnataka.\n"
            "Karnataka is a state in India.\n"
            "India is located in Asia.\n"
            "Paris is the capital of France.\n"
            "France is located in Europe.\n"
            "Canada borders United States.\n"
            "United States borders Mexico.\n"
            "Stanford University was established in 1891.\n"
            "Harvard University was established in 1636.\n"
            "Python was developed by Guido van Rossum.\n"
            "C was developed by Dennis Ritchie.\n"
            "Python is a high-level programming language.\n"
        )
        try:
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)

            with open(sample_path, 'rb') as f:
                from werkzeug.datastructures import FileStorage
                file_storage = FileStorage(stream=f, filename=sample_filename, content_type='text/plain')
                result = app.service.process_uploaded_file(file_storage, force=True)

            return jsonify(result), 200
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to load sample document: {str(e)}'
            }), 500

    @app.route('/ask', methods=['POST'])
    def ask_question():
        """
        Answer a natural language question through deterministic parsing,
        fact retrieval, symbolic reasoning (if required), answer generation,
        and transparent explanation generation.
        """
        question_text = ''
        if request.is_json:
            data = request.get_json(silent=True) or {}
            question_text = data.get('question', '')
        else:
            question_text = request.form.get('question', '')

        if not question_text or not question_text.strip():
            return jsonify({
                'status': 'error',
                'message': 'Question text cannot be empty.',
                'answer_status': 'error',
                'answer_text': 'Please enter a valid question.'
            }), 400

        result = app.service.answer_question(question_text)
        return jsonify(result), 200

    @app.route('/api/evaluate', methods=['POST', 'GET'])
    def evaluate_system():
        """
        Trigger the comprehensive 21-case evaluation benchmark across the
        full symbolic AI pipeline and return detailed metrics.
        """
        result = app.service.run_system_evaluation()
        return jsonify(result), 200

    @app.route('/api/history', methods=['GET'])
    def get_query_history():
        """Return persisted question answering interactions."""
        try:
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
        except ValueError:
            limit = 50
            offset = 0

        history = app.service.get_query_history(limit=limit, offset=offset)
        total_queries = app.service.repo.count_queries()
        return jsonify({
            'status': 'success',
            'total_queries': total_queries,
            'limit': limit,
            'offset': offset,
            'history': history
        }), 200

    @app.route('/api/workspace/reset', methods=['POST'])
    def reset_workspace():
        """
        Destructive NUKE DATA endpoint:
        Requires explicit confirmation keyword "DELETE".
        Safely clears uploaded document files and database facts/records.
        """
        confirmation = ""
        if request.is_json:
            data = request.get_json(silent=True) or {}
            confirmation = data.get('confirmation', '')
        else:
            confirmation = request.form.get('confirmation', '')

        result = app.service.reset_workspace(confirmation)
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code

    @app.route('/api/export/json', methods=['GET'])
    def export_workspace_json():
        """Download comprehensive structured workspace data as UTF-8 JSON."""
        data = app.service.export_workspace_json()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        return Response(
            json_str,
            mimetype='application/json',
            headers={
                'Content-Disposition': 'attachment; filename="documind_workspace_report.json"',
                'Cache-Control': 'no-cache'
            }
        )

    @app.route('/api/export/pdf', methods=['GET'])
    def export_workspace_pdf():
        """Download professional academic workspace analysis report as PDF."""
        pdf_bytes = app.service.export_workspace_pdf()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='documind_workspace_report.pdf'
        )

    # ==========================================================================
    # HTTP Error Handlers
    # ==========================================================================

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'status': 'error',
            'message': str(error.description if hasattr(error, 'description') else 'Bad Request')
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'status': 'error',
            'message': 'Requested resource was not found on this server.'
        }), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'status': 'error',
            'message': 'File exceeds maximum permitted size of 16 MB.'
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            'status': 'error',
            'message': 'An internal server error occurred.'
        }), 500

    return app


app = create_app()

if __name__ == '__main__':
    # Local development server for academic demonstration
    app.run(host='127.0.0.1', port=5000, debug=True)
