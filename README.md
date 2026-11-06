# DocuMind-AI – Knowledge-Based Document Question Answering System

DocuMind-AI is an Artificial Intelligence college assignment project designed to demonstrate classical, symbolic knowledge representation and reasoning for document-based question answering.

The system accepts documents in PDF, DOCX, and TXT formats, extracts their raw textual content, converts that text into structured knowledge representations (Subject-Predicate-Object triples and entity-attribute relationships), and answers user queries using deterministic classification, relational search, rule-based reasoning, and inference.

---

## 🚫 Mandatory Academic Restrictions

This project strictly adheres to classical Symbolic AI principles. Modern NLP, transformer models, and Retrieval-Augmented Generation (RAG) pipelines are **strictly prohibited**:

### Strictly Prohibited:
- ❌ **No NLP libraries** (e.g., spaCy, NLTK, Stanford CoreNLP).
- ❌ **No pretrained language models** (e.g., BERT, RoBERTa, T5, GPT).
- ❌ **No text embeddings or semantic similarity** (e.g., cosine similarity over vectors).
- ❌ **No vector databases or vector search** (e.g., Chroma, FAISS, Pinecone, Qdrant).
- ❌ **No RAG frameworks** (e.g., LangChain, LlamaIndex).
- ❌ **No LLM-based generative question answering**.

### Required Classical AI Techniques:
- ✅ **Structured Knowledge Representation**: Fact extraction into Subject-Predicate-Object (SPO) triples and entity-attribute schemas.
- ✅ **Relational Storage**: SQLite database for persistent knowledge storage and structured querying.
- ✅ **Deterministic Question Classification**: Rule-based keyword matching and grammatical pattern heuristics to identify question intent (e.g., WHO, WHAT, WHERE, WHEN, IS-A verification).
- ✅ **Rule-Based Reasoning**: Classical inference engine supporting forward chaining (deductive closure) and backward chaining (goal-driven proof).
- ✅ **Template-Based Answer Synthesis**: Deterministic, hallucination-free generation using structured response templates populated from verified facts.
- ✅ **Raw Text Extraction**: Restricted solely to file I/O libraries (PyMuPDF for PDF and python-docx for DOCX) to extract raw character strings.

---

## 🛠️ Technology Stack

- **Language:** Python 3 (Python 3.9+)
- **Backend:** Flask web framework
- **Frontend:** Semantic HTML5, modern CSS3, vanilla JavaScript
- **Database:** SQLite3 (standard library `sqlite3`)
- **Document Extractors:**
  - `PyMuPDF` (`fitz`): Raw text extraction from PDF files
  - `python-docx`: Raw text extraction from DOCX files
  - Python standard file I/O: Plain text (`.txt`) reading

---

## 📁 Project Folder Structure

```text
DocuMind-AI/
│
├── app.py                      # Flask application entry point and routing
├── config.py                   # Central configuration (paths, limits, security)
├── requirements.txt            # Minimal Python dependencies
├── README.md                   # Project documentation and specifications
├── .gitignore                  # Version control ignore rules
│
├── database/                   # SQLite database storage directory
│   └── .gitkeep
│
├── uploads/                    # Temporary raw document uploads
│   └── .gitkeep
│
├── modules/                    # Modular AI subsystem packages
│   ├── __init__.py
│   ├── document_processor/     # Raw text extraction (PyMuPDF, python-docx)
│   ├── knowledge_base/         # SPO triples, schemas, SQLite persistence
│   ├── question_processor/     # Deterministic question classification & keywords
│   ├── reasoning_engine/       # Inference engine: forward and backward chaining
│   ├── answer_generator/       # Template-based deterministic answer synthesis
│   └── evaluation/             # Metrics: retrieval precision, recall, latency
│
├── templates/                  # Jinja2 HTML templates for the web interface
│   └── index.html
│
├── static/                     # Web assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
└── tests/                      # Unit and integration test suites
    ├── __init__.py
    └── test_initialization.py
```

---

## 🚀 Setup and Installation Instructions

### 1. Prerequisites
Ensure you have **Python 3.9 or higher** installed on your system.

### 2. Clone or Open the Workspace
Navigate to the root of the project directory:
```bash
cd "c:\Users\prakr\Downloads\AI assignment"
```

### 3. Create a Python Virtual Environment
If not already created:
- **Windows (PowerShell/CMD):**
  ```powershell
  python -m venv venv
  ```
- **Linux/macOS:**
  ```bash
  python3 -m venv venv
  ```

### 4. Activate the Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### 5. Install Dependencies
Install only the permitted, necessary dependencies:
```bash
pip install -r requirements.txt
```

### 6. Run the Application
Start the local Flask development server:
```bash
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

### 7. Run Verification Tests
To verify project initialization and configurations:
```bash
python -m unittest discover tests
```

---

## 📋 Architectural Subsystems Roadmap

1. **Document Processor (`modules/document_processor`):**
   Reads uploaded PDFs, DOCXs, and TXTs, converting document pages into clean raw text streams while filtering encoding artifacts.
2. **Knowledge Base (`modules/knowledge_base`):**
   Parses raw text into structured `(subject, predicate, object)` triples, manages database schema, indexes entities, and stores fact relations.
3. **Question Processor (`modules/question_processor`):**
   Determines query type (fact retrieval, entity check, relationship query) through deterministic rules and regex token matching.
4. **Reasoning Engine (`modules/reasoning_engine`):**
   Applies logical rules (e.g., transitive, symmetric, inverse properties) to deduce implicit facts using forward and backward chaining.
5. **Answer Generator (`modules/answer_generator`):**
   Maps inferred and retrieved triples directly into user-friendly response templates.
6. **Evaluation (`modules/evaluation`):**
   Tests query accuracy, precision, recall, and execution time across benchmark question-document pairs.
