/**
 * DocuMind-AI: Modern Frontend Application Logic
 * Academic Knowledge-Based Document Question Answering System
 * Course: AID201A – Artificial Intelligence
 * Institution: M S Ramaiah University of Applied Sciences
 * Visual Direction: Hybrid AI Workstation + Expressive Animations
 */

let currentEvalCases = [];

const MASCOT_STATES = {
    idle: {
        file: 'mascot-idle.svg',
        badge: 'Archie • Ready',
        badgeClass: 'badge-success',
        desc: 'Deterministic Grammar Parsing + Symbolic Forward & Backward Chaining Inference.'
    },
    reading: {
        file: 'mascot-reading.svg',
        badge: 'Archie • Reading Document',
        badgeClass: 'badge-indigo',
        desc: 'Scanning document text and extracting structural syntax lines.'
    },
    processing: {
        file: 'mascot-processing.svg',
        badge: 'Archie • Parsing Query',
        badgeClass: 'badge-cyan',
        desc: 'Evaluating relational triples against deterministic grammar rules.'
    },
    reasoning: {
        file: 'mascot-reasoning.svg',
        badge: 'Archie • Deriving Proof',
        badgeClass: 'badge-purple',
        desc: 'Applying Modus Ponens forward chaining and Horn clause deductions.'
    },
    answered: {
        file: 'mascot-answered.svg',
        badge: 'Archie • Fact Verified',
        badgeClass: 'badge-success',
        desc: 'Deterministic answer derived with complete verifiable sentence provenance.'
    },
    unknown: {
        file: 'mascot-unknown.svg',
        badge: 'Archie • Open-World Unknown',
        badgeClass: 'badge-warning',
        desc: 'Query verified as consistent under Open-World Assumption: fact not asserted in KB.'
    },
    unsupported: {
        file: 'mascot-unsupported.svg',
        badge: 'Archie • Scope Boundary',
        badgeClass: 'badge-danger',
        desc: 'Input outside the 9 supported question categories or strict grammar scope.'
    },
    reset: {
        file: 'mascot-reset.svg',
        badge: 'Archie • Garden Rested',
        badgeClass: 'badge-neutral',
        desc: 'Workspace database purged. Knowledge base ready for new document ingestion.'
    }
};

function setMascotState(state) {
    const info = MASCOT_STATES[state] || MASCOT_STATES.idle;
    const imgEl = document.getElementById('qa-mascot-avatar');
    const badgeEl = document.getElementById('mascot-state-badge');
    const descEl = document.getElementById('mascot-state-description');

    if (imgEl) {
        imgEl.src = `/static/illustrations/mascot/${info.file}`;
    }
    if (badgeEl) {
        badgeEl.textContent = info.badge;
        badgeEl.className = `badge ${info.badgeClass}`;
    }
    if (descEl) {
        descEl.textContent = info.desc;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initMobileMenu();
    initFileUpload();
    initKnowledgeBaseSearch();
    initQuestionAnswering();
    initDropdownListeners();
    initGraphVisualizer();
    refreshDashboardStats();
    loadQuestionHistory();
});

// =============================================================================
// 1. Navigation & Tab Switching
// =============================================================================

const PAGE_META = {
    'tab-dashboard': {
        title: 'System Dashboard',
        description: 'Overview of knowledge base contents, ingestion statistics, and deductive engine status.'
    },
    'tab-documents': {
        title: 'Document Management',
        description: 'Upload PDF, DOCX, and TXT files for deterministic text extraction and SPO fact generation.'
    },
    'tab-kb': {
        title: 'Knowledge Base Explorer',
        description: 'Inspect relational Subject-Predicate-Object (SPO) triples and sentence provenance in SQLite.'
    },
    'tab-qa': {
        title: 'Ask DocuMind (Q&A & Symbolic Reasoning)',
        description: 'Query the knowledge base using deterministic natural language parsing and forward/backward inference.'
    },
    'tab-evaluation': {
        title: 'System Evaluation & Benchmarks',
        description: 'Deterministic 21-case empirical evaluation audit measuring accuracy, retrieval, reasoning, and provenance.'
    }
};

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
            // Close mobile menu if open
            const sidebar = document.getElementById('app-sidebar');
            if (sidebar) sidebar.classList.remove('mobile-open');
        });
    });
}

function initMobileMenu() {
    const toggleBtn = document.getElementById('mobile-menu-toggle');
    const sidebar = document.getElementById('app-sidebar');
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-open');
        });
    }
}

function switchTab(tabId) {
    // Update active nav button
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    const targetNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
    if (targetNav) targetNav.classList.add('active');

    // Update active tab section
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const targetContent = document.getElementById(tabId);
    if (targetContent) targetContent.classList.add('active');

    // Update Page Header Titles
    const meta = PAGE_META[tabId] || { title: 'DocuMind AI', description: '' };
    const pageTitleEl = document.getElementById('page-title');
    const pageDescEl = document.getElementById('page-description');
    if (pageTitleEl) pageTitleEl.textContent = meta.title;
    if (pageDescEl) pageDescEl.textContent = meta.description;

    // Trigger tab-specific refresh actions
    if (tabId === 'tab-dashboard') {
        refreshDashboardStats();
    } else if (tabId === 'tab-kb') {
        loadKnowledgeBaseFacts();
    } else if (tabId === 'tab-qa') {
        loadQuestionHistory();
    }
}

// =============================================================================
// 2. Document Upload & Ingestion
// =============================================================================

function initFileUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileSelectedBar = document.getElementById('file-selected-bar');
    const selectedFileName = document.getElementById('selected-file-name');
    const selectedFileSize = document.getElementById('selected-file-size');
    const clearFileBtn = document.getElementById('clear-file-btn');
    const uploadForm = document.getElementById('upload-form');
    const uploadSubmitBtn = document.getElementById('upload-submit-btn');
    const uploadSpinner = document.getElementById('upload-spinner');
    const uploadAlert = document.getElementById('upload-alert');
    const uploadResultCard = document.getElementById('upload-result-card');

    if (!dropZone || !fileInput) return;

    // Open file selector on click
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag-and-drop feedback
    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
            setMascotState('reading');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
            if (name === 'dragleave') setMascotState('idle');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            fileInput.files = dt.files;
            handleFileSelection(dt.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    if (clearFileBtn) {
        clearFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.value = '';
            fileSelectedBar.style.display = 'none';
            uploadSubmitBtn.disabled = true;
            if (uploadAlert) uploadAlert.style.display = 'none';
            if (uploadResultCard) uploadResultCard.style.display = 'none';
        });
    }

    function handleFileSelection(file) {
        const cleanName = normalizeFilename(file.name);
        selectedFileName.textContent = cleanName;
        if (selectedFileSize) {
            const kbSize = (file.size / 1024).toFixed(1);
            selectedFileSize.textContent = `(${kbSize} KB)`;
        }
        fileSelectedBar.style.display = 'flex';
        uploadSubmitBtn.disabled = false;
        if (uploadAlert) uploadAlert.style.display = 'none';
    }

    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!fileInput.files || fileInput.files.length === 0) return;

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        uploadSubmitBtn.disabled = true;
        uploadSpinner.style.display = 'inline-block';
        if (uploadAlert) uploadAlert.style.display = 'none';
        if (uploadResultCard) uploadResultCard.style.display = 'none';
        setMascotState('reading');

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json().then(data => ({ status: response.status, body: data })))
        .then(({ status, body }) => {
            uploadSubmitBtn.disabled = false;
            uploadSpinner.style.display = 'none';

            if (status === 200 && body.status === 'success') {
                renderUploadSuccess(body);
                refreshDashboardStats();
                setMascotState('answered');
            } else {
                renderUploadError(body.message || 'Failed to process document.');
                setMascotState('unsupported');
            }
        })
        .catch(err => {
            uploadSubmitBtn.disabled = false;
            uploadSpinner.style.display = 'none';
            renderUploadError(`Upload network failure: ${err.message}`);
            setMascotState('unsupported');
        });
    });

    function renderUploadSuccess(data) {
        if (uploadAlert) {
            uploadAlert.className = 'alert-box alert-success';
            uploadAlert.innerHTML = `<strong>Success!</strong> ${escapeHtml(data.message)}`;
            uploadAlert.style.display = 'block';
        }

        if (uploadResultCard) {
            uploadResultCard.style.display = 'block';
            const metaDiv = document.getElementById('upload-result-meta');
            if (metaDiv) {
                metaDiv.innerHTML = `
                    <span class="badge badge-indigo">Doc ID: #${data.document_id}</span>
                    <span class="badge badge-neutral">Format: ${escapeHtml(data.file_format.toUpperCase())}</span>
                    <span class="badge badge-neutral">${data.char_count} chars</span>
                    <span class="badge badge-neutral">${data.word_count} words</span>
                    <span class="badge badge-success">${data.facts_extracted} Facts Extracted</span>
                `;
            }

            const tbody = document.querySelector('#upload-preview-facts-table tbody');
            if (tbody && data.extracted_facts) {
                tbody.innerHTML = '';
                data.extracted_facts.forEach(f => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${escapeHtml(f.subject)}</strong></td>
                        <td><span class="predicate-badge">${escapeHtml(f.predicate)}</span></td>
                        <td><strong>${escapeHtml(f.object)}</strong></td>
                        <td><span class="source-sentence-quote">&ldquo;${escapeHtml(f.source_sentence)}&rdquo;</span></td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        }
    }

    function renderUploadError(message) {
        if (uploadAlert) {
            uploadAlert.className = 'alert-box alert-error';
            uploadAlert.innerHTML = `<strong>Extraction Error:</strong> ${escapeHtml(message)}`;
            uploadAlert.style.display = 'block';
        }
    }
}

// =============================================================================
// 3. Knowledge Base Explorer
// =============================================================================

function initKnowledgeBaseSearch() {
    const filterInput = document.getElementById('kb-filter-input');
    if (!filterInput) return;

    filterInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        const rows = document.querySelectorAll('#kb-facts-tbody tr');

        let matchCount = 0;
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            if (text.includes(query)) {
                row.style.display = '';
                matchCount++;
            } else {
                row.style.display = 'none';
            }
        });

        const badge = document.getElementById('kb-count-badge');
        if (badge) {
            badge.textContent = `Showing ${matchCount} fact(s)`;
        }
    });
}

function loadKnowledgeBaseFacts() {
    const badge = document.getElementById('kb-count-badge');
    const tbody = document.getElementById('kb-facts-tbody');
    if (badge) badge.textContent = 'Loading facts…';

    fetch('/api/facts?limit=100')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                kbFactsCache = data.facts || [];
                if (typeof updateKbGraphFromFacts === 'function') {
                    updateKbGraphFromFacts(kbFactsCache);
                }
                if (badge) badge.textContent = `Total Facts: ${data.total_facts}`;
                if (tbody) {
                    tbody.innerHTML = '';
                    if (data.facts.length === 0) {
                        tbody.innerHTML = `
                            <tr>
                                <td colspan="6" class="empty-table-state">
                                    <div class="empty-state-wrap">
                                        <img src="/static/illustrations/kb-empty.svg" alt="" class="empty-state-art" aria-hidden="true">
                                        <p>The knowledge base is currently empty. Upload documents to extract facts.</p>
                                        <button class="btn btn-sm btn-primary" onclick="switchTab('tab-documents')">Go to Document Ingestion</button>
                                    </div>
                                </td>
                            </tr>
                        `;
                    } else {
                        data.facts.forEach(f => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td><code class="code-id">#${f.fact_id}</code></td>
                                <td class="entity-subject"><strong>${escapeHtml(f.subject)}</strong></td>
                                <td><span class="predicate-badge">${escapeHtml(f.predicate)}</span></td>
                                <td class="entity-object"><strong>${escapeHtml(f.object)}</strong></td>
                                <td><small class="text-muted">${escapeHtml(normalizeFilename(f.filename))}</small></td>
                                <td><span class="source-sentence-quote">&ldquo;${escapeHtml(f.source_sentence)}&rdquo;</span></td>
                            `;
                            tbody.appendChild(tr);
                        });
                    }
                }
            }
        })
        .catch(err => {
            if (badge) badge.textContent = 'Failed to load facts';
            console.error('KB fetch error:', err);
        });
}

// =============================================================================
// 4. Ask DocuMind: Q&A & Symbolic Reasoning Pipeline
// =============================================================================

function setQuery(text) {
    const input = document.getElementById('question-input');
    if (input) {
        input.value = text;
        input.focus();
    }
}

function initQuestionAnswering() {
    const askForm = document.getElementById('ask-form');
    const input = document.getElementById('question-input');
    const btn = document.getElementById('ask-btn');
    const spinner = document.getElementById('ask-spinner');
    const answerContainer = document.getElementById('answer-container');

    if (!askForm) return;

    askForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const question = input.value.trim();
        if (!question) return;

        btn.disabled = true;
        spinner.style.display = 'inline-block';
        setMascotState('processing');

        fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        })
        .then(res => res.json())
        .then(data => {
            btn.disabled = false;
            spinner.style.display = 'none';
            renderAnswerResponse(data);
            loadQuestionHistory();
            refreshDashboardStats();
        })
        .catch(err => {
            btn.disabled = false;
            spinner.style.display = 'none';
            setMascotState('unsupported');
            alert(`Query processing failure: ${err.message}`);
        });
    });
}

function renderAnswerResponse(data) {
    lastAnswerData = data;
    const answerContainer = document.getElementById('answer-container');
    const badgesEl = document.getElementById('answer-status-badges');
    const latencyEl = document.getElementById('answer-latency');
    const answerTextEl = document.getElementById('answer-text');
    const parsedBarEl = document.getElementById('parsed-query-bar');
    const stepsContainer = document.getElementById('reasoning-steps-container');
    const supportingBox = document.getElementById('supporting-facts-box');
    const supportingTbody = document.querySelector('#supporting-facts-table tbody');
    const explanationPre = document.getElementById('explanation-text');

    if (!answerContainer) return;

    answerContainer.style.display = 'block';

    // 1. Telemetry
    latencyEl.textContent = `Latency: ${data.execution_time_ms || 0} ms`;

    // 2. Status Badges
    badgesEl.innerHTML = '';
    const status = data.answer_status;
    const type = data.answer_type;

    if (status === 'answered') {
        badgesEl.innerHTML += `<span class="badge badge-success">&#10003; Answered</span>`;
        if (type === 'inferred') {
            setMascotState('reasoning');
            setTimeout(() => { setMascotState('answered'); }, 1200);
        } else {
            setMascotState('answered');
        }
    } else if (status === 'unknown') {
        badgesEl.innerHTML += `<span class="badge badge-warning">&#9888; Unknown (Open-World)</span>`;
        setMascotState('unknown');
    } else if (status === 'unsupported') {
        badgesEl.innerHTML += `<span class="badge badge-danger">&#10005; Out-of-Scope</span>`;
        setMascotState('unsupported');
    } else {
        badgesEl.innerHTML += `<span class="badge badge-danger">Status: ${status}</span>`;
        setMascotState('idle');
    }

    if (type === 'direct') {
        badgesEl.innerHTML += `<span class="badge badge-indigo">Direct Retrieval</span>`;
    } else if (type === 'inferred') {
        badgesEl.innerHTML += `<span class="badge badge-cyan">Inferred via Deduction</span>`;
    }

    if (data.confidence && data.confidence !== 'none') {
        badgesEl.innerHTML += `<span class="badge badge-neutral">Confidence: ${data.confidence}</span>`;
    }

    // 3. Highlighted Answer Text
    answerTextEl.textContent = data.answer_text || 'No answer generated.';

    // 4. Query Audit Strip
    if (data.query_info) {
        const q = data.query_info;
        const catStr = q.category || 'N/A';
        const subjStr = q.subject || '?';
        const predStr = q.predicate || '?';
        const objStr = q.object || '?';
        const targetStr = q.expected_answer_field || 'N/A';
        parsedBarEl.textContent = `Category: [${catStr.toUpperCase()}] • SPO Triple: (${subjStr}, ${predStr}, ${objStr}) • Expected Target: [${targetStr}]`;
        parsedBarEl.style.display = 'block';
    } else {
        parsedBarEl.style.display = 'none';
    }

    // 5. Build Numbered Vertical Reasoning Timeline
    buildReasoningTimeline(stepsContainer, data);

    // 6. Populate Supporting Ground Facts Table
    supportingTbody.innerHTML = '';
    if (data.supporting_facts && data.supporting_facts.length > 0) {
        supportingBox.style.display = 'block';
        data.supporting_facts.forEach(f => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code class="code-id">#${f.fact_id}</code></td>
                <td><strong>${escapeHtml(f.subject)}</strong></td>
                <td><span class="predicate-badge">${escapeHtml(f.predicate)}</span></td>
                <td><strong>${escapeHtml(f.object)}</strong></td>
                <td><small class="text-muted">${escapeHtml(normalizeFilename(f.filename || 'Doc #' + f.document_id))}</small></td>
                <td><span class="source-sentence-quote">&ldquo;${escapeHtml(f.source_sentence || '')}&rdquo;</span></td>
            `;
            supportingTbody.appendChild(tr);
        });
    } else {
        supportingBox.style.display = 'none';
    }

    // 7. Full Verifiable Explanation
    explanationPre.textContent = data.explanation || 'No explanation report generated.';

    // Smooth scroll to answer
    answerContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function buildReasoningTimeline(container, data) {
    container.innerHTML = '';
    const status = data.answer_status;
    const type = data.answer_type;
    const qInfo = data.query_info || {};

    // -------------------------------------------------------------------------
    // STEP 1: Query Ingestion & Classification
    // -------------------------------------------------------------------------
    const step1 = document.createElement('div');
    step1.className = 'timeline-step';
    step1.setAttribute('data-step-num', '1');
    const catDisplay = (qInfo.category || 'Unclassified').toUpperCase();
    const isSupported = (status !== 'unsupported');

    step1.innerHTML = `
        <div class="step-label-row">
            <span class="step-phase-title">Step 1 — Question Classification & Deterministic Parsing</span>
            <span class="badge ${isSupported ? 'badge-indigo' : 'badge-danger'}">
                ${isSupported ? 'Grammar Match' : 'Unsupported Query'}
            </span>
        </div>
        <div class="step-content-body">
            Input Question: <em>"${escapeHtml(data.question)}"</em>.<br>
            Deterministic Question Category: <strong>${escapeHtml(catDisplay)}</strong>.
            ${qInfo.subject || qInfo.predicate || qInfo.object ? `
                <div style="margin-top: 0.4rem; font-family: var(--font-mono); font-size: 0.78rem; color: #475569;">
                    Mapped Structured Query: Subject=[<strong>${escapeHtml(qInfo.subject || '?')}</strong>], Predicate=[<strong>${escapeHtml(qInfo.predicate || '?')}</strong>], Object=[<strong>${escapeHtml(qInfo.object || '?')}</strong>] &rarr; Target: [${escapeHtml(qInfo.expected_answer_field || '?')}]
                </div>
            ` : ''}
        </div>
    `;
    container.appendChild(step1);

    if (status === 'unsupported') {
        const stepHalt = document.createElement('div');
        stepHalt.className = 'timeline-step';
        stepHalt.setAttribute('data-step-num', '2');
        stepHalt.innerHTML = `
            <div class="step-label-row">
                <span class="step-phase-title">Out-of-Scope Classification</span>
                <span class="badge badge-danger">Execution Halted</span>
            </div>
            <div class="step-content-body">
                The question does not conform to the 9 supported factual grammar categories.
                Under classical AI principles, execution safely terminates without probabilistic hallucination.
            </div>
        `;
        container.appendChild(stepHalt);
        return;
    }

    // -------------------------------------------------------------------------
    // STEP 2: Ground Knowledge Base Retrieval
    // -------------------------------------------------------------------------
    const step2 = document.createElement('div');
    step2.className = 'timeline-step';
    step2.setAttribute('data-step-num', '2');
    const factsCount = (data.supporting_facts || []).length;
    const isDirect = (type === 'direct');

    step2.innerHTML = `
        <div class="step-label-row">
            <span class="step-phase-title">Step 2 — Ground Fact Retrieval</span>
            <span class="badge ${factsCount > 0 ? 'badge-success' : 'badge-warning'}">
                ${factsCount > 0 ? `${factsCount} Fact(s) Matched` : 'Zero Direct Facts'}
            </span>
        </div>
        <div class="step-content-body">
            ${isDirect
                ? `Direct ground proposition verified in SQLite database. Complete sentence provenance preserved.`
                : (factsCount > 0
                    ? `Ground premise facts retrieved to support deductive inference in Working Memory.`
                    : `No direct factual triple satisfied the goal. Forward/backward chaining reasoning activated.`)}
        </div>
    `;
    container.appendChild(step2);

    // -------------------------------------------------------------------------
    // STEP 3: Symbolic Reasoning (If Inferred)
    // -------------------------------------------------------------------------
    if (type === 'inferred') {
        const step3 = document.createElement('div');
        step3.className = 'timeline-step active-rule-step';
        step3.setAttribute('data-step-num', '3');
        const rulesList = (data.applied_rule_ids || []).map((rid, i) => {
            const rname = (data.applied_rule_names && data.applied_rule_names[i]) ? data.applied_rule_names[i] : rid;
            return `<strong>${rid}</strong> (${rname})`;
        }).join(', ');

        step3.innerHTML = `
            <div class="step-label-row">
                <span class="step-phase-title">Step 3 — Symbolic Deductive Inference</span>
                <span class="badge badge-cyan">Modus Ponens Applied</span>
            </div>
            <div class="step-content-body">
                Production rules fired: ${rulesList}.<br>
                Goal proved via formal variable unification across ground premises in Working Memory.
                <div class="rule-highlight-box">
                    <span class="rule-name-tag">Active Inference Rule: ${(data.applied_rule_ids || []).join(', ')}</span>
                    <span class="rule-binding-line">
                        ${data.supporting_facts && data.supporting_facts.length > 0
                            ? data.supporting_facts.map(f => `Premise #${f.fact_id}: (${f.subject}, ${f.predicate}, ${f.object})`).join(' &bull; ')
                            : 'Premises unified from relational storage'}
                    </span>
                </div>
            </div>
        `;
        container.appendChild(step3);
    }

    // -------------------------------------------------------------------------
    // STEP 4: Final Conclusion Synthesis
    // -------------------------------------------------------------------------
    const step4 = document.createElement('div');
    step4.className = 'timeline-step';
    step4.setAttribute('data-step-num', type === 'inferred' ? '4' : '3');

    let conclusionTitle = 'Verified Response Synthesis';
    let conclusionBadge = 'badge-success';
    let conclusionBadgeText = 'Verified';
    let conclusionText = '';

    if (status === 'answered') {
        conclusionText = `Synthesized verifiable answer: <strong>"${escapeHtml(data.answer_text)}"</strong>.`;
        if (type === 'inferred') {
            conclusionBadge = 'badge-cyan';
            conclusionBadgeText = 'Deduced';
        }
    } else if (status === 'unknown') {
        conclusionTitle = 'Open-World Epistemological Resolution';
        conclusionBadge = 'badge-warning';
        conclusionBadgeText = 'Unknown';
        conclusionText = `Under the Open-World Assumption (OWA), lack of proof does not establish falsehood. The query proposition is correctly resolved as unrecorded.`;
    }

    step4.innerHTML = `
        <div class="step-label-row">
            <span class="step-phase-title">${conclusionTitle}</span>
            <span class="badge ${conclusionBadge}">${conclusionBadgeText}</span>
        </div>
        <div class="step-content-body">
            ${conclusionText}
        </div>
    `;
    container.appendChild(step4);
}

function toggleExplanationDetails() {
    const panel = document.getElementById('explanation-details-panel');
    const arrow = document.getElementById('explanation-arrow');
    if (!panel) return;

    if (panel.style.display === 'block') {
        panel.style.display = 'none';
        if (arrow) arrow.innerHTML = '&#9660;';
    } else {
        panel.style.display = 'block';
        if (arrow) arrow.innerHTML = '&#9650;';
    }
}

// =============================================================================
// 5. Question Interaction History
// =============================================================================

function loadQuestionHistory() {
    const listEl = document.getElementById('query-history-list');
    if (!listEl) return;

    fetch('/api/history?limit=15')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success' && data.history) {
                if (data.history.length === 0) {
                    listEl.innerHTML = '<div class="history-empty-placeholder">No interactions recorded in this workspace session yet.</div>';
                    return;
                }

                listEl.innerHTML = '';
                data.history.forEach(item => {
                    const row = document.createElement('div');
                    row.className = 'history-item-row';
                    const statusClass = item.answer_status === 'answered' ? 'badge-success' : (item.answer_status === 'unknown' ? 'badge-warning' : 'badge-danger');
                    
                    row.innerHTML = `
                        <div class="history-left-col">
                            <span class="history-q-text">${escapeHtml(item.question_text)}</span>
                            <span class="history-a-text">${escapeHtml(item.answer_text || 'No answer')} &bull; <small class="text-muted">${escapeHtml(item.timestamp || '')}</small></span>
                        </div>
                        <div class="history-right-col">
                            <span class="badge ${statusClass}">${item.answer_status}</span>
                            <button class="btn btn-xs btn-outline" onclick="setQuery('${escapeHtml(item.question_text).replace(/'/g, "\\'")}')">Ask Again</button>
                        </div>
                    `;
                    listEl.appendChild(row);
                });
            }
        })
        .catch(err => console.error('Failed to load question history:', err));
}

// =============================================================================
// 6. System Benchmark & Evaluation (Module 5 Stage 3)
// =============================================================================

function executeEvaluation() {
    const runBtn = document.getElementById('run-eval-btn');
    const spinner = document.getElementById('eval-spinner');
    const loadingState = document.getElementById('eval-loading-state');
    const resultsContainer = document.getElementById('eval-results-container');

    runBtn.disabled = true;
    spinner.style.display = 'inline-block';
    loadingState.style.display = 'block';
    resultsContainer.style.display = 'none';
    setMascotState('processing');

    fetch('/api/evaluate', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        runBtn.disabled = false;
        spinner.style.display = 'none';
        loadingState.style.display = 'none';

        if (data.status === 'success') {
            renderEvaluationDashboard(data);
            setMascotState('answered');
        } else {
            alert('Evaluation benchmark failed to complete.');
            setMascotState('idle');
        }
    })
    .catch(err => {
        runBtn.disabled = false;
        spinner.style.display = 'none';
        loadingState.style.display = 'none';
        setMascotState('idle');
        alert(`Evaluation execution error: ${err.message}`);
    });
}

function renderEvaluationDashboard(data) {
    const resultsContainer = document.getElementById('eval-results-container');
    resultsContainer.style.display = 'block';

    const s = data.summary;
    currentEvalCases = data.cases || [];

    // Overall Success Rate
    const successRate = ((s.successful_cases / s.total_cases) * 100).toFixed(1);
    document.getElementById('eval-success-rate').textContent = `${successRate}%`;
    document.getElementById('eval-cases-count').textContent = `${s.successful_cases} / ${s.total_cases} passed`;

    // Answer Accuracy
    document.getElementById('eval-answer-acc').textContent = `${s.answer_accuracy}%`;

    // Retrieval Metrics (Preserving N/A conventions cleanly!)
    const f1Str = s.retrieval_f1 !== null ? `${s.retrieval_f1}%` : 'N/A';
    document.getElementById('eval-retrieval-f1').textContent = f1Str;
    const pStr = s.retrieval_precision !== null ? `${s.retrieval_precision}%` : 'N/A';
    const rStr = s.retrieval_recall !== null ? `${s.retrieval_recall}%` : 'N/A';
    document.getElementById('eval-retrieval-pr').textContent = `Precision: ${pStr} | Recall: ${rStr}`;

    // Reasoning Accuracy
    document.getElementById('eval-reasoning-acc').textContent = `${s.reasoning_accuracy}%`;
    document.getElementById('eval-rule-prec').textContent = `Rule Precision: ${s.rule_precision}%`;

    // Provenance Accuracy
    document.getElementById('eval-provenance-acc').textContent = `${s.provenance_accuracy}%`;

    // Latency
    document.getElementById('eval-latency').textContent = `${s.average_execution_time_ms} ms`;
    document.getElementById('eval-total-time').textContent = `Total: ${s.total_execution_time_ms} ms`;

    // Render Table
    renderCasesTable(currentEvalCases);

    // Markdown Report
    document.getElementById('eval-report-pre').textContent = data.report_markdown || '';
}

function renderCasesTable(cases) {
    const casesTbody = document.getElementById('eval-cases-tbody');
    if (!casesTbody) return;
    casesTbody.innerHTML = '';

    cases.forEach(c => {
        const tr = document.createElement('tr');
        const badgeClass = c.is_overall_success ? 'badge-success' : 'badge-danger';
        const badgeText = c.is_overall_success ? 'PASS' : 'FAIL';
        const rulesStr = (c.applied_rule_ids && c.applied_rule_ids.length > 0) ? c.applied_rule_ids.join(', ') : 'None';

        tr.innerHTML = `
            <td><code class="code-id">${c.case_id}</code></td>
            <td><span class="badge badge-neutral">${escapeHtml(c.category)}</span></td>
            <td><strong>${escapeHtml(c.question)}</strong></td>
            <td><span class="badge badge-neutral">${c.actual_status}</span></td>
            <td><span class="badge badge-neutral">${c.expected_status}</span></td>
            <td><span class="predicate-badge">${rulesStr}</span></td>
            <td><small class="text-muted">${c.execution_time_ms} ms</small></td>
            <td><span class="badge ${badgeClass}">${badgeText}</span></td>
        `;
        casesTbody.appendChild(tr);
    });
}

function filterEvalCases(filterType) {
    document.querySelectorAll('.filter-btn-group .btn').forEach(b => b.classList.remove('active'));
    if (event && event.target) {
        event.target.classList.add('active');
    }

    if (filterType === 'pass') {
        renderCasesTable(currentEvalCases.filter(c => c.is_overall_success));
    } else if (filterType === 'fail') {
        renderCasesTable(currentEvalCases.filter(c => !c.is_overall_success));
    } else {
        renderCasesTable(currentEvalCases);
    }
}

function toggleReportAccordion() {
    const body = document.getElementById('report-accordion-body');
    const icon = document.getElementById('accordion-toggle-icon');
    if (!body) return;

    if (body.style.display === 'block') {
        body.style.display = 'none';
        if (icon) icon.innerHTML = '&#9660;';
    } else {
        body.style.display = 'block';
        if (icon) icon.innerHTML = '&#9650;';
    }
}

function copyReportMarkdown() {
    const pre = document.getElementById('eval-report-pre');
    if (pre && navigator.clipboard) {
        navigator.clipboard.writeText(pre.textContent).then(() => {
            alert('Academic Evaluation Report copied to clipboard!');
        });
    }
}

// =============================================================================
// 7. Modals: Nuke Data & Download Results
// =============================================================================

function openNukeModal() {
    const modal = document.getElementById('nuke-modal');
    const input = document.getElementById('nuke-confirm-input');
    const btn = document.getElementById('nuke-execute-btn');
    const msg = document.getElementById('nuke-status-msg');

    if (modal) {
        modal.style.display = 'flex';
        if (input) {
            input.value = '';
            input.focus();
        }
        if (btn) btn.disabled = true;
        if (msg) msg.style.display = 'none';
    }
}

function closeNukeModal() {
    const modal = document.getElementById('nuke-modal');
    if (modal) modal.style.display = 'none';
}

function handleNukeInput(val) {
    const btn = document.getElementById('nuke-execute-btn');
    if (btn) {
        btn.disabled = (val.trim() !== 'DELETE');
    }
}

function executeNukeWorkspace() {
    const input = document.getElementById('nuke-confirm-input');
    const btn = document.getElementById('nuke-execute-btn');
    const spinner = document.getElementById('nuke-spinner');
    const msgArea = document.getElementById('nuke-status-msg');

    if (!input || input.value.trim() !== 'DELETE') return;

    btn.disabled = true;
    if (spinner) spinner.style.display = 'inline-block';

    fetch('/api/workspace/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirmation: 'DELETE' })
    })
    .then(res => res.json().then(data => ({ status: res.status, body: data })))
    .then(({ status, body }) => {
        btn.disabled = false;
        if (spinner) spinner.style.display = 'none';

        if (status === 200 && body.status === 'success') {
            closeNukeModal();
            showGlobalToast(body.message, 'success');
            setMascotState('reset');
            
            // Reset all workspace views
            refreshDashboardStats();
            loadKnowledgeBaseFacts();
            loadQuestionHistory();

            // Clear answer container
            const ansContainer = document.getElementById('answer-container');
            if (ansContainer) ansContainer.style.display = 'none';

            // Clear upload results
            const upResult = document.getElementById('upload-result-card');
            if (upResult) upResult.style.display = 'none';
            const upAlert = document.getElementById('upload-alert');
            if (upAlert) upAlert.style.display = 'none';
            const fileSelectedBar = document.getElementById('file-selected-bar');
            if (fileSelectedBar) fileSelectedBar.style.display = 'none';
            const fileInput = document.getElementById('file-input');
            if (fileInput) fileInput.value = '';

            // Clear evaluation results view
            const evalRes = document.getElementById('eval-results-container');
            if (evalRes) evalRes.style.display = 'none';
        } else {
            if (msgArea) {
                msgArea.className = 'nuke-msg-area alert-box alert-error';
                msgArea.textContent = body.message || 'Reset failed.';
                msgArea.style.display = 'block';
            }
        }
    })
    .catch(err => {
        btn.disabled = false;
        if (spinner) spinner.style.display = 'none';
        if (msgArea) {
            msgArea.className = 'nuke-msg-area alert-box alert-error';
            msgArea.textContent = `Network error: ${err.message}`;
            msgArea.style.display = 'block';
        }
    });
}

function openExportModal() {
    const modal = document.getElementById('export-modal');
    if (modal) modal.style.display = 'flex';
}

function closeExportModal() {
    const modal = document.getElementById('export-modal');
    if (modal) modal.style.display = 'none';
}

function toggleHeaderExportDropdown(e) {
    e.stopPropagation();
    const menu = document.getElementById('header-export-menu');
    if (menu) {
        menu.style.display = (menu.style.display === 'flex' || menu.style.display === 'block') ? 'none' : 'block';
    }
}

function initDropdownListeners() {
    document.addEventListener('click', (e) => {
        const menu = document.getElementById('header-export-menu');
        if (menu && menu.style.display !== 'none') {
            if (!menu.contains(e.target)) {
                menu.style.display = 'none';
            }
        }
    });

    // Close modals on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeNukeModal();
            closeExportModal();
        }
    });
}

function showGlobalToast(message, type = 'success') {
    const toast = document.getElementById('global-alert-toast');
    if (!toast) return;

    toast.className = `alert-toast-bar toast-${type}`;
    toast.innerHTML = `
        <span>${escapeHtml(message)}</span>
        <button type="button" class="btn btn-xs btn-outline" onclick="this.parentElement.style.display='none'">&times;</button>
    `;
    toast.style.display = 'flex';

    setTimeout(() => {
        if (toast) toast.style.display = 'none';
    }, 7000);
}

// =============================================================================
// 8. Dashboard Statistics Live Refresh
// =============================================================================

function refreshDashboardStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            const totalDocsEl = document.getElementById('stat-total-docs');
            const succDocsEl = document.getElementById('stat-successful-docs');
            const failDocsEl = document.getElementById('stat-failed-docs');
            const totalFactsEl = document.getElementById('stat-total-facts');

            if (totalDocsEl) totalDocsEl.textContent = data.total_documents;
            if (succDocsEl) succDocsEl.textContent = data.successful_documents;
            if (failDocsEl) failDocsEl.textContent = `${data.failed_documents} extraction failures`;
            if (totalFactsEl) totalFactsEl.textContent = data.total_facts;

            // Update recent documents table
            const tbody = document.querySelector('#recent-docs-table tbody');
            if (tbody && data.recent_documents) {
                tbody.innerHTML = '';
                if (data.recent_documents.length > 0) {
                    data.recent_documents.forEach(doc => {
                        const tr = document.createElement('tr');
                        const statusBadge = doc.extraction_status === 'success' 
                            ? '<span class="badge badge-success">&#10003; Success</span>'
                            : `<span class="badge badge-danger">&#9888; ${escapeHtml(doc.extraction_status)}</span>`;

                        tr.innerHTML = `
                            <td><code class="code-id">#${doc.document_id}</code></td>
                            <td>
                                <div class="doc-file-cell">
                                    <span class="file-icon-badge format-${doc.file_format.toLowerCase()}">${doc.file_format}</span>
                                    <span class="doc-name-text">${escapeHtml(normalizeFilename(doc.filename))}</span>
                                </div>
                            </td>
                            <td><span class="badge badge-neutral">${escapeHtml(doc.file_format)}</span></td>
                            <td class="text-muted"><small>${escapeHtml(doc.upload_date)}</small></td>
                            <td>${statusBadge}</td>
                            <td><strong>${doc.fact_count}</strong> facts</td>
                        `;
                        tbody.appendChild(tr);
                    });
                } else {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="6" class="empty-table-state">
                                <div class="empty-state-wrap">
                                    <img src="/static/illustrations/documents-empty.svg" alt="" class="empty-state-art" aria-hidden="true">
                                    <p>No documents uploaded yet. Upload a document to build the knowledge base.</p>
                                    <button class="btn btn-sm btn-primary" onclick="switchTab('tab-documents')">Upload First Document</button>
                                </div>
                            </td>
                        </tr>`;
                }
            }
        })
        .catch(err => console.error('Failed to refresh dashboard stats:', err));
}

// =============================================================================
// Utilities
// =============================================================================

function normalizeFilename(filename) {
    if (!filename) return '';
    let name = String(filename).trim();
    const exts = ['.txt', '.pdf', '.docx'];
    let cleaned = true;
    while (cleaned) {
        cleaned = false;
        for (const ext of exts) {
            for (const prev of exts) {
                const combo = (prev + ext).toLowerCase();
                if (name.toLowerCase().endsWith(combo)) {
                    name = name.slice(0, -ext.length);
                    cleaned = true;
                    break;
                }
            }
            if (cleaned) break;
        }
    }
    return name;
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// =============================================================================
// Theme Switching (Obsidian Dark / Botanical Parchment)
// =============================================================================

function initTheme() {
    const saved = localStorage.getItem('documind_theme');
    if (saved === 'dark' || (!saved && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        applyTheme('dark');
    } else {
        applyTheme('light');
    }
}

function toggleTheme() {
    const isDark = document.body.classList.contains('dark-theme');
    applyTheme(isDark ? 'light' : 'dark');
}

function applyTheme(theme) {
    const lightIcon = document.querySelector('.theme-icon-light');
    const darkIcon = document.querySelector('.theme-icon-dark');
    const label = document.getElementById('theme-label-text');

    if (theme === 'dark') {
        document.body.classList.add('dark-theme');
        localStorage.setItem('documind_theme', 'dark');
        if (lightIcon) lightIcon.style.display = 'none';
        if (darkIcon) darkIcon.style.display = 'inline-block';
        if (label) label.textContent = 'Botanical Light';
    } else {
        document.body.classList.remove('dark-theme');
        localStorage.setItem('documind_theme', 'light');
        if (lightIcon) lightIcon.style.display = 'inline-block';
        if (darkIcon) darkIcon.style.display = 'none';
        if (label) label.textContent = 'Obsidian Dark';
    }
    if (graphViewMode === 'graph' && typeof renderGraph === 'function') {
        requestAnimationFrame(renderGraph);
    }
}

// =============================================================================
// Academic Sample Document Loader
// =============================================================================

function loadSampleDocument() {
    const btn = document.getElementById('load-sample-btn');
    const spinner = document.getElementById('sample-spinner');
    if (btn) btn.disabled = true;
    if (spinner) spinner.style.display = 'inline-block';
    setMascotState('reading');

    fetch('/api/load-sample', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (btn) btn.disabled = false;
            if (spinner) spinner.style.display = 'none';

            if (data.status === 'success') {
                showGlobalToast(`Ingested sample corpus: "${data.filename}" (${data.facts_extracted} SPO facts extracted)`, 'success');
                setMascotState('answered');
                refreshDashboardStats();
                loadKnowledgeBaseFacts();
                setTimeout(() => { switchTab('tab-kb'); }, 700);
            } else {
                showGlobalToast(data.message || 'Error loading sample document', 'error');
                setMascotState('unsupported');
            }
        })
        .catch(err => {
            if (btn) btn.disabled = false;
            if (spinner) spinner.style.display = 'none';
            setMascotState('unsupported');
            showGlobalToast(`Network error: ${err.message}`, 'error');
        });
}

// =============================================================================
// Answer & Proof Copy Actions
// =============================================================================

let lastAnswerData = null;

function copyCurrentAnswer() {
    if (!lastAnswerData || !lastAnswerData.answer_text) {
        showGlobalToast('No answer to copy', 'warning');
        return;
    }
    navigator.clipboard.writeText(lastAnswerData.answer_text).then(() => {
        const label = document.getElementById('copy-answer-label');
        if (label) {
            const old = label.textContent;
            label.textContent = 'Copied!';
            setTimeout(() => { label.textContent = old; }, 1800);
        }
        showGlobalToast('Answer copied to clipboard!', 'success');
    }).catch(err => {
        showGlobalToast('Clipboard access denied', 'error');
    });
}

function copyDeductiveProof() {
    if (!lastAnswerData) {
        showGlobalToast('No deductive proof to copy', 'warning');
        return;
    }
    const text = (lastAnswerData.explanation && lastAnswerData.explanation.full_report) 
        ? lastAnswerData.explanation.full_report 
        : (lastAnswerData.answer_text || 'No proof available.');
    navigator.clipboard.writeText(text).then(() => {
        const label = document.getElementById('copy-proof-label');
        if (label) {
            const old = label.textContent;
            label.textContent = 'Copied Proof!';
            setTimeout(() => { label.textContent = old; }, 1800);
        }
        showGlobalToast('Deductive proof report copied to clipboard!', 'success');
    }).catch(err => {
        showGlobalToast('Clipboard access denied', 'error');
    });
}

// =============================================================================
// Interactive Knowledge Graph Canvas Visualizer
// =============================================================================

let kbGraphNodes = [];
let kbGraphEdges = [];
let kbFactsCache = [];
let graphViewMode = 'table';
let graphScale = 1.0;
let graphPanX = 0;
let graphPanY = 0;
let isDraggingNode = null;
let isPanningGraph = false;
let panStartX = 0;
let panStartY = 0;
let hoveredNode = null;
let canvasEl = null;
let ctx = null;
let simStep = 0;
const MAX_SIM_STEPS = 200;

function setKbViewMode(mode) {
    graphViewMode = mode;
    const btnTable = document.getElementById('btn-view-table');
    const btnGraph = document.getElementById('btn-view-graph');
    const tablePanel = document.getElementById('kb-table-panel');
    const graphPanel = document.getElementById('kb-graph-panel');

    if (mode === 'graph') {
        if (btnTable) btnTable.classList.remove('active');
        if (btnGraph) btnGraph.classList.add('active');
        if (tablePanel) tablePanel.style.display = 'none';
        if (graphPanel) graphPanel.style.display = 'flex';
        initKbGraphCanvas();
        if (kbFactsCache.length > 0) {
            updateKbGraphFromFacts(kbFactsCache);
        } else {
            loadKnowledgeBaseFacts();
        }
    } else {
        if (btnTable) btnTable.classList.add('active');
        if (btnGraph) btnGraph.classList.remove('active');
        if (tablePanel) tablePanel.style.display = 'block';
        if (graphPanel) graphPanel.style.display = 'none';
    }
}

function initGraphVisualizer() {
    canvasEl = document.getElementById('kb-graph-canvas');
    if (!canvasEl) return;
    ctx = canvasEl.getContext('2d');

    const viewport = document.getElementById('canvas-viewport');
    if (!viewport) return;

    function resizeCanvas() {
        const rect = viewport.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvasEl.width = rect.width * dpr;
        canvasEl.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        canvasEl.style.width = rect.width + 'px';
        canvasEl.style.height = rect.height + 'px';
        renderGraph();
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Mouse Interactions
    viewport.addEventListener('mousedown', e => {
        const mousePos = getCanvasMousePos(e);
        const node = findNodeAt(mousePos.x, mousePos.y);
        if (node) {
            isDraggingNode = node;
            isDraggingNode.isFixed = true;
        } else {
            isPanningGraph = true;
            panStartX = e.clientX - graphPanX;
            panStartY = e.clientY - graphPanY;
        }
    });

    window.addEventListener('mousemove', e => {
        if (!canvasEl) return;
        const mousePos = getCanvasMousePos(e);
        if (isDraggingNode) {
            isDraggingNode.x = (mousePos.x - graphPanX) / graphScale;
            isDraggingNode.y = (mousePos.y - graphPanY) / graphScale;
            renderGraph();
        } else if (isPanningGraph) {
            graphPanX = e.clientX - panStartX;
            graphPanY = e.clientY - panStartY;
            renderGraph();
        } else {
            const prev = hoveredNode;
            hoveredNode = findNodeAt(mousePos.x, mousePos.y);
            if (prev !== hoveredNode) {
                viewport.style.cursor = hoveredNode ? 'pointer' : 'grab';
                renderGraph();
            }
        }
    });

    window.addEventListener('mouseup', () => {
        if (isDraggingNode) {
            isDraggingNode.isFixed = false;
            isDraggingNode = null;
        }
        isPanningGraph = false;
    });

    viewport.addEventListener('wheel', e => {
        e.preventDefault();
        const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
        zoomKbGraph(zoomFactor, e.clientX, e.clientY);
    }, { passive: false });

    viewport.addEventListener('click', e => {
        const mousePos = getCanvasMousePos(e);
        const node = findNodeAt(mousePos.x, mousePos.y);
        if (node) {
            openNodeInspector(node);
        }
    });
}

function initKbGraphCanvas() {
    if (!canvasEl) {
        initGraphVisualizer();
    }
}

function getCanvasMousePos(e) {
    const viewport = document.getElementById('canvas-viewport');
    const rect = viewport.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function findNodeAt(x, y) {
    const worldX = (x - graphPanX) / graphScale;
    const worldY = (y - graphPanY) / graphScale;
    for (let i = kbGraphNodes.length - 1; i >= 0; i--) {
        const n = kbGraphNodes[i];
        const dx = worldX - n.x;
        const dy = worldY - n.y;
        if (dx * dx + dy * dy <= (n.radius + 4) * (n.radius + 4)) {
            return n;
        }
    }
    return null;
}

function updateKbGraphFromFacts(facts) {
    if (!facts || facts.length === 0) {
        kbGraphNodes = [];
        kbGraphEdges = [];
        renderGraph();
        return;
    }

    const nodeMap = new Map();
    kbGraphEdges = [];

    facts.forEach(f => {
        const subName = (f.subject || '').trim();
        const objName = (f.object || '').trim();
        const pred = (f.predicate || '').trim();
        if (!subName || !objName) return;

        if (!nodeMap.has(subName)) {
            nodeMap.set(subName, {
                id: subName,
                name: subName,
                x: (Math.random() - 0.5) * 500,
                y: (Math.random() - 0.5) * 400,
                vx: 0,
                vy: 0,
                degree: 0,
                radius: 18,
                facts: []
            });
        }
        if (!nodeMap.has(objName)) {
            nodeMap.set(objName, {
                id: objName,
                name: objName,
                x: (Math.random() - 0.5) * 500,
                y: (Math.random() - 0.5) * 400,
                vx: 0,
                vy: 0,
                degree: 0,
                radius: 18,
                facts: []
            });
        }

        const subNode = nodeMap.get(subName);
        const objNode = nodeMap.get(objName);
        subNode.degree++;
        objNode.degree++;
        subNode.facts.push(f);
        objNode.facts.push(f);

        kbGraphEdges.push({
            source: subNode,
            target: objNode,
            predicate: pred,
            fact: f
        });
    });

    kbGraphNodes = Array.from(nodeMap.values());
    kbGraphNodes.forEach(n => {
        n.radius = Math.min(32, Math.max(16, 14 + n.degree * 2.5));
    });

    const nodeBadge = document.getElementById('graph-node-count-badge');
    const edgeBadge = document.getElementById('graph-edge-count-badge');
    if (nodeBadge) nodeBadge.textContent = `${kbGraphNodes.length} Entities`;
    if (edgeBadge) edgeBadge.textContent = `${kbGraphEdges.length} Predicates`;

    resetKbGraphView();
    simStep = 0;
    runSimulationLoop();
}

function runSimulationLoop() {
    if (simStep >= MAX_SIM_STEPS) {
        renderGraph();
        return;
    }

    simStep++;
    const k = 140; // ideal spring distance

    // 1. Repulsion between all nodes
    for (let i = 0; i < kbGraphNodes.length; i++) {
        for (let j = i + 1; j < kbGraphNodes.length; j++) {
            const n1 = kbGraphNodes[i];
            const n2 = kbGraphNodes[j];
            const dx = n2.x - n1.x;
            const dy = n2.y - n1.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            if (dist < 380) {
                const force = (k * k) / (dist * dist);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                if (!n1.isFixed) { n1.vx -= fx; n1.vy -= fy; }
                if (!n2.isFixed) { n2.vx += fx; n2.vy += fy; }
            }
        }
    }

    // 2. Attraction along edges
    for (let i = 0; i < kbGraphEdges.length; i++) {
        const e = kbGraphEdges[i];
        const dx = e.target.x - e.source.x;
        const dy = e.target.y - e.source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist * dist) / k;
        const fx = (dx / dist) * force * 0.08;
        const fy = (dy / dist) * force * 0.08;
        if (!e.source.isFixed) { e.source.vx += fx; e.source.vy += fy; }
        if (!e.target.isFixed) { e.target.vx -= fx; e.target.vy -= fy; }
    }

    // 3. Center gravity and apply damping
    for (let i = 0; i < kbGraphNodes.length; i++) {
        const n = kbGraphNodes[i];
        if (!n.isFixed) {
            n.vx -= n.x * 0.015;
            n.vy -= n.y * 0.015;
            n.x += n.vx * 0.15;
            n.y += n.vy * 0.15;
            n.vx *= 0.85;
            n.vy *= 0.85;
        }
    }

    renderGraph();
    requestAnimationFrame(runSimulationLoop);
}

function renderGraph() {
    if (!ctx || !canvasEl) return;
    const viewport = document.getElementById('canvas-viewport');
    if (!viewport) return;

    const w = viewport.clientWidth;
    const h = viewport.clientHeight;
    ctx.clearRect(0, 0, w, h);

    const isDark = document.body.classList.contains('dark-theme');
    const bgEdgeColor = isDark ? 'rgba(79, 115, 101, 0.45)' : 'rgba(185, 199, 187, 0.65)';
    const textPredColor = isDark ? '#b5c7bd' : '#63736b';

    ctx.save();
    ctx.translate(graphPanX, graphPanY);
    ctx.scale(graphScale, graphScale);

    // 1. Draw Edges
    for (let i = 0; i < kbGraphEdges.length; i++) {
        const e = kbGraphEdges[i];
        const isHovered = hoveredNode && (hoveredNode === e.source || hoveredNode === e.target);
        
        ctx.beginPath();
        ctx.strokeStyle = isHovered ? (isDark ? '#e2b66f' : '#315c50') : bgEdgeColor;
        ctx.lineWidth = isHovered ? 2.5 : 1.2;
        ctx.moveTo(e.source.x, e.source.y);
        ctx.lineTo(e.target.x, e.target.y);
        ctx.stroke();

        // Edge label (predicate)
        const midX = (e.source.x + e.target.x) / 2;
        const midY = (e.source.y + e.target.y) / 2;
        ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.fillStyle = isHovered ? (isDark ? '#f4e5c8' : '#172b2a') : textPredColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(e.predicate, midX, midY - 6);
    }

    // 2. Draw Nodes
    for (let i = 0; i < kbGraphNodes.length; i++) {
        const n = kbGraphNodes[i];
        const isHovered = (n === hoveredNode);

        // Node circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        
        if (isDark) {
            ctx.fillStyle = isHovered ? '#416f60' : '#203b35';
            ctx.strokeStyle = isHovered ? '#e2b66f' : '#4f7365';
        } else {
            ctx.fillStyle = isHovered ? '#8bae9b' : '#ffffff';
            ctx.strokeStyle = isHovered ? '#315c50' : '#8bae9b';
        }
        ctx.lineWidth = isHovered ? 3 : 2;
        ctx.fill();
        ctx.stroke();

        // Node label
        ctx.font = `${Math.max(11, Math.min(13, n.radius - 2))}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
        ctx.fillStyle = isDark ? '#edf3ee' : '#20312d';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        const label = n.name.length > 15 ? n.name.slice(0, 13) + '…' : n.name;
        ctx.fillText(label, n.x, n.y);
    }

    ctx.restore();
}

function zoomKbGraph(factor, clientX, clientY) {
    const viewport = document.getElementById('canvas-viewport');
    if (!viewport) return;
    const rect = viewport.getBoundingClientRect();
    const cx = clientX ? (clientX - rect.left) : (viewport.clientWidth / 2);
    const cy = clientY ? (clientY - rect.top) : (viewport.clientHeight / 2);

    const prevScale = graphScale;
    graphScale = Math.min(2.8, Math.max(0.35, graphScale * factor));
    graphPanX = cx - (cx - graphPanX) * (graphScale / prevScale);
    graphPanY = cy - (cy - graphPanY) * (graphScale / prevScale);
    renderGraph();
}

function resetKbGraphView() {
    const viewport = document.getElementById('canvas-viewport');
    if (!viewport) return;
    graphScale = 1.0;
    graphPanX = viewport.clientWidth / 2;
    graphPanY = viewport.clientHeight / 2;
    renderGraph();
}

function openNodeInspector(node) {
    const modal = document.getElementById('graph-node-modal');
    const titleEl = document.getElementById('inspector-entity-title');
    const listEl = document.getElementById('inspector-facts-list');
    if (!modal || !titleEl || !listEl) return;

    titleEl.textContent = node.name;
    listEl.innerHTML = '';

    if (!node.facts || node.facts.length === 0) {
        listEl.innerHTML = '<div style="color: var(--text-muted);">No facts associated with this entity.</div>';
    } else {
        node.facts.forEach(f => {
            const item = document.createElement('div');
            item.className = 'inspector-fact-item';
            item.innerHTML = `
                <div><strong>${escapeHtml(f.subject)}</strong> <span class="predicate-badge" style="font-size: 0.7rem;">${escapeHtml(f.predicate)}</span> <strong>${escapeHtml(f.object)}</strong></div>
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem;">&ldquo;${escapeHtml(f.source_sentence)}&rdquo;</div>
            `;
            listEl.appendChild(item);
        });
    }

    modal.style.display = 'flex';
}

function closeNodeInspector() {
    const modal = document.getElementById('graph-node-modal');
    if (modal) modal.style.display = 'none';
}
