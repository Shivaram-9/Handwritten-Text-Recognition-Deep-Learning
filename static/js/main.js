/**
 * NeuralText — AI Handwritten Text Recognition Platform
 * Interactive Client Controller & Background Canvas Animation
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Background Neural Canvas Animation ---
    initNeuralCanvas();

    // --- 2. DOM Elements Selection ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const browseBtn = document.getElementById('browseBtn');
    
    // Preview Controls
    const previewEmptyState = document.getElementById('previewEmptyState');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const replaceBtn = document.getElementById('replaceBtn');
    const removeBtn = document.getElementById('removeBtn');
    const recognizeBtn = document.getElementById('recognizeBtn');

    // Feedback & Loading
    const loadingState = document.getElementById('loadingState');
    const loadingStatusText = document.getElementById('loadingStatusText');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const errorCloseBtn = document.getElementById('errorCloseBtn');

    // Output & Metrics
    const resultSection = document.getElementById('resultSection');
    const recognizedText = document.getElementById('recognizedText');
    const confidenceScoreText = document.getElementById('confidenceScoreText');
    const confidenceProgressBar = document.getElementById('confidenceProgressBar');
    const metricPrep = document.getElementById('metricPrep');
    const metricInfer = document.getElementById('metricInfer');
    const metricTotal = document.getElementById('metricTotal');

    // Result Actions
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const clearBtn = document.getElementById('clearBtn');

    // Pipeline Visualizer Elements
    const visOriginal = document.getElementById('visOriginal');
    const visPerspective = document.getElementById('visPerspective');
    const visClahe = document.getElementById('visClahe');
    const visLines = document.getElementById('visLines');
    const placeholderStep1 = document.getElementById('placeholderStep1');
    const placeholderStep2 = document.getElementById('placeholderStep2');
    const placeholderStep3 = document.getElementById('placeholderStep3');
    const placeholderStep4 = document.getElementById('placeholderStep4');

    // Session History Elements
    const historyCount = document.getElementById('historyCount');
    const historyTableBody = document.getElementById('historyTableBody');
    const emptyHistoryRow = document.getElementById('emptyHistoryRow');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');

    // Theme Switcher & Mobile Nav
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeIcon = document.getElementById('themeIcon');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navMenu = document.getElementById('navMenu');

    // State
    let currentFile = null;
    let currentDataUrl = null;
    let sessionHistory = [];

    // --- 3. Theme Toggle Setup ---
    initTheme();

    function initTheme() {
        const savedTheme = localStorage.getItem('neuraltext_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);

        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                const current = document.documentElement.getAttribute('data-theme');
                const next = current === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', next);
                localStorage.setItem('neuraltext_theme', next);
                updateThemeIcon(next);
            });
        }
    }

    function updateThemeIcon(theme) {
        if (!themeIcon) return;
        themeIcon.setAttribute('data-lucide', theme === 'dark' ? 'sun' : 'moon');
        if (window.lucide) lucide.createIcons();
    }

    // Mobile nav toggle
    if (mobileMenuBtn && navMenu) {
        mobileMenuBtn.addEventListener('click', () => {
            navMenu.classList.toggle('nav-menu-open');
        });
    }

    // --- 4. Drag & Drop File Handling ---
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (dropZone) {
        dropZone.addEventListener('click', (e) => {
            if (e.target !== browseBtn) {
                fileInput.click();
            }
        });

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, preventDefaults, false);
            document.body.addEventListener(name, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, () => dropZone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, () => dropZone.classList.remove('dragover'), false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleFileSelect(dt.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                handleFileSelect(this.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        hideError();
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a high-contrast JPG, PNG, or WEBP image.');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showError('File exceeds 16 MB. Please choose a smaller image.');
            return;
        }

        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            currentDataUrl = e.target.result;
            imagePreview.src = currentDataUrl;
            
            // Show preview, hide placeholder
            previewEmptyState.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            recognizeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Replace / Remove buttons
    if (replaceBtn) {
        replaceBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUploadState();
        });
    }

    function resetUploadState() {
        currentFile = null;
        currentDataUrl = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        previewEmptyState.classList.remove('hidden');
        recognizeBtn.disabled = true;
    }

    // --- 5. Run AI Model Execution ---
    if (recognizeBtn) {
        recognizeBtn.addEventListener('click', runInference);
    }

    async function runInference() {
        if (!currentFile) {
            showError('Please upload a handwriting image first.');
            return;
        }

        // UI state: loading
        hideError();
        recognizeBtn.disabled = true;
        loadingState.classList.remove('hidden');
        loadingStatusText.textContent = 'Preprocessing image & binarizing strokes...';

        const stages = [
            'Normalizing contrast & skew angle...',
            'Adaptive line segmentation...',
            'Convolutional feature extraction...',
            'Bidirectional LSTM temporal decoding...',
            'Connectionist Temporal Classification greedy decoding...'
        ];
        let stageIdx = 0;
        const progressTimer = setInterval(() => {
            stageIdx = (stageIdx + 1) % stages.length;
            loadingStatusText.textContent = stages[stageIdx];
        }, 450);

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressTimer);
            loadingState.classList.add('hidden');
            recognizeBtn.disabled = false;

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || 'Server error during recognition.');
                return;
            }

            // Display results smoothly
            renderPredictionResults(data);

            // Record into session history
            addToHistory({
                thumb: currentDataUrl,
                text: data.recognized_text || '',
                confidence: data.confidence !== undefined ? data.confidence : 0,
                latency: data.latency && data.latency.total_ms ? data.latency.total_ms : 0,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            });

            // Smooth scroll down to result card
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (err) {
            clearInterval(progressTimer);
            loadingState.classList.add('hidden');
            recognizeBtn.disabled = false;
            showError('Network connectivity error. Please ensure the server is running.');
            console.error('Inference error:', err);
        }
    }

    // --- 6. Render Prediction & Pipeline Visualizer ---
    function renderPredictionResults(data) {
        resultSection.classList.remove('hidden');

        // Text & Confidence
        recognizedText.textContent = data.recognized_text || '(No text detected)';
        const conf = typeof data.confidence === 'number' ? data.confidence : 0;
        confidenceScoreText.textContent = `${conf.toFixed(1)}%`;
        confidenceProgressBar.style.width = `${Math.min(100, Math.max(0, conf))}%`;

        // Color gauge based on confidence
        if (conf >= 80) {
            confidenceProgressBar.style.background = 'linear-gradient(90deg, #10b981, #06b6d4)';
        } else if (conf >= 55) {
            confidenceProgressBar.style.background = 'linear-gradient(90deg, #f59e0b, #10b981)';
        } else {
            confidenceProgressBar.style.background = 'linear-gradient(90deg, #ef4444, #f59e0b)';
        }

        // Latencies
        if (data.latency) {
            metricPrep.textContent = `${data.latency.preprocessing_ms || 0} ms`;
            metricInfer.textContent = `${data.latency.inference_ms || 0} ms`;
            metricTotal.textContent = `${data.latency.total_ms || 0} ms`;
        }

        // Render 4-step Pipeline Visualizer
        const imgs = data.pipeline_images || {};

        // Step 1: Original
        if (imgs.original) {
            visOriginal.src = `data:image/jpeg;base64,${imgs.original}`;
            visOriginal.classList.remove('hidden');
            placeholderStep1.classList.add('hidden');
        } else if (currentDataUrl) {
            visOriginal.src = currentDataUrl;
            visOriginal.classList.remove('hidden');
            placeholderStep1.classList.add('hidden');
        }

        // Step 2: Perspective / Normalization
        if (imgs.perspective) {
            visPerspective.src = `data:image/jpeg;base64,${imgs.perspective}`;
            visPerspective.classList.remove('hidden');
            placeholderStep2.classList.add('hidden');
        } else if (imgs.preprocessed) {
            visPerspective.src = `data:image/jpeg;base64,${imgs.preprocessed}`;
            visPerspective.classList.remove('hidden');
            placeholderStep2.classList.add('hidden');
        }

        // Step 3: CLAHE & Adaptive Thresholding
        if (imgs.clahe_thresh) {
            visClahe.src = `data:image/jpeg;base64,${imgs.clahe_thresh}`;
            visClahe.classList.remove('hidden');
            placeholderStep3.classList.add('hidden');
        } else if (imgs.preprocessed) {
            visClahe.src = `data:image/jpeg;base64,${imgs.preprocessed}`;
            visClahe.classList.remove('hidden');
            placeholderStep3.classList.add('hidden');
        }

        // Step 4: Multi-line Segments
        visLines.innerHTML = '';
        if (Array.isArray(imgs.lines) && imgs.lines.length > 0) {
            placeholderStep4.classList.add('hidden');
            imgs.lines.forEach((b64, idx) => {
                const lineImg = document.createElement('img');
                lineImg.src = `data:image/jpeg;base64,${b64}`;
                lineImg.alt = `Segment ${idx + 1}`;
                lineImg.className = 'pipeline-line-crop';
                visLines.appendChild(lineImg);
            });
        } else {
            placeholderStep4.classList.remove('hidden');
        }

        if (window.lucide) lucide.createIcons();
    }

    // --- 7. Result Action Buttons ---
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const text = recognizedText.textContent;
            if (!text || text === 'Waiting for input...') return;

            navigator.clipboard.writeText(text).then(() => {
                const originalText = copyBtn.innerHTML;
                copyBtn.innerHTML = '<i data-lucide="check"></i><span>Copied!</span>';
                if (window.lucide) lucide.createIcons();
                setTimeout(() => {
                    copyBtn.innerHTML = originalText;
                    if (window.lucide) lucide.createIcons();
                }, 1800);
            }).catch(() => {
                showError('Could not copy to clipboard.');
            });
        });
    }

    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            const text = recognizedText.textContent;
            if (!text || text === 'Waiting for input...') return;

            const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `neuraltext-transcription-${Date.now()}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            resultSection.classList.add('hidden');
            recognizedText.textContent = 'Waiting for input...';
            confidenceScoreText.textContent = '0.0%';
            confidenceProgressBar.style.width = '0%';
            metricPrep.textContent = '-- ms';
            metricInfer.textContent = '-- ms';
            metricTotal.textContent = '-- ms';
            resetPipelinePlaceholders();
        });
    }

    function resetPipelinePlaceholders() {
        visOriginal.classList.add('hidden');
        visPerspective.classList.add('hidden');
        visClahe.classList.add('hidden');
        placeholderStep1.classList.remove('hidden');
        placeholderStep2.classList.remove('hidden');
        placeholderStep3.classList.remove('hidden');
        placeholderStep4.classList.remove('hidden');
        visLines.innerHTML = '';
    }

    // --- 8. Session History Table ---
    function addToHistory(item) {
        sessionHistory.unshift(item);
        if (sessionHistory.length > 20) sessionHistory.pop();
        renderHistoryTable();
    }

    function renderHistoryTable() {
        if (!historyTableBody) return;

        if (sessionHistory.length === 0) {
            historyTableBody.innerHTML = `
                <tr class="empty-table-row" id="emptyHistoryRow">
                    <td colspan="7" class="empty-table-cell">
                        <div class="empty-history-box">
                            <i data-lucide="inbox"></i>
                            <span>No recognition history recorded in this session.</span>
                        </div>
                    </td>
                </tr>
            `;
            if (historyCount) historyCount.textContent = '0 entries';
            if (window.lucide) lucide.createIcons();
            return;
        }

        if (historyCount) {
            historyCount.textContent = `${sessionHistory.length} ${sessionHistory.length === 1 ? 'entry' : 'entries'}`;
        }

        historyTableBody.innerHTML = '';
        sessionHistory.forEach((entry, idx) => {
            const tr = document.createElement('tr');
            
            const confClass = entry.confidence >= 80 ? 'pill-high' : entry.confidence >= 55 ? 'pill-mid' : 'pill-low';

            tr.innerHTML = `
                <td style="font-weight: 600; color: var(--text-muted);">${idx + 1}</td>
                <td>
                    <img src="${entry.thumb}" alt="Thumb" class="history-thumb-img">
                </td>
                <td class="history-text-cell" title="${escapeHtml(entry.text)}">
                    ${escapeHtml(entry.text)}
                </td>
                <td>
                    <span class="confidence-pill ${confClass}">${entry.confidence.toFixed(1)}%</span>
                </td>
                <td style="color: var(--text-muted); font-size: 0.85rem;">
                    ${entry.latency} ms
                </td>
                <td style="color: var(--text-muted); font-size: 0.85rem;">
                    ${entry.timestamp}
                </td>
                <td style="text-align: center;">
                    <button class="btn btn-secondary btn-icon-only copy-history-btn" data-index="${idx}" title="Copy Text">
                        <i data-lucide="copy"></i>
                    </button>
                    <button class="btn btn-secondary btn-icon-only delete-history-btn" data-index="${idx}" title="Delete Record">
                        <i data-lucide="trash-2"></i>
                    </button>
                </td>
            `;
            historyTableBody.appendChild(tr);
        });

        // Wire copy and delete events
        document.querySelectorAll('.copy-history-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const i = parseInt(btn.getAttribute('data-index'), 10);
                const rec = sessionHistory[i];
                if (rec) {
                    navigator.clipboard.writeText(rec.text).then(() => {
                        btn.innerHTML = '<i data-lucide="check"></i>';
                        if (window.lucide) lucide.createIcons();
                        setTimeout(() => {
                            btn.innerHTML = '<i data-lucide="copy"></i>';
                            if (window.lucide) lucide.createIcons();
                        }, 1200);
                    });
                }
            });
        });

        document.querySelectorAll('.delete-history-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const i = parseInt(btn.getAttribute('data-index'), 10);
                sessionHistory.splice(i, 1);
                renderHistoryTable();
            });
        });

        if (window.lucide) lucide.createIcons();
    }

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            if (sessionHistory.length === 0) return;
            if (confirm('Clear all session history records?')) {
                sessionHistory = [];
                renderHistoryTable();
            }
        });
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // --- 9. Error Helpers ---
    function showError(msg) {
        if (errorText) errorText.textContent = msg;
        if (errorMessage) errorMessage.classList.remove('hidden');
    }

    function hideError() {
        if (errorMessage) errorMessage.classList.add('hidden');
    }

    if (errorCloseBtn) {
        errorCloseBtn.addEventListener('click', hideError);
    }

    // --- 10. Neural Network Interactive Canvas ---
    function initNeuralCanvas() {
        const canvas = document.getElementById('neuralCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let width = 0;
        let height = 0;
        let particles = [];
        let animationFrameId = null;

        function resize() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
            buildParticles();
        }

        window.addEventListener('resize', resize);

        const particleCount = Math.min(65, Math.max(25, Math.floor(window.innerWidth / 30)));
        const maxDist = 130;

        function buildParticles() {
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    vx: (Math.random() - 0.5) * 0.45,
                    vy: (Math.random() - 0.5) * 0.45,
                    radius: Math.random() * 1.8 + 0.8,
                    baseAlpha: Math.random() * 0.4 + 0.2
                });
            }
        }

        resize();

        function step() {
            ctx.clearRect(0, 0, width, height);

            // Connect neighboring particles
            for (let i = 0; i < particles.length; i++) {
                const p1 = particles[i];
                p1.x += p1.vx;
                p1.y += p1.vy;

                if (p1.x < 0) p1.x = width;
                if (p1.x > width) p1.x = 0;
                if (p1.y < 0) p1.y = height;
                if (p1.y > height) p1.y = 0;

                // Draw node
                ctx.beginPath();
                ctx.arc(p1.x, p1.y, p1.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 229, 255, ${p1.baseAlpha})`;
                ctx.fill();

                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dx = p1.x - p2.x;
                    const dy = p1.y - p2.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < maxDist) {
                        const alpha = (1 - dist / maxDist) * 0.18;
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(0, 229, 255, ${alpha})`;
                        ctx.lineWidth = 0.8;
                        ctx.stroke();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(step);
        }

        step();
    }
});
