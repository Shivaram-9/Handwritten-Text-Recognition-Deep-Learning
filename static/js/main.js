/**
 * NeuralText — AI Handwritten Text Recognition Platform
 * Interactive Client Controller & Particle Flow Background Canvas
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Background Neural Canvas Animation ---
    initNeuralCanvas();

    // --- 2. DOM Elements Selection ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const chooseImageBtn = document.getElementById('chooseImageBtn');
    const browseLink = document.getElementById('browseLink');
    
    // Preview Controls
    const previewArea = document.getElementById('previewArea');
    const previewDefaultNote = document.getElementById('previewDefaultNote');
    const imagePreview = document.getElementById('imagePreview');
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
    const gaugeProgressCircle = document.getElementById('gaugeProgressCircle');
    const metricPrep = document.getElementById('metricPrep');
    const metricInfer = document.getElementById('metricInfer');
    const metricTotal = document.getElementById('metricTotal');

    // Result Actions
    const copyBtn = document.getElementById('copyBtn');
    const inlineCopyBtn = document.getElementById('inlineCopyBtn');
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
    const historyTableBody = document.getElementById('historyTableBody');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const sampleRowCopy = document.getElementById('sampleRowCopy');
    const sampleRowDel = document.getElementById('sampleRowDel');

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
            navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        });
    }

    // --- 4. File Selection & Drag & Drop Handling ---
    if (chooseImageBtn) {
        chooseImageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (browseLink) {
        browseLink.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (dropZone) {
        dropZone.addEventListener('click', (e) => {
            if (e.target !== chooseImageBtn && e.target !== browseLink) {
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
            imagePreview.classList.remove('hidden');
            
            // Hide sample handwriting lines inside the sticky note
            const textLines = document.querySelector('.paper-content-lines');
            if (textLines) textLines.classList.add('hidden');
        };
        reader.readAsDataURL(file);
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
        imagePreview.classList.add('hidden');
        const textLines = document.querySelector('.paper-content-lines');
        if (textLines) textLines.classList.remove('hidden');
    }

    // --- 5. Run AI Model Execution ---
    if (recognizeBtn) {
        recognizeBtn.addEventListener('click', runInference);
    }

    async function runInference() {
        if (!currentFile) {
            showError('Please upload a handwriting image first or use the sample.');
            return;
        }

        hideError();
        recognizeBtn.disabled = true;
        loadingState.classList.remove('hidden');
        loadingStatusText.textContent = 'Preprocessing image & normalizing contrast...';

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

            renderPredictionResults(data);

            // Add to session history table
            addToHistory({
                thumb: currentDataUrl,
                text: data.recognized_text || '',
                confidence: data.confidence !== undefined ? data.confidence : 0,
                latency: data.latency && data.latency.total_ms ? data.latency.total_ms : 0,
                timestamp: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) + ' ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            });

            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (err) {
            clearInterval(progressTimer);
            loadingState.classList.add('hidden');
            recognizeBtn.disabled = false;
            showError('Network connectivity error. Please ensure the backend is running.');
            console.error('Inference error:', err);
        }
    }

    // --- 6. Render Prediction Results & Update Radial Gauge ---
    function renderPredictionResults(data) {
        resultSection.classList.remove('hidden');

        // Update transcription text
        recognizedText.textContent = data.recognized_text || '(No text detected)';
        
        // Update confidence
        const conf = typeof data.confidence === 'number' ? data.confidence : 0;
        confidenceScoreText.textContent = `${conf.toFixed(2)}%`;

        // Update SVG radial circle (perimeter = 2 * PI * 50 ~= 314.15)
        if (gaugeProgressCircle) {
            const circumference = 314.15;
            const offset = circumference - (conf / 100) * circumference;
            gaugeProgressCircle.style.strokeDashoffset = offset;
        }

        // Latencies
        if (data.latency) {
            metricPrep.textContent = `${data.latency.preprocessing_ms || 0} ms`;
            metricInfer.textContent = `${data.latency.inference_ms || 0} ms`;
            metricTotal.textContent = `${data.latency.total_ms || 0} ms`;
        }

        // Render 4-step Pipeline Images
        const imgs = data.pipeline_images || {};

        if (imgs.original) {
            visOriginal.src = `data:image/jpeg;base64,${imgs.original}`;
            visOriginal.classList.remove('hidden');
            placeholderStep1.classList.add('hidden');
        } else if (currentDataUrl) {
            visOriginal.src = currentDataUrl;
            visOriginal.classList.remove('hidden');
            placeholderStep1.classList.add('hidden');
        }

        if (imgs.perspective) {
            visPerspective.src = `data:image/jpeg;base64,${imgs.perspective}`;
            visPerspective.classList.remove('hidden');
            placeholderStep2.classList.add('hidden');
        } else if (imgs.preprocessed) {
            visPerspective.src = `data:image/jpeg;base64,${imgs.preprocessed}`;
            visPerspective.classList.remove('hidden');
            placeholderStep2.classList.add('hidden');
        }

        if (imgs.clahe_thresh) {
            visClahe.src = `data:image/jpeg;base64,${imgs.clahe_thresh}`;
            visClahe.classList.remove('hidden');
            placeholderStep3.classList.add('hidden');
        } else if (imgs.preprocessed) {
            visClahe.src = `data:image/jpeg;base64,${imgs.preprocessed}`;
            visClahe.classList.remove('hidden');
            placeholderStep3.classList.add('hidden');
        }

        visLines.innerHTML = '';
        if (Array.isArray(imgs.lines) && imgs.lines.length > 0) {
            placeholderStep4.classList.add('hidden');
            imgs.lines.forEach((b64, idx) => {
                const lineImg = document.createElement('img');
                lineImg.src = `data:image/jpeg;base64,${b64}`;
                lineImg.alt = `Segment ${idx + 1}`;
                lineImg.className = 'cv-step-real-img';
                visLines.appendChild(lineImg);
            });
        } else {
            placeholderStep4.classList.remove('hidden');
        }

        if (window.lucide) lucide.createIcons();
    }

    // --- 7. Result Action Handlers ---
    function setupCopy(btnElement) {
        if (!btnElement) return;
        btnElement.addEventListener('click', () => {
            const text = recognizedText.textContent;
            if (!text) return;

            navigator.clipboard.writeText(text).then(() => {
                const originalHtml = btnElement.innerHTML;
                btnElement.innerHTML = '<i data-lucide="check"></i><span>Copied!</span>';
                if (window.lucide) lucide.createIcons();
                setTimeout(() => {
                    btnElement.innerHTML = originalHtml;
                    if (window.lucide) lucide.createIcons();
                }, 1800);
            }).catch(() => {
                showError('Could not copy to clipboard.');
            });
        });
    }

    setupCopy(copyBtn);
    setupCopy(inlineCopyBtn);

    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            const text = recognizedText.textContent;
            if (!text) return;

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
            recognizedText.textContent = '';
            confidenceScoreText.textContent = '0.00%';
            if (gaugeProgressCircle) gaugeProgressCircle.style.strokeDashoffset = 314.15;
            metricPrep.textContent = '0 ms';
            metricInfer.textContent = '0 ms';
            metricTotal.textContent = '0 ms';
        });
    }

    // Sample row handlers from template
    if (sampleRowCopy) {
        sampleRowCopy.addEventListener('click', () => {
            navigator.clipboard.writeText("Code is like humor. When you have to explain it, it's bad. - Cory House");
        });
    }
    if (sampleRowDel) {
        sampleRowDel.addEventListener('click', function() {
            const row = this.closest('tr');
            if (row) row.remove();
        });
    }

    // --- 8. Session History Table Management ---
    function addToHistory(item) {
        sessionHistory.unshift(item);
        if (sessionHistory.length > 15) sessionHistory.pop();
        renderHistoryTable();
    }

    function renderHistoryTable() {
        if (!historyTableBody) return;

        historyTableBody.innerHTML = '';
        sessionHistory.forEach((entry, idx) => {
            const tr = document.createElement('tr');
            tr.className = 'history-row-item';
            
            const confClass = entry.confidence >= 80 ? 'pill-high' : entry.confidence >= 55 ? 'pill-mid' : 'pill-low';
            const latencySec = (entry.latency / 1000).toFixed(2);
            const previewHtml = entry.thumb 
                ? `<img src="${entry.thumb}" alt="Thumb" class="history-thumb-img">`
                : `<div class="thumb-mini-note"><span>${escapeHtml(entry.text.substring(0, 15))}...</span></div>`;

            tr.innerHTML = `
                <td class="col-num">${idx + 1}</td>
                <td>
                    <div class="history-thumb-wrap">
                        ${previewHtml}
                    </div>
                </td>
                <td class="history-text-col">
                    <span class="recognized-snippet" title="${escapeHtml(entry.text)}">
                        ${escapeHtml(entry.text.length > 40 ? entry.text.substring(0, 40) + '...' : entry.text)}
                    </span>
                </td>
                <td>
                    <span class="confidence-pill ${confClass}">${entry.confidence.toFixed(2)}%</span>
                </td>
                <td class="metric-time-col">${latencySec} s</td>
                <td class="metric-date-col">${entry.timestamp}</td>
                <td class="actions-col">
                    <button type="button" class="btn-table-action copy-dyn-btn" data-index="${idx}" title="Copy Text">
                        <i data-lucide="copy"></i>
                    </button>
                    <button type="button" class="btn-table-action btn-table-del del-dyn-btn" data-index="${idx}" title="Delete Record">
                        <i data-lucide="trash-2"></i>
                    </button>
                </td>
            `;
            historyTableBody.appendChild(tr);
        });

        document.querySelectorAll('.copy-dyn-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const i = parseInt(btn.getAttribute('data-index'), 10);
                const rec = sessionHistory[i];
                if (rec) navigator.clipboard.writeText(rec.text);
            });
        });

        document.querySelectorAll('.del-dyn-btn').forEach(btn => {
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
            if (confirm('Clear all session history records?')) {
                sessionHistory = [];
                historyTableBody.innerHTML = '';
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

    // --- 10. Futuristic Animated Neural Canvas ---
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

        const particleCount = Math.min(75, Math.max(30, Math.floor(window.innerWidth / 28)));
        const maxDist = 140;

        function buildParticles() {
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    radius: Math.random() * 2.0 + 0.9,
                    baseAlpha: Math.random() * 0.45 + 0.25
                });
            }
        }

        resize();

        function step() {
            ctx.clearRect(0, 0, width, height);

            for (let i = 0; i < particles.length; i++) {
                const p1 = particles[i];
                p1.x += p1.vx;
                p1.y += p1.vy;

                if (p1.x < 0) p1.x = width;
                if (p1.x > width) p1.x = 0;
                if (p1.y < 0) p1.y = height;
                if (p1.y > height) p1.y = 0;

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
                        const alpha = (1 - dist / maxDist) * 0.22;
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`;
                        ctx.lineWidth = 0.9;
                        ctx.stroke();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(step);
        }

        step();
    }

    // --- 10. Navigation Links & Scrollspy & Copy Email ---
    const copyEmailBtn = document.getElementById('copyEmailBtn');
    if (copyEmailBtn) {
        copyEmailBtn.addEventListener('click', async () => {
            const email = 'ymvshiva1784@gmail.com';
            try {
                await navigator.clipboard.writeText(email);
                const originalText = copyEmailBtn.innerHTML;
                copyEmailBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                copyEmailBtn.style.borderColor = 'var(--neon-green)';
                copyEmailBtn.style.color = 'var(--neon-green)';
                setTimeout(() => {
                    copyEmailBtn.innerHTML = originalText;
                    copyEmailBtn.style.borderColor = '';
                    copyEmailBtn.style.color = '';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy email:', err);
            }
        });
    }

    // Scrollspy for navigation active pill state
    const navPills = document.querySelectorAll('.nav-pill');
    const sections = ['hero', 'about', 'how-it-works', 'examples', 'contact-section'];

    function updateActiveNav() {
        const scrollPosition = window.scrollY + 200;
        let currentSectionId = '';

        for (const sectionId of sections) {
            const el = document.getElementById(sectionId);
            if (el) {
                const top = el.offsetTop;
                const height = el.offsetHeight;
                if (scrollPosition >= top && scrollPosition < top + height) {
                    currentSectionId = sectionId;
                    break;
                }
            }
        }

        if (currentSectionId) {
            navPills.forEach(pill => {
                const href = pill.getAttribute('href');
                if (href === `#${currentSectionId}`) {
                    pill.classList.add('active');
                } else {
                    pill.classList.remove('active');
                }
            });
        }
    }

    // Smooth scrolling click handler for navigation pills
    navPills.forEach(pill => {
        pill.addEventListener('click', (e) => {
            const href = pill.getAttribute('href');
            if (href && href.startsWith('#')) {
                const targetId = href.substring(1);
                const targetEl = document.getElementById(targetId);
                if (targetEl) {
                    e.preventDefault();
                    targetEl.scrollIntoView({ behavior: 'smooth' });
                    history.pushState(null, null, href);
                    navPills.forEach(p => p.classList.remove('active'));
                    pill.classList.add('active');
                }
            }
        });
    });

    window.addEventListener('scroll', updateActiveNav, { passive: true });
    updateActiveNav();
});
