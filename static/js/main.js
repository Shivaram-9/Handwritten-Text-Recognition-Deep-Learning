document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const replaceBtn = document.getElementById('replaceBtn');
    const removeBtn = document.getElementById('removeBtn');
    const recognizeBtn = document.getElementById('recognizeBtn');
    const loadingState = document.getElementById('loadingState');
    const resultSection = document.getElementById('resultSection');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const recognizedText = document.getElementById('recognizedText');
    const confidenceScoreText = document.getElementById('confidenceScoreText');
    const confidenceProgressBar = document.getElementById('confidenceProgressBar');
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const historySection = document.getElementById('historySection');
    const historyList = document.getElementById('historyList');
    const historyCount = document.getElementById('historyCount');
    
    // Theme Toggling
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const htmlEl = document.documentElement;
    const themeIcon = document.getElementById('themeIcon');
    
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        htmlEl.setAttribute('data-theme', 'light');
        themeIcon.setAttribute('data-lucide', 'moon');
    }
    
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = htmlEl.getAttribute('data-theme');
        if (currentTheme === 'dark') {
            htmlEl.setAttribute('data-theme', 'light');
            themeIcon.setAttribute('data-lucide', 'moon');
        } else {
            htmlEl.setAttribute('data-theme', 'dark');
            themeIcon.setAttribute('data-lucide', 'sun');
        }
        lucide.createIcons();
    });

    // File Management State
    let currentFile = null;
    let predictionHistory = [];

    // --- Drag and Drop Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            if (!currentFile) dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // Click to Browse (only if no file is selected to avoid conflict with preview buttons)
    dropZone.addEventListener('click', (e) => {
        if(!currentFile && e.target !== replaceBtn && e.target !== removeBtn) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        hideError();
        resultSection.classList.add('hidden');
        
        if (files.length === 0) return;
        
        const file = files[0];
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a JPG or PNG image.');
            return;
        }

        currentFile = file;
        previewFile(file);
    }

    function previewFile(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            recognizeBtn.disabled = false;
        }
    }

    // --- Replace / Remove File ---
    replaceBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUI();
    });

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        uploadContent.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        recognizeBtn.disabled = true;
        resultSection.classList.add('hidden');
        hideError();
    }

    // --- API Call & UI Status ---
    recognizeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Prep
        recognizeBtn.disabled = true;
        resultSection.classList.add('hidden');
        hideError();
        
        const statusText = document.getElementById('loadingStatusText');
        loadingState.classList.remove('hidden');
        
        // Ensure progress bar resets visually immediately
        confidenceProgressBar.style.width = '0%';

        const formData = new FormData();
        formData.append('file', currentFile);
        
        // Real-time status simulation
        const statuses = [
            "Uploading Image...", 
            "Preprocessing via OpenCV...", 
            "Extracting Features (CNN)...", 
            "Sequential Modeling (BiLSTM)...", 
            "Decoding CTC Matrix..."
        ];
        let statusIndex = 0;
        statusText.textContent = statuses[0];
        const statusInterval = setInterval(() => {
            statusIndex++;
            if (statusIndex < statuses.length) {
                statusText.textContent = statuses[statusIndex];
            }
        }, 400);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            clearInterval(statusInterval);

            setTimeout(() => {
                loadingState.classList.add('hidden');

                if (response.ok) {
                    statusText.textContent = "Prediction Complete!";
                    displayResults(data);
                    addToHistory(data.recognized_text, data.confidence);
                } else {
                    showError(data.error || 'Prediction failed. Please try again.');
                }
            }, 600); // Slight delay for UX smoothness

        } catch (err) {
            clearInterval(statusInterval);
            loadingState.classList.add('hidden');
            showError('Network error. Check connection and try again.');
        } finally {
            recognizeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        recognizedText.textContent = data.recognized_text;
        confidenceScoreText.textContent = `${data.confidence}%`;
        
        // Trigger CSS width transition for the bar
        setTimeout(() => {
            confidenceProgressBar.style.width = `${data.confidence}%`;
            // Change color dynamically based on confidence
            if(data.confidence < 60) {
                confidenceProgressBar.style.background = 'linear-gradient(90deg, #EF4444, #F87171)';
            } else if (data.confidence < 85) {
                confidenceProgressBar.style.background = 'linear-gradient(90deg, #F59E0B, #FBBF24)';
            } else {
                confidenceProgressBar.style.background = 'linear-gradient(90deg, var(--success), #34D399)';
            }
        }, 100);

        // Metrics
        if (data.latency) {
            document.getElementById('metricTotal').textContent = `${data.latency.total_ms || 0} ms`;
            document.getElementById('metricPrep').textContent = `${data.latency.preprocessing_ms || 0} ms`;
            document.getElementById('metricInfer').textContent = `${data.latency.inference_ms || 0} ms`;
        }

        resultSection.classList.remove('hidden');
    }

    // --- History Management ---
    function addToHistory(text, confidence) {
        const timestamp = new Date().toLocaleTimeString();
        predictionHistory.unshift({ text, confidence, timestamp });
        
        // Keep only last 5 items to prevent DOM bloat
        if(predictionHistory.length > 5) predictionHistory.pop();
        
        renderHistory();
    }

    function renderHistory() {
        if(predictionHistory.length > 0) {
            historySection.classList.remove('hidden');
            historyCount.textContent = `${predictionHistory.length} items`;
            
            historyList.innerHTML = '';
            predictionHistory.forEach(item => {
                const el = document.createElement('div');
                el.className = 'history-item';
                el.innerHTML = `
                    <div class="history-text" title="${item.text}">${item.text}</div>
                    <div class="history-meta">
                        <span><i data-lucide="check-circle" style="width:14px; height:14px; vertical-align:middle; color:var(--success);"></i> ${item.confidence}%</span>
                        <span><i data-lucide="clock" style="width:14px; height:14px; vertical-align:middle;"></i> ${item.timestamp}</span>
                    </div>
                `;
                // Re-render past result on click (without image for now)
                el.addEventListener('click', () => {
                    recognizedText.textContent = item.text;
                    confidenceScoreText.textContent = `${item.confidence}%`;
                    confidenceProgressBar.style.width = `${item.confidence}%`;
                    window.scrollTo({ top: resultSection.offsetTop - 100, behavior: 'smooth' });
                });
                historyList.appendChild(el);
            });
            lucide.createIcons();
        }
    }

    // --- Action Buttons ---
    copyBtn.addEventListener('click', () => {
        const text = recognizedText.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i data-lucide="check"></i>';
            lucide.createIcons();
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                lucide.createIcons();
            }, 2000);
        });
    });

    downloadBtn.addEventListener('click', () => {
        const text = recognizedText.textContent;
        const blob = new Blob([text], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `NeuralText_Result_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    });

    function showError(msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
    }
    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
