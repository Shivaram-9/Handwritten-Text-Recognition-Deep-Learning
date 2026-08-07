document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const removeBtn = document.getElementById('removeBtn');
    const recognizeBtn = document.getElementById('recognizeBtn');
    const loadingState = document.getElementById('loadingState');
    const resultSection = document.getElementById('resultSection');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const recognizedText = document.getElementById('recognizedText');
    const confidenceScore = document.getElementById('confidenceScore');
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    
    // Theme Toggling
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const htmlEl = document.documentElement;
    const themeIcon = document.getElementById('themeIcon');
    
    // Set initial icon
    if(htmlEl.getAttribute('data-theme') === 'dark') {
        themeIcon.setAttribute('data-lucide', 'sun');
    } else {
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
        lucide.createIcons(); // Re-render icon
    });

    // File Management State
    let currentFile = null;

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
            dropZone.classList.add('dragover');
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

    // --- Click to Browse ---
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    // --- File Handling ---
    function handleFiles(files) {
        hideError();
        resultSection.classList.add('hidden');
        
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Validate type
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
            // Hide upload content, show preview
            document.querySelector('.upload-content').classList.add('hidden');
            previewContainer.classList.remove('hidden');
            // Enable recognize button
            recognizeBtn.disabled = false;
            recognizeBtn.classList.remove('disabled');
        }
    }

    // --- Remove File ---
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering browse
        resetUI();
    });

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        document.querySelector('.upload-content').classList.remove('hidden');
        previewContainer.classList.add('hidden');
        recognizeBtn.disabled = true;
        recognizeBtn.classList.add('disabled');
        resultSection.classList.add('hidden');
        hideError();
    }

    // --- API Call ---
    recognizeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State Update
        recognizeBtn.disabled = true;
        recognizeBtn.classList.add('disabled');
        resultSection.classList.add('hidden');
        hideError();
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loadingState.classList.add('hidden');

            if (response.ok) {
                // Success
                recognizedText.textContent = data.recognized_text;
                confidenceScore.textContent = `${data.confidence}%`;
                resultSection.classList.remove('hidden');
            } else {
                // API Error
                showError(data.error || 'Failed to recognize text. Please try again.');
            }

        } catch (err) {
            loadingState.classList.add('hidden');
            showError('Network error. Please check your connection and try again.');
        } finally {
            recognizeBtn.disabled = false;
            recognizeBtn.classList.remove('disabled');
        }
    });

    // --- Action Buttons ---
    copyBtn.addEventListener('click', () => {
        const text = recognizedText.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i data-lucide="check"></i> Copied!';
            lucide.createIcons();
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                lucide.createIcons();
            }, 2000);
        }).catch(err => {
            showError('Failed to copy text.');
        });
    });

    downloadBtn.addEventListener('click', () => {
        const text = recognizedText.textContent;
        const blob = new Blob([text], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `HTR_Result_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    });

    // --- Error Handling ---
    function showError(msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
