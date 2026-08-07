<div align="center">
  
# ✍️ Handwritten Text Recognition Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-API-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

*An Enterprise-Grade, End-to-End Computer Vision System for digitizing handwritten text via CNNs, Bidirectional LSTMs, and CTC Loss.*

</div>

---

## 📖 Project Overview
This project is an advanced Machine Learning pipeline designed to recognize and transcribe handwritten text from images. By leveraging a deep neural network architecture combining Convolutional Neural Networks (CNNs) for feature extraction and Bidirectional Long Short-Term Memory (BiLSTM) networks for sequence modeling, the system achieves robust transcription accuracy.

It is paired with a production-ready, Glassmorphism-styled Flask web interface, enabling users to upload images and receive millisecond-latency transcriptions natively in the browser.

## ✨ Features
- **🖼️ Drag & Drop UI**: A modern, responsive, and mobile-friendly Glassmorphism web interface.
- **🧠 Deep Learning Engine**: CNN + BiLSTM + Connectionist Temporal Classification (CTC) Decoder.
- **⚡ XLA Optimization**: Accelerated Linear Algebra (JIT compilation) for lightning-fast inference.
- **🛡️ Production Hardened**: Protected against XSS, clickjacking, and OOM-DOS attacks via middleware constraints.
- **🚀 Scalable WSGI Server**: Multi-threaded deployment configuration powered by `Waitress`.
- **📊 Comprehensive Evaluation**: Built-in metrics calculator for Character Error Rate (CER) and Word Error Rate (WER).

## 🛠️ Technology Stack
- **AI/ML Core**: TensorFlow / Keras (Functional API)
- **Computer Vision**: OpenCV, Pillow (PIL)
- **Data Engineering**: NumPy, Pandas, scikit-learn
- **Backend**: Python 3.11, Flask, Waitress
- **Frontend**: HTML5, CSS3, Vanilla JS
- **Testing & QA**: Pytest, Pytest-cov

## 🏗️ System Architecture
The machine learning pipeline flows as follows:
1. **Preprocessing**: Grayscale conversion, Gaussian Blur, Adaptive Thresholding, and normalized padding to `128x32` tensors.
2. **Feature Extraction**: Deep CNN layers with Batch Normalization and MaxPooling extract spatial feature maps.
3. **Sequence Modeling**: The spatial features are squeezed and passed through Bidirectional LSTMs to capture sequential context.
4. **Decoding**: The dense Softmax output is mapped to string text using a Greedy CTC Decoder.

## 📂 Folder Structure
```text
Handwritten-Text-Recognition/
├── app.py                     # Main Flask Server & API Router
├── config.py                  # Global Configuration & Environment Parser
├── src/                       # Machine Learning Source Code
│   ├── cnn_extractor.py
│   ├── rnn_module.py
│   ├── model.py               # Assembles CNN+RNN+CTC
│   ├── predictor.py           # Inference & Caching Engine
│   ├── trainer.py             # Distributed Training Pipeline
│   └── evaluator.py           # CER/WER Metric Calculator
├── tests/                     # Automated QA Suite
├── docs/                      # Deployment Checklists & Audit Reports
├── static/ & templates/       # Frontend UI Assets
└── models/                    # Saved weights (e.g. best_htr_model.h5)
```

## 💻 Installation Guide
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shivaram-9/Handwritten-Text-Recognition-Deep-Learning.git
   cd Handwritten-Text-Recognition-Deep-Learning
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage Guide
To run the server locally or in production:
```bash
python app.py
```
By default, the application runs on `http://0.0.0.0:5000/`. It uses the built-in Flask debugger in development (`FLASK_ENV=development`), and scales up to `waitress` in production.

## 🔌 API Documentation
**Endpoint**: `POST /predict`
- **Content-Type**: `multipart/form-data`
- **Payload**: `file` (Image blob: .png, .jpg, .jpeg)
- **Response**:
```json
{
  "recognized_text": "Hello World",
  "confidence": 98.72
}
```

## ☁️ Deployment Guide
The system is fundamentally platform-agnostic (AWS, GCP, Heroku, Docker).
1. Copy `.env.example` to `.env`.
2. Configure your `SECRET_KEY`, `PORT`, and `HOST`.
3. Start the application. The system will automatically detect the absence of the dev flag and launch the production Waitress WSGI server.
4. *Refer to `docs/deployment_checklist.md` for pre-flight deployment checks.*

## 📸 Screenshots
*(Placeholder: Add screenshots of the dark-mode Glassmorphism UI here)*
- `[Screenshot 1: Landing Page]`
- `[Screenshot 2: Upload & Inference Result]`

## 🔮 Future Enhancements
- 🌍 **Multilingual Support**: Expanding the CTC character map to support localized datasets.
- 📷 **Real-Time Camera Feed**: WebRTC integration for live text scanning.
- 🔄 **Continuous Learning**: User-flagged corrections fed back into a retraining pipeline.

## 🐛 Troubleshooting Guide
- **TensorFlow AVX Warnings**: If your CPU is extremely old, TF may crash. Downgrade TF or compile it from source.
- **Port 5000 in Use**: Change the `PORT` variable in your `.env` file to 8080 or 8000.
- **Waitress Not Found**: Ensure you ran `pip install -r requirements.txt` cleanly.

## ❓ FAQ
**Q: Can it read cursive writing?**
A: Yes. The Bidirectional LSTM is specifically designed to understand contextual sequencing, which excels at cursive chaining.

**Q: Is the API rate limited?**
A: Currently, no. If deploying to the public internet, it is highly recommended to place the Waitress server behind an NGINX reverse proxy configured with strict rate limiting.

## 🤝 Contributing Guide
Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Credits
- **IAM Handwriting Database**: For providing the foundational dataset.
- Developed by **Shivaram-9** as an Engineering Capstone Project.
