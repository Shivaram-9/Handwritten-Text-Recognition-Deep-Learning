# Handwritten Text Recognition Using Deep Learning

## Project Overview
This is a Final Year Engineering Capstone Project focused on recognizing handwritten text from images using Deep Learning techniques. The project implements a robust neural network architecture to transcribe images of handwriting into digital text, featuring a web interface for easy interaction.

## Features
- **Image Upload & Processing:** Upload images of handwritten text for instant transcription.
- **Deep Learning Model:** Utilizes state-of-the-art neural network architectures (e.g., CNN + RNN/LSTM + CTC Loss).
- **Web Interface:** A simple and intuitive Flask-based frontend for inference.
- **High Accuracy:** Preprocessed data and optimized model for improved prediction accuracy.

## Tech Stack
- **Language:** Python 3.11
- **Deep Learning Framework:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Data Manipulation:** NumPy, Pandas
- **Image Processing:** Pillow
- **Machine Learning Utilities:** scikit-learn
- **Web Framework:** Flask
- **Visualization:** Matplotlib
- **Utilities:** tqdm

## Folder Structure
```text
Handwritten-Text-Recognition-Deep-Learning/
├── data/                  # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── models/                # Saved trained models (.h5)
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── src/                   # Main source code (model definition, training scripts)
├── config.py              # Configuration settings for model and Flask app
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignored files and directories
└── README.md              # Project documentation
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shivaram-9/Handwritten-Text-Recognition-Deep-Learning.git
   cd Handwritten-Text-Recognition-Deep-Learning
   ```
2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the server locally or in production:
```bash
python app.py
```
By default, the application runs on `http://0.0.0.0:5000/`. It automatically uses the built-in Flask server in development, and falls back to `waitress` in production.

## Deployment Notes
The application has been audited and prepared for enterprise production deployment. It is fundamentally platform-agnostic and will run cleanly on AWS (EC2/Elastic Beanstalk), Heroku, Google Cloud, Docker, or bare-metal Linux/Windows servers.

**Environment Variables**:
- The system is fully configurable without modifying the code.
- Copy `.env.example` to `.env` or set the variables directly in your deployment provider's dashboard.
- Key configurations include `PORT`, `HOST`, `FLASK_ENV`, `ENABLE_XLA` (TensorFlow Optimization), and `ENABLE_CACHING`.

**Deployment Best Practices**:
1. Ensure your host has sufficient RAM (min 2GB recommended) for loading the Deep Learning model weights.
2. If using Docker, expose the port defined in your `PORT` environment variable (default: `5000`).
3. The server runs safely using `Waitress` as the WSGI server by default in production. You may wrap it in NGINX if SSL/HTTPS termination is required.

## Future Scope
- Support for multiple languages.
- Real-time text recognition via camera feed.
- Continuous model training feedback loop.
