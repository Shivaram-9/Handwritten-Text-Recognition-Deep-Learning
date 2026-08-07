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
*Instructions for running the application and training the model will be added here as the development progresses.*

## Future Scope
- Support for multiple languages.
- Real-time text recognition via camera feed.
- Deployment to cloud platforms (e.g., AWS, Heroku, or GCP) for public access.
- Continuous model training feedback loop.
