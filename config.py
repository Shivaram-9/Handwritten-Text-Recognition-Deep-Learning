import os

class Config:
    """Base configuration class."""
    # Basic Config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Project Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Model Configuration
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 32
    BATCH_SIZE = 64
    EPOCHS = 50
    MAX_TEXT_LENGTH = 32
    VOCAB_SIZE = 80  # English chars + numbers + punctuation + CTC blank token
    
    # Flask App Config
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
