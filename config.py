import os

# Suppress verbose TensorFlow C++ logs (0 = all, 1 = no INFO, 2 = no INFO/WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Config:
    """Base configuration class."""
    # Basic Config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Project Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Model Configuration
    # Model Configuration
    IMAGE_WIDTH = int(os.environ.get('IMAGE_WIDTH', 128))
    IMAGE_HEIGHT = int(os.environ.get('IMAGE_HEIGHT', 32))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
    EPOCHS = int(os.environ.get('EPOCHS', 50))
    MAX_TEXT_LENGTH = int(os.environ.get('MAX_TEXT_LENGTH', 32))
    VOCAB_SIZE = int(os.environ.get('VOCAB_SIZE', 80))
    
    # Flask App Config
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'static', 'uploads'))
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    
    # Production Optimizations
    ENABLE_XLA = os.environ.get('ENABLE_XLA', 'True').lower() == 'true'
    ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'True').lower() == 'true'
    
    # WSGI Config
    PORT = int(os.environ.get('PORT', 5000))
    HOST = os.environ.get('HOST', '0.0.0.0')
