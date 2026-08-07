import os
import uuid
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from src.predictor import HTRPredictor
from config import Config

# Ensure logs directory exists
os.makedirs(os.path.join(Config.BASE_DIR, 'logs'), exist_ok=True)

# Configure API logging
log_file = os.path.join(Config.BASE_DIR, 'logs', 'api.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HTR_Flask_API")

# Use correct paths for Flask static and templates
app = Flask(__name__, 
            static_folder=os.path.join(Config.BASE_DIR, 'static'),
            template_folder=os.path.join(Config.BASE_DIR, 'templates'))

app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Ensure upload directory exists to prevent IOError on save
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the ML predictor globally to keep the model weights loaded in memory
try:
    logger.info("Initializing HTR Predictor Engine...")
    predictor = HTRPredictor()
    logger.info("HTR Predictor Engine ready.")
except Exception as e:
    logger.error(f"Failed to initialize HTR Predictor Engine: {e}")
    logger.error(traceback.format_exc())
    predictor = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Validates the uploaded file extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """
    Root Endpoint - Serves the Web Frontend
    """
    return render_template('index.html')

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """
    API Documentation Endpoint
    """
    docs = {
        "api_name": "Handwritten Text Recognition API",
        "version": "1.0",
        "endpoints": {
            "GET /api/docs": "Displays API documentation.",
            "POST /predict": "Accepts an image file via multipart/form-data with key 'file'. Returns transcribed text and confidence.",
            "POST /health": "Returns API and ML model health status."
        }
    }
    return jsonify(docs), 200

@app.route('/health', methods=['GET', 'POST'])
def health_check():
    """
    Health Check Endpoint - Verifies that both the Flask server and ML backend are running.
    """
    status = "healthy" if predictor is not None else "degraded"
    response = {
        "status": status,
        "model_loaded": predictor is not None
    }
    return jsonify(response), 200 if status == "healthy" else 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Inference Endpoint - Processes an uploaded image and returns the recognized text.
    """
    if predictor is None:
        return jsonify({"error": "ML Engine not initialized or model weights missing."}), 503
        
    if 'file' not in request.files:
        logger.warning("Upload attempt rejected: No 'file' key in form-data.")
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Upload attempt rejected: Empty filename.")
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filepath = None
        try:
            # Generate a unique, secure filename to prevent race conditions or collisions
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file temporarily
            file.save(filepath)
            logger.info(f"File saved successfully for inference: {filepath}")
            
            # Dispatch to the underlying Deep Learning predictor module
            text, conf = predictor.predict_image(filepath)
            
            if text is None:
                return jsonify({"error": "Failed to process image or image corrupted."}), 422
                
            # Scale confidence to percentage (0-100) and round to 2 decimal places as requested
            confidence_pct = round(conf * 100, 2)
            
            return jsonify({
                "recognized_text": text,
                "confidence": confidence_pct
            }), 200
            
        except Exception as e:
            logger.error(f"Error during prediction API execution: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "Internal server error during processing."}), 500
            
        finally:
            # Cleanup: Always delete the temporary image file to prevent server disk bloat
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Cleaned up temporary file: {filepath}")
                except Exception as cleanup_err:
                    logger.error(f"Failed to cleanup file {filepath}: {cleanup_err}")
    else:
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        return jsonify({"error": "Invalid file type. Allowed: jpg, jpeg, png"}), 415


# Security Headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Global Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    logger.warning("Upload rejected: Exceeded maximum file size.")
    return jsonify({"error": f"File exceeds maximum permitted size of {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB"}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found. Refer to GET / for documentation."}), 404
    
@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "HTTP Method not allowed on this endpoint."}), 405

if __name__ == '__main__':
    # Determine environment
    env = os.environ.get('FLASK_ENV', 'production')
    
    port = getattr(Config, 'PORT', 5000)
    host = getattr(Config, 'HOST', '0.0.0.0')
    
    if env == 'development':
        logger.info(f"Starting Handwritten Text Recognition API in DEVELOPMENT mode on {host}:{port}...")
        app.run(host=host, port=port, debug=True)
    else:
        logger.info(f"Starting Handwritten Text Recognition API in PRODUCTION mode (Waitress) on {host}:{port}...")
        try:
            from waitress import serve
            # Waitress is a production WSGI server for Windows/Linux
            serve(app, host=host, port=port, threads=4)
        except ImportError:
            logger.warning("Waitress not installed. Falling back to Flask dev server. Run 'pip install waitress'.")
            app.run(host=host, port=port, debug=False)
