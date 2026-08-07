import os
import io
import pytest
from app import app
from config import Config

@pytest.fixture
def client():
    """Sets up the Flask test client."""
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = os.path.join(Config.BASE_DIR, 'tests', 'test_uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.test_client() as client:
        yield client

def test_root_endpoint(client):
    """Test if the root endpoint serves the HTML UI properly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data

def test_api_docs_endpoint(client):
    """Test API docs endpoint."""
    response = client.get('/api/docs')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert "endpoints" in data

def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code in [200, 503]
    assert response.is_json
    data = response.get_json()
    assert "status" in data

def test_empty_upload(client):
    """Test the predict endpoint without providing a file."""
    response = client.post('/predict', data={})
    assert response.status_code == 400
    assert b"No file part" in response.data

def test_invalid_image_type(client):
    """Test uploading a non-image file."""
    data = {
        'file': (io.BytesIO(b"fake data"), 'test.txt')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 415
    assert b"Invalid file type" in response.data

def test_corrupted_image(client):
    """Test uploading a file that pretends to be a PNG but contains garbage data."""
    data = {
        'file': (io.BytesIO(b"this is not a valid png image format data"), 'corrupted.png')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    # Preprocessor should fail and return 422 Unprocessable Entity
    assert response.status_code in [422, 503] # 503 if ML engine not initialized in CI environment
