# Deployment Checklist

Before taking this Handwritten Text Recognition (HTR) system live on any cloud provider or bare-metal server, verify the following steps:

## Infrastructure Preparation
- [ ] **Python Version**: Ensure the host server is running Python 3.11.x.
- [ ] **Memory**: The host environment has at least 2GB of RAM (Required for TensorFlow + Model Weights).
- [ ] **Storage**: Adequate disk space is available for saving processed images and logs.

## Application Configuration
- [ ] **Environment Variables**: A `.env` file (or provider configuration) is created based on `.env.example`.
- [ ] **Secret Key**: `SECRET_KEY` is overridden with a cryptographically secure random string.
- [ ] **Flask Environment**: `FLASK_ENV` is set to `production`.
- [ ] **Port Configuration**: `PORT` environment variable matches the exposed port of the hosting provider.

## Dependencies & Pre-computation
- [ ] **Requirements**: Run `pip install -r requirements.txt` cleanly without missing C++ build tools.
- [ ] **Model Weights**: Ensure `models/best_htr_model.h5` exists and is tracked (or mounted via volume if stored externally).

## Architecture & Security
- [ ] **WSGI Server**: Confirm the app initializes via `Waitress` instead of the Flask Dev Server (watch terminal logs during boot).
- [ ] **Reverse Proxy (Optional but Recommended)**: The Waitress server is running safely behind an NGINX or Apache reverse proxy for SSL/HTTPS termination.
- [ ] **File Permissions**: The web server process has write access to the `logs/` and `static/uploads/` directories.
