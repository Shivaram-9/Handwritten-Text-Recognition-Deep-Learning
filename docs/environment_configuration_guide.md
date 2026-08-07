# Environment Configuration Guide

The Handwritten Text Recognition (HTR) system relies on environment variables to seamlessly switch between local development and cloud production without touching the source code.

## How to use Environment Variables
Create a file named `.env` in the root of the project (copy it from `.env.example`).
If deploying on platforms like AWS, Heroku, or Render, set these variables directly in their respective configuration dashboards.

### Core Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `FLASK_ENV` | `production` | Set to `development` to enable live-reloading and the Flask debugger. |
| `SECRET_KEY` | *None* | Cryptographic key used by Flask. **Must be overridden in production**. |
| `PORT` | `5000` | The port the application binds to. Overwrite this if your host (like Heroku) assigns a dynamic `$PORT`. |
| `HOST` | `0.0.0.0` | The host IP to bind to. `0.0.0.0` exposes it externally; `127.0.0.1` restricts it to local loops. |

### Deep Learning Optimizations

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENABLE_XLA` | `True` | Accelerates matrix multiplication logic via TensorFlow's JIT compiler. Turn to `False` if experiencing memory faults on extremely constrained hardware. |
| `ENABLE_CACHING` | `True` | Caches predictions in an LRU cache to bypass the neural network for identical image requests. |

### Model Architecture Constants
*WARNING: These must exactly match the architecture of the trained `.h5` model.*

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `IMAGE_WIDTH` | `128` | The padded width required by the CNN. |
| `IMAGE_HEIGHT` | `32` | The padded height required by the CNN. |
| `MAX_TEXT_LENGTH` | `32` | Maximum character length supported by the sequence model. |
| `VOCAB_SIZE` | `80` | Number of supported unique characters (including CTC blanks). |

### Security Boundaries

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MAX_CONTENT_LENGTH` | `16777216` | Limits the maximum upload size of an image in bytes (16MB default). Prevents memory overflow attacks. |
