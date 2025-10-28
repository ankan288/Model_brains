# Simple CNN model — small web frontend

This small project hosts a minimal web UI and a Flask backend to let anyone upload an image and get predictions from your `simple_cnn_model.h5` model.

Files added
- `app.py` — Flask server that loads `simple_cnn_model.h5` and provides `/predict`.
- `static/index.html` — Single-page frontend for uploading images and viewing results.
- `requirements.txt` — Python packages required.

Quick start (PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the server:

```powershell
python .\app.py
```

4. Open the frontend in your browser:

http://localhost:5000/

Notes
- The server will attempt to infer the model input size from the loaded model. If the model has an unusual input shape the server falls back to 224x224 RGB.
- Predictions are returned as a JSON object with `predictions` (top entries) and `raw` (full output array).

Deploying a live demo on Railway
--------------------------------

This project is ready to deploy to Railway (or similar Git-based hosts). I added a `Procfile`, `Dockerfile`, and `requirements-pinned.txt` to help deployment.

Quick Railway steps (after you push this repo to GitHub):

1. Create a Railway account at https://railway.app and connect your GitHub account.
2. Click "New Project" -> "Deploy from GitHub" and select this repository.
3. Set the start command (Railway UI) to:

```
gunicorn --workers 4 --bind 0.0.0.0:$PORT app:app
```

4. Add any environment variables (none required by default). If your model files are stored externally, add their URLs as env vars.

5. Deploy. Railway will build the app (it may use the Dockerfile) and provide a public URL that anyone can access.

Notes on models & storage
-------------------------
- If your model files are large, do NOT commit them to a public repo. Use Git LFS or host them on S3/Drive and add code to download them at runtime.
- I included `requirements-pinned.txt` with the exact package versions from the current venv to make deployments reproducible.


If you want class label names, you can edit `app.py` to map indices to strings before returning the JSON.
