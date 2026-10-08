# Shakira_mam_Real-Time-Deep-Learning-Framework-for-Crack-Detection

M.Tech project for concrete crack detection using a CNN image classifier, with a FastAPI backend and React dashboard.

## Structure

- `backend/` – Python FastAPI server and API endpoints
- `frontend/` – React + Vite client application
- `backend/train_crack_model.py` – trains the crack/no-crack CNN from the local image dataset
- `backend/app/crack_model.py` – CNN architecture and image prediction
- `.gitignore` – ignore generated files

## Train the concrete crack model

The trainer uses the dataset already stored beside this repository at:
`../Dataset/Concrete Crack Images for Classification/`, with `Positive/` (crack) and `Negative/` (no crack) class folders. It creates a reproducible stratified 70%/15%/15% split and saves the best validation checkpoint to `backend/models/crack_classifier.pth`.

From the `ProjectRepository` directory, install the backend requirements and train:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe backend\train_crack_model.py --epochs 15 --batch-size 64
```

The trainer automatically selects CUDA when available, otherwise CPU. It prints per-epoch validation accuracy and final test accuracy. For a quick pipeline check, use `--epochs 1 --batch-size 16`.

After training, start the backend as described below. Submit an image to `POST /api/crack/predict` as multipart form data with the field name `image`; the API returns `crack` or `no_crack` and confidence. The route returns HTTP 503 until the checkpoint has been trained.

The dataset images are stored outside the Git repository and the generated `.pth` model is ignored by Git. See `backend/data/README.md` for dataset citation and license details.

## Backend quick start

```bash
cd ProjectRepository
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend quick start

```bash
cd ProjectRepository\frontend
npm install
npm run dev -- --host 0.0.0.0
```

## App URLs

- Backend: http://localhost:8000/api/health
- Frontend: http://localhost:5173
