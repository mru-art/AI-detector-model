# S.C.R.A.M: Spectral Classification, Replay and Art Masking

S.C.R.A.M estimates whether a piece of art is **human-made or AI-generated**. It also shows which areas of the
image look machine-made, and it keeps learning as new image generators appear, without forgetting the old ones
(continual learning with a replay memory).

```
frontend/            React + Vite site: analyzer, model results dashboard
backend/             FastAPI server: runs the trained model (POST /detect)
vision_transformer/  ViT-B/16 detector: training, evaluation, continual learning, dataset importer
local_dev/           .env and docker-compose for optional Postgres / MinIO
```

## Quick start

You need Python 3.11+, Node 20+ and a trained checkpoint (see *Getting a model*).

```bash
# 1. Backend (from the repo root)
pip install torch torchvision            # GPU build: see https://pytorch.org/get-started/locally/
pip install -r local_dev/requirements.txt
uvicorn backend.api.main:app --reload --env-file local_dev/.env

# 2. Frontend (second terminal)
cd frontend
npm install
npx vite --config vite.config.local.ts   # http://localhost:5173, proxies /api to the backend
```

Open http://localhost:5173 and drop an image. The API docs are at http://localhost:8000/docs.

To work on the UI without a backend, use `VITE_USE_MOCK=true npm run dev`. Results are then made up and clearly labelled.

### Getting a model

The backend uses the newest checkpoint it finds:

- `vision_transformer/continual_learning/results/<method>/final_model.pth` (continual learning), or
- `vision_transformer/runs/<run>/best_model.pth` (offline `train.py`).

Checkpoints are git-ignored (they're about 330 MB), so copy yours in, or point `MODEL_PATH` at one. With no
model, the site still loads, and the analyzer explains that no model is loaded.

### Configuration (all optional)

| Variable | What it does |
|---|---|
| `MODEL_PATH` | Checkpoint to use instead of the newest one |
| `DATABASE_URL` | Save detection history. `sqlite+aiosqlite:///./scram.db` needs no server |
| `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, … | MinIO/S3, for artworks and the protection service |
| `STORE_UPLOADS` | Keep uploaded images in storage (off by default: only a checksum is saved) |
| `CORS_ORIGINS` | Allowed origins, comma-separated (default `*`) |
| `MAX_UPLOAD_MB` | Upload limit (default 15) |
| `MAX_TILES` | Patches scored for the region map (default 16). Lower is faster on CPU |

## Training on art datasets

The original model was trained on Defactify (mostly photos). To make it an **art** detector,
`vision_transformer/art_datasets.py` adds AI-vs-human art datasets to the training folder:

- **Human-made art** goes into the existing `real/` class.
- **AI art** from each dataset becomes a new class. Each new class is learned as its own
  continual-learning experience.

| Source key | Dataset | Becomes |
|---|---|---|
| `anipro` | [AniPrO](https://huggingface.co/datasets/YAN-LIU05/AniPro), 15k anime images | `real`, `anipro_text2image`, `anipro_inpainting` |
| `human-ai-artwork` | [Human-AI Artwork](https://www.kaggle.com/datasets/adamelkholy/human-ai-artwork), 270k images | `real`, `ai_artwork` |
| `real-fake-art` | [Real and Fake (AI-Generated) Art Images](https://www.kaggle.com/datasets/doctorstrange420/real-and-fake-ai-generated-art-images-dataset) | `real`, `ai_art_realfake` |
| `art500k` | [ART500K](https://deepart.hkust.edu.hk/ART500K/art500k.html), human artworks only (request access) | `real` |
| `folder` | Any other dataset labelled by folder names or a CSV | `real`, `ai_art_other` (or `--ai-class NAME`) |

```bash
cd vision_transformer
pip install kagglehub huggingface_hub pyzipper    # downloads; Kaggle needs ~/.kaggle/kaggle.json

# 1. Download
python art_datasets.py download anipro --dest D:/raw/anipro          # also unlocks the encrypted zips
python art_datasets.py download human-ai-artwork --dest D:/raw/haa
python art_datasets.py download real-fake-art --dest D:/raw/rfa
#    ART500K: download it from the HKUST site yourself and extract it.

# 2. Check how images will be labelled (writes nothing)
python art_datasets.py import human-ai-artwork --src D:/raw/haa --dry-run

# 3. Import (hard links when possible, so no extra disk space; safe to re-run)
python art_datasets.py import anipro --src D:/raw/anipro
python art_datasets.py import human-ai-artwork --src D:/raw/haa --max-per-class 15000
python art_datasets.py import real-fake-art --src D:/raw/rfa
python art_datasets.py import art500k --src D:/raw/art500k --max-per-class 20000

# 4. Add the printed CLASS_TO_IDX / NUM_CLASSES / GENERATORS lines to
#    continual_learning/cl_common/config.py, then teach the saved model only the new classes:
python continual_learning/replay/add_generator.py
```

### Training in Google Colab

No GPU at home? Use `notebooks/train_art_detector_colab.ipynb`. It downloads and imports the datasets, trains
one AI class at a time (saved to Google Drive after each, so a disconnect only loses the current class),
shows the metrics, lets you test your own images, and exports the files the app needs.

1. Upload the project zip to `My Drive/scram/`.
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab's 🔑 Secrets panel.
3. Open the notebook in Colab, choose a T4 GPU runtime, and run all the cells. Tick `QUICK_TEST` first for a
   ~20-minute check.

How the importer works:

- Images are labelled from the nearest folder name that clearly says *ai / fake / generated* or
  *human / real / original*. Images with unclear labels are listed and skipped. For a dataset labelled by a
  CSV, pass `--csv labels.csv` (optionally with `--path-col`, `--label-col` and `--ai-value`).
- Each dataset is split 80/10/10 into train, validation and test. Related images stay in the same split:
  for example, an AniPrO original and its inpainted copy never end up on both sides.
- A manifest of every imported file is written to `<dataset>/_imports/<source>.csv`.
- Use `--max-per-class` to keep the classes balanced. The Kaggle set alone is about 270k images.

**Licences.** AniPrO is CC BY-NC 4.0 and ART500K is for non-commercial research only. Check each Kaggle
dataset's licence before using a model trained on it commercially.

## API

| Endpoint | Returns |
|---|---|
| `POST /detect` (multipart `file`) | Verdict, P(AI), confidence, probability per source, per-patch region map, notes |
| `GET /health` | Whether the model is loaded, its classes, and whether the database and storage are on |
| `GET /metrics/continual?method=replay` | Accuracy after every continual-learning experience (dashboard data) |
| `POST /evaluation/run`, `GET /evaluation/{id}` | Candidate-checkpoint evaluation hook |
| `/artworks…` | Only when both a database and storage are configured |

The verdict uses the same 5-crop averaging as `vision_transformer/predict.py`, so the API and the CLI agree. The
region map scores up to 16 patches of the **original** pixels. Resizing would erase the traces generators leave.

After a new training run, refresh the dashboard snapshot the site falls back on when the API is unreachable:

```bash
python -m backend.metrics > frontend/src/data/metrics-snapshot.json
```

## Limitations

- Results are probabilities, not proof.
- Screenshots, resizing and heavy JPEG compression weaken the signals the detector relies on.
- Accuracy drops on generators the model hasn't learned yet.
