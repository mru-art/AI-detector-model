# SCRAM: Free Deployment Guide

Prepared for Shreyas S. Covers deploying `scram_github_ready.zip` (repo `ai-detector-model`) at no cost.

## Overview

| Part | Where it runs | Cost | How it deploys |
|---|---|---|---|
| Backend (detector and cloak API) | Hugging Face Space (Docker, CPU) | Free | `deploy/publish_space.py` |
| Frontend (website) | GitHub Pages | Free | `.github/workflows/deploy-frontend.yml` on each push to `main` |
| Database and S3 storage | Not needed | n/a | Optional. They switch on only if `DATABASE_URL` or `S3_ENDPOINT_URL` is set |

## What the zip contains

- `ml/masking/`: the art-cloaking code (`cloak.py`, `encoder.py`, `health.py`, `cli.py`, README)
- `backend/api/cloak.py` and `tests/test_masking.py`, `tests/test_cloak_api.py`
- `ml/continual/`, `ml/models/`, `vision_transformer/` (training and inference), `vision_transformer/continual_learning/` (replay, KD, EWC, naive, distill_loop)
- `frontend/`, `deploy/`, `notebooks/`
- Only `vision_transformer/generator_head.pth` (1.6 MB) is included as weights

## Before you push

1. **Check `local_dev/.env`.** It is in the zip and holds `DATABASE_URL` and S3 key and secret variables. If any value is a real credential, delete the file, add `.env` to `.gitignore`, rotate the keys, and ship a `.env.example` with fake values. A pushed secret stays in git history. If they are only local defaults such as `minioadmin`, you can push as is. (I did not see the values.)
2. **Do not commit `best_model.pth`** (about 340 MB). `.gitignore` already blocks `*.pth`, except `generator_head.pth`, which is intended.
3. Add a README line saying where to download `best_model.pth`, for example a Hugging Face model repo or a GitHub Release.

## Is `best_model.pth` needed?

Needed to **run** the system, not to push or import the code.

- Detection: `backend/inference.py` loads it through `find_checkpoint()`. Without it, it raises `ModelUnavailable`.
- Cloaking: `backend/api/cloak.py` and `ml/masking/encoder.py` use the same checkpoint.
- Lookup order: the `MODEL_PATH` environment variable, then `vision_transformer/continual_learning/results/<method>/final_model.pth`, then `vision_transformer/runs/<run>/best_model.pth`.
- `generator_head.pth` is optional. It only adds generator-name prediction.

## Deployment steps

1. Push the repo to GitHub. In Settings → Pages, set the source to **GitHub Actions**.
2. Create a Hugging Face account. Under Settings → Access Tokens, create a token with **Write** access.
3. On your PC:
   ```
   pip install huggingface_hub
   hf auth login
   ```
4. Publish the backend:
   ```
   python deploy/publish_space.py --space YOUR_HF_NAME/scram-detector --model path/to/best_model.pth
   ```
   The upload is about 330 MB. The first build takes 5 to 10 minutes.
5. Check `https://YOUR_HF_NAME-scram-detector.hf.space/health`. It should show `"status":"ok"`.
6. Point the website at the backend (see Problem 1 below), then re-run **Actions → Deploy frontend to GitHub Pages**.
7. Open your GitHub Pages site. The header should say "Detector ready".

Notes: the Space must stay public. Free Spaces sleep after about 2 days without visitors, and the first load after that takes 1 to 2 minutes.

## Two problems to fix first

### Problem 1: the frontend points at the wrong backend

`.github/workflows/deploy-frontend.yml` hardcodes:

```
VITE_API_BASE_URL: https://scram-backend.onrender.com
```

Nothing reads the `API_URL` repository variable that the setup guide and the publish script tell you to create, so the site would stay "Detector offline".

**Fix:** change that line to

```
VITE_API_BASE_URL: ${{ vars.API_URL }}
```

and add a repository variable named `API_URL` (Settings → Secrets and variables → Actions → Variables) set to your Space address.

### Problem 2: the cloak endpoint will fail on the Space

`deploy/publish_space.py` uploads only `backend/`, `vision_transformer/models.py` and `vision_transformer/dataset.py`. It does not upload `ml/`. The server still starts, because `backend/api/cloak.py` imports `ml.masking` only when a request arrives, so detection works. Every cloak request would error.

**Fix:** in the `stage()` function, copy `ml/masking/` (and `ml/__init__.py`) into the Space folder. Then check that `deploy/huggingface/requirements.txt` includes any extra packages the masking code imports. I have not checked what those are, so confirm by running `ml/masking` in a clean environment.

## After deployment

- Retrain and republish: run `publish_space.py` again, and the Space rebuilds with the new model.
- The cloak is optimized against SCRAM's own encoder. The health score is an optimistic upper bound, and protection against other scrapers' models is untested.
- Replace or remove the old fixed-pattern `backend/services/protection.py`.
- The health score's Identity attack always contributes 20 points, which biases the average. Removing it from the average is a pending fix.
- Nothing yet wires the cloak into the frontend upload page, so there is no on-screen health bar.
- Paper deadline: BMS College, **24 November 2026**.

## Source

Based on the contents of `scram_github_ready.zip`: `deploy/publish_space.py`, `deploy/huggingface/Dockerfile`, `.github/workflows/deploy-frontend.yml`, `SETUP_GUIDE.md`, `backend/inference.py`, `backend/api/cloak.py`. I read these files but did not run the code or deploy anything.
