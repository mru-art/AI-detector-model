# S.C.R.A.M setup guide

Follow these parts in order. By the end you'll have:

- the new code on GitHub, with the website building again
- the detector online, so the site stops saying **Detector offline**
- a model trained on art, telling human-made art from AI-generated art

| Part | What you do | Time |
|---|---|---|
| 1 | Get accounts and tools ready | 15 min |
| 2 | Put the new code on GitHub | 10 min |
| 3 | Try it on your PC (optional) | 15 min |
| 4 | Put the detector online | 20 min |
| 5 | Train the art model in Google Colab | 3–5 h, mostly waiting |
| 6 | Switch the website to the art model | 15 min |

---

## Part 1: Get accounts and tools ready

**Accounts (all free):**

- [ ] **GitHub**, where your repo already is
- [ ] **Hugging Face**, at [huggingface.co/join](https://huggingface.co/join). It hosts the detector online.
- [ ] **Kaggle**, at [kaggle.com](https://www.kaggle.com). Two of the art datasets come from here.
- [ ] **Google**, for Colab and Drive

**On your PC:**

- [ ] **Python 3.11 or newer**: [python.org/downloads](https://www.python.org/downloads/). Tick *Add Python to PATH* when installing.
- [ ] **Git**: [git-scm.com](https://git-scm.com/downloads)
- [ ] **Node.js 20 or newer** (only needed for Part 3): [nodejs.org](https://nodejs.org)

**Tokens (keep these private):**

1. **Hugging Face token:** go to huggingface.co → your avatar → *Settings → Access Tokens → Create new token*. Choose **Write** and copy it.
2. **Kaggle token:** go to kaggle.com → your avatar → *Settings → API → Create New Token*. A `kaggle.json` file downloads. Open it in Notepad; it contains your `username` and `key`.

---

## Part 2: Put the new code on GitHub

The zip has the improved project. Copying files over your repo never deletes old files, so some must be removed by hand.

1. Unzip `AI-detector-model-improved.zip` somewhere, for example `Downloads\AI-detector-model-main`.
2. Copy **everything inside** that folder into your local repo folder and choose *Replace* when asked. Don't copy `node_modules` if you see one.
3. Open a terminal (PowerShell) **in your repo folder** and delete the files that no longer exist:

   ```powershell
   git rm frontend/src/pages/CL-Dashboard.tsx
   git rm frontend/src/components/artwork-inspector.tsx
   git rm backend/api/continual_learning.py
   ```

   If one says *did not match any files*, it's already gone. That's fine.

4. Commit and push:

   ```powershell
   git add -A
   git commit -m "New detector backend, redesigned site, art dataset tools"
   git push
   ```

5. On GitHub, open the **Actions** tab. The *Deploy frontend to GitHub Pages* run should turn green within about 2 minutes.

✅ **Check:** open your GitHub Pages site. It shows the new design. It will still say *Detector offline* until Part 4.

> **Your trained model is not on GitHub.** `.pth` files are excluded from git because they're too big, which is normal.
> Find yours on your PC: `vision_transformer\continual_learning\results\replay\final_model.pth`. You need it in Parts 3 and 4.

---

## Part 3: Try it on your PC (optional)

Skip to Part 4 if you only want the website online.

**Terminal 1: the backend**, in the repo folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
#   no NVIDIA GPU?  pip install torch torchvision
pip install -r local_dev/requirements.txt
uvicorn backend.api.main:app --reload --env-file local_dev/.env
```

Wait for `Detector loaded ...` in the output, then open http://localhost:8000/health. It should say `"status":"ok"`.

**Terminal 2: the website:**

```powershell
cd frontend
npm install
npx vite --config vite.config.local.ts
```

Open http://localhost:5173 and drop an image. The header dot should be blue and say **Detector ready**.

---

## Part 4: Put the detector online

The website needs the backend running somewhere public. You'll use a free **Hugging Face Space**.

1. In the repo folder (with the `.venv` from Part 3 active, or any Python):

   ```powershell
   pip install huggingface_hub
   hf auth login
   ```

   Paste your **Write** token when asked. On an older version, use `huggingface-cli login` instead.

2. Publish, replacing `YOUR_HF_NAME` with your Hugging Face username:

   ```powershell
   python deploy/publish_space.py --space YOUR_HF_NAME/scram-detector
   ```

   It finds `final_model.pth` automatically. If it's somewhere else, add `--model C:\path\to\final_model.pth`.
   The upload is about 330 MB, so give it a few minutes. At the end it prints your API address, like
   `https://YOUR_HF_NAME-scram-detector.hf.space`. **Copy it.**

3. Open `https://huggingface.co/spaces/YOUR_HF_NAME/scram-detector` and wait for **Running**. The first build takes 5–10 minutes. You can watch the *Logs* tab.

4. ✅ **Check:** open `https://YOUR_HF_NAME-scram-detector.hf.space/health`. It should show `"status":"ok"`.

5. **Point the website at it.** In your GitHub repo:
   - *Settings → Secrets and variables → Actions → **Variables** tab → New repository variable*
   - Name: `API_URL`
   - Value: your address from step 2, with no `/` at the end
   - Save, then go to *Actions → Deploy frontend to GitHub Pages → Run workflow*.

6. ✅ **Check:** reload your GitHub Pages site. The header says **Detector ready**. Drop an image to try it.

> The Space must stay **public**, or the website can't reach it.
> Free Spaces sleep after about 2 days without visitors. The first visit after that takes 1–2 minutes, and the site shows a waking-up message.

At this point the site works, but with the **photo** model, which wasn't trained on art. Part 5 fixes that.

---

## Part 5: Train the art model in Google Colab

### 5.1 Prepare Google Drive and Colab

1. In Google Drive, create a folder **`scram`** in *My Drive*.
2. Upload **`AI-detector-model-improved.zip`** into it, so you have `My Drive/scram/AI-detector-model-improved.zip`.
3. Go to [colab.research.google.com](https://colab.research.google.com), choose *File → Upload notebook*, and pick **`train_art_detector_colab.ipynb`**. It's also in the zip under `notebooks/`.
4. Choose *Runtime → Change runtime type → **T4 GPU** → Save*.
5. Add your secrets: click the **🔑 key icon** in the left sidebar, then *Add new secret* for each:

   | Name | Value |
   |---|---|
   | `KAGGLE_USERNAME` | `username` from kaggle.json |
   | `KAGGLE_KEY` | `key` from kaggle.json |
   | `HF_TOKEN` | your Hugging Face **Write** token |

   Turn on **Notebook access** for all three.

### 5.2 Do a quick test first (about 20 minutes)

1. In the **Settings** cell, tick **`QUICK_TEST`** and leave everything else as it is.
2. Choose *Runtime → Run all*. Allow Google Drive access when asked.
3. Watch **step 7**, the dataset import. For each dataset it prints a table like:

   ```
     class                        train  validation        test
     ai_artwork                    4800         600         600
     real                          4800         600         600
   ```

   - ✅ Each dataset shows both `real` and an AI class with numbers above zero.
   - ⚠️ If it says **"images skipped because their label is unclear"** for most of a dataset, stop and send me that output. The folder names didn't match and I'll adjust the importer.

4. When it finishes, **step 10** shows a results table. The numbers will be poor because this is a tiny test. What matters is that it ran to the end.

### 5.3 Do the full training (about 2–4 hours on a T4)

1. In **Settings**, untick `QUICK_TEST`. Optionally fill in **`PUBLISH_SPACE`** with `YOUR_HF_NAME/scram-detector` to put the art model online automatically at the end.
2. Choose *Runtime → Run all*.
3. Keep the browser tab open. Colab disconnects idle tabs.

**If Colab disconnects**, nothing is lost except the class that was training. Reconnect, choose T4 GPU again, and click *Run all*. Downloads repeat, but training **continues from the last finished class**.

**If you see `CUDA out of memory`**, set `BATCH_SIZE` to `16` and run all again.

**If you run out of disk**, untick one dataset (for example `USE_HUMAN_AI_ARTWORK`) or lower `MAX_PER_CLASS` to `4000`.

### 5.4 Read the results (step 10)

| Metric | What it means | A good sign |
|---|---|---|
| AI vs human accuracy | Share of test images labelled correctly as human or AI | above 85% |
| AI vs human ROC-AUC | How well it ranks AI images above human ones (1.0 = perfect) | above 0.90 |
| Exact source accuracy | Named the exact class (human, or which AI dataset) | lower is fine |
| Forgetting | Accuracy lost on earlier classes after learning later ones | under 5 pts |
| Per-class table | Accuracy for each class separately | no class far below the others |

Then try your own images in **step 11**. Use a few human artworks you know and a few AI images.

---

## Part 6: Switch the website to the art model

1. **Put it online.**
   - If you filled in `PUBLISH_SPACE`, **step 13** already did this. Wait for the Space to show *Running* again.
   - Otherwise, download `final_model.pth` and `metrics.json` from `My Drive/scram/results_art_only/replay/` and run:

     ```powershell
     python deploy/publish_space.py --space YOUR_HF_NAME/scram-detector --model C:\path\to\final_model.pth
     ```

2. **Update the dashboard numbers in your repo.** From `My Drive/scram/`, download:
   - `results_art_only/replay/metrics.json` → put it in `vision_transformer/continual_learning/results/replay/metrics.json`, replacing the old one
   - `metrics-snapshot.json` → put it in `frontend/src/data/metrics-snapshot.json`, replacing the old one

   Then push:

   ```powershell
   git add -A
   git commit -m "Art model results"
   git push
   ```

3. ✅ **Check:** after the GitHub Action finishes, the **Model results** page shows the art classes, and the analyzer uses the art model.

---

## Troubleshooting

| You see | What to do |
|---|---|
| **Detector offline** on the website | Check that `API_URL` is set (Part 4, step 5) and that you re-ran the workflow afterwards. Then open `…hf.space/health`. If it doesn't load, check the Space's *Logs* tab. |
| **No model loaded** | The Space is running without a model. Re-run `publish_space.py` with `--model` pointing at your `final_model.pth`. |
| The site keeps saying the detector is waking up | The Space was asleep. Wait 1–2 minutes. If it's still loading after 5 minutes, check the Space logs. |
| GitHub build fails with `Cannot find module '@/components/ui/...'` | The old files are still there. Run the three `git rm` commands in Part 2. |
| The Space build fails | Open the Space's *Logs* tab, copy the last red lines, and send them to me. |
| `publish_space.py` says *Not logged in* | Run `hf auth login` again with a **Write** token. |
| Kaggle download fails with `401` / `403` | Check that `KAGGLE_USERNAME` and `KAGGLE_KEY` are correct and that *Notebook access* is on. |
| AniPrO download asks you to log in | Open the [AniPrO page](https://huggingface.co/datasets/YAN-LIU05/AniPro) while logged in and accept the terms if shown, then run again. Make sure `HF_TOKEN` is set. |
| Colab: `No GPU` | *Runtime → Change runtime type → T4 GPU*. On the free tier GPUs are sometimes unavailable, so try again later. |
| Many *label is unclear* warnings | Send me the output of step 7. The importer needs to know that dataset's folder names. |

---

## Licences

**AniPrO** is CC BY-NC 4.0 and **ART500K** is for non-commercial research only. That's fine for a university project.
For anything commercial, check each dataset's licence first.
