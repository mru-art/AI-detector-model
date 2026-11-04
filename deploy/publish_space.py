"""Put the detector online as a free Hugging Face Space (Docker, CPU).

    pip install huggingface_hub
    python deploy/publish_space.py --space YOUR_HF_NAME/scram-detector

It uploads the backend, the model and its metrics to the Space. Hugging Face then builds it
(about 5-10 minutes the first time) and serves it at https://YOUR_HF_NAME-scram-detector.hf.space

Then point the website at that address: on GitHub, open Settings -> Secrets and variables ->
Actions -> Variables, add API_URL = https://YOUR_HF_NAME-scram-detector.hf.space, and re-run the
"Deploy frontend" workflow.

Needs a Hugging Face token with WRITE access (huggingface.co -> Settings -> Access Tokens), given
with --token, the HF_TOKEN environment variable, or `huggingface-cli login`.

Run it again whenever you train a better model: the Space rebuilds with the new one.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "huggingface"
MODEL_DEST = Path("vision_transformer/continual_learning/results/replay")


def default_model() -> Path | None:
    for p in (ROOT / "vision_transformer/continual_learning/results/replay/final_model.pth",
              *sorted((ROOT / "vision_transformer/continual_learning/results").glob("*/final_model.pth")),
              *sorted((ROOT / "vision_transformer/runs").glob("*/best_model.pth"))):
        if p.is_file():
            return p
    return None


def space_url(space: str) -> str:
    owner, name = space.split("/", 1)
    host = re.sub(r"[^a-z0-9-]", "-", f"{owner}-{name}".lower())
    return f"https://{host}.hf.space"


def stage(model: Path, metrics: Path | None, out: Path) -> None:
    """Copy exactly what the Space needs into `out`."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.db")
    shutil.copytree(ROOT / "backend", out / "backend", ignore=ignore)
    vit = out / "vision_transformer"
    vit.mkdir(parents=True)
    for f in ("models.py", "dataset.py"):          # the model definition and preprocessing
        shutil.copy2(ROOT / "vision_transformer" / f, vit / f)
    dest = out / MODEL_DEST
    dest.mkdir(parents=True)
    shutil.copy2(model, dest / "final_model.pth")
    if metrics and metrics.is_file():
        shutil.copy2(metrics, dest / "metrics.json")   # powers the dashboard's /metrics/continual
    for f in ("Dockerfile", "requirements.txt", "README.md"):
        shutil.copy2(TEMPLATE / f, out / f)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="YOUR_HF_NAME/space-name, e.g. shreyas/scram-detector")
    ap.add_argument("--model", type=Path, help="checkpoint (.pth). Default: the newest one in the project")
    ap.add_argument("--metrics", type=Path, help="metrics.json for the dashboard. Default: next to the model")
    ap.add_argument("--token", help="Hugging Face write token (default: HF_TOKEN or your saved login)")
    ap.add_argument("--private", action="store_true", help="make the Space private (the website can't reach it then)")
    ap.add_argument("--out", type=Path, help="only build the Space folder here, don't upload")
    a = ap.parse_args(argv)

    model = a.model or default_model()
    if model is None or not model.is_file():
        raise SystemExit("No model found. Pass --model path/to/final_model.pth "
                         "(train one first, e.g. with notebooks/train_art_detector_colab.ipynb).")
    metrics = a.metrics or model.with_name("metrics.json")
    print(f"Model:   {model} ({model.stat().st_size / 1e6:.0f} MB)")
    print(f"Metrics: {metrics if metrics.is_file() else 'none (the dashboard will use its built-in snapshot)'}")

    if a.out:
        if a.out.exists():
            shutil.rmtree(a.out)
        stage(model, metrics, a.out)
        print(f"Space folder built in {a.out}")
        return

    if not a.space or "/" not in a.space:
        raise SystemExit("Pass --space YOUR_HF_NAME/space-name")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    api = HfApi(token=a.token)
    try:
        who = api.whoami()["name"]
    except Exception:
        raise SystemExit("Not logged in to Hugging Face. Pass --token, set HF_TOKEN, or run `huggingface-cli login` "
                         "(the token needs WRITE access).")
    print(f"Logged in as {who}")

    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(model, metrics, Path(tmp))
        print("Uploading (the model is ~330 MB, so this can take a few minutes) ...")
        api.upload_folder(repo_id=a.space, repo_type="space", folder_path=tmp,
                          commit_message=f"Deploy detector with {model.parent.name}/{model.name}",
                          delete_patterns=["backend/**"])  # drop backend files that no longer exist

    url = space_url(a.space)
    print(f"""
Uploaded. Hugging Face is now building the Space (5-10 minutes the first time):
  build log:  https://huggingface.co/spaces/{a.space}
  API:        {url}/health

Last step, point the website at it:
  GitHub repo -> Settings -> Secrets and variables -> Actions -> Variables -> New variable
    Name:  API_URL
    Value: {url}
  then Actions -> "Deploy frontend to GitHub Pages" -> Run workflow.""")


if __name__ == "__main__":
    sys.exit(main())
