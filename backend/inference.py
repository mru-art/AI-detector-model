"""Runs the trained ViT detector on an uploaded image.

The model, its preprocessing and its checkpoints all come from ``vision_transformer/``,
so the API gives exactly the same answer as ``python vision_transformer/predict.py``.

Which checkpoint is used
  1. ``MODEL_PATH`` (environment variable), if set.
  2. Otherwise the newest saved model that predict.py would find:
     vision_transformer/continual_learning/results/<method>/final_model.pth
     or vision_transformer/runs/<run>/best_model.pth

What one call returns
  * the verdict, from the same 5-crop test-time augmentation predict.py uses
  * the probability of every class the model knows (real + each generator)
  * a coarse "where does it look AI-made" map: the image is covered by up to
    MAX_TILES 224x224 crops of the ORIGINAL pixels (no resizing, which would wipe
    out generator fingerprints) and each crop gets its own P(AI). Crops with
    P(AI) >= 0.5 are returned as flagged regions.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VIT_DIR = REPO_ROOT / "vision_transformer"

MAX_TILES = int(os.getenv("MAX_TILES", "16"))  # crops for the region map; lower = faster on CPU
MAX_PIXELS = 40_000_000  # refuse decompression bombs

# Friendly names for the class folders. Unknown names fall back to a tidied folder name.
PRETTY_NAMES: dict[str, str] = {
    "real": "Real / human-made",
    "dalle3": "DALL·E 3",
    "midjourney": "Midjourney",
    "sd21": "Stable Diffusion 2.1",
    "sd3": "Stable Diffusion 3",
    "sdxl": "Stable Diffusion XL",
    "nanobanana": "Nano Banana",
    "gpt_image_2": "GPT Image 2",
    "flux": "FLUX",
    "anipro_text2image": "Anime text-to-image (AniPro)",
    "anipro_inpainting": "Anime AI inpainting (AniPro)",
    "ai_artwork": "AI artwork (Human-AI Artwork set)",
    "ai_art_realfake": "AI artwork (Real & Fake Art set)",
    "ai_art_other": "AI artwork (other)",
    "ai_generated": "AI-generated",
}


def pretty_name(name: str) -> str:
    return PRETTY_NAMES.get(name.lower(), name.replace("_", " ").title())


class ModelUnavailable(RuntimeError):
    """No checkpoint could be loaded; the message says why and how to fix it."""


class InvalidImage(ValueError):
    """The upload is not a readable image."""


def _load_module(name: str, path: Path) -> ModuleType:
    # Load vision_transformer/*.py by file path, so their flat module names
    # (models, dataset, config ...) never shadow anything in the backend.
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded_score(model_path: Path) -> float | None:
    import json

    for fname, read in (
        ("metrics.json", lambda j: j["summary"]["final_macro_f1"]),
        ("test_metrics.json", lambda j: j["classification"]["macro_f1"]),
        ("val_metrics.json", lambda j: j["classification"]["macro_f1"]),
    ):
        f = model_path.parent / fname
        if f.is_file():
            try:
                return float(read(json.loads(f.read_text())))
            except Exception:
                continue
    return None


def find_checkpoint() -> Path | None:
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    candidates: list[Path] = []
    cl_root = VIT_DIR / "continual_learning"
    for results in ("results", "results_smoke"):  # smoke-test models only if nothing else exists
        found = sorted((cl_root / results).glob("*/final_model.pth"))
        if found:
            candidates += found
            break
    candidates += sorted((VIT_DIR / "runs").glob("*/best_model.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@dataclass
class ModelInfo:
    loaded: bool = False
    loading: bool = False
    path: str | None = None
    version: str | None = None
    task: str | None = None
    classes: list[str] = field(default_factory=list)
    device: str | None = None
    macro_f1: float | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "version": self.version,
            "task": self.task,
            "classes": [{"id": c, "label": pretty_name(c)} for c in self.classes],
            "device": self.device,
            "macro_f1": self.macro_f1,
            "error": self.error,
        }


class Detector:
    def __init__(self) -> None:
        self.info = ModelInfo()
        self._lock = threading.Lock()
        self._model = None
        self._ckpt: dict[str, Any] | None = None
        self._torch = None
        self._vit_dataset: ModuleType | None = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Load the checkpoint once. Safe to call from several threads."""
        with self._lock:
            if self._model is not None:
                return
            self.info.loading = True
            try:
                self._load_locked()
            except ModelUnavailable as exc:
                self.info.error = str(exc)
                LOGGER.warning("Detector unavailable: %s", exc)
            except Exception as exc:  # broken checkpoint, missing torch ...
                self.info.error = f"Could not load the model: {exc}"
                LOGGER.exception("Detector failed to load")
            finally:
                self.info.loading = False

    def _load_locked(self) -> None:
        path = find_checkpoint()
        if path is None:
            raise ModelUnavailable(
                "No trained model found. Train one (vision_transformer/train.py or a "
                "continual_learning method), or set MODEL_PATH to a .pth checkpoint."
            )
        if not path.is_file():
            raise ModelUnavailable(f"MODEL_PATH points to a file that does not exist: {path}")

        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable(
                "PyTorch is not installed. Install torch and torchvision to run the detector."
            ) from exc

        models = _load_module("scram_vit_models", VIT_DIR / "models.py")
        self._vit_dataset = _load_module("scram_vit_dataset", VIT_DIR / "dataset.py")

        device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        threads = os.getenv("TORCH_THREADS")
        if threads:
            torch.set_num_threads(int(threads))

        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = models.create_model(len(ckpt["class_names"]), pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        self._torch, self._model, self._ckpt = torch, model, ckpt
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        self.info = ModelInfo(
            loaded=True,
            path=str(rel),
            version=f"{path.parent.name}/{path.stem}",
            task=ckpt.get("task", "multiclass"),
            classes=list(ckpt["class_names"]),
            device=device,
            macro_f1=_recorded_score(path),
        )
        LOGGER.info("Detector loaded %s on %s (%d classes)", rel, device, len(ckpt["class_names"]))

    # ------------------------------------------------------------------ inference
    def _probs(self, batch) -> Any:
        torch = self._torch
        assert torch is not None and self._model is not None
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            logits = self._model(batch.to(device)).float()
        return torch.softmax(logits, dim=1).cpu().numpy()

    def analyze(self, data: bytes) -> dict[str, Any]:
        if self._model is None:
            self.load()
        if self._model is None:
            raise ModelUnavailable(self.info.error or "The model is not loaded.")

        import numpy as np
        from PIL import Image, ImageOps, UnidentifiedImageError

        started = time.perf_counter()
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        try:
            with Image.open(io.BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im)  # phone photos: respect orientation
                img = im.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InvalidImage("That file isn’t an image we can read.") from exc

        torch = self._torch
        ds = self._vit_dataset
        assert torch is not None and ds is not None and self._ckpt is not None
        ckpt = self._ckpt
        cfg = ckpt.get("config", {})
        size = int(cfg.get("image_size", 224))
        preprocess = cfg.get("preprocess", "crop")
        task = ckpt.get("task", "multiclass")
        names: list[str] = list(ckpt["class_names"])
        source_names: list[str] = list(ckpt.get("source_names", names))
        real_idx = (
            0 if task == "binary"
            else next((i for i, n in enumerate(names) if n.lower() == "real"), 0)
        )

        def p_ai_of(probs: np.ndarray) -> np.ndarray:
            return probs[:, 1] if task == "binary" else 1.0 - probs[:, real_idx]

        # 1) Verdict: same transform as predict.py (5 crops for "crop" models).
        tta = preprocess == "crop"
        tf = ds.build_transform(preprocess, size, train=False, tta=tta)
        x = tf(img)
        x = x if x.ndim == 4 else x.unsqueeze(0)
        crop_probs = self._probs(x)
        probs = crop_probs.mean(0)
        crop_ai = p_ai_of(crop_probs)
        p_ai = float(probs[1]) if task == "binary" else float(1.0 - probs[real_idx])
        is_ai = p_ai >= 0.5

        # 2) Region map from crops of the original pixels.
        tiles = self._tile_map(img, size, preprocess, p_ai_of) if MAX_TILES > 0 else None

        # 3) Package the answer.
        if task == "binary":
            sources = [
                {"id": "real", "label": pretty_name("real"), "probability": 1 - p_ai, "is_real": True},
                {"id": "ai_generated", "label": pretty_name("ai_generated"), "probability": p_ai, "is_real": False},
            ]
        else:
            sources = [
                {"id": n, "label": pretty_name(n), "probability": float(probs[i]), "is_real": i == real_idx}
                for i, n in enumerate(names)
            ]
        sources.sort(key=lambda s: -s["probability"])

        generator = None
        top_ai = next((s for s in sources if not s["is_real"]), None)
        if is_ai and task != "binary" and top_ai is not None:
            generator = top_ai["label"]

        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum() / math.log(len(probs)))
        agree = int(((crop_ai >= 0.5) == is_ai).sum())
        notes: list[str] = []
        if max(p_ai, 1 - p_ai) < 0.7:
            notes.append("This is close to a coin flip, so treat the verdict with caution.")
        if len(crop_ai) > 1 and agree < len(crop_ai):
            notes.append(
                f"Different parts of the image disagree ({agree} of {len(crop_ai)} crops back the verdict)."
            )
        if is_ai and task != "binary" and top_ai is not None and top_ai["probability"] < 0.6 * p_ai:
            notes.append(
                "The exact generator is uncertain. It may be one this model hasn’t learned yet."
            )
        if task != "binary" and entropy >= 0.85:
            notes.append(
                "The model is unsure overall (its probability is spread across many sources), "
                "so even the AI-vs-human score is unreliable."
            )
        if not is_ai and tiles and any(r["ai_probability"] >= 0.5 for r in tiles["regions"]):
            notes.append(
                "Some areas look AI-made even though the image as a whole looks real. "
                "This can happen with partly edited (inpainted) images."
            )
        if min(img.size) < size:
            notes.append(
                f"The image is smaller than {size}px, so the model had very little detail to work with."
            )

        return {
            "prediction": "AI" if is_ai else "REAL",
            "ai_probability": p_ai,
            "confidence": max(p_ai, 1 - p_ai),
            "generator": generator,
            "sources": sources,
            "uncertainty": entropy,
            "crops": {"agree": agree, "total": int(len(crop_ai))},
            "tiles": tiles,
            "notes": notes,
            "image": {"width": img.width, "height": img.height},
            "model_version": self.info.version or "unknown",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    def _tile_map(self, img, size: int, preprocess: str, p_ai_of) -> dict[str, Any] | None:
        """P(AI) for a regular grid of size x size crops of the original image."""
        import numpy as np

        torch = self._torch
        ds = self._vit_dataset
        assert torch is not None and ds is not None
        if preprocess != "crop":
            return None  # "resize" models only ever see the whole image

        padded = ds.PadToMin(size)(img)
        w, h = padded.size
        per_side = max(1, int(math.isqrt(MAX_TILES)))
        cols = min(per_side, max(1, math.ceil(w / size)))
        rows = min(per_side, max(1, math.ceil(h / size)))
        xs = [0] if cols == 1 else [round(i * (w - size) / (cols - 1)) for i in range(cols)]
        ys = [0] if rows == 1 else [round(j * (h - size) / (rows - 1)) for j in range(rows)]

        norm = ds.T.Compose([ds.T.ToTensor(), ds.T.Normalize(ds.MEAN, ds.STD)])
        boxes = [(x, y) for y in ys for x in xs]
        batch = torch.stack([norm(padded.crop((x, y, x + size, y + size))) for x, y in boxes])
        values: list[float] = []
        for start in range(0, len(batch), 16):
            values += p_ai_of(self._probs(batch[start:start + 16])).tolist()

        # Coordinates relative to the ORIGINAL image (PadToMin centres small images).
        ox, oy = (w - img.width) / 2, (h - img.height) / 2
        regions = []
        for (x, y), v in zip(boxes, values):
            x0, y0 = max(0.0, x - ox), max(0.0, y - oy)
            x1, y1 = min(img.width, x + size - ox), min(img.height, y + size - oy)
            regions.append({
                "x": x0 / img.width, "y": y0 / img.height,
                "w": (x1 - x0) / img.width, "h": (y1 - y0) / img.height,
                "ai_probability": float(v),
            })
        grid = np.array(values).reshape(rows, cols).tolist()
        return {"rows": rows, "cols": cols, "values": grid, "regions": regions}


detector = Detector()
