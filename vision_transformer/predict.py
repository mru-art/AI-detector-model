"""Classify one or more images (or a whole folder).
   python predict.py photo.jpg
   python predict.py folder_with_images --model runs\\<run>\\best_model.pth
   python predict.py photo.jpg --show"""
import argparse
import os

import torch
from PIL import Image

from config import DEVICE
from dataset import IMG_EXT, build_transform
from utils import latest_best_model, load_checkpoint


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in IMG_EXT)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="image file(s) or folder(s)")
    ap.add_argument("--model", default=None, help="best_model.pth (default: newest run in ./runs)")
    ap.add_argument("--show", action="store_true", help="display each image with its prediction")
    ap.add_argument("--no_tta", action="store_true")
    a = ap.parse_args()

    model_path = a.model or latest_best_model()
    if not model_path:
        raise SystemExit("No trained model found. Train first or pass --model path\\to\\best_model.pth")
    device = torch.device(DEVICE)
    model, ckpt = load_checkpoint(model_path, device)
    cfg, names, task = ckpt["config"], ckpt["class_names"], ckpt["task"]
    real_idx = next(i for i, n in enumerate(ckpt["source_names"]) if n.lower() == "real")
    tta = (not a.no_tta) and cfg["preprocess"] == "crop"
    tf = build_transform(cfg["preprocess"], cfg["image_size"], train=False, tta=tta)
    print(f"Model: {model_path} | task: {task} | classes: {names}\n")

    paths = collect(a.paths) or [input("Enter image path: ").strip().strip('"')]
    for path in paths:
        img = Image.open(path).convert("RGB")
        x = tf(img)
        x = x if x.ndim == 4 else x.unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x.to(device)).float(), dim=1).mean(0).cpu()
        # probability that the image is AI-generated
        p_ai = probs[1].item() if task == "binary" else 1.0 - probs[real_idx].item()
        verdict = "AI-GENERATED" if p_ai >= 0.5 else "REAL"
        print(f"{os.path.basename(path)}: {verdict} (P(AI) = {p_ai * 100:.2f}%)")
        if task == "multiclass":
            top = int(probs.argmax())
            print(f"   most likely source: {names[top]} ({probs[top] * 100:.2f}%)")
            print("   " + " | ".join(f"{n}: {p * 100:.1f}%" for n, p in zip(names, probs.tolist())))
        if a.show:
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.title(f"{verdict} - P(AI) {p_ai * 100:.1f}%")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
