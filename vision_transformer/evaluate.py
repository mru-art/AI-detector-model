"""Evaluate a trained model on the test split.
   python evaluate.py --model runs\\<run>\\best_model.pth --data_dir D:\\defactify"""
import argparse
import os
from dataclasses import fields

import torch

from config import Config, DEVICE
from dataset import make_loader
from engine import full_evaluation
from utils import load_checkpoint, save_json
from visualize import save_binary_roc, save_confusion_matrix, save_roc_curve


def run_test(model, cfg, device, out_dir, prefix="test"):
    amp = device.type == "cuda"
    ds, loader = make_loader(cfg, "test", train=False, tta=cfg.tta and cfg.preprocess == "crop")
    print(f"\nTest images: {len(ds)} | TTA(5-crop): {cfg.tta and cfg.preprocess == 'crop'}")
    r = full_evaluation(model, loader, cfg, device, amp, desc="Testing")

    print(f"\n=== TEST RESULTS ({cfg.task}) ===")
    print(r["report"])
    print({k: round(float(v), 4) for k, v in r["metrics"].items()})
    b = r["binary"]
    print("\n=== REAL vs AI-GENERATED ===")
    print({k: round(float(v), 4) for k, v in b.items() if k != "per_source_accuracy"})
    print("per-source accuracy (real = correctly called real, generators = correctly called AI):")
    for name, v in b["per_source_accuracy"].items():
        print(f"  {name:12s} {v:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    save_json({"test_loss": r["loss"], "classification": r["metrics"], "real_vs_ai": b},
              os.path.join(out_dir, f"{prefix}_metrics.json"))
    with open(os.path.join(out_dir, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(r["report"])
    save_confusion_matrix(r["cm"], ds.class_names, os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    save_roc_curve(r["labels"], r["probs"], ds.class_names, os.path.join(out_dir, f"{prefix}_roc_curve.png"))
    save_binary_roc(r["is_ai"], r["ai_score"], os.path.join(out_dir, f"{prefix}_real_vs_ai_roc.png"))
    print(f"\nSaved test results to {out_dir}")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to best_model.pth")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_per_class", type=int, default=0)
    a = p.parse_args()

    device = torch.device(DEVICE)
    model, ckpt = load_checkpoint(a.model, device)
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in known})
    if a.data_dir:
        cfg.data_dir = a.data_dir
    if a.batch_size:
        cfg.batch_size = a.batch_size
    if a.num_workers is not None:
        cfg.num_workers = a.num_workers
    cfg.tta, cfg.max_per_class = a.tta, a.max_per_class
    run_test(model, cfg, device, os.path.dirname(os.path.abspath(a.model)))


if __name__ == "__main__":
    main()
