"""run_experiment(strategy, name): learn the 5 experiences one after another with the given
CL strategy, test after each one, and save everything to results/<name>/."""
import json
import os

import torch
from sklearn.metrics import confusion_matrix

from cl_common import config as cfg
from cl_common.data import experience_dataset
from cl_common.evaluation import (backward_transfer, evaluate_seen, forgetting, print_table)
from cl_common.trainer import train_stage
from models import create_model          # your existing ViT-B/16 + 512 -> 256 -> 6 head
from utils import set_seed
from visualize import save_confusion_matrix


def run_experiment(strategy, name):
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    out_dir = os.path.join(cfg.RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n########## {name.upper()} | device: {device} ##########")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found -> ViT-B/16 on CPU is extremely slow.")

    model = create_model(cfg.NUM_CLASSES, pretrained=True).to(device)
    history = []       # one entry per experience: accuracies on all classes seen so far

    for k, gen in enumerate(cfg.GENERATORS):
        print(f"\n=== Experience {k + 1}/{len(cfg.GENERATORS)}: real + {gen} ===")
        data = experience_dataset(k)
        if k == 0:     # very first experience: warm up the brand-new head before touching the backbone
            train_stage(model, strategy.make_loader(data), strategy, cfg.HEAD_EPOCHS, True, "head warm-up")
        train_stage(model, strategy.make_loader(data), strategy, cfg.EPOCHS_PER_EXPERIENCE, False, f"E{k + 1}")

        strategy.after_experience(model, data)      # replay: fill memory | EWC: measure importance | KD: copy teacher

        history.append(evaluate_seen(model, k))
        h = history[-1]
        print(f"  -> after E{k + 1}: average accuracy {h['avg_acc'] * 100:.1f}% | "
              f"per class: " + ", ".join(f"{n} {a * 100:.1f}%" for n, a in h["acc"].items()))

    # ------------------------------------------------------------------ report
    last = len(history) - 1
    report = [print_table("\n6-way accuracy per class (did it name the exact source?)", history, "acc"),
              print_table("\nReal-vs-AI accuracy per class (real called real / generator called AI)", history, "ai_det")]
    summary = {
        "final_avg_acc": history[last]["avg_acc"],
        "final_macro_f1": history[last]["macro_f1"],
        "final_avg_real_vs_ai": history[last]["avg_ai_det"],
        "final_forgetting": forgetting([h["acc"] for h in history], last),
        "final_backward_transfer": backward_transfer([h["acc"] for h in history], last),
    }
    print("\nSUMMARY:", {k: (round(v, 4) if v is not None else None) for k, v in summary.items()})

    metrics = {"method": name, "summary": summary,
               "acc": [h["acc"] for h in history], "ai_det": [h["ai_det"] for h in history],
               "avg_acc": [h["avg_acc"] for h in history], "avg_ai_det": [h["avg_ai_det"] for h in history],
               "forgetting": [forgetting([h["acc"] for h in history], t) for t in range(len(history))]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "tables.txt"), "w") as f:
        f.write("\n".join(report) + f"\n\nSUMMARY: {summary}\n")

    cm = confusion_matrix(history[last]["labels"], history[last]["preds"], labels=list(range(cfg.NUM_CLASSES)))
    save_confusion_matrix(cm, cfg.CLASS_NAMES, os.path.join(out_dir, "confusion_matrix_final.png"))
    torch.save({"model": model.state_dict(), "class_names": cfg.CLASS_NAMES, "source_names": cfg.CLASS_NAMES,
                "task": "multiclass", "config": {"preprocess": cfg.PREPROCESS, "image_size": cfg.IMAGE_SIZE}},
               os.path.join(out_dir, "final_model.pth"))
    print(f"\nSaved results to {out_dir}")
    return metrics
