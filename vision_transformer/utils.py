import json
import os
import random

import numpy as np
import torch

from models import create_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True makes cuDNN much slower and gains nothing for a ViT; benchmark speeds up conv_proj
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_experiment_dirs(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=conv)


def save_checkpoint(path, model, cfg, dataset):
    """Self-contained checkpoint: weights + everything predict.py / test.py need."""
    from dataclasses import asdict
    torch.save({
        "model": model.state_dict(),
        "class_names": dataset.class_names,
        "source_names": dataset.source_names,
        "task": cfg.task,
        "config": asdict(cfg),
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = create_model(len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def latest_best_model(runs_dir="runs"):
    cands = [os.path.join(runs_dir, d, "best_model.pth") for d in os.listdir(runs_dir)] \
        if os.path.isdir(runs_dir) else []
    cands = [c for c in cands if os.path.isfile(c)]
    return max(cands, key=os.path.getmtime) if cands else None
