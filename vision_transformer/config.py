"""All hyper-parameters live here. Every field can be overridden from the
command line, e.g.  python train.py --data_dir D:\\defactify --batch_size 16"""
import argparse
import os
from dataclasses import dataclass, fields
from datetime import datetime

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # ---------------- data ----------------
    data_dir: str = "C:/7th_sem_project/ai_art_and_image_detector_cl/AI_art_and_image_detector_with_continual_Learning/dataset"       # folder that contains train/ validation/ test/
    # "multiclass": 6-way (real, dalle3, midjourney, sd21, sd3, sdxl). The real-vs-AI
    #               decision is derived from it and reported separately.
    # "binary":     real vs AI-generated only.
    task: str = "multiclass"
    # "crop":   train on random 224 crops of the ORIGINAL pixels (best for AI detection,
    #           resizing wipes out the generator fingerprints).
    # "resize": squash the whole image to 224x224 (faster to decode, usually less accurate).
    preprocess: str = "crop"
    image_size: int = 224
    max_per_class: int = 0          # 0 = use everything; e.g. 300 for a quick smoke test
    jpeg_prob: float = 0.3          # train-time re-compression (robustness + less format shortcut)
    blur_prob: float = 0.1

    # ---------------- loader ----------------
    batch_size: int = 32            # lower to 16 if you run out of GPU memory
    num_workers: int = min(4, os.cpu_count() or 2)   # each worker is a separate process that loads PyTorch;
                                                      # too many -> Windows error 1455 (paging file too small)

    # ---------------- training ----------------
    pretrained: bool = True
    epochs_head: int = 2            # phase 1: backbone frozen
    epochs_fine: int = 10           # phase 2: everything unfrozen (early stopping may end sooner)
    lr_head: float = 3e-4
    lr_fine: float = 5e-5           # LR of the top layer; lower layers get lr * layer_decay^depth
    layer_decay: float = 0.8
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    patience: int = 3               # early stopping on validation macro-F1
    seed: int = 42

    # ---------------- evaluation ----------------
    tta: bool = True                # 5-crop averaging for the FINAL test run only
    skip_test: bool = False         # don't run the test split after training

    # ---------------- bookkeeping ----------------
    output_dir: str = "runs"
    run_name: str = ""
    resume: str = ""                # path of a run folder to continue, e.g. runs\vit_defactify_...

    def __post_init__(self):
        if not self.run_name:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"vit_defactify_{self.task}_{stamp}"

    @property
    def save_dir(self):
        return os.path.join(self.output_dir, self.run_name)

    @property
    def best_path(self):
        return os.path.join(self.save_dir, "best_model.pth")

    @property
    def last_path(self):
        return os.path.join(self.save_dir, "last.pth")

    @property
    def tb_dir(self):
        return os.path.join(self.save_dir, "tensorboard_logs")


def get_config(argv=None):
    parser = argparse.ArgumentParser(description="ViT AI-generated image detector (Defactify)")
    defaults = Config()
    for f in fields(Config):
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    args = parser.parse_args(argv)
    cfg = Config(**vars(args))
    assert cfg.task in ("multiclass", "binary"), "--task must be multiclass or binary"
    assert cfg.preprocess in ("crop", "resize"), "--preprocess must be crop or resize"
    return cfg
