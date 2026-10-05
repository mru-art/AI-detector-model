import math
import os
import time
from dataclasses import fields

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import DEVICE, Config, get_config
from dataset import make_loader
from engine import full_evaluation, train_epoch
from models import create_model, layerwise_param_groups, set_head_only
from evaluate import run_test
from utils import create_experiment_dirs, save_checkpoint, save_json, set_seed
from visualize import plot_training_curves, save_confusion_matrix, save_roc_curve

HIST_KEYS = ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1", "val_bin_acc"]


def main():
    cfg = get_config()
    blob = None
    if cfg.resume:  # continue an interrupted run
        run_dir = os.path.normpath(cfg.resume)
        blob = torch.load(os.path.join(run_dir, "last.pth"), map_location="cpu", weights_only=False)
        keep = {"data_dir": cfg.data_dir, "num_workers": cfg.num_workers, "batch_size": cfg.batch_size}
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in blob["config"].items() if k in known}, )
        for k, v in keep.items():
            setattr(cfg, k, v)
        cfg.output_dir, cfg.run_name = os.path.split(run_dir)
        cfg.resume = run_dir
        print(f"Resuming {run_dir}: phase {blob['state']['phase']}, epoch {blob['state']['epoch']}")

    set_seed(cfg.seed)
    create_experiment_dirs(cfg)
    device = torch.device(DEVICE)
    amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {amp} | task: {cfg.task} | preprocess: {cfg.preprocess}")
    if not amp:
        print("WARNING: no CUDA GPU found -> training ViT-B/16 on CPU will be extremely slow. "
              "Install the CUDA build of PyTorch (see README).")

    train_ds, train_loader = make_loader(cfg, "train", train=True)
    val_ds, val_loader = make_loader(cfg, "val", train=False)
    assert train_ds.class_names == val_ds.class_names, "train/validation class folders differ"
    class_names = train_ds.class_names
    counts = train_ds.class_counts()
    print(f"Classes: {class_names}\nTrain images per class: {counts.tolist()} | validation: {len(val_ds)}")
    save_json({"class_names": class_names, "source_names": train_ds.source_names, "config": cfg.__dict__},
              os.path.join(cfg.save_dir, "run_info.json"))

    model = create_model(len(class_names), pretrained=cfg.pretrained).to(device)
    if blob:
        model.load_state_dict(blob["model"])

    class_w = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    train_loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    writer = SummaryWriter(cfg.tb_dir)

    state = {"phase": 1, "epoch": 0, "best": -1.0, "bad": 0, "history": {k: [] for k in HIST_KEYS}}
    if blob:
        state = blob["state"]
        scaler.load_state_dict(blob["scaler"])

    def save_last(phase, optimizer, scheduler):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict(), "state": state, "config": cfg.__dict__}, cfg.last_path)

    def fit(phase, n_epochs, optimizer, scheduler):
        name = "Head" if phase == 1 else "Fine"
        hist = state["history"]
        for ep in range(state["epoch"], n_epochs):
            if phase == 2 and state["bad"] >= cfg.patience:
                print("Early stopping (restored from checkpoint).")
                break
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, train_loss_fn, device, scaler,
                                          scheduler, cfg.grad_clip, amp, desc=f"{name} {ep + 1}/{n_epochs}")
            r = full_evaluation(model, val_loader, cfg, device, amp, desc="Validating")
            f1 = r["metrics"]["macro_f1"]
            for k, v in zip(HIST_KEYS, (tr_loss, r["loss"], tr_acc, r["acc"], f1, r["binary"]["accuracy"])):
                hist[k].append(float(v))
            gstep = len(hist["train_loss"])
            for k in HIST_KEYS:
                writer.add_scalar(f"{name}/{k}", hist[k][-1], gstep)
            writer.add_scalar("Metrics/val_mcc", r["metrics"]["mcc"], gstep)
            writer.add_scalar("Metrics/val_auc", r["metrics"]["roc_auc"], gstep)
            writer.add_scalar("LR", optimizer.param_groups[-1]["lr"], gstep)
            print(f"[{name} {ep + 1}/{n_epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {r['loss']:.4f} acc {r['acc']:.4f} macroF1 {f1:.4f} "
                  f"real-vs-AI acc {r['binary']['accuracy']:.4f} | {time.time() - t0:.0f}s")

            if f1 > state["best"]:
                state["best"], state["bad"] = f1, 0
                save_checkpoint(cfg.best_path, model, cfg, train_ds)
                save_json({"epoch": gstep, "classification": r["metrics"], "real_vs_ai": r["binary"]},
                          os.path.join(cfg.save_dir, "val_metrics.json"))
                with open(os.path.join(cfg.save_dir, "val_classification_report.txt"), "w") as f:
                    f.write(r["report"])
                save_confusion_matrix(r["cm"], class_names, os.path.join(cfg.save_dir, "val_confusion_matrix.png"))
                save_roc_curve(r["labels"], r["probs"], class_names, os.path.join(cfg.save_dir, "val_roc_curve.png"))
                print("  -> best model saved")
            elif phase == 2:
                state["bad"] += 1
            plot_training_curves(hist, os.path.join(cfg.save_dir, "training_curves.png"), cfg.epochs_head)
            state["epoch"] = ep + 1
            save_last(phase, optimizer, scheduler)
            if phase == 2 and state["bad"] >= cfg.patience:
                print("Early stopping triggered.")
                break

    # ======================= PHASE 1: head only =======================
    if state["phase"] == 1:
        if cfg.epochs_head > 0:
            print("\n=== PHASE 1: TRAINING HEAD (backbone frozen) ===")
            set_head_only(model, True)
            opt = torch.optim.AdamW(model.heads.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
            if blob and blob["state"]["phase"] == 1:
                opt.load_state_dict(blob["optimizer"])
            fit(1, cfg.epochs_head, opt, None)
        state["phase"], state["epoch"] = 2, 0
        blob = None  # optimizer state of phase 1 must not leak into phase 2

    # ======================= PHASE 2: full fine-tuning =======================
    if cfg.epochs_fine > 0:
        print("\n=== PHASE 2: FULL FINE-TUNING ===")
        set_head_only(model, False)
        opt = torch.optim.AdamW(layerwise_param_groups(model, cfg.lr_fine, cfg.layer_decay, cfg.weight_decay))
        total = cfg.epochs_fine * len(train_loader)
        warm = max(1, int(total * cfg.warmup_ratio))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        if blob and blob["state"]["phase"] == 2:
            opt.load_state_dict(blob["optimizer"])
            sched.load_state_dict(blob["scheduler"])
        fit(2, cfg.epochs_fine, opt, sched)
    writer.close()

    print(f"\nTraining complete. Best validation macro-F1: {state['best']:.4f}\nSaved to: {cfg.save_dir}")

    if not cfg.skip_test and os.path.isfile(cfg.best_path):
        ckpt = torch.load(cfg.best_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        run_test(model, cfg, device, cfg.save_dir)


if __name__ == "__main__":
    main()
