"""Knowledge-distillation continual loop with a small exemplar memory.

To add a generator the model has not seen yet:
  1. The saved model is frozen as the TEACHER.
  2. The model (with one new output) is trained on the new generator's images plus the
     stored exemplars of every old class, with cross-entropy on all labels.
  3. KD keeps the model's answers on the old classes equal to the teacher's, for every image.
  4. A few new images of the generator are stored as its exemplars.

Nothing is retrained from scratch: the teacher is the saved checkpoint, and the memory is a
few hundred image paths per class. Progress is saved after every generator, so an
interrupted run can be resumed by running it again."""
import copy
import json
import os

import torch
from torch.utils.data import ConcatDataset

from cl_common import config as cfg
from cl_common.data import ImageDataset, get_items, make_loader
from cl_common.evaluation import evaluate_seen, print_experience_line
from cl_common.runner import CLError, expand_head, save_results
from cl_common.trainer import train_stage
from distill_loop.memory import ExemplarMemory
from kd.strategy import KD
from models import create_model
from utils import set_seed

NAME = "distill_loop"          # results/distill_loop/ holds the continuing model and its memory


def _paths(split, class_name):
    return [p for p, _ in get_items(split, [class_name])]


def _rebuild_history(metrics_path, learned):
    with open(metrics_path) as f:
        old = json.load(f)
    if len(old["acc"]) != len(learned):
        raise CLError(f"metrics.json has {len(old['acc'])} experiences but the model learned {len(learned)}.")
    return [{"acc": old["acc"][i], "ai_det": old["ai_det"][i], "avg_acc": old["avg_acc"][i],
             "avg_ai_det": old["avg_ai_det"][i], "detailed": (old.get("detailed") or [None] * len(learned))[i]}
            for i in range(len(learned))]


def run_update(source, name=NAME, device=None):
    """Add every generator in config.GENERATORS that the model has not learned yet.

    Continues from results/<name>/ if it exists; otherwise starts from results/<source>/
    (e.g. a model trained with replay_kd) and builds the exemplar memory from that model's
    existing training images. Returns the metrics of the final model."""
    set_seed(cfg.SEED)
    device = torch.device(device or cfg.DEVICE)
    out_dir = os.path.join(cfg.RESULTS_DIR, name)
    memory_path = os.path.join(out_dir, "exemplars.json")

    start = out_dir if os.path.exists(os.path.join(out_dir, "final_model.pth")) else os.path.join(cfg.RESULTS_DIR, source)
    ckpt_path, metrics_path = os.path.join(start, "final_model.pth"), os.path.join(start, "metrics.json")
    if not (os.path.exists(ckpt_path) and os.path.exists(metrics_path)):
        raise CLError(f"No trained model found in {start}. Train one first (e.g. replay_kd\\run.py) "
                      f"or pass the right --from name.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    learned = ckpt.get("generators") or [n for n in ckpt["class_names"] if n != "real"]
    if cfg.GENERATORS[:len(learned)] != learned:
        raise CLError(f"The saved model learned {learned}, but GENERATORS in config.py starts with "
                      f"{cfg.GENERATORS[:len(learned)]}. Keep the old generators in the same order and only append new ones.")
    saved_map = ckpt.get("class_to_idx") or {}
    if any(cfg.CLASS_TO_IDX.get(n) != i for n, i in saved_map.items()):
        raise CLError("CLASS_TO_IDX in config.py changed for an existing class. Only append new classes with new numbers.")
    pending = cfg.GENERATORS[len(learned):]
    if not pending:
        raise CLError(f"Nothing to add: the model already learned every generator in config.py {learned}.")

    history = _rebuild_history(metrics_path, learned)

    # Teacher = the model exactly as it is now. The student gets one extra output per new generator.
    teacher = create_model(len(ckpt["class_names"]), pretrained=False)
    teacher.load_state_dict(ckpt["model"])
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    model = copy.deepcopy(teacher)
    expand_head(model, cfg.NUM_CLASSES)
    model.to(device)

    memory = ExemplarMemory.load(memory_path, cfg.DISTILL_EXEMPLARS_PER_CLASS) if os.path.exists(memory_path) \
        else ExemplarMemory(cfg.DISTILL_EXEMPLARS_PER_CLASS)
    for c in ["real"] + learned:                       # only fills classes that have no exemplars yet
        memory.ensure(c, _paths("train", c), cfg.SEED)
    missing = memory.missing_files()
    if missing:
        raise CLError(f"{len(missing)} exemplar images are missing, e.g. {missing[0]}. "
                      f"Delete {memory_path} to rebuild the memory from the current dataset.")

    print(f"\n########## distillation loop | knows {learned} | adding {pending} | "
          f"{cfg.DISTILL_EXEMPLARS_PER_CLASS} exemplars/class | device: {device} ##########")

    metrics = None
    for gen in pending:
        k = len(learned)
        print(f"\n=== Adding {gen} (experience {k + 1}/{len(cfg.GENERATORS)}) ===")
        new_items = get_items("train", [gen])
        old_items = memory.labelled(cfg.CLASS_TO_IDX)
        old_classes = sorted(cfg.CLASS_TO_IDX[c] for c in ["real"] + learned)

        new_ds = ImageDataset(new_items, train=True)
        if old_items:
            old_ds = ImageDataset(old_items, train=True)
            copies = max(1, len(new_ds) // len(old_items))      # exemplars repeated so old classes stay in every epoch
            loader = make_loader(ConcatDataset([new_ds] + [old_ds] * copies), train=True)
        else:
            loader = make_loader(new_ds, train=True)

        kd = KD()
        kd.teacher = teacher
        kd.old_classes = old_classes
        train_stage(model, loader, kd, cfg.HEAD_EPOCHS, True, f"{gen} head")   # new output row first, backbone frozen
        train_stage(model, loader, kd, cfg.DISTILL_EPOCHS, False, gen)

        memory.ensure(gen, [p for p, _ in new_items], cfg.SEED)   # keep a few of the new generator's images
        memory.save(memory_path)

        learned.append(gen)
        history.append(evaluate_seen(model, k))
        print_experience_line(history[-1], k)

        teacher = copy.deepcopy(model).eval()                    # next update's teacher
        for p in teacher.parameters():
            p.requires_grad = False
        metrics = save_results(name, model, history)            # save after every generator
    return metrics
