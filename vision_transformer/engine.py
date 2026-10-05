import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import binary_view, calculate_metrics


def train_epoch(model, loader, optimizer, loss_fn, device, scaler, scheduler=None,
                grad_clip=1.0, amp=False, desc="Train"):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    params = [p for p in model.parameters() if p.requires_grad]
    loop = tqdm(loader, desc=desc, leave=False)
    for x, y, _ in loop:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (out.argmax(1) == y).sum().item()
        n += bs
        loop.set_postfix(loss=f"{total_loss / n:.4f}", acc=f"{correct / n:.4f}")
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device, amp=False, desc="Eval"):
    """Handles normal batches [B,3,H,W] and 5-crop TTA batches [B,5,3,H,W] (probabilities averaged)."""
    model.eval()
    probs_all, y_all, s_all, loss_sum = [], [], [], 0.0
    for x, y, s in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if x.ndim == 5:
                b, c = x.shape[:2]
                out = model(x.flatten(0, 1)).float()
                probs = F.softmax(out, dim=1).view(b, c, -1).mean(1)
            else:
                probs = F.softmax(model(x).float(), dim=1)
        loss_sum += F.nll_loss(torch.log(probs.clamp_min(1e-8)), y, reduction="sum").item()
        probs_all.append(probs.cpu().numpy())
        y_all.append(y.cpu().numpy())
        s_all.append(np.asarray(s))
    probs = np.concatenate(probs_all)
    labels = np.concatenate(y_all)
    srcs = np.concatenate(s_all)
    preds = probs.argmax(1)
    return {"loss": loss_sum / len(labels), "acc": float((preds == labels).mean()),
            "preds": preds, "labels": labels, "srcs": srcs, "probs": probs}


def full_evaluation(model, loader, cfg, device, amp, desc="Eval"):
    ds = loader.dataset
    r = evaluate(model, loader, device, amp, desc)
    metrics, cm, report = calculate_metrics(r["labels"], r["preds"], r["probs"], ds.class_names)
    binary, is_ai, ai_score = binary_view(r["srcs"], r["probs"], ds.source_names, cfg.task)
    r.update(metrics=metrics, cm=cm, report=report, binary=binary, is_ai=is_ai, ai_score=ai_score)
    return r
