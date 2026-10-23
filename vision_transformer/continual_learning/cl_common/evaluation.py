"""Evaluation after every experience + the continual-learning metrics."""
import numpy as np
import torch
from sklearn.metrics import f1_score

from cl_common import config as cfg
from cl_common.data import make_loader, seen_test_dataset


@torch.no_grad()
def predict(model, loader):
    device = torch.device(cfg.DEVICE)
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def evaluate_seen(model, k):
    """Test the model on every class seen up to experience k (0-based).
    Two views per class:
      acc    - 6-way accuracy: did it name the exact source (real / dalle3 / ...)?
      ai_det - real-vs-AI accuracy: real must be called real, a generator must be called 'not real'."""
    loader = make_loader(seen_test_dataset(k), train=False)
    preds, labels = predict(model, loader)
    acc, ai_det = {}, {}
    for name in ["real"] + cfg.GENERATORS[:k + 1]:
        mask = labels == cfg.CLASS_TO_IDX[name]
        acc[name] = float((preds[mask] == labels[mask]).mean())
        ai_det[name] = float((preds[mask] == 0).mean() if name == "real" else (preds[mask] != 0).mean())
    return {"acc": acc, "ai_det": ai_det,
            "avg_acc": float(np.mean(list(acc.values()))),       # every seen class counts equally
            "avg_ai_det": float(np.mean(list(ai_det.values()))),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "preds": preds, "labels": labels}


# ----------------------------------------------------------------------------- CL metrics
# acc_matrix[t][name] = accuracy on class `name` after experience t.  (generator j is learned in experience j)
def forgetting(acc_matrix, t):
    """Average over the generators learned before experience t of
       (best accuracy it ever had) - (accuracy now)."""
    if t == 0:
        return None
    drops = []
    for j, gen in enumerate(cfg.GENERATORS[:t]):
        best = max(acc_matrix[s][gen] for s in range(j, t))
        drops.append(best - acc_matrix[t][gen])
    return float(np.mean(drops))


def backward_transfer(acc_matrix, t):
    """Average over old generators of (accuracy now) - (accuracy right after learning it).
    Negative = learning new generators hurt the old ones."""
    if t == 0:
        return None
    return float(np.mean([acc_matrix[t][g] - acc_matrix[j][g] for j, g in enumerate(cfg.GENERATORS[:t])]))


def print_table(title, matrix, key):
    names = ["real"] + cfg.GENERATORS
    head = f"{'':12s}" + "".join(f"{'after E' + str(t + 1):>10s}" for t in range(len(matrix)))
    lines = [title, head]
    for n in names:
        row = "".join(f"{matrix[t][key][n] * 100:9.1f}%" if n in matrix[t][key] else f"{'-':>10s}"
                      for t in range(len(matrix)))
        lines.append(f"{n:12s}{row}")
    avg = "".join(f"{m['avg_' + key] * 100:9.1f}%" for m in matrix)
    lines.append(f"{'AVERAGE':12s}{avg}")
    text = "\n".join(lines)
    print(text)
    return text
