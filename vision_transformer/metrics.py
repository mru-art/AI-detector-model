import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def calculate_metrics(y_true, y_pred, y_probs, class_names):
    k = len(class_names)
    labels = list(range(k))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if k == 2:
            m["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            m["roc_auc"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:  # e.g. a class missing from a tiny debug subset
        m["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   digits=4, zero_division=0)
    return m, cm, report


def binary_view(source_targets, probs, source_names, task, thr=0.5):
    """Real-vs-AI view of the predictions, whatever the training task was.
    Also returns the detection rate for every individual generator."""
    real_idx = next(i for i, n in enumerate(source_names) if n.lower() == "real")
    is_ai = (source_targets != real_idx).astype(int)
    ai_score = probs[:, 1] if task == "binary" else 1.0 - probs[:, real_idx]
    pred_ai = (ai_score >= thr).astype(int)

    out = {
        "accuracy": accuracy_score(is_ai, pred_ai),
        "precision_ai": precision_score(is_ai, pred_ai, zero_division=0),
        "recall_ai": recall_score(is_ai, pred_ai, zero_division=0),
        "f1_ai": f1_score(is_ai, pred_ai, zero_division=0),
        "real_recall": recall_score(1 - is_ai, 1 - pred_ai, zero_division=0),
    }
    if len(np.unique(is_ai)) == 2:
        out["roc_auc"] = roc_auc_score(is_ai, ai_score)
        out["avg_precision"] = average_precision_score(is_ai, ai_score)
        fpr, tpr, _ = roc_curve(is_ai, ai_score)
        fnr = 1 - tpr
        i = int(np.argmin(np.abs(fnr - fpr)))
        out["eer"] = float((fpr[i] + fnr[i]) / 2)
    per_source = {}
    for i, name in enumerate(source_names):
        sel = source_targets == i
        if sel.any():
            # real: fraction correctly called real; generators: fraction correctly called AI
            per_source[name] = float((pred_ai[sel] == (0 if i == real_idx else 1)).mean())
    out["per_source_accuracy"] = per_source
    return out, is_ai, ai_score
